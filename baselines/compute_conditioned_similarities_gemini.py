import os
import re
import json
import argparse
import random
import torch
import numpy as np
from transformers import set_seed
from tqdm import tqdm
from PIL import Image
from ratelimit import limits, sleep_and_retry
from google import genai
import time
import cv2
import io
from google.genai import types

def set_all_seeds(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    set_seed(seed)


def read_video_pairs(pairs_jsonl):
    video_pairs = []
    with open(pairs_jsonl, "r") as f:
        for line in f:
            pair = json.loads(line)
            video_pairs.append({
                "video1": pair["video1"],
                "video2": pair["video2"],
            })
    
    return video_pairs


def wait_for_active(client, file_ref, timeout_s=600, poll_interval=50):
    """
    Polls file_ref.name until file.state == ACTIVE.
    Raises an exception if state == FAILED or on timeout.
    """
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        file_ref = client.files.get(name=file_ref.name)  
        state = file_ref.state.name

        if state == "ACTIVE":
            return file_ref
        if state == "FAILED":
            raise RuntimeError(f"Video processing failed: {file_ref.error}")

        print(f"Waiting for {file_ref.name} to become ACTIVE (currently {state})…")
        time.sleep(poll_interval)

    raise TimeoutError(f"Timed out waiting for {file_ref.name} to become ACTIVE")


_ONE_MINUTE = 60

@sleep_and_retry
@limits(calls=5, period=_ONE_MINUTE)
def call_gemini_model(client, contents):
    """
    This function will pause automatically if 15 calls/minute are exceeded.
    """
    resp = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=contents,
    )
    return resp.text



SYSPROMPT_CONCEPT_TEMPLATE = (
    "You are an AI designed to compare two videos based on the visual concept of '{concept_name}'.\n\n"
    "The input consists of a sequence of concatenated frames: the first half represents Video 1, and the second half represents Video 2.\n"
    "Your task is to evaluate how similar these two videos are with respect to the concept '{concept_name}'.\n"
    "Output a single similarity score between 1 and 5, where 1 means completely different and 5 means perfectly similar in terms of '{concept_name}'.\n"
    "Do not explain your reasoning. Only output the numerical score.\n"
)


def extract_frame_as_bytes(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video file: {video_path}")

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    middle_frame_index = frame_count // 2
    cap.set(cv2.CAP_PROP_POS_FRAMES, middle_frame_index)

    success, frame = cap.read()
    cap.release()

    if not success:
        raise ValueError(f"Could not read frame from: {video_path}")

    # Convert to RGB and save to a BytesIO object
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(frame_rgb)
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG")
    buffer.seek(0)
    return buffer


def uniform_sample(l, n):
    gap = len(l) / n
    idxs = [int(i * gap + gap / 2) for i in range(n)]
    return [l[i] for i in idxs]


def parse_answer_to_score(answer):
    """
    Parse the answer from the model to a score.
    """
    answer = answer.split("\n")[0]
    try:
        score = float(answer)
    except ValueError:
        print(f"Could not parse answer: {answer}")
        score = 0.0
    return score

def compute_vqa_based_similarity(video_pairs, args, output_file):


    prompt = SYSPROMPT_CONCEPT_TEMPLATE.format(concept_name=re.sub(r'(?<!^)(?=[A-Z])', ' ', args.concept_name))
    print(f"\033[93mUsing system prompt\n\n{prompt}\033[0m")
    print(f"\033[92mConcept: {re.sub(r'(?<!^)(?=[A-Z])', ' ', args.concept_name)}\033[0m")
    
    already_computed = []
    if os.path.exists(output_file):
        with open(output_file, "r") as f:
            already_computed = json.load(f)
    
    video_pairs_sim = []
    for video_pair in tqdm(video_pairs):
        
        # check if video pair (e.g. {'video1': 'videos_trimmed/education/1754_007.mp4', 
        # 'video2': 'videos_trimmed/education/2385_006.mp4'} ) is already computed
        if any(
            video_pair["video1"] == pair["video1"] and video_pair["video2"] == pair["video2"]
            for pair in already_computed
        ):  
            # add the already computed similarity to the list
            video_pairs_sim.append(
                {
                    "video1": video_pair["video1"],
                    "video2": video_pair["video2"],
                    "similarity": next(
                        pair["similarity"]
                        for pair in already_computed
                        if pair["video1"] == video_pair["video1"]
                        and pair["video2"] == video_pair["video2"]
                    ),
                }
            )
            print(f"Already computed similarity for {video_pair['video1']} and {video_pair['video2']}, skipping...")
            continue


        client = genai.Client(api_key=args.apikey)            

        if not args.single_frame:
            video1 = client.files.upload(
                file=os.path.join(args.video_dir, video_pair["video1"])
            )
            video2 = client.files.upload(
                file=os.path.join(args.video_dir, video_pair["video2"])
            )

            video1 = wait_for_active(client, video1)
            video2 = wait_for_active(client, video2)
            answer = call_gemini_model(
                client,
                contents=[
                    video1,
                    video2,
                    prompt,
                ]
            )
        else:
            frame1_bytes = extract_frame_as_bytes(os.path.join(args.video_dir, video_pair["video1"]))
            frame2_bytes = extract_frame_as_bytes(os.path.join(args.video_dir, video_pair["video2"]))
            
            parts = [
                 types.Part.from_bytes(data=frame1_bytes.getvalue(), mime_type="image/jpeg"),
                 types.Part.from_bytes(data=frame2_bytes.getvalue(), mime_type="image/jpeg"),
                 prompt  
            ]                
            answer = call_gemini_model(
                client,
                contents=parts
            )

        sim = parse_answer_to_score(answer[0])
        video_pairs_sim.append(
            {
                "video1": video_pair["video1"],
                "video2": video_pair["video2"],
                "similarity": sim,
            }
        )

        
        with open(output_file, "w") as f:
            json.dump(video_pairs_sim, f, indent=4)
            
    
    return video_pairs_sim



def main(args):

    set_all_seeds(42)
    print(f"\033[93mComputing similarities with model: {args.vqa_model_path}\033[0m")
    video_pairs = read_video_pairs(args.pairs_jsonl)

    concept_name = re.sub(r'(?<!^)(?=[A-Z])', ' ', args.concept_name).replace(" ", "_")
    output_dir = os.path.join(args.conditioned_similarity_dir, concept_name, args.vqa_model_path.split('/')[-1])
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_file = os.path.join(
        output_dir, f"{args.vqa_model_path.split('/')[-1]}_{concept_name}_similarities.json"
    )

    video_pairs_sim = compute_vqa_based_similarity(video_pairs, args, output_file)
    

    with open(output_file, "w") as f:
        json.dump(video_pairs_sim, f, indent=4)
    print(f"\033[92mSimilarities saved to: {output_file}\033[0m")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--video_dir",
        type=str,
        default="convisbench/",
        help="Directory where videos are stored.",
    )
    parser.add_argument(
        "--vqa_model_path",
        type=str,
        default="Gemini-2.0-Flash",
        required=False,
    )
    parser.add_argument(
        "--conditioned_similarity_dir",
        type=str,
        default="baselines/computed_conditioned_similarities/",
        required=False,
    )
    parser.add_argument(
        "--apikey",
        type=str,
        default="",
        required=True,
    )

    parser.add_argument(
        "--pairs_jsonl",
        type=str,
        default="convisbench/ConVIS.jsonl",
        required=False,
    )

    parser.add_argument(
        "--concept_name",
        type=str,
        default="Location",
        required=True,
    )
    parser.add_argument(
        "--fps_factor",
        type=int,
        default=1,
        required=False,
    )
    parser.add_argument(
        "--max_frames_num",
        type=int,
        default=16,
        required=False,
    )
    parser.add_argument(
        "--single_frame",
        type=bool,
        default=False,
        required=False,
    )
    args = parser.parse_args()

    main(args)
