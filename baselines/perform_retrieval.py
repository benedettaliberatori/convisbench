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
from decord import VideoReader, cpu
from models.mplugowl3_model import mPLUGOwl3Model
from models.internvl2_5_model import InternVL2Model
import warnings

warnings.filterwarnings("ignore", category=UserWarning)


def set_all_seeds(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    set_seed(seed)


def read_video_sets(sets_json):
    with open(sets_json, "r") as f:
        video_sets_json = json.load(f)

    video_sets = []
    for k, v in video_sets_json.items():
        video_sets.append(
            {"anchor": "videos/" + k, "targets": ["videos/" + t for t in v]}
        )

    return video_sets


SYSPROMPT_CONCEPT_TEMPLATE = (
    "You are an AI designed to compare two videos based on the visual concept of '{concept_name}'.\n\n"
    "The input consists of a sequence of concatenated frames: the first half represents Video 1, and the second half represents Video 2.\n"
    "Your task is to evaluate how similar these two videos are with respect to the concept '{concept_name}'.\n"
    "Output a single similarity score between 1 and 5, where 1 means completely different and 5 means perfectly similar in terms of '{concept_name}'.\n"
    "Do not explain your reasoning. Only output the numerical score.\n"
)


def load_model(model_path):
    """
    Load the Video-LLM model based on the model path.
    """
    if model_path in ["Qwen/Qwen2.5-VL-7B-Instruct", "Qwen/Qwen2.5-VL-3B-Instruct"]:
        from models.qwen2_vl_model import Qwen2VLModel

        model_vqa = Qwen2VLModel(model_name=args.vqa_model_version)
        model_vqa.model.eval()
    elif model_path == "mPLUG/mPLUG-Owl3-7B-240728":
        model_vqa = mPLUGOwl3Model()
        model_vqa.model.eval()
    elif model_path in [
        "OpenGVLab/InternVL2_5-4B",
        "OpenGVLab/InternVL2_5-8B",
        "OpenGVLab/InternVL3-8B",
    ]:
        model_vqa = InternVL2Model(model_name=args.vqa_model_version)
        model_vqa.model.eval()
    elif args.vqa_model_path in [
        "lmms-lab/llava-onevision-qwen2-7b-ov",
        "lmms-lab/llava-onevision-qwen2-0.5b-ov",
        "lmms-lab/LLaVA-Video-7B-Qwen2",
    ]:
        from models.llavaov_model import LLaVAOneVisionModel

        model_vqa = LLaVAOneVisionModel(model_name=args.vqa_model_version)
        model_vqa.model.eval()
    else:
        raise NotImplementedError(f"Model {model_path} currently not implemented")
    return model_vqa


def approximate_smart_resize(height, width, image_factor, min_pixels, max_pixels):
    """
    Function used to have the same preprocessing of Qwen
    when model takes in input frames and not videos in .mp4 format.
    """
    target_pixels = max(min_pixels, min(max_pixels, height * width // image_factor))
    aspect_ratio = width / height
    new_height = int((target_pixels / aspect_ratio) ** 0.5)
    new_width = int(new_height * aspect_ratio)
    return new_width, new_height


def uniform_sample(l, n):
    gap = len(l) / n
    idxs = [int(i * gap + gap / 2) for i in range(n)]
    return [l[i] for i in idxs]


def encode_concat_video_Qwen(video1_path, video2_path, fps_factor=1, max_frames_num=16):
    """
    Load and encode two videos, salmpe uniformly max_frames_num frames / 2 from each video.
    """
    video1_path = os.path.join(args.video_dir, video1_path)
    video2_path = os.path.join(args.video_dir, video2_path)

    max_frames_num_per_video = max_frames_num // 2
    all_frames = []
    for video_path in [video1_path, video2_path]:
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file {video_path} not found.")

        vr = VideoReader(video_path, ctx=cpu(0))
        sample_fps = max(
            1, round(vr.get_avg_fps() / fps_factor)
        )  # sample 'fps_factor' frames per second
        frame_idx = [i for i in range(0, len(vr), sample_fps)]

        if len(frame_idx) > max_frames_num_per_video:
            frame_idx = uniform_sample(frame_idx, max_frames_num_per_video)

        frames = vr.get_batch(frame_idx).asnumpy()
        frames = [Image.fromarray(v.astype("uint8")) for v in frames]

        orig_width, orig_height = frames[0].size

        IMAGE_FACTOR = 28
        VIDEO_MIN_PIXELS = 128 * 28 * 28
        VIDEO_MAX_PIXELS = 768 * 28 * 28
        FRAME_FACTOR = 2
        VIDEO_TOTAL_PIXELS = 640 * 360 * 28 * 28

        nframes = len(frames)
        max_pixels = max(
            min(VIDEO_MAX_PIXELS, VIDEO_TOTAL_PIXELS / nframes * FRAME_FACTOR),
            int(VIDEO_MIN_PIXELS * 1.05),
        )

        # Compute target size
        target_width, target_height = approximate_smart_resize(
            orig_height,
            orig_width,
            image_factor=IMAGE_FACTOR,
            min_pixels=VIDEO_MIN_PIXELS,
            max_pixels=max_pixels,
        )

        # Resize all frames uniformly
        frames = [
            frame.resize((target_width, target_height), Image.BICUBIC)
            for frame in frames
        ]

        if len(frames) < max_frames_num_per_video:
            frames.extend([frames[-1]] * (max_frames_num_per_video - len(frames)))
        all_frames.extend(frames)

    return all_frames


def encode_concat_video_mPLUG(
    video1_path, video2_path, fps_factor=1, max_frames_num=16
):
    """
    Load and encode two videos, salmpe uniformly max_frames_num frames / 2 from each video.
    """
    video1_path = os.path.join(args.video_dir, video1_path)
    video2_path = os.path.join(args.video_dir, video2_path)

    max_frames_num_per_video = max_frames_num // 2
    all_frames = []
    for video_path in [video1_path, video2_path]:
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file {video_path} not found.")

        vr = VideoReader(video_path, ctx=cpu(0))
        sample_fps = max(
            1, round(vr.get_avg_fps() / fps_factor)
        )  # sample 'fps_factor' frames per second
        frame_idx = [i for i in range(0, len(vr), sample_fps)]

        if len(frame_idx) > max_frames_num_per_video:
            frame_idx = uniform_sample(frame_idx, max_frames_num_per_video)

        frames = vr.get_batch(frame_idx).asnumpy()
        frames = [Image.fromarray(v.astype("uint8")) for v in frames]

        if len(frames) < max_frames_num_per_video:
            frames.extend([frames[-1]] * (max_frames_num_per_video - len(frames)))
        all_frames.extend(frames)

    return all_frames


def encode_concat_video_InternVL(
    model_vqa, video1_path, video2_path, fps_factor, max_frames_num
):

    video1_path = os.path.join(args.video_dir, video1_path)
    video2_path = os.path.join(args.video_dir, video2_path)

    video_frames1 = model_vqa.encode_video(video1_path, fps_factor, max_frames_num)
    video_frames2 = model_vqa.encode_video(video2_path, fps_factor, max_frames_num)

    video_frame1, video_num_patches1 = video_frames1
    video_frame2, video_num_patches2 = video_frames2

    video_frames = torch.cat([video_frame1, video_frame2], dim=0)
    video_num_patches = video_num_patches1 + video_num_patches2
    video_frames = [video_frames, video_num_patches]
    return video_frames


def encode_concat_video_LLavaOV(model_vqa, video1_path, video2_path, max_frames_num):

    video1_path = os.path.join(args.video_dir, video1_path)
    video2_path = os.path.join(args.video_dir, video2_path)

    video_frames1 = model_vqa.load_images([video1_path], max_frames_num)
    video_frames2 = model_vqa.load_images([video2_path], max_frames_num)
    video_frames = torch.cat([video_frames1[0], video_frames2[0]], dim=0)
    return video_frames


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


def compute_vqa_based_similarity(video_sets, args, output_file):

    model_vqa = load_model(args.vqa_model_path)
    prompt = SYSPROMPT_CONCEPT_TEMPLATE.format(
        concept_name=re.sub(r"(?<!^)(?=[A-Z])", " ", args.concept_name)
    )

    print(f"\033[93mUsing system prompt\n\n{prompt}\033[0m")
    print(
        f"\033[92mConcept: {re.sub(r'(?<!^)(?=[A-Z])', ' ', args.concept_name)}\033[0m"
    )

    already_computed = []
    if os.path.exists(output_file):
        with open(output_file, "r") as f:
            already_computed = json.load(f)

    video_ranking_sim = {}
    for video_set in tqdm(video_sets):

        anchor = video_set["anchor"]
        targets = video_set["targets"]

        generation_method_name = "generate"
        generation_method = getattr(model_vqa, generation_method_name)

        for target in targets:
            if args.vqa_model_path == "mPLUG/mPLUG-Owl3-7B-240728":
                concat_video_frames = encode_concat_video_mPLUG(
                    anchor, target, args.fps_factor, args.max_frames_num
                )

                answer = model_vqa.generate(
                    [concat_video_frames],
                    max_new_tokens=1,
                    use_system_prompt=True,
                    conditioned_system_prompt=prompt,
                )

            elif args.vqa_model_path in [
                "Qwen/Qwen2.5-VL-7B-Instruct",
                "Qwen/Qwen2.5-VL-3B-Instruct",
            ]:
                concat_video_frames = encode_concat_video_Qwen(
                    anchor, target, args.fps_factor, args.max_frames_num
                )

                processed_data = [{"type": "video", "video": concat_video_frames}]
                answer = generation_method(
                    processed_data,
                    max_new_tokens=1,
                    use_system_prompt=True,
                    conditioned_system_prompt=prompt,
                )

            elif args.vqa_model_path in [
                "OpenGVLab/InternVL2_5-4B",
                "OpenGVLab/InternVL2_5-8B",
                "OpenGVLab/InternVL3-8B",
            ]:

                concat_video_frames = encode_concat_video_InternVL(
                    model_vqa, anchor, target, args.fps_factor, args.max_frames_num
                )
                answer = model_vqa.generate(
                    concat_video_frames,
                    max_new_tokens=1,
                    use_system_prompt=True,
                    conditioned_system_prompt=prompt,
                )

            elif args.vqa_model_path in [
                "lmms-lab/llava-onevision-qwen2-7b-ov",
                "lmms-lab/llava-onevision-qwen2-0.5b-ov",
                "lmms-lab/LLaVA-Video-7B-Qwen2",
            ]:

                concat_video_frames = encode_concat_video_LLavaOV(
                    model_vqa, anchor, target, args.max_frames_num
                )
                answer = model_vqa.generate(
                    concat_video_frames,
                    max_new_tokens=1,
                    use_system_prompt=True,
                    conditioned_system_prompt=prompt,
                )
            else:
                raise NotImplementedError(
                    f"Model {args.vqa_model_path} currently not implemented"
                )

            sim = parse_answer_to_score(answer[0])

            if anchor not in video_ranking_sim.keys():
                video_ranking_sim[anchor] = {}
            if target not in video_ranking_sim[anchor].keys():
                video_ranking_sim[anchor][target] = sim

    return video_ranking_sim


MODEL_2_DIR = {
    "mplug-owl3-7b": "mPLUG-Owl3-7B-240728",
    "internvl2.5-8b": "InternVL2_5-8B",
    "internvl2.5-4b": "InternVL2_5-4B",
    "internvl3-8b": "InternVL3-8B",
    "internvl3-4b": "InternVL3-4B",
    "qwen2.5-vl-3b": "Qwen2.5-VL-3B-Instruct",
    "qwen2.5-vl-7b": "Qwen2.5-VL-7B-Instruct",
    "llava-onevision-qwen2-7b-ov": "LLaVA-OneVision-Qwen2-7B",
    "llava-onevision-qwen2-0.5b-ov": "LLaVA-OneVision-Qwen2-0.5B",
    "llava-onevision-qwen2-7b-ov-hf": "LLaVA-OneVision-Qwen2-7B-HF",
    "llava-video-7b": "LLaVA-NeXT-Video-7B",
}


def main(args):

    set_all_seeds(42)
    print(
        f"\033[93mComputing similarities with model: {args.vqa_model_path}, version {args.vqa_model_version}\033[0m"
    )
    video_sets = read_video_sets(args.to_be_ranked_json)

    model_dir = MODEL_2_DIR.get(
        args.vqa_model_version, args.vqa_model_path.split("/")[-1]
    )

    concept_name = re.sub(r"(?<!^)(?=[A-Z])", " ", args.concept_name).replace(" ", "_")
    output_dir = os.path.join(args.retrieval_scores_dir, concept_name, model_dir)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_file = os.path.join(output_dir, f"{model_dir}_{concept_name}_ranking.json")

    print(f"\033[93mOutput file: {output_file}\033[0m")

    video_pairs_sim = compute_vqa_based_similarity(video_sets, args, output_file)

    with open(output_file, "w") as f:
        json.dump(video_pairs_sim, f, indent=4)
    print(f"\033[92mSimilarities saved to: {output_file}\033[0m")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Perform Retrieval.")

    parser.add_argument(
        "--video_dir",
        type=str,
        default="convisbench/",
        help="Directory where videos are stored.",
    )
    parser.add_argument(
        "--vqa_model_path",
        type=str,
        default="Qwen/Qwen2.5-VL-7B-Instruct",
        required=False,
    )
    parser.add_argument(
        "--retrieval_scores_dir",
        type=str,
        default="baselines/computed_retrieval_scores/",
        required=False,
    )

    parser.add_argument(
        "--to_be_ranked_json",
        type=str,
        default="human_annotations/to_be_ranked.json",
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
        "--vqa_model_version",
        type=str,
        default="internvl2.5-8b",
        required=False,
    )
    args = parser.parse_args()

    main(args)
