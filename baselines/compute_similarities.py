import os
import json
import argparse
import random
import torch
import numpy as np
from transformers import set_seed
from tqdm import tqdm

from baselines.baselines_utils import (
    load_CLIP_model,
    load_InternVideo_model,
)


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

cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)


def compute_video2video_similarity(video_pairs, args):
    """
    Compute the similarity between video pairs using the video features.    
    """
    video_pairs_sim = []
    for video_pair in tqdm(video_pairs):
        video1_id = video_pair["video1"].split('/')[-1].split(".")[0]
        video2_id = video_pair["video2"].split('/')[-1].split(".")[0]

        video1_features = torch.load(
            os.path.join(
                args.feature_dir, args.model_path.split('/')[-1], f"{video1_id}.pt"
            ),
            weights_only=True,
        )
        video2_features = torch.load(
            os.path.join(
                args.feature_dir, args.model_path.split('/')[-1], f"{video2_id}.pt"
            ),
            weights_only=True,
        )

        if args.model_path == "facebook/dinov2-large":
            video1_features = video1_features.mean(dim=0, keepdim=True)
            video2_features = video2_features.mean(dim=0, keepdim=True)

        sim = cos(video1_features, video2_features).item()
        sim = (sim + 1) / 2

        video_pairs_sim.append(
            {
                "video1": video_pair["video1"],
                "video2": video_pair["video2"],
                "similarity": sim,
            }
        )

    return video_pairs_sim


def compute_text2text_similarity(video_pairs, args):
    """
    Compute the similarity between video pairs using the text features of generated captions.
    """
    from sentence_transformers import SentenceTransformer
    text_model = SentenceTransformer("all-MiniLM-L6-v2")

    video_pairs_sim = []

    caption_dir = os.path.join(args.captions_dir, args.model_path.split("/")[-1])
    print(f"Loading captions from {caption_dir}")
    for video_pair in tqdm(video_pairs):
        video1_id = video_pair["video1"].split('/')[-1].split(".")[0]
        video2_id = video_pair["video2"].split('/')[-1].split(".")[0]

        video1_caption = os.path.join(caption_dir, f"{video1_id}.txt")
        video2_caption = os.path.join(caption_dir, f"{video2_id}.txt")
        with open(video1_caption, "r") as f:
            video1_caption = f.read().strip()
        with open(video2_caption, "r") as f:
            video2_caption = f.read().strip()

        caption1_embedding = text_model.encode(video1_caption)
        caption2_embedding = text_model.encode(video2_caption)
        sim = text_model.similarity(caption1_embedding, caption2_embedding).item()
        sim = (sim + 1) / 2

        video_pairs_sim.append(
            {
                "video1": video_pair["video1"],
                "video2": video_pair["video2"],
                "similarity": sim,
            }
        )
    return video_pairs_sim


def compute_crossmodal_similarity(video_pairs, args):
    """
    Computing the similarity between video pairs using CLIPScore.
    """

    if args.model_path == "openai/clip-vit-large-patch14":
        vlm_model, vlm_processor = load_CLIP_model(args.model_path)
    elif args.model_path == "InternVideo/InternVideo-MM-L-14":
        vlm_model, vlm_tokenizer = load_InternVideo_model(args.model_path)

    caption_dir = os.path.join(
        args.captions_dir, args.model_path_captions_crossmodal.split("/")[-1]
    )
    print(f"Loading captions from {caption_dir}")

    video_pairs_sim = []
    for video_pair in tqdm(video_pairs):
        video1_id = video_pair["video1"].split('/')[-1].split(".")[0]
        video2_id = video_pair["video2"].split('/')[-1].split(".")[0]

        video1_features = torch.load(
            os.path.join(
                args.feature_dir, args.model_path.split('/')[-1], f"{video1_id}.pt"
            ),
            weights_only=True,
        ).to("cuda")
        video2_features = torch.load(
            os.path.join(
                args.feature_dir, args.model_path.split('/')[-1], f"{video2_id}.pt"
            ),
            weights_only=True,
        ).to("cuda")

        video1_caption = os.path.join(caption_dir, f"{video1_id}.txt")
        video2_caption = os.path.join(caption_dir, f"{video2_id}.txt")
        with open(video1_caption, "r") as f:
            video1_caption = f.read().strip()
        with open(video2_caption, "r") as f:
            video2_caption = f.read().strip()

        if args.model_path == "openai/clip-vit-large-patch14":
            video1_features = video1_features.mean(dim=0, keepdim=True)
            video2_features = video2_features.mean(dim=0, keepdim=True)

            inputs1 = vlm_processor(
                text=[video1_caption],
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=77,
            ).to("cuda")
            inputs2 = vlm_processor(
                text=[video2_caption],
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=77,
            ).to("cuda")
            with torch.no_grad():
                caption1_features = vlm_model.get_text_features(**inputs1)
                caption2_features = vlm_model.get_text_features(**inputs2)

        elif args.model_path == "InternVideo/InternVideo-MM-L-14":

            tokens1 = vlm_tokenizer(video1_caption, truncate=True).to(video1_features.device)
            tokens2 = vlm_tokenizer(video2_caption, truncate=True).to(video2_features.device)

            with torch.no_grad():
                caption1_features = vlm_model.encode_text(tokens1)
                caption2_features = vlm_model.encode_text(tokens2)

        else:
            raise NotImplementedError(
                f"Model {args.model_path} currently not implemented"
            )
        

        video2text = cos(video1_features, caption2_features).item()
        text2video = cos(video2_features, caption1_features).item()
        sim = (video2text + text2video) / 2
        sim = (sim + 1) / 2
        video_pairs_sim.append(
            {
                "video1": video_pair["video1"],
                "video2": video_pair["video2"],
                "similarity": sim,
            }
        )

    return video_pairs_sim

def compute_vqascore_similarity(video_pairs, args):
    """
    Computing the similarity between video pairs using VQAScore.
    """
    from baselines.t2v_metrics.t2v_metrics.vqascore import VQAScore
    llava_ov_score = VQAScore(model='llava-onevision-qwen2-7b-ov')
    video_pairs_sim = []

    caption_dir = os.path.join(
        args.captions_dir, args.model_path_captions_crossmodal.split("/")[-1]
    )
    print(f"Loading captions from {caption_dir}")


    for video_pair in tqdm(video_pairs):
        video1_path = os.path.join(args.video_dir, video_pair["video1"])
        video2_path = os.path.join(args.video_dir, video_pair["video2"])
        

        video1_id = video_pair["video1"].split('/')[-1].split(".")[0]
        video2_id = video_pair["video2"].split('/')[-1].split(".")[0]
        video1_caption = os.path.join(caption_dir, f"{video1_id}.txt")
        video2_caption = os.path.join(caption_dir, f"{video2_id}.txt")
        with open(video1_caption, "r") as f:
            video1_caption = f.read().strip()
        with open(video2_caption, "r") as f:
            video2_caption = f.read().strip()

        
        score = llava_ov_score(
            images = [video1_path],
            texts=[video2_caption],
            num_frames=16,
            question_template='Is this figure showing "{}"? Please answer yes or no.',
            answer_template='Yes'
        ) + llava_ov_score(
            images = [video2_path],
            texts=[video1_caption],
            num_frames=16,
            question_template='Is this video showing "{}"? Please answer yes or no.',
            answer_template='Yes'
        )
        score = score / 2 
         
        score = score.item() #  No need to normalize in [0,1] because already a probability
        
        video_pairs_sim.append(
            {
                "video1": video_pair["video1"],
                "video2": video_pair["video2"],
                "similarity": score,
            }
        )
    return video_pairs_sim


def main(args):

    set_all_seeds(42)
    print(f"\033[93mComputing similarities with model: {args.model_path}\033[0m")
    video_pairs = read_video_pairs(args.pairs_jsonl)

    if args.model_path in ["facebook/dinov2-large", "MCG-NJU/videomae-large"]:
        video_pairs_sim = compute_video2video_similarity(video_pairs, args)
    elif args.model_path in [
        "mPLUG/mPLUG-Owl3-7B-240728",
        "OpenGVLab/InternVL2_5-8B",
        "Qwen/Qwen2.5-VL-7B-Instruct",
    ]:
        video_pairs_sim = compute_text2text_similarity(video_pairs, args)
    elif args.model_path in [
        "openai/clip-vit-large-patch14",
        "InternVideo/InternVideo-MM-L-14",
        ]:
        video_pairs_sim = compute_crossmodal_similarity(video_pairs, args)
    elif args.model_path == "VQAScore/LLaVA-OneVision":
        video_pairs_sim = compute_vqascore_similarity(video_pairs, args)

    else:
        raise NotImplementedError(f"Model {args.model_path} currently not implemented")

    output_dir = os.path.join(args.similarity_dir, args.model_path.split('/')[-1])
    if args.model_path in [
        "openai/clip-vit-large-patch14",
        "InternVideo/InternVideo-MM-L-14",
        "VQAScore/LLaVA-OneVision",
    ]:
        output_dir = (
            output_dir + f"_{args.model_path_captions_crossmodal.split('/')[-1]}"
        )
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"\033[92mOutput directory created: {output_dir}\033[0m")

    
    output_file = os.path.join(
        output_dir, f"{args.model_path.split('/')[-1]}_similarities.json"
    )
    with open(output_file, "w") as f:
        json.dump(video_pairs_sim, f, indent=4)
    print(f"\033[92mSimilarities saved to: {output_file}\033[0m")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--feature_dir",
        type=str,
        default="visual_features/",
        help="Directory where features are stored.",
    )
    parser.add_argument(
        "--captions_dir",
        type=str,
        default="captions/",
        help="Directory where features are stored.",
    )
    parser.add_argument(
        "--video_dir",
        type=str,
        default="convisbench/",
        help="Directory where videos are stored.",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        help="Path to the model.",
    )
    parser.add_argument(
        "--similarity_dir",
        type=str,
        default="baselines/computed_similarities/",
        required=False,
    )

    parser.add_argument(
        "--pairs_jsonl",
        type=str,
        default="convisbench/ConVIS.jsonl",
        required=False,
    )
    parser.add_argument(
        "--model_path_captions_crossmodal",
        type=str,
        default="mPLUG/mPLUG-Owl3-7B-240728",
        help="Path to the model used to generate captions for cross-modal approaches.",
    )

    args = parser.parse_args()

    main(args)
