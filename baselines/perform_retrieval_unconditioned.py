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
from baselines.baselines_utils import (
    load_CLIP_model,
    load_InternVideo_model,
)


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
        video_sets.append({"anchor": k, "targets": v})
    return video_sets


cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)

def compute_video2video_retrieval(video_sets, args, output_file):

    video_ranking_sim = {}
    for video_set in tqdm(video_sets):
        anchor = video_set["anchor"]
        targets = video_set["targets"]
 
        anchor_features = torch.load(
            os.path.join(
                args.feature_dir, args.model_path.split('/')[-1], f"{anchor.split('.')[0]}.pt"
            ),
            weights_only=True,
        )
        if args.model_path == "facebook/dinov2-large":
            anchor_features = anchor_features.mean(dim=0, keepdim=True)
        for target in targets:

            target_features = torch.load(
                os.path.join(
                    args.feature_dir, args.model_path.split('/')[-1], f"{target.split('.')[0]}.pt"
                ),
                weights_only=True,
            )

            if args.model_path == "facebook/dinov2-large":
                target_features = target_features.mean(dim=0, keepdim=True)

            sim = cos(anchor_features, target_features).item()
            sim = (sim + 1) / 2

            if anchor not in video_ranking_sim.keys():
                video_ranking_sim[anchor] = {}
            if target not in video_ranking_sim[anchor].keys():
                video_ranking_sim[anchor][target] = sim

    return video_ranking_sim


def compute_text2text_retrieval(video_sets, args, output_file):
    from sentence_transformers import SentenceTransformer
    text_model = SentenceTransformer("all-MiniLM-L6-v2")

    caption_dir = os.path.join(args.captions_dir, args.model_path.split("/")[-1])
    print(f"Loading captions from {caption_dir}")

    video_ranking_sim = {}
    for video_set in tqdm(video_sets):
        anchor = video_set["anchor"]
        targets = video_set["targets"]
 
        anchor_path = os.path.join(args.video_dir, anchor)
        anchor_caption = os.path.join(caption_dir, f"{anchor.split('.')[0]}.txt")
        with open(anchor_caption, 'r') as f:
            anchor_caption = f.read().strip()

        anchor_embedding = text_model.encode(anchor_caption)

        for target in targets:
            target_path = os.path.join(args.video_dir, target)
            target_caption = os.path.join(caption_dir, f"{target.split('.')[0]}.txt")
            with open(target_caption, "r") as f:
                target_caption = f.read().strip()

            target_embedding = text_model.encode(target_caption)
            sim = text_model.similarity(anchor_embedding, target_embedding).item()
            sim = (sim + 1) / 2

            if anchor not in video_ranking_sim.keys():
                video_ranking_sim[anchor] = {}
            if target not in video_ranking_sim[anchor].keys():
                video_ranking_sim[anchor][target] = sim

    return video_ranking_sim



def compute_crossmodal_retrieval(video_sets, args, output_file):

    if args.model_path == "openai/clip-vit-large-patch14":
        vlm_model, vlm_processor = load_CLIP_model(args.model_path)
    elif args.model_path == "InternVideo/InternVideo-MM-L-14":
        vlm_model, vlm_tokenizer = load_InternVideo_model(args.model_path)

    caption_dir = os.path.join(
        args.captions_dir, args.model_path_captions_crossmodal.split("/")[-1]
    )
    print(f"Loading captions from {caption_dir}")
    video_ranking_sim = {}
    for video_set in tqdm(video_sets):

        anchor = video_set["anchor"]
        targets = video_set["targets"]
        anchor_path = os.path.join(args.video_dir, anchor)
        anchor_caption = os.path.join(caption_dir, f"{anchor.split('.')[0]}.txt")

        with open(anchor_caption, 'r') as f:
            anchor_caption = f.read().strip()

        anchor_features = torch.load(
            os.path.join(
                args.feature_dir, args.model_path.split('/')[-1], f"{anchor.split('.')[0]}.pt"
            ),
            weights_only=True,
        ).to("cuda")

        if args.model_path == "openai/clip-vit-large-patch14":
            anchor_features = anchor_features.mean(dim=0, keepdim=True)
        
        for target in targets:
            
            target_path = os.path.join(args.video_dir, target)
            target_caption = os.path.join(caption_dir, f"{target.split('.')[0]}.txt")
            with open(target_caption, "r") as f:
                target_caption = f.read().strip()

            target_features = torch.load(
                os.path.join(
                    args.feature_dir, args.model_path.split('/')[-1], f"{target.split('.')[0]}.pt"
                ),
                weights_only=True,
            ).to("cuda")

            if args.model_path == "openai/clip-vit-large-patch14":
                target_features = target_features.mean(dim=0, keepdim=True)

                inputs1 = vlm_processor(
                    text=[anchor_caption],
                    return_tensors="pt",
                    padding="max_length",
                    truncation=True,
                    max_length=77,
                ).to("cuda")
                inputs2 = vlm_processor(
                    text=[target_caption],
                    return_tensors="pt",
                    padding="max_length",
                    truncation=True,
                    max_length=77,
                ).to("cuda")
                with torch.no_grad():
                    anchor_caption_features = vlm_model.get_text_features(**inputs1)
                    target_caption_features = vlm_model.get_text_features(**inputs2)

            elif args.model_path == "InternVideo/InternVideo-MM-L-14":

                tokens1 = vlm_tokenizer(anchor_caption, truncate=True).to(anchor_features.device)
                tokens2 = vlm_tokenizer(target_caption, truncate=True).to(target_features.device)

                with torch.no_grad():
                    anchor_caption_features = vlm_model.encode_text(tokens1)
                    target_caption_features = vlm_model.encode_text(tokens2)

            else:
                raise NotImplementedError(
                    f"Model {args.model_path} currently not implemented"
                )
        

            video2text = cos(anchor_features, target_caption_features).item()
            text2video = cos(target_features, anchor_caption_features).item()
            sim = (video2text + text2video) / 2
            sim = (sim + 1) / 2



            if anchor not in video_ranking_sim.keys():
                video_ranking_sim[anchor] = {}
            if target not in video_ranking_sim[anchor].keys():
                video_ranking_sim[anchor][target] = sim



    return video_ranking_sim


def compute_vqascore_retrieval(video_sets, args, output_file):
    from t2v_metrics.t2v_metrics.vqascore import VQAScore
    llava_ov_score = VQAScore(model='llava-onevision-qwen2-7b-ov')

    caption_dir = os.path.join(
        args.captions_dir, args.model_path_captions_crossmodal.split("/")[-1]
    )

    video_ranking_sim = {}
    for video_set in tqdm(video_sets):

        anchor = video_set["anchor"]
        targets = video_set["targets"]

        anchor_path = os.path.join(args.video_dir, anchor)


        anchor_caption = os.path.join(caption_dir, f"{anchor.split('.')[0]}.txt")
        with open(anchor_caption, "r") as f:
            anchor_caption = f.read().strip()

        for target in targets:
            
            target_path = os.path.join(args.video_dir, target)
            target_caption = os.path.join(caption_dir, f"{target.split('.')[0]}.txt")
            with open(target_caption, "r") as f:
                target_caption = f.read().strip()

            score = llava_ov_score(
                images = [anchor_path],
                texts=[target_path],
                num_frames=16,
                question_template='Is this video showing "{}"? Please answer yes or no.',
                answer_template='Yes'
            ) + llava_ov_score(
                images = [target_path],
                texts=[anchor_path],
                num_frames=16,
                question_template='Is this video showing "{}"? Please answer yes or no.',
                answer_template='Yes'
            )

            score = score / 2 
            sim = score.item()


            if anchor not in video_ranking_sim.keys():
                video_ranking_sim[anchor] = {}
            if target not in video_ranking_sim[anchor].keys():
                video_ranking_sim[anchor][target] = sim

            
        
            
    return video_ranking_sim




MODEL_2_DIR = {
    "mplug-owl3-7b": "mPLUG-Owl3-7B-240728",
    "internvl2.5-8b" : "InternVL2_5-8B",
    "internvl2.5-4b" : "InternVL2_5-4B",
    "internvl3-8b" : "InternVL3-8B",
    "internvl3-4b" : "InternVL3-4B",
    "qwen2.5-vl-3b" : "Qwen2.5-VL-3B-Instruct",
    "qwen2.5-vl-7b" : "Qwen2.5-VL-7B-Instruct",
    "llava-onevision-qwen2-7b-ov" : "LLaVA-OneVision-Qwen2-7B",
    "llava-onevision-qwen2-0.5b-ov": "LLaVA-OneVision-Qwen2-0.5B",
    "llava-onevision-qwen2-7b-ov-hf": "LLaVA-OneVision-Qwen2-7B-HF",
    "llava-video-7b" : "LLaVA-NeXT-Video-7B"
}

def main(args):

    set_all_seeds(42)
    print(f"\033[93mComputing similarities with model: {args.model_path}\033[0m")
    video_sets = read_video_sets(args.to_be_ranked_json)

    output_dir = os.path.join(args.retrieval_scores_dir, args.model_path.split('/')[-1]) 

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_file = os.path.join(
        output_dir, f"{args.model_path.split('/')[-1]}_unconditioned_ranking.json"
    )
    print(f"\033[93mOutput file: {output_file}\033[0m")


    if args.model_path in ["facebook/dinov2-large", "MCG-NJU/videomae-large"]:
        video_ranks = compute_video2video_retrieval(video_sets, args, output_file)
    elif args.model_path in [
        "mPLUG/mPLUG-Owl3-7B-240728",
        "OpenGVLab/InternVL2_5-8B",
        "Qwen/Qwen2.5-VL-7B-Instruct",
    ]:
        video_ranks = compute_text2text_retrieval(video_sets, args, output_file)
    elif args.model_path in [
        "openai/clip-vit-large-patch14",
        "InternVideo/InternVideo-MM-L-14",
        ]:
        video_ranks = compute_crossmodal_retrieval(video_sets, args, output_file)
    elif args.model_path == "VQAScore/LLaVA-OneVision":
        video_ranks = compute_vqascore_retrieval(video_sets, args, output_file)

    else:
        raise NotImplementedError(f"Model {args.model_path} currently not implemented")



    with open(output_file, "w") as f:
        json.dump(video_ranks, f, indent=4)
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
        "--retrieval_scores_dir", 
        type=str,
        default="baselines/computed_retrieval_scores_unconditioned/",
        required=False,
    )
    parser.add_argument(
        "--to_be_ranked_json",
        type=str,
        default="human_annotations/to_be_ranked.json",
        required=False,
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--feature_dir",
        type=str,
        default="visual_features/",
        help="Directory where features are stored.",
    )
    parser.add_argument(
        "--max_frames_num",
        type=int,
        default=16,
        required=False,
    )
    parser.add_argument(
        "--captions_dir",
        type=str,
        default="captions/",
        help="Directory where features are stored.",
    )
    parser.add_argument(
        "--model_path_captions_crossmodal",
        type=str,
        default="mPLUG/mPLUG-Owl3-7B-240728",
    )


    args = parser.parse_args()

    main(args)
