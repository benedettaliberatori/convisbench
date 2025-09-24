import os
import json
import argparse
import random
import torch
import numpy as np
from transformers import set_seed
from baselines.baselines_utils import (
    load_DINOv2_model,
    load_VideoMAE_model,
    load_CLIP_model,
    load_InternVideo_model,
)
from baselines.video_dataset import VideoDataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

def set_all_seeds(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    set_seed(seed)


def main(args):
    """
    Extract visual features from videos.
    """
    set_all_seeds(42)

    if args.model_path in ["facebook/dinov2-large", "openai/clip-vit-large-patch14"]:
        video_dataset = VideoDataset(args.video_dir, args.pairs_json)
    elif args.model_path in ["MCG-NJU/videomae-large", "InternVideo/InternVideo-MM-L-14"]:
        use_internvideo = args.model_path == "InternVideo/InternVideo-MM-L-14"
        video_dataset = VideoDataset(
            args.video_dir,
            args.pairs_json,
            max_num_frames=args.max_num_frames,
            use_internvideo=use_internvideo,
        )
    else:
        raise NotImplementedError(f"Model {args.model_path} currently not implemented")

    video_loader = DataLoader(
        video_dataset, batch_size=args.batch_size, shuffle=False
    )
    output_dir = os.path.join(args.out_video_dir, args.model_path.split("/")[-1])

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"\033[92mOutput directory created: {output_dir}\033[0m")

    print(
        f"\033[96mExtracting visual features from {len(video_dataset.video_list)} videos."
    )
    print(f"\033[92mOutput directory: {output_dir}")
    print(f"\033[93mModel path: {args.model_path}\033[0m")

    if args.model_path == "facebook/dinov2-large":
        chunk_size = 2000
        model, processor = load_DINOv2_model(args.model_path)
        for video_batch in tqdm(video_loader):
            video_ids = video_batch["video_id"][0]
            video_paths = video_batch["video_path"][0]
            video_frames = video_batch["video_frames"][0]

            if os.path.exists(os.path.join(output_dir, video_ids + ".pt")):
                print(f"Visual features already exist for {video_ids}. Skipping...")
                continue

            video_frames_list = [
                video_frames[i : i + chunk_size]
                for i in range(0, len(video_frames), chunk_size)
            ]
            features = []
            for video_frames_chunk in video_frames_list:
                inputs = processor(images=video_frames_chunk, return_tensors="pt").to(
                    "cuda"
                )
                with torch.no_grad():
                    outputs = model(**inputs)

                features_chunk = outputs.last_hidden_state.mean(dim=1)
                features.append(features_chunk)

            features = torch.cat(features, dim=0)
            assert (
                len(video_frames) == features.shape[0]
            ), f"Video frames: {len(video_frames_chunk)}, Features: {features.shape[0]}"
            features = features.cpu()
            torch.save(features, os.path.join(output_dir, video_ids + ".pt"))

    elif args.model_path == "MCG-NJU/videomae-large":
        model, processor = load_VideoMAE_model(args.model_path)
        print(
            f"\033[92mThis model process {model.config.num_frames} frames for each video.\033[0m"
        )
        assert (
            args.max_num_frames == model.config.num_frames
        ), f"Model {args.model_path} requires {model.config.num_frames} frames, but got {args.max_num_frames}."

        for video_batch in tqdm(video_loader):
            video_ids = video_batch["video_id"][0]
            video_paths = video_batch["video_path"][0]
            video_frames = video_batch["video_frames"][0]

            if os.path.exists(os.path.join(output_dir, video_ids + ".pt")):
                print(f"Visual features already exist for {video_ids}. Skipping...")
                continue

            inputs = processor(list(video_frames), return_tensors="pt").to("cuda")
            with torch.no_grad():
                outputs = model(**inputs)
            last_hidden_states = outputs.last_hidden_state
            features = last_hidden_states.mean(dim=1)
            features = features.cpu()
            torch.save(features, os.path.join(output_dir, video_ids + ".pt"))

    elif args.model_path == "openai/clip-vit-large-patch14":
        chunk_size = 1500
        model, processor = load_CLIP_model(args.model_path)

        for video_batch in tqdm(video_loader):
            video_ids = video_batch["video_id"][0]
            video_paths = video_batch["video_path"][0]
            video_frames = video_batch["video_frames"][0]

            if os.path.exists(os.path.join(output_dir, video_ids + ".pt")):
                print(f"Visual features already exist for {video_ids}. Skipping...")
                continue

            video_frames_list = [
                video_frames[i : i + chunk_size]
                for i in range(0, len(video_frames), chunk_size)
            ]
            features = []
            for video_frames_chunk in video_frames_list:
                inputs = processor(images=video_frames_chunk, return_tensors="pt").to(
                    "cuda"
                )
                with torch.no_grad():
                    features_chunk = model.get_image_features(**inputs)
                features.append(features_chunk)
            features = torch.cat(features, dim=0)

            assert (
                len(video_frames) == features.shape[0]
            ), f"Video frames: {len(video_frames_chunk)}, Features: {features.shape[0]}"

            features = features.detach().cpu()
            torch.save(features, os.path.join(output_dir, video_ids + ".pt"))
    elif args.model_path == "InternVideo/InternVideo-MM-L-14":

        model, tokenizer = load_InternVideo_model(args.model_path)

        for video_batch in tqdm(video_loader):
            video_ids = video_batch["video_id"][0]
            video_paths = video_batch["video_path"][0]
            video_frames = video_batch["video_frames"][0]

            if os.path.exists(os.path.join(output_dir, video_ids + ".pt")):
                print(f"Visual features already exist for {video_ids}. Skipping...")
                continue

            video_frames = video_frames.unsqueeze(0).to("cuda")
            with torch.no_grad():
                features = model.encode_video(video_frames)
            features = features.cpu()
            torch.save(features, os.path.join(output_dir, video_ids + ".pt"))

    else:
        raise NotImplementedError(f"Model {args.model_path} currently not implemented")

    print(f"\033[92mVisual features saved to {output_dir}.\033[0m")
    print(f"\033[93mTotal videos processed: {len(video_dataset.video_list)}\033[0m")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract visual features from videos.")

    parser.add_argument(
        "--out_video_dir",
        type=str,
        default="visual_features/",
        help="Directory where features should be stored.",
    )
    parser.add_argument(
        "--pairs_json",
        type=str,
        default="convisbench/ConVIS.jsonl",
        required=False,
        help="Path to the JSONL file with video pairs.",
    )
    parser.add_argument(
        "--video_dir",
        type=str,
        default="convisbench",
        required=False,
        help="Directory with all the original videos.",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        help="Path to the model to use.",
    )
    parser.add_argument(
        "--max_num_frames",
        type=int,
        default=16,
        required=False,
        help="Maximum number of frames to load from each video, if using a video encoder.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        required=False,
        help="Batch size for processing videos.",
    )
    args = parser.parse_args()

    main(args)
