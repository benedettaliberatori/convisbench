import os
import json
import ffmpeg
import numpy as np
from torch.utils.data import Dataset
from decord import VideoReader, cpu
from baselines.InternVideo.internvideo import load_video as load_video_internvideo


class VideoDataset(Dataset):
    def __init__(self, video_dir, pairs_jsonl, max_num_frames=None, use_internvideo=False):
        """
        Args:
            video_dir (string): Directory with all the original videos.
            pairs_jsonl (string): Path to the JSONL file with video pairs.
            max_num_frames (int, optional): Maximum number of frames to load from each video, if video encoder.
            use_internvideo (bool, optional): Whether to use InternVideo for loading videos.
        """
        self.video_dir = video_dir
        self.pairs_jsonl = pairs_jsonl    
        self.max_num_frames = max_num_frames
        self.use_internvideo = use_internvideo
        self._load_video_list()

    def _load_video_list(self):
        """
        Get all video paths from the JSONL with the annotated pairs.
        """
        all_videos = []
        with open(self.pairs_jsonl, "r") as f:
            for line in f:
                pair = json.loads(line)
                all_videos.extend([pair["video1"], pair["video2"]])

        all_videos = list(set(all_videos))
        all_videos = [os.path.join(self.video_dir, video) for video in all_videos]
        self.video_list = all_videos


    def __len__(self):
        return len(self.video_list)
    
    def _load_video(self, video_path):
        """
        Load video frames from the given path.
        """       

        if not os.path.isfile(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
        try:
            if self.use_internvideo:
                frames = load_video_internvideo(video_path)
                return frames

            video_reader = VideoReader(video_path, ctx=cpu(0))
            total_frames = len(video_reader)
            if not self.max_num_frames:
                frames = video_reader.get_batch(range(total_frames)).asnumpy()
            else:
                if total_frames >= self.max_num_frames:
                    frame_indices = np.linspace(0, total_frames - 1, self.max_num_frames, dtype=int)
                    frames = video_reader.get_batch(frame_indices).asnumpy()
                else:
                    frame_indices = np.linspace(0, total_frames - 1, total_frames, dtype=int)
                    frames = video_reader.get_batch(frame_indices).asnumpy()
                    pad_count = self.max_num_frames - total_frames
                    last_frame = frames[-1]
                    pad_frames = np.repeat(last_frame[np.newaxis, ...], pad_count, axis=0)
                    frames = np.concatenate([frames, pad_frames], axis=0)
            return frames
        except Exception as e:
            raise RuntimeError(f"Failed to load video {video_path}: {e}")

    def __getitem__(self, idx):
        video_path = self.video_list[idx]
        video_id = video_path.split("/")[-1].split(".")[0]
        try:
            video_frames = self._load_video(video_path)
        except Exception as e:
            print(f"[ERROR] {e}")
            return video_id, None

        return {
            "video_id": video_id,
            "video_path": video_path,
            "video_frames": video_frames
        }