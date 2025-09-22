import torch
import numpy as np
from PIL import Image
from typing import List, Union
from transformers import AutoModel, AutoTokenizer
from decord import VideoReader, cpu
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode


INTERNVL2_MODELS = {
    'internvl2.5-4b': {
        'tokenizer': {
            'path': 'OpenGVLab/InternVL2_5-4B',
            'trust_remote_code': True,
            'use_fast': False,
        },
        'model': {
            'pretrained_model_name_or_path': 'OpenGVLab/InternVL2_5-4B',
            'torch_dtype': torch.bfloat16,
            'low_cpu_mem_usage': True,
            'use_flash_attn': True,
            'trust_remote_code': True,
            "device_map": "auto",
        },
    },

    "internvl2.5-8b": {
        "tokenizer": {
            "path": "OpenGVLab/InternVL2_5-8B",
            "trust_remote_code": True,
            "use_fast": False,
        },
        "model": {
            "pretrained_model_name_or_path": "OpenGVLab/InternVL2_5-8B",
            "torch_dtype": torch.bfloat16,
            "low_cpu_mem_usage": True,
            "use_flash_attn": True,
            "trust_remote_code": True,
            "device_map": "auto",
        },
    },
    "internvl3-8b": {
        "tokenizer": {
            "path": "OpenGVLab/InternVL3-8B",
            "trust_remote_code": True,
            "use_fast": False,
        },
        "model": {
            "pretrained_model_name_or_path": "OpenGVLab/InternVL3-8B",
            "torch_dtype": torch.bfloat16,
            "low_cpu_mem_usage": True,
            "use_flash_attn": True,
            "trust_remote_code": True,
            "device_map": "auto",
        },
    },
}

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class InternVL2Model:
    def __init__(self, model_name="internvl2.5-8b", device="cuda", cache_dir=None):
        assert (
            model_name in INTERNVL2_MODELS
        ), f"Model {model_name} not found in INTERNVL2_MODELS"
        self.model_name = model_name
        self.device = device
        self.cache_dir = cache_dir
        self.model_info = INTERNVL2_MODELS[model_name]
        print(f"Loading {INTERNVL2_MODELS[model_name]["model"]["pretrained_model_name_or_path"]} model...")
        self.load_model()

    def load_model(self):
        tokenizer_path = self.model_info["tokenizer"]["path"]

        self.model = AutoModel.from_pretrained(**self.model_info["model"]).eval()

        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path, **self.model_info["tokenizer"]
        )

        self.device = next(
            self.model.parameters()
        ).device  # If there are multiple GPUs put the model on the first parameters GPU

    def build_transform(self, input_size=448):
        transform = T.Compose(
            [
                T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
                T.Resize(
                    (input_size, input_size), interpolation=InterpolationMode.BICUBIC
                ),
                T.ToTensor(),
                T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]
        )
        return transform

    def find_closest_aspect_ratio(
        self, aspect_ratio, target_ratios, width, height, image_size
    ):
        best_ratio_diff = float("inf")
        best_ratio = (1, 1)
        area = width * height
        for ratio in target_ratios:
            target_aspect_ratio = ratio[0] / ratio[1]
            ratio_diff = abs(aspect_ratio - target_aspect_ratio)
            if ratio_diff < best_ratio_diff:
                best_ratio_diff = ratio_diff
                best_ratio = ratio
            elif ratio_diff == best_ratio_diff:
                if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                    best_ratio = ratio
        return best_ratio

    def dynamic_preprocess(
        self, image, min_num=1, max_num=12, image_size=448, use_thumbnail=False
    ):
        orig_width, orig_height = image.size
        aspect_ratio = orig_width / orig_height

        target_ratios = set(
            (i, j)
            for n in range(min_num, max_num + 1)
            for i in range(1, n + 1)
            for j in range(1, n + 1)
            if i * j <= max_num and i * j >= min_num
        )
        target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

        target_aspect_ratio = self.find_closest_aspect_ratio(
            aspect_ratio, target_ratios, orig_width, orig_height, image_size
        )

        target_width = image_size * target_aspect_ratio[0]
        target_height = image_size * target_aspect_ratio[1]
        blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

        resized_img = image.resize((target_width, target_height))
        processed_images = []
        for i in range(blocks):
            box = (
                (i % (target_width // image_size)) * image_size,
                (i // (target_width // image_size)) * image_size,
                ((i % (target_width // image_size)) + 1) * image_size,
                ((i // (target_width // image_size)) + 1) * image_size,
            )
            split_img = resized_img.crop(box)
            processed_images.append(split_img)

        if use_thumbnail and len(processed_images) != 1:
            thumbnail_img = image.resize((image_size, image_size))
            processed_images.append(thumbnail_img)
        return processed_images

    def load_image(self, image_file, input_size=448, max_num=12):
        if image_file.lower().endswith(".npy"):
            np_array = np.load(image_file)
            if np_array.ndim == 3:
                image = Image.fromarray(np_array.astype("uint8"), "RGB")
            else:
                raise ValueError(f"Unexpected shape for NumPy array in {image_file}")
        else:
            image = Image.open(image_file).convert("RGB")

        transform = self.build_transform(input_size=input_size)
        images = self.dynamic_preprocess(
            image, image_size=input_size, use_thumbnail=True, max_num=max_num
        )
        pixel_values = [transform(image) for image in images]
        pixel_values = torch.stack(pixel_values)
        return pixel_values

    def get_video_index(self, bound, fps, max_frame, first_idx=0, num_segments=32):
        if bound:
            start, end = bound[0], bound[1]
        else:
            start, end = -100000, 100000
        start_idx = max(first_idx, round(start * fps))
        end_idx = min(round(end * fps), max_frame)
        seg_size = float(end_idx - start_idx) / num_segments
        frame_indices = np.array(
            [
                int(start_idx + (seg_size / 2) + np.round(seg_size * idx))
                for idx in range(num_segments)
            ]
        )
        return frame_indices

    def load_video(
        self, video_path, bound=None, input_size=448, max_num=1, num_segments=32
    ):
        vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
        max_frame = len(vr) - 1
        fps = float(vr.get_avg_fps())
        pixel_values_list, num_patches_list = [], []
        transform = self.build_transform(input_size=input_size)
        frame_indices = self.get_video_index(
            bound, fps, max_frame, first_idx=0, num_segments=num_segments
        )

        for frame_index in frame_indices:
            img = Image.fromarray(vr[frame_index].asnumpy()).convert("RGB")
            img = self.dynamic_preprocess(
                img, image_size=input_size, use_thumbnail=True, max_num=max_num
            )
            pixel_values = [transform(tile) for tile in img]
            pixel_values = torch.stack(pixel_values)
            num_patches_list.append(pixel_values.shape[0])
            pixel_values_list.append(pixel_values)
        pixel_values = torch.cat(pixel_values_list)
        return pixel_values, num_patches_list

    def load_images(
        self, paths: List[str], fps_factor: int = 1, num_frames: int = 16
    ) -> List[Union[torch.Tensor, List[torch.Tensor]]]:
        processed_data = []
        num_patches_list = []
        for path in paths:
            if path.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):  # Video file
                video_frames, video_num_patches = self.load_video(
                    path, num_segments=num_frames
                )
                processed_data.append(video_frames)
                num_patches_list.append(video_num_patches)  # You want a list of lists
            else:  # Image file or .npy file
                image_tensor = self.load_image(path)
                processed_data.append(image_tensor)
                num_patches_list.append(image_tensor.shape[0])
        return processed_data, num_patches_list

    def encode_video(self, video_path, fps_factor, max_frames_num):
        video_frames, video_num_patches = self.load_video(
            video_path, num_segments=max_frames_num
        )
        return [video_frames, video_num_patches]

    def generate(
        self, images: List[str], max_new_tokens: int = 1024,
        use_system_prompt: bool = False,
        conditioned_system_prompt: str = None,
    ) -> torch.Tensor:

        pixel_values, num_patches_list = images            

        question = "Describe this video in detail."
        video_prefix = "".join(
            [f"Frame{i+1}: <image>\n" for i in range(len(num_patches_list))]
        )

        if use_system_prompt:
            question = f"{conditioned_system_prompt}\n{video_prefix}Output a similarity score from 1 to 5."
        else:
            question = video_prefix + question

        generation_config = dict(max_new_tokens=max_new_tokens, do_sample=True, pad_token_id=self.tokenizer.eos_token_id)
        pixel_values = pixel_values.to(torch.bfloat16).cuda()

        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            outputs = self.model.chat(
                tokenizer=self.tokenizer,
                pixel_values=pixel_values,
                question=question,
                num_patches_list=num_patches_list,
                generation_config=generation_config,
            )

        return outputs
    
    @torch.no_grad()
    def batch_video_generate(
        self, images: List[str], max_new_tokens: int = 1024,
        use_system_prompt: bool = False,
        conditioned_system_prompt: str = None,
    ) -> torch.Tensor:

        batch_pixel_values = [el[0] for el in images]
        batch_num_patches_list = [el[1] for el in images]
        batch_pixel_values = [p.to(torch.bfloat16).cuda() for p in batch_pixel_values]

        batch_questions = []
        for num_patches_list in batch_num_patches_list:
            video_prefix = "".join(
                [f"Frame{i+1}: <image>\n" for i in range(len(num_patches_list))]
            )
            if use_system_prompt:
                question = f"{conditioned_system_prompt}\n{video_prefix}Output a similarity score from 1 to 5."
            else:
                question = video_prefix + "Describe this video in detail."

            batch_questions.append(question)

        pixel_values_batch = torch.cat(batch_pixel_values, dim=0)
        num_patches_list_batch = [p.size(0) for p in batch_pixel_values]

        generation_config = dict(max_new_tokens=max_new_tokens, do_sample=True)

        
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            outputs = self.model.batch_chat(
                self.tokenizer,
                pixel_values=pixel_values_batch,
                num_patches_list=num_patches_list_batch,
                questions=batch_questions,
                generation_config=generation_config
            )

            

        return outputs
