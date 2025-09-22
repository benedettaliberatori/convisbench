import torch
import numpy as np
from PIL import Image
from typing import List, Union
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from decord import VideoReader, cpu
from models.prompts import CAPTIONING_SYSPROMPT

QWEN2_VL_MODELS = {
    "qwen2.5-vl-3b": {
        "tokenizer": {
            "path": "Qwen/Qwen2.5-VL-3B-Instruct",
        },
        "model": {
            "path": "Qwen/Qwen2.5-VL-3B-Instruct",
            "torch_dtype": torch.bfloat16,
            "attn_implementation": "sdpa", 
        },
    },
    "qwen2.5-vl-7b": {
        "tokenizer": {
            "path": "Qwen/Qwen2.5-VL-7B-Instruct",
        },
        "model": {
            "path": "Qwen/Qwen2.5-VL-7B-Instruct",
            "torch_dtype": torch.bfloat16,
            "attn_implementation": "sdpa", 
        },
    },
}


class Qwen2VLModel:
    def __init__(self, model_name="qwen2.5-vl-7b", device="cuda", cache_dir=None):
        assert (
            model_name in QWEN2_VL_MODELS
        ), f"Model {model_name} not found in QWEN2_VL_MODELS"
        self.model_name = model_name
        self.device = device
        self.cache_dir = cache_dir
        self.model_info = QWEN2_VL_MODELS[model_name]
        self.load_model()

    def load_model(self):
        model_path = self.model_info["model"]["path"]
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=self.model_info["model"]["torch_dtype"],
            attn_implementation=self.model_info["model"]["attn_implementation"],
            device_map="auto",
        )

        self.processor = AutoProcessor.from_pretrained(
            self.model_info["tokenizer"]["path"]
        )
        self.model.eval()
        self.device = next(self.model.parameters()).device

    def load_images(
        self, paths: List[str], fps_factor: int = 1, num_frames: int = 16
    ) -> List[Union[torch.Tensor, List[torch.Tensor]]]:
        processed_data = []
        for path in paths:
            if path.lower().endswith(
                (".mp4", ".avi", ".mov", ".mkv")
            ):  # Video file path
                # video_frames = self.load_video(path, num_frames)
                processed_data.append(
                    {
                        "type": "video",
                        "video": path,
                        "max_pixels": 360 * 420,
                        "fps": 1.0,
                    }
                )
            elif path.lower().endswith(".npy"):  # NumPy file
                np_array = np.load(path)
                if np_array.ndim == 3:  # Single image
                    image = Image.fromarray(np_array.astype("uint8"), "RGB")
                    processed_data.append({"type": "image", "image": image})
                elif np_array.ndim == 4:  # Multiple frames
                    frames = [
                        Image.fromarray(frame.astype("uint8"), "RGB")
                        for frame in np_array
                    ]
                    processed_data.append({"type": "video", "video": frames})
                else:
                    raise ValueError(f"Unexpected shape for NumPy array in {path}")
            else:  # Regular image file
                image = Image.open(path).convert("RGB")
                processed_data.append({"type": "image", "image": image})
        return processed_data

    def load_video(self, video_path, max_frames_num):
        print(f"Going into load_video method.")
        vr = VideoReader(video_path, ctx=cpu(0))
        total_frame_num = len(vr)
        uniform_sampled_frames = np.linspace(
            0, total_frame_num - 1, max_frames_num, dtype=int
        )
        frame_idx = uniform_sampled_frames.tolist()
        spare_frames = vr.get_batch(frame_idx).numpy()
        return [Image.fromarray(frame) for frame in spare_frames]

    @torch.no_grad()
    def generate(
        self,
        images: List[str],
        num_frames: int = 16,
        max_new_tokens: int = 256,
        use_system_prompt: bool = False,
        conditioned_system_prompt: str = None, 
    ) -> List[str]:

        text = "Describe this video in detail."

        if conditioned_system_prompt is not None:
            prompt_to_use = conditioned_system_prompt
        else:
            prompt_to_use = CAPTIONING_SYSPROMPT

        generated_texts = []
        for data in images:

            if use_system_prompt:
                text = "Output a similarity score from 1 to 5."
                messages = [
                {
                    "role": "system",
                    "content": prompt_to_use, 
                },
                {
                    "role": "user",
                    "content": [{"type": "text", "text": text}, data],
                },
            ]   

            else:
                messages = [
                        {"role": "user", "content": [data, {"type": "text", "text": text}]}
                    ]
            

                
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

            image_inputs, video_inputs = process_vision_info(messages)

            inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            inputs = inputs.to(self.device)

            with torch.inference_mode():
                generated_ids = self.model.generate(
                    **inputs, 
                    max_new_tokens=max_new_tokens, 
                )
                generated_ids_trimmed = [
                    out_ids[len(in_ids) :]
                    for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                ]
                text = self.processor.batch_decode(
                    generated_ids_trimmed,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )[0].strip()
                generated_texts.append(text)

        return generated_texts

    @torch.no_grad()
    def compute_logprobs(
        self,
        images: List[str],
        num_frames: int = 16,
        max_new_tokens: int = 256,
        use_system_prompt: bool = False,
        conditioned_system_prompt: str = None, 
        ) -> List[str]:

        text = "Describe this video in detail."

        if conditioned_system_prompt is not None:
            prompt_to_use = conditioned_system_prompt
        else:
            prompt_to_use = CAPTIONING_SYSPROMPT
        
        generated_probs = []
        for data in images:
            if use_system_prompt:
                text = "Answer the question with Yes or No."
                messages = [
                {
                    "role": "system",
                    "content": prompt_to_use, 
                },
                {
                    "role": "user",
                    "content": [{"type": "text", "text": text}, data],
                },
            ]   

            else:
                messages = [
                        {"role": "user", "content": [data, {"type": "text", "text": text}]}
                    ]



            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
                

            image_inputs, video_inputs = process_vision_info(messages)

            inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            inputs = inputs.to(self.device)

            with torch.inference_mode():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=1,
                    do_sample=False, 
                    output_scores=True,
                    return_dict_in_generate=True
                )
                answer = 'Yes'
                scores = outputs.scores[0]
                probs = torch.nn.functional.softmax(scores, dim=-1)
                yes_token_id = self.processor.tokenizer.encode(answer)[0]
                lm_prob = probs[0, yes_token_id].item()
                generated_probs.append(lm_prob)



        return generated_probs