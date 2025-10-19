import os
import re
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
import matplotlib.pyplot as plt
from collections import defaultdict
import pandas as pd

from scipy.stats import kendalltau
from scipy.stats import rankdata
from scipy.stats import spearmanr
from tabulate import tabulate
from textwrap import dedent



def set_all_seeds(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    set_seed(seed)


def get_gt_conditioned_similarities(user_inputs_path):
    """
    Load user inputs from a JSON file and compute ground truth similarities.
    """
    with open(user_inputs_path, "r") as f:
        data = json.load(f)

    
    pair_scores = {}

    for entry in data:
        pair_id = entry["video_pair_id"]
        inputs = entry.get("user_inputs", {})
        keys = ["MainAction", "MainSubjects", "Location", "OrderofActions", "MainObjects"]
        similarities = {key: float(inputs.get(key, None)) for key in keys}  
        similarities["OrderOfActions"] = similarities.pop("OrderofActions", None)
        keys[3] = "OrderOfActions"  
        if not pair_id in pair_scores:
            pair_scores[pair_id] = {key: [] for key in keys}  # Initialize with None
        for key in keys:
            pair_scores[pair_id][key].append(similarities[key]) # store the similarity for this key

    output = []
    for pair_id, scores in pair_scores.items(): 
        avg_similarity = {key: np.mean(scores[key]) for key in scores}  # Compute average similarity
    
        video1, video2 = pair_id.split("-")
        output.append({
            "video1": f"videos_trimmed/{video1}",
            "video2": f"videos_trimmed/{video2}",
            "conditioned_similarity": avg_similarity
        })

    with open("gt_conditioned_similarities.json", "w") as f:
        json.dump(output, f, indent=4)

    return output

def normalize_pair(video1, video2):
    """
    Normalize the pair by removing any additional directory info.
    """
    v1 = os.path.basename(video1)  # e.g. "3863_003.mp4"
    v2 = os.path.basename(video2)  # e.g. "883_000.mp4"
    return tuple([v1, v2])


def main(args):
    """
    Main function to extract visual features from videos.
    """
    set_all_seeds(42)
    
    if args.compute_gt:
       gt_conditioned_similarities = get_gt_conditioned_similarities(args.user_inputs) 

    gt_sim_dict = {}
    for entry in gt_conditioned_similarities:
        pair = normalize_pair(entry["video1"], entry["video2"])
        for key, value in entry["conditioned_similarity"].items():
            if key not in gt_sim_dict:
                gt_sim_dict[key] = {}
            gt_sim_dict[key][pair] = value
    

    keys_original = ["MainAction", "MainSubjects", "Location", "MainObjects", "OrderOfActions"]
    keys=[re.sub(r'(?<!^)(?=[A-Z])', ' ', key) for key in keys_original]
    keys = [key.replace(" ", "_") for key in keys]
    models = [  
                "facebook/dinov2-large", 
                "MCG-NJU/videomae-large",
                "mPLUG/mPLUG-Owl3-7B-240728",
                "OpenGVLab/InternVL2_5-8B",
                "Qwen/Qwen2.5-VL-7B-Instruct",
                "openai/clip-vit-large-patch14",
                "InternVideo/InternVideo-MM-L-14",
                "VQAScore/LLaVA-OneVision",
                "mPLUG-Owl3-7B-240728", 
                "LLaVA-OneVision-Qwen2-0.5B",
                "LLaVA-OneVision-Qwen2-7B",
                "LLaVA-NeXT-Video-7B",
                "LLaVA-OneVision-Qwen2-7B-HF",
                "Qwen2.5-VL-3B-Instruct", 
                "Qwen2.5-VL-7B-Instruct", 
                "InternVL2_5-4B", 
                "InternVL2_5-8B",  
                "InternVL3-8B",
                "Gemini-2.0-Flash",
    ]
    


    models_2_table = {
        "InternVL2_5-8B": "InternVL2.5-8B",
        "InternVL2_5-4B": "InternVL2.5-4B",
        "InternVL3-8B": "InternVL3-8B",
        "LLaVA-NeXT-Video-7B": "LLaVA-NeXT-Video-7B",
        "LLaVA-OneVision-Qwen2-0.5B": "LLaVA-OneVision-Qwen2-0.5B",
        "LLaVA-OneVision-Qwen2-7B-HF": "LLaVA-OneVision-Qwen2-7B-HF",
        "LLaVA-OneVision-Qwen2-7B": "LLaVA-OneVision-Qwen2-7B",
        "mPLUG-Owl3-7B-240728": "mPLUG-Owl3-7B",
        "Qwen2.5-VL-3B-Instruct": "Qwen2.5-VL-3B",
        "Qwen2.5-VL-7B-Instruct": "Qwen2.5-VL-7B",
        "Qwen2.5-VL-7B-Instruct_32": "Qwen2.5-VL-7B_32",
        "Qwen2.5-VL-7B-Instruct_new": "Qwen2.5-VL-7B_new",
        "Gemini-2.0-Flash": "Gemini-2.0-Flash",
        "VQAScore/LLaVA-OneVision" : "VQAScore",
        "openai/clip-vit-large-patch14": "CLIPScore (CLIP)",
        "InternVideo/InternVideo-MM-L-14" : "CLIPScore (InternVideo)",
        "facebook/dinov2-large": "DINOv2", 
        "MCG-NJU/videomae-large" : "VideoMAE",
        "mPLUG/mPLUG-Owl3-7B-240728": "SBERT (mPLUG-Owl3-7B)",
        "OpenGVLab/InternVL2_5-8B" : "SBERT (InternVL2_5-8B)",
        "Qwen/Qwen2.5-VL-7B-Instruct": "SBERT (Qwen2.5-VL-7B)",
    }

    model_2_dir = {
         "VQAScore/LLaVA-OneVision" : "LLaVA-OneVision_mPLUG-Owl3-7B-240728",
        "openai/clip-vit-large-patch14": "clip-vit-large-patch14_mPLUG-Owl3-7B-240728",
        "InternVideo/InternVideo-MM-L-14" : "InternVideo-MM-L-14_mPLUG-Owl3-7B-240728",
        "facebook/dinov2-large": "dinov2-large", 
        "MCG-NJU/videomae-large" : "videomae-large",
        "mPLUG/mPLUG-Owl3-7B-240728": "mPLUG-Owl3-7B-240728",
        "OpenGVLab/InternVL2_5-8B" : "InternVL2_5-8B",
        "Qwen/Qwen2.5-VL-7B-Instruct": "Qwen2.5-VL-7B-Instruct",
    }

    computed_similarities = {models[i]: {key: {} for key in keys} for i in range(len(models))}

    for model in models:
        for key in keys:
            if model in [
                        "VQAScore/LLaVA-OneVision", 
                        "openai/clip-vit-large-patch14", 
                        "InternVideo/InternVideo-MM-L-14", 
                        "facebook/dinov2-large", 
                        "MCG-NJU/videomae-large",
                        "mPLUG/mPLUG-Owl3-7B-240728",
                        "OpenGVLab/InternVL2_5-8B",
                        "Qwen/Qwen2.5-VL-7B-Instruct",
                        ]:

                with open(os.path.join(args.unconditioned_similarity_dir, model_2_dir[model], f"{model.split("/")[-1]}_similarities.json"), "r") as f:
                    data = json.load(f)
            else:
                with open(os.path.join(args.similarity_dir, key, model, f"{model}_{key}_similarities.json"), "r") as f:
                    data = json.load(f)

            for entry in data:
                video1 = entry["video1"]
                video2 = entry["video2"]
                similarity = entry["similarity"]
                # ONLY here convert those that could not be parsed
                if similarity == 0.0:
                    similarity = 3.0
                    
                pair = normalize_pair(video1, video2)
                computed_similarities[model][key][pair] = similarity
    

    results_dict = defaultdict(dict)

    for model in models:
        for key in keys:
            computed_sim = computed_similarities[model][key]
            gt_sim = gt_sim_dict[key.replace("_", "")]  # Replace "_" with "" to match the keys in gt_sim_dict
            
            computed_scores, gt_scores = [], []

            for pair in computed_sim.keys():
                if pair in gt_sim:
                    computed_scores.append(computed_sim[pair])
                    gt_scores.append(gt_sim[pair])

            if computed_scores and gt_scores:
                spearman_result = spearmanr(computed_scores, gt_scores)
                kendall_result = kendalltau(computed_scores, gt_scores)


            results_dict[model][key] = {
                "spearman": round(spearman_result.correlation * 100, 2),
                "kendall": round(kendall_result.correlation * 100, 2),
            }
        
            
        keys = ['Main_Action', 'Main_Subjects', 'Main_Objects', 'Location', 'Order_Of_Actions']
        key_titles = {
            'Main_Action': 'Main Action',
            'Main_Subjects': 'Main Subjects',
            'Main_Objects': 'Main Objects',
            'Location': 'Location',
            'Order_Of_Actions': 'Order of Actions'
        }

        
    latex = dedent(r"""
        \begin{table}[ht!]
        \def\arraystretch{1.15}
        \centering
        \caption{Baselines for video similarity conditioned on concepts.}
        \scriptsize
        \begin{tabularx}{\textwidth}{
        >{\raggedright\arraybackslash}p{3.2cm}""" +
        ">{\\centering\\arraybackslash}X" * (2 * len(keys)) +
        r"""}
        \toprule
        \textbf{Model} & """ + " & ".join(
            [f"\\multicolumn{{2}}{{c}}{{\\texttt{{{key_titles[key]}}}}}" for key in keys]
        ) + r" \\" + "\n" + \
        " & " + " & ".join(["$\\rho$ & $\\tau$" for _ in keys]) + r" \\" + "\n" + \
        r"\midrule" + "\n" + \
        rf"\multicolumn{{{1 + 2 * len(keys)}}}{{c}}{{\textit{{VQA-based}}}} \\[-1ex] \\" + "\n"
    )

    # Add data rows
    for model, values in results_dict.items():
        row = models_2_table[model].replace("_", r"\_") + " & "
        for key in keys:
            spearman = values[key]['spearman']
            kendall = values[key]['kendall']
            row += f"{spearman:.1f} & {kendall:.2f} & "
        latex += row.rstrip(" & ") + r" \\" + "\n"

    latex += r"""\bottomrule
        \end{tabularx}
        \label{tab:semantic-vqa}
        \end{table}
        """

    with open("tables/conditioned_correlation_table.tex", "w") as f:
        f.write(latex)


    headers = ["Model"]
    for key in keys:
        headers.append(f"{key_titles[key]} (ρ)")
        headers.append(f"{key_titles[key]} (τ)")

    
    rows = []
    for model in models:
        model_name = models_2_table[model]
        row = [model_name]
        for key in keys:
            spearman = results_dict[model][key]["spearman"]
            kendall = results_dict[model][key]["kendall"]
            row.extend([f"{spearman:.1f}", f"{kendall:.2f}"])
        rows.append(row)

    
    print(f"\033[93mCorrelation metrics computed for all models and concepts:\033[0m")
    print(tabulate(rows, headers=headers, tablefmt="fancy_grid"))
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract visual features from videos.")

    parser.add_argument(
        "--similarity_dir",
        type=str,
        default="baselines/computed_conditioned_similarities/",
        required=False,
    )
    parser.add_argument(
        "--unconditioned_similarity_dir",
        type=str,
        default="baselines/computed_similarities/",
        required=False,
    )
    parser.add_argument(
        "--compute_gt",
        type=bool,
        default=True,
        required=False,
    )
    parser.add_argument(
        "--user_inputs",
        type=str,
        default="human_annotations/vicos.user_inputs.json",
        required=False,
        help="Path to the user inputs JSON file.",
    
    )

    args = parser.parse_args()

    main(args)
