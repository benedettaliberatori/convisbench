import os
import re
import json
import argparse
import random
import torch
import numpy as np
from transformers import set_seed
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import pandas as pd

from scipy.stats import kendalltau
from scipy.stats import rankdata
from scipy.stats import spearmanr
from tabulate import tabulate
from textwrap import dedent
from sklearn.metrics import ndcg_score


def set_all_seeds(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    set_seed(seed)


def get_gt_rankings(gt_rankings):
    with open(gt_rankings, "r") as f:
        gt_rankings = json.load(f)

    keys = ["MainAction", "MainSubjects", "Location", "MainObjects", "OrderOfActions"]
    rg_rankings_per_category = {key: [] for key in keys}

    for entry in gt_rankings:
        for key in keys:
            if key in entry:
                rg_rankings_per_category[key].append({entry["anchor"]: entry[key]})

    return rg_rankings_per_category


def rank_values(field_dict):
    items = list(field_dict.items())
    sorted_items = sorted(items, key=lambda x: -x[1])
    ranks = {}
    rank = 1
    last_value = None
    value_to_rank = {}

    for key, value in sorted_items:
        if value not in value_to_rank:
            value_to_rank[value] = rank
            rank += 1
        ranks[key] = value_to_rank[value]

    ordered_ranks = {k: ranks[k] for k in field_dict.keys()}
    return ordered_ranks


def main(args):

    set_all_seeds(42)

    gt_rankings_per_category = get_gt_rankings(args.gt_rankings)

    keys_original = [
        "MainAction",
        "MainSubjects",
        "Location",
        "MainObjects",
        "OrderOfActions",
    ]
    keys = [re.sub(r"(?<!^)(?=[A-Z])", " ", key) for key in keys_original]
    keys = [key.replace(" ", "_") for key in keys]
    models = [
        "mPLUG-Owl3-7B-240728",
        "LLaVA-OneVision-Qwen2-0.5B",
        "LLaVA-OneVision-Qwen2-7B",
        "LLaVA-NeXT-Video-7B",
        "Qwen2.5-VL-3B-Instruct",
        "Qwen2.5-VL-7B-Instruct_new",
        "InternVL2_5-8B",
        "InternVL2_5-4B",
        "InternVL3-8B",
        "VQAScore/LLaVA-OneVision",
        "openai/clip-vit-large-patch14",
        "InternVideo/InternVideo-MM-L-14",
        "facebook/dinov2-large", 
        "MCG-NJU/videomae-large",
        "mPLUG/mPLUG-Owl3-7B-240728",
        "OpenGVLab/InternVL2_5-8B",
        "Qwen/Qwen2.5-VL-7B-Instruct",

    ]

    models_2_table = {
        "InternVL2_5-8B": "InternVL2.5-8B",
        "InternVL2_5-4B": "InternVL2.5-4B",
        "InternVL3-8B": "InternVL3-8B",
        "LLaVA-OneVision-Qwen2-0.5B": "LLaVA-OneVision-Qwen2-0.5B",
        "LLaVA-OneVision-Qwen2-7B-HF": "LLaVA-OneVision-Qwen2-7B-HF",
        "LLaVA-OneVision-Qwen2-7B": "LLaVA-OneVision-Qwen2-7B",
        "mPLUG-Owl3-7B-240728": "mPLUG-Owl3-7B",
        "Qwen2.5-VL-3B-Instruct": "Qwen2.5-VL-3B",
        "Qwen2.5-VL-7B-Instruct_new": "Qwen2.5-VL-7B",
        "Qwen2.5-VL-7B-Instruct_32": "Qwen2.5-VL-7B_32",
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

    computed_rankings = {
        models[i]: {key: {} for key in keys} for i in range(len(models))
    }

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
                with open(
                    os.path.join(
                        args.retrieval_scores_unconditioned_dir, model.split('/')[-1], f"{model.split('/')[-1]}_unconditioned_ranking.json"
                    ),
                    "r",
                ) as f:
                    data = json.load(f)
            else:
                
                with open(
                    os.path.join(
                        args.retrieval_scores_dir, key, model, f"{model}_{key}_ranking.json"
                    ),
                    "r",
                ) as f:
                    data = json.load(f)

            top1_correct = 0
            bottom1_correct = 0
            recall_all = 0
            recall_bottom_all = 0

            precision_all = 0
            precision_bottom_all = 0
            accuracy_all = 0
            accuracy_bottom_all = 0

            f1_all = 0
            f1_bottom_all = 0

            total_accuracy = 0
            mse_all = 0
            count = 0
            ndgcs = []
            taus = []

            for anchor, ranked_targets in data.items():
                gt_rankings = [
                    list(d.values())[0]
                    for d in gt_rankings_per_category[key.replace("_", "")]
                    if anchor in d
                ]
                if len(gt_rankings) == 0:
                    continue

                assert (
                    len(gt_rankings) == 1
                ), f"Expected 1 gt ranking, got {len(gt_rankings), gt_rankings}"


                ranked_targets = rank_values(ranked_targets)
                
                # Hit@1 = Recall@1
                gt_rankings = gt_rankings[0]
                gt_max = min(gt_rankings.values())
                gt_top_set = {k for k, v in gt_rankings.items() if v == gt_max}
                model_top1 = min(ranked_targets, key=ranked_targets.get)
                if model_top1 in gt_top_set:
                    top1_correct += 1

                # Hit@1 Bottom
                gt_min = max(gt_rankings.values())
                gt_bottom_set = {k for k, v in gt_rankings.items() if v == gt_min}
                model_bottom1 = max(ranked_targets, key=ranked_targets.get)
                if model_bottom1 in gt_bottom_set:
                    bottom1_correct += 1

                # NGDC
                gt_items = list(gt_rankings.keys())
                gt_relevance = [gt_rankings[k] for k in gt_items]
                pred_scores = [ranked_targets[k] for k in gt_items]

                ndcg = ndcg_score([gt_relevance], [pred_scores])
                ndgcs.append(ndcg)


                max_gt_rank = min(gt_rankings.values())
                relevant_items = {k for k, v in gt_rankings.items() if v == max_gt_rank}

                predicted_items = {
                    k
                    for k, v in ranked_targets.items()
                    if v == min(ranked_targets.values())
                }

                correct = relevant_items & predicted_items

                recall = len(correct) / len(relevant_items) if relevant_items else 0.0
                recall_all += recall
                accuracy = (
                    len(correct) / len(predicted_items) if predicted_items else 0.0
                )
                accuracy_all += accuracy
                precision = (
                    len(correct) / len(predicted_items) if predicted_items else 0.0
                )
                precision_all += precision
                
                f1 = (
                    2 * recall * precision / (recall + precision)
                    if (recall + precision) > 0
                    else 0
                )
                f1_all += f1

                # Recall/Accuracy (of Bottom prediction)
                max_gt_rank = max(gt_rankings.values())
                relevant_items = {k for k, v in gt_rankings.items() if v == max_gt_rank}
                predicted_items = {
                    k
                    for k, v in ranked_targets.items()
                    if v == max(ranked_targets.values())
                }
                correct = relevant_items & predicted_items
                recall_bottom = len(correct) / len(relevant_items) if relevant_items else 0.0
                recall_bottom_all += recall_bottom
                accuracy_bottom = (
                    len(correct) / len(predicted_items) if predicted_items else 0.0
                )
                precision_bottom = (
                    len(correct) / len(predicted_items) if predicted_items else 0.0
                )

                f1_bottom = (
                    2 * recall_bottom * precision_bottom
                    / (recall_bottom + precision_bottom)
                    if (recall_bottom + precision_bottom) > 0
                    else 0
                )
                f1_bottom_all += f1_bottom

                precision_bottom_all += precision_bottom
                accuracy_bottom_all += accuracy_bottom
                gt_items = list(gt_rankings.keys())
                gt_scores = [gt_rankings[k] for k in gt_items]
                pred_scores = [ranked_targets[k] for k in gt_items]
                tau, _ = kendalltau(gt_scores, pred_scores)
                count += 1

                taus.append(tau)

                gt_rankings = {k: v for k, v in sorted(gt_rankings.items(), key=lambda item: item[0])}
                ranked_targets = {k: v for k, v in sorted(ranked_targets.items(), key=lambda item: item[0])}

                gt_rankings_values = list(gt_rankings.values())
                ranked_targets_values = list(ranked_targets.values())
                gt_rankings_values = [float(x) for x in gt_rankings_values]
                ranked_targets_values = [float(x) for x in ranked_targets_values]

                accuracy = np.sum(
                    np.array(ranked_targets_values) == np.array(gt_rankings_values)
                ) / len(gt_rankings_values)

                total_accuracy += accuracy
                mse = np.mean(
                    np.square(
                        np.array(ranked_targets_values) - np.array(gt_rankings_values)
                    )
                )
                mse_all += mse
                

            taus = [tau for tau in taus if not np.isnan(tau)] 

            computed_rankings[model][key] = {
                "top1_correct": top1_correct,
                "bottom1_correct": bottom1_correct,
                "count": count,
                "Hit@1 Top": (top1_correct / count) * 100,
                "Hit@1 Bottom": (bottom1_correct / count) * 100,
                "Kendall Tau": np.mean(taus),
                "NDCG": np.mean(ndgcs),
                "Recall (Top)": (recall_all / count) * 100,
                "Precision (Top)": (precision_all / count) * 100,
                "Accuracy (Top)": (accuracy_all / count) * 100,
                "F1 (Top)": (f1_all / count) * 100,
                "Recall (Bottom)": (recall_bottom_all / count) * 100,
                "Precision (Bottom)": (precision_bottom_all / count) * 100,
                "Accuracy (Bottom)": (accuracy_bottom_all / count) * 100,
                "F1 (Bottom)": (f1_bottom_all / count) * 100,
                "Overall Accuracy": (total_accuracy / count) * 100,
                "MSE": (mse_all / count),
            }

    from pprint import pprint

    with open("computed_rankings_tau.json", "w") as f:
        json.dump(computed_rankings, f, indent=4)

    print("Computed rankings:")
    for model in models:
        for key in keys:
            result = computed_rankings[model].get(key, {})
            top1 = result.get("Top-1 Accuracy", 0)
            bottom1 = result.get("Bottom-1 Accuracy", 0)
            tau = result.get("Kendall Tau", 0)
            print(f"{key}: Top-1: {top1:.2f}, Bottom-1: {bottom1:.2f}, Tau: {tau:.2f}")

    keys = [
        "Main_Action",
        "Main_Subjects",
        "Main_Objects",
        "Location",
        "Order_Of_Actions",
    ]
    key_titles = {
        "Main_Action": "Main Action",
        "Main_Subjects": "Main Subjects",
        "Main_Objects": "Main Objects",
        "Location": "Location",
        "Order_Of_Actions": "Order of Actions",
    }

    latex = dedent(
    r"""
    \begin{table}[ht!]
    \def\arraystretch{1.15}
    \centering
    \caption{Recall, Precision, and F1 (Top) for models on video retrieval conditioned on concepts.}
    \scriptsize
    \begin{tabularx}{\textwidth}{
    >{\raggedright\arraybackslash}p{3.2cm}"""
    + ">{\\centering\\arraybackslash}X" * (3 * len(keys[:-1])) 
    + r"""}
    \toprule
    \textbf{Model} & """
    + " & ".join(
        [
            f"\\multicolumn{{3}}{{c}}{{\\texttt{{{key_titles[key]}}}}}"
            for key in keys[:-1]
        ]
    )
    + r" \\"
    + "\n"
    + " & "
    + " & ".join(["Rec & Prec & F1" for _ in keys[:-1]])
    + r" \\"
    + "\n"
    + r"\midrule"
    + "\n"
)

    for model in models:
        model_name = models_2_table.get(model, model)
        row = [model_name]
        for key in keys[:-1]:  
            result = computed_rankings[model].get(key, {})
            top1 = result.get("Recall (Top)", 0)
            prec = result.get("Precision (Top)", 0)
            f1 = result.get("F1 (Top)", 0)
            row.append(f"{top1:.1f}")
            row.append(f"{prec:.1f}")
            row.append(f"{f1:.1f}")
        latex += " & ".join(row) + r" \\" + "\n"

    latex += r"""\bottomrule
    \end{tabularx}
    \end{table}
    """
    with open("tables/retrieval_table_recall_precision_f1.tex", "w") as f:
        f.write(latex)


    headers = ["Model"]
    for key in keys[:-1]:
        headers.extend([
            f"{key_titles[key]} - Rec@1",
            f"{key_titles[key]} - Prec@1",
            f"{key_titles[key]} - F1@1"
        ])

    table_data = []
    for model in models:
        model_name = models_2_table.get(model, model)
        row = [model_name]
        for key in keys[:-1]:
            result = computed_rankings[model].get(key, {})
            recall = result.get("Recall (Top)", 0)
            precision = result.get("Precision (Top)", 0)
            f1 = result.get("F1 (Top)", 0)
            row.extend([f"{recall:.2f}", f"{precision:.2f}", f"{f1:.2f}"])
        table_data.append(row)


    print("\nRecall@1, Precision@1, and F1@1 Table:")
    print(tabulate(table_data, headers=headers, tablefmt="grid"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract visual features from videos.")

    parser.add_argument(
        "--retrieval_scores_dir",
        type=str,
        default="baselines/computed_retrieval_scores/",
        required=False,
    )
    parser.add_argument(
        "--retrieval_scores_unconditioned_dir",
        type=str,
        default="baselines/computed_retrieval_scores_unconditioned/",
        required=False,
    )
    parser.add_argument(
        "--gt_rankings",
        type=str,
        default="human_annotations/ranked.json",
        required=False,
    )
    parser.add_argument(
        "--verbose",
        type=bool,
        default=False,
        required=False,
    )

    args = parser.parse_args()

    main(args)
