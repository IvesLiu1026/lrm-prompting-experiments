import os
import re
import json
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from datasets import load_dataset

ds = load_dataset("cais/mmlu", "all")
subject_map = {i: ds['test'][i]['subject'] for i in range(len(ds['test']))}

def load_jsonl_to_list(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]

def save_flip_subject_csv(flip_subject_stats, output_path, mode="run"):
    rows = []
    for subj, stats in flip_subject_stats.items():
        total = stats["flip_success"] + stats["backfire"]
        success_ratio = stats["flip_success"] / total if total > 0 else 0
        rows.append({
            "Subject": subj,
            "Flip Success": stats["flip_success"],
            "Backfire": stats["backfire"],
            "Total Flips": total,
            "Success Ratio": round(success_ratio, 4)
        })
    df = pd.DataFrame(rows)
    df = df.sort_values("Total Flips", ascending=False)
    if mode == "run":
        df.to_csv(output_path, index=False)
    return df

def plot_flip_subjects(df, prompt_name):
    top_df = df.head(20)
    plt.figure(figsize=(10, 6))
    plt.barh(top_df["Subject"], top_df["Flip Success"], label="Flip Success", color="green")
    plt.barh(top_df["Subject"], top_df["Backfire"], left=top_df["Flip Success"], label="Backfire", color="red")
    plt.xlabel("Count")
    plt.title(f"Top 20 Subjects by Flip Success and Backfire ({prompt_name})")
    plt.legend()
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(f"{plot_folder}/{prompt_name}_flip_subjects.png")
    plt.close()

def compare_word_counts_per_question(baseline_file, prompt_file, prompt_name, mode="run"):
    def get_word_counts(data):
        result = {}
        for item in data:
            idx = item.get("index")
            response = item.get("response", "")
            total_words = len(response.strip().split())
            think_words = 0
            if "<think>" in response and "</think>" in response:
                think_blocks = re.findall(r"<think>(.*?)</think>", response, flags=re.DOTALL)
                think_words = sum(len(block.strip().split()) for block in think_blocks)
            result[idx] = {
                "total_words": total_words,
                "think_words": think_words,
                "subject": subject_map.get(idx, "Unknown")
            }
        return result

    base_data = load_jsonl_to_list(baseline_file)
    prompt_data = load_jsonl_to_list(prompt_file)
    base_counts = get_word_counts(base_data)
    prompt_counts = get_word_counts(prompt_data)

    rows = []
    for idx in sorted(set(base_counts) & set(prompt_counts)):
        b = base_counts[idx]
        p = prompt_counts[idx]
        rows.append({
            "index": idx,
            "subject": b["subject"],
            "baseline_total": b["total_words"],
            "prompt_total": p["total_words"],
            "diff_total": p["total_words"] - b["total_words"],
            "baseline_think": b["think_words"],
            "prompt_think": p["think_words"],
            "diff_think": p["think_words"] - b["think_words"],
        })

    df = pd.DataFrame(rows)
    if mode == "run":
        os.makedirs("utils", exist_ok=True)
        df.to_csv(f"utils/word_diff_standard_vs_{prompt_name}.csv", index=False)

    os.makedirs("plots", exist_ok=True)

    # Violin plots
    plt.figure(figsize=(14, 6))
    sns.violinplot(data=df, x="diff_total", y="subject", density_norm="width", inner="quartile")
    plt.title(f"Distribution of Total Word Count Differences (standard → {prompt_name})")
    plt.tight_layout()
    if mode == "run":
        plt.savefig(f"plots/violin_diff_total_{prompt_name}.png")
        plt.close()
    else:
        plt.savefig(f"plots_test/violin_diff_total_{prompt_name}.png")
        plt.close()

    plt.figure(figsize=(14, 6))
    sns.violinplot(data=df, x="diff_think", y="subject", density_norm="width", inner="quartile", color="orange")
    plt.title(f"Distribution of <think> Word Count Differences (standard → {prompt_name})")
    plt.tight_layout()
    if mode == "run":
        plt.savefig(f"plots/violin_diff_think_{prompt_name}.png")
        plt.close()
    else:
        plt.savefig(f"plots_test/violin_diff_think_{prompt_name}.png")
        plt.close()

    # Scatter plot
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=df, x="baseline_total", y="prompt_total", alpha=0.5)
    sns.regplot(data=df, x="baseline_total", y="prompt_total", scatter=False, color="red")
    plt.title(f"Total Word Count: Baseline vs. {prompt_name}")
    plt.xlabel("Baseline Word Count")
    plt.ylabel(f"{prompt_name} Word Count")
    plt.tight_layout()
    if mode == "run":
        plt.savefig(f"plots/scatter_total_baseline_vs_{prompt_name}.png")
        plt.close()
    else:
        plt.savefig(f"plots_test/violin_diff_total_baseline_vs_{prompt_name}.png")
        plt.close()


def analyze_single_output(prompt_name, data, baseline_map, mode="run"):
    think_word_count = 0
    think_response_count = 0
    total = correct = think_count = wait_token_count = total_word_count = 0
    index_missing_list = response_ans_missing_list = response_missing_list = []
    index_missing_count = response_ans_missing_count = response_missing_count = 0
    output_all_index = set(item.get("index") for item in data if item.get("index") is not None)
    dataset_all_index = set(range(len(ds['test'])))
    total_latency = completion_tokens = 0
    subject_stats = defaultdict(lambda: {"total": 0, "correct": 0})
    flip_subject_stats = defaultdict(lambda: {"flip_success": 0, "backfire": 0})

    index_missing_list = list(dataset_all_index - output_all_index)
    index_missing_count = len(index_missing_list)

    for item in data:
        idx = item.get("index")
        correct_flag = item.get("correct")
        response = item.get("response", "")
        response_ans = item.get("response_ans")
        total += 1

        if correct_flag:
            correct += 1
        if "<think>" in response and "</think>" in response:
            think_count += 1
            think_blocks = re.findall(r"<think>(.*?)</think>", response, flags=re.DOTALL)
            think_words = sum(len(block.strip().split()) for block in think_blocks)
            think_word_count += think_words
            if think_words > 0:
                think_response_count += 1
        if response_ans is None:
            response_ans_missing_count += 1
            response_ans_missing_list.append(idx)
        if response is None:
            response_missing_count += 1
            response_missing_list.append(idx)

        wait_token_count += response.lower().count("wait")
        total_word_count += len(response.split())

        token_info = item.get("token_usage", {})
        completion_tokens += token_info.get("completion", 0)
        total_latency += item.get("time_usage", 0)

        subj = subject_map.get(idx, "Unknown")
        subject_stats[subj]["total"] += 1
        subject_stats[subj]["correct"] += int(correct_flag)

        if baseline_map and idx in baseline_map and correct_flag is not None:
            baseline_correct = baseline_map[idx]
            if baseline_correct and not correct_flag:
                flip_subject_stats[subj]["backfire"] += 1
            elif not baseline_correct and correct_flag:
                flip_subject_stats[subj]["flip_success"] += 1

    def pct(x): return f"{x} ({x/total:.2%})" if total > 0 else "0 (0.00%)"
    flip_success = sum(d["flip_success"] for d in flip_subject_stats.values())
    backfire = sum(d["backfire"] for d in flip_subject_stats.values())
    stay_correct = sum(1 for item in data if baseline_map.get(item.get("index")) is True and item.get("correct") is True) if baseline_map else 0
    flip_failure = sum(1 for item in data if baseline_map.get(item.get("index")) is False and item.get("correct") is False) if baseline_map else 0

    
    print(f"\nPrompt: {prompt_name}")
    print(f"Total: {total}")
    print(f"Index Missing: {index_missing_count} | Response Ans Missing: {response_ans_missing_count} | Response Missing: {response_missing_count}")
    print(f"Index Missing List: {index_missing_list}")
    print(f"Response Ans Missing List: {response_ans_missing_list}")
    print(f"Response Missing List: {response_missing_list}")
    print(f"<think>: {pct(think_count)} | Think Words Avg: {think_word_count / think_response_count:.2f}")
    print(f"Accuracy: {pct(correct)}")
    print(f"'Wait' token count: {wait_token_count} | Avg per question: {wait_token_count / total:.2f}")
    print(f"Words: {total_word_count / total:.2f} | Tokens: {completion_tokens / total:.2f}")
    print(f"Latency: {total_latency / total:.2f}s")
    if baseline_map:
        print("Flip:")
        print(f"   Success: {pct(flip_success)}")
        print(f"   Failure: {pct(flip_failure)}")
        print(f"   Stayed Correct: {pct(stay_correct)}")
        print(f"   Backfire: {pct(backfire)}")

    if mode == "run":
        missing_dir = "log/missing_lists"
        os.makedirs(missing_dir, exist_ok=True)

        missing_output = {
            "index_missing_list": index_missing_list,
            "response_ans_missing_list": response_ans_missing_list,
            "response_missing_list": response_missing_list
        }
        with open(os.path.join(missing_dir, f"{prompt_name}_missing.json"), "w", encoding="utf-8") as f:
            json.dump(missing_output, f, indent=2)

        

    return {item["index"]: item["correct"] for item in data if item.get("correct") is not None}

        


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--i", type=str, nargs="*", help="Prompt name(s) without _output.json")
    parser.add_argument("--all", action="store_true", help="Analyze all prompts")
    parser.add_argument("--folder", type=str, default="output", help="Folder containing outputs")
    parser.add_argument("--mode", type=str, choices=["run", "test"], default="run", help="Mode: run or test")
    args = parser.parse_args()
    
    if args.mode == "run":
        plot_folder = "output/plots"
    else:
        plot_folder = "output/plots_test"

    os.makedirs(plot_folder, exist_ok=True)

    files = os.listdir(args.folder)
    if "standard_output.json" not in files:
        raise FileNotFoundError("❌ standard_output.json not found in specified folder.")

    baseline_path = os.path.join(args.folder, "standard_output.json")
    baseline_data = load_jsonl_to_list(baseline_path)
    baseline_map = analyze_single_output("standard", baseline_data, {}, mode=args.mode)

    input_files = []
    if args.all:
        input_files = [f.replace("_output.json", "") for f in files if f.endswith("_output.json") and f != "standard_output.json"]
    elif args.i:
        input_files = args.i
    else:
        raise ValueError("Please provide --i or --all")

    for name in input_files:
        path = os.path.join(args.folder, f"{name}_output.json")
        data = load_jsonl_to_list(path)
        analyze_single_output(name, data, baseline_map, mode=args.mode)
        compare_word_counts_per_question(baseline_path, path, name, mode=args.mode)
