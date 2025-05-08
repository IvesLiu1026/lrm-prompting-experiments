import os
import re
import json
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from datasets import load_dataset
# from scipy.stats import wilcoxon
# from statsmodels.stats.contingency_tables import mcnemar

# Load MMLU dataset
ds = load_dataset("cais/mmlu", "all")
subject_map = {i: ds['test'][i]['subject'] for i in range(len(ds['test']))}

# === File utilities ===
def load_jsonl_to_list(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]

# === Save per-subject flip stats ===
def save_flip_subject_csv(flip_subject_stats, output_path):
    rows = []
    for subj, stats in flip_subject_stats.items():
        total = stats["flip_success"] + stats["backfire"] + stats["stay_correct"] + stats["flip_failure"]
        success_ratio = stats["flip_success"] / total if total > 0 else 0
        rows.append({
            "Subject": subj,
            "Flip Success": stats["flip_success"],
            "Flip Failure": stats["flip_failure"],
            "Stay Correct": stats["stay_correct"],
            "Backfire": stats["backfire"],
            "Total Flips": total,
            "Success Ratio": round(success_ratio, 4)
        })
    df = pd.DataFrame(rows)
    df = df.sort_values("Total Flips", ascending=False)
    df.to_csv(output_path, index=False)
    return df

# === Plot per-subject bar chart ===
def plot_flip_subjects(df, prompt_name, plot_folder):
    top_df = df
    plt.figure(figsize=(18, 10))
    plt.barh(top_df["Subject"], top_df["Flip Success"], label="Flip Success", color="green")
    plt.barh(top_df["Subject"], top_df["Flip Failure"], left=top_df["Flip Success"], label="Flip Failure", color="blue")
    plt.barh(top_df["Subject"], top_df["Stay Correct"], left=top_df["Flip Success"] + top_df["Flip Failure"], label="Stay Correct", color="gray")
    plt.barh(top_df["Subject"], top_df["Backfire"], left=top_df["Flip Success"] + top_df["Flip Failure"] + top_df["Stay Correct"], label="Backfire", color="red")
    plt.xlabel("Count")
    plt.title(f"All Subjects - Flip Outcomes ({prompt_name})")
    plt.legend()
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(f"{plot_folder}/flip/{prompt_name}_flip_subjects.png")
    plt.close()

# only backfire and flip success
# def plot_flip_subjects(df, prompt_name, plot_folder):
#     top_df = df
#     plt.figure(figsize=(18, 10))

#     plt.barh(top_df["Subject"], top_df["Flip Success"], label="Flip Success ✅", color="green")
#     plt.barh(top_df["Subject"], top_df["Backfire"], 
#              left=top_df["Flip Success"], label="Backfire ❌", color="red")

#     plt.xlabel("Count")
#     plt.title(f"All Subjects - Flip Success vs. Backfire ({prompt_name})")
#     plt.legend()
#     plt.gca().invert_yaxis()
#     plt.tight_layout()
#     plt.savefig(f"{plot_folder}/flip/{prompt_name}_flip_subjects.png")
#     plt.close()

# === Plot scatter: baseline vs. prompt word count ===
def plot_wordcount_scatter(baseline_file, prompt_file, prompt_name, plot_folder):
    base_data = load_jsonl_to_list(baseline_file)
    prompt_data = load_jsonl_to_list(prompt_file)

    def get_word_map(data):
        return {
            item["index"]: len(item.get("response", "").split())
            for item in data if item.get("index") is not None
        }

    base_map = get_word_map(base_data)
    prompt_map = get_word_map(prompt_data)

    rows = []
    for idx in sorted(set(base_map) & set(prompt_map)):
        rows.append({
            "index": idx,
            "baseline_total": base_map[idx],
            "prompt_total": prompt_map[idx],
        })
    df = pd.DataFrame(rows)

    plt.figure(figsize=(18, 10))
    sns.scatterplot(data=df, x="baseline_total", y="prompt_total", alpha=0.5)
    sns.regplot(data=df, x="baseline_total", y="prompt_total", scatter=False, color="red")

    plt.gca().set_aspect('equal', adjustable='box')  # 👈 Add this line for equal scaling

    plt.title(f"Total Word Count: Baseline vs. {prompt_name}")
    plt.xlabel("Baseline Word Count")
    plt.ylabel(f"{prompt_name} Word Count")
    plt.tight_layout()
    os.makedirs(f"{plot_folder}/scatter", exist_ok=True)
    plt.savefig(f"{plot_folder}/scatter/{prompt_name}_scatter_total_words.png")
    plt.close()


# === Analyze and plot missing data per subject ===
def analyze_missing_by_subject(prompt_name, index_missing_list, response_missing_list, response_ans_missing_list, subject_map, plot_folder, stats_folder):
    missing_subject_stats = {
        "index_missing": defaultdict(int),
        "response_missing": defaultdict(int),
        "answer_missing": defaultdict(int)
    }

    for idx in index_missing_list:
        subj = subject_map.get(idx, "Unknown")
        missing_subject_stats["index_missing"][subj] += 1

    for idx in response_missing_list:
        subj = subject_map.get(idx, "Unknown")
        missing_subject_stats["response_missing"][subj] += 1

    for idx in response_ans_missing_list:
        subj = subject_map.get(idx, "Unknown")
        missing_subject_stats["answer_missing"][subj] += 1

    missing_plot_dir = os.path.join(plot_folder, "missing")
    os.makedirs(missing_plot_dir, exist_ok=True)
    os.makedirs(f"{stats_folder}/missing_details", exist_ok=True)

    for miss_type, counter in missing_subject_stats.items():
        df_miss = pd.DataFrame({
            "Subject": list(counter.keys()),
            "Count": list(counter.values())
        }).sort_values("Count", ascending=False)

        df_miss.to_csv(f"{stats_folder}/missing_details/{prompt_name}_{miss_type}.csv", index=False)

        plt.figure(figsize=(18, 10))
        sns.barplot(data=df_miss, y="Subject", x="Count", color="orange")
        plt.title(f"Top Subjects - {miss_type.replace('_', ' ').title()} ({prompt_name})")
        plt.tight_layout()
        plt.savefig(f"{missing_plot_dir}/{prompt_name}_{miss_type}_bar.png")
        plt.close()

# === Main analysis logic ===
def analyze_single_output(prompt_name, data, baseline_map, stats_folder, missing_folder):
    total = correct = think_word_count = think_count = think_response_count = wait_token_count = total_word_count = total_latency = completion_tokens = 0
    index_missing_list = response_ans_missing_list = response_missing_list = []
    output_all_index = set(item.get("index") for item in data if item.get("index") is not None)
    dataset_all_index = set(range(len(ds['test'])))

    subject_stats = defaultdict(lambda: {"total": 0, "correct": 0})
    flip_subject_stats = defaultdict(lambda: {
        "flip_success": 0, "flip_failure": 0,
        "stay_correct": 0, "backfire": 0
    })

    index_missing_list = list(dataset_all_index - output_all_index)

    for item in data:
        idx = item.get("index")
        response = item.get("response", "")
        response_ans = item.get("response_ans")
        correct_flag = item.get("correct")

        total += 1
        if correct_flag:
            correct += 1

        if "<think>" in response and "</think>" in response:
            think_count += 1
            think_blocks = re.findall(r"<think>(.*?)</think>", response, flags=re.DOTALL)
            words = sum(len(block.strip().split()) for block in think_blocks)
            think_word_count += words
            if words > 0:
                think_response_count += 1

        if response_ans is None:
            response_ans_missing_list.append(idx)
        if response is None:
            response_missing_list.append(idx)

        wait_token_count += response.lower().count("wait")
        total_word_count += len(response.split())
        completion_tokens += item.get("token_usage", {}).get("completion", 0)
        total_latency += item.get("time_usage", 0)

        subj = subject_map.get(idx, "Unknown")
        subject_stats[subj]["total"] += 1
        subject_stats[subj]["correct"] += int(correct_flag)

        if baseline_map and idx in baseline_map and correct_flag is not None:
            baseline_correct = baseline_map[idx]
            if not baseline_correct and correct_flag:
                flip_subject_stats[subj]["flip_success"] += 1
            elif not baseline_correct and not correct_flag:
                flip_subject_stats[subj]["flip_failure"] += 1
            elif baseline_correct and correct_flag:
                flip_subject_stats[subj]["stay_correct"] += 1
            elif baseline_correct and not correct_flag:
                flip_subject_stats[subj]["backfire"] += 1

    summary = {
        "Prompt": prompt_name,
        "Total": total,
        "Total_correct": correct,
        "Accuracy": round(correct / total, 4),
        "Total_words_avg": round(total_word_count / total, 2),
        "Think_words_avg": round(think_word_count / think_response_count, 2) if think_response_count else 0,
        "Total_wait_tokens": wait_token_count,
        "Wait_tokens_avg": round(wait_token_count / total, 2),
        "Time(s)": round(total_latency / total, 2),
        "Flip_success": sum(d["flip_success"] for d in flip_subject_stats.values()),
        "Flip_failure": sum(d["flip_failure"] for d in flip_subject_stats.values()),
        "Stay_correct": sum(d["stay_correct"] for d in flip_subject_stats.values()),
        "Backfire": sum(d["backfire"] for d in flip_subject_stats.values()),
        "Index_missing_list": index_missing_list,
        "Response_missing_list": response_missing_list,
        "Response_ans_missing_list": response_ans_missing_list
    }
    
    # log all the info
    print(f"\n📊 Prompt Summary: {prompt_name}")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print(f"🧮 Total Questions       : {total:>5}")
    print(f"✅ Correct               : {correct:>5}")
    print(f"🎯 Accuracy              : {summary['Accuracy']:.2%}")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print(f"📝 Avg. Words            : {summary['Total_words_avg']:>5}")
    print(f"🧠 Avg. <think> Words    : {summary['Think_words_avg']:>5}")
    print(f"⏳ Total Latency (s)     : {summary['Time(s)']:>5}")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print(f"⏳ Total 'wait' tokens   : {summary['Total_wait_tokens']:>5}")
    print(f"⏱️ Avg. 'wait' per Q     : {summary['Wait_tokens_avg']:>5}")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print(f"🟩 Flip Success          : {summary['Flip_success']:>5}")
    print(f"🟦 Flip Failure          : {summary['Flip_failure']:>5}")
    print(f"⬜ Stay Correct          : {summary['Stay_correct']:>5}")
    print(f"🟥 Backfire              : {summary['Backfire']:>5}")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print(f"❗ Index Missing         : {len(summary['Index_missing_list']):>5}")
    print(f"❗ Response Missing      : {len(summary['Response_missing_list']):>5}")
    print(f"❗ Answer Missing        : {len(summary['Response_ans_missing_list']):>5}")
    print("==================================================\n")


    os.makedirs(missing_folder, exist_ok=True)
    with open(f"{missing_folder}/{prompt_name}_missing.json", "w", encoding="utf-8") as f:
        json.dump({
            "index_missing_list": index_missing_list,
            "response_ans_missing_list": response_ans_missing_list,
            "response_missing_list": response_missing_list
        }, f, indent=2)

    os.makedirs(stats_folder, exist_ok=True)
    
    analyze_missing_by_subject(
        prompt_name,
        index_missing_list,
        response_missing_list,
        response_ans_missing_list,
        subject_map,
        plot_folder,
        stats_folder
    )

    pd.DataFrame([summary]).to_csv(f"{stats_folder}/{prompt_name}_stats.csv", index=False)

    return {item["index"]: item["correct"] for item in data if item.get("correct") is not None}, flip_subject_stats

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--i", type=str, nargs="*", help="Prompt name(s) without _output.json")
    parser.add_argument("--all", action="store_true", help="Analyze all prompts")
    parser.add_argument("--folder", type=str, default="output", help="Folder containing outputs")
    parser.add_argument("--mode", type=str, choices=["run", "test"], default="run", help="Mode: run or test")
    args = parser.parse_args()

    # Folder setup
    plot_folder = f"{args.folder}/plots" if args.mode == "run" else f"{args.folder}/plots_test"
    stats_folder = f"{args.folder}/stats"
    flip_csv_folder = f"{args.folder}/flip_csvs"
    missing_folder = f"log/missing_lists"
    os.makedirs(f"{plot_folder}/flip", exist_ok=True)
    os.makedirs(stats_folder, exist_ok=True)
    os.makedirs(flip_csv_folder, exist_ok=True)

    # Baseline
    files = os.listdir(args.folder)
    if "standard_output.json" not in files:
        raise FileNotFoundError("❌ standard_output.json not found in specified folder.")

    baseline_path = os.path.join(args.folder, "standard_output.json")
    baseline_data = load_jsonl_to_list(baseline_path)
    baseline_map, _ = analyze_single_output("standard", baseline_data, {}, stats_folder, missing_folder)

    input_files = [f.replace("_output.json", "") for f in files if f.endswith("_output.json") and f != "standard_output.json"] if args.all else args.i
    if not input_files:
        raise ValueError("Please provide --i or --all")

    for name in input_files:
        path = os.path.join(args.folder, f"{name}_output.json")
        data = load_jsonl_to_list(path)
        correct_map, flip_stats = analyze_single_output(name, data, baseline_map, stats_folder, missing_folder)

        csv_path = f"{flip_csv_folder}/{name}_flip.csv"
        df = save_flip_subject_csv(flip_stats, csv_path)
        plot_flip_subjects(df, name, plot_folder)
        plot_wordcount_scatter(baseline_path, path, name, plot_folder)
