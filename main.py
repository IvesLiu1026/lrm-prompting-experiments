import argparse
import os, json, re
import time
from time import time as timer
from pathlib import Path
from datetime import datetime
from datasets import load_dataset
from together import Together
from dotenv import load_dotenv
from tqdm import tqdm
from prompts.prompts import prompt_map
import openai
from google import genai
from google.genai import types

# ====== API Setup ======
def get_client(provider, api_key):
    load_dotenv()
    if provider == "together":
        return Together(api_key=os.getenv(api_key))
    elif provider == "gemini":
        genai.Client(api_key=os.getenv(api_key))
        return genai
    elif provider == "nvidia":
        from openai import OpenAI
        return OpenAI(
            base_url="https://integrate.api.nvidia.com/v1",
            api_key=os.getenv(api_key)
        )
    else:
        raise ValueError("❌ Unsupported provider. Use 'together' or 'gemini'.")

# ====== Logger Setup ======
def create_logger(name, folder, shard_id):
    path = Path(f"log/{folder}/{name}_shard{shard_id}.log")
    return open(path, "a", encoding="utf-8")

def log_line(logger, msg, mode):
    if mode == "test":
        print(msg)
    if logger:
        logger.write(msg + "\n")

# ====== API Call and Response Handling ======
def call_api(client, provider, model_name, prompt, use_stream=False):
    try:
        if provider == "gemini":
            response = client.models.generate_content(
                model="models/gemini-2.0-flash",
                contents=prompt
            )
        elif provider == "together":
            response = client.chat.completions.create(
                model=model_name,
                temperature=0.1,
                max_new_tokens=8192,
                messages=[{"role": "user", "content": prompt}],
                stream=use_stream
            )
        elif provider == "nvidia":
            response = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_new_tokens=8192,
                stream=use_stream
            )
        else:
            raise ValueError("Unsupported provider")
        return response, None
    except Exception as e:
        return None, str(e)


def extract_response_text(response, provider, use_stream=False):
    if provider == "gemini":
        return response.text
    elif provider == "together":
        if use_stream:
            text_parts = []
            for chunk in response:
                delta = chunk.choices[0].delta.content if chunk.choices[0].delta else ""
                if delta:
                    text_parts.append(delta)
            return "".join(text_parts)
        else:
            return response.choices[0].message.content

def extract_response_ans(text):
    match = re.search(r"The answer is\s*\((\w)\)", text)
    if match:
        return ord(match.group(1).lower()) - ord('a')
    return None

def extract_token_usage(response):
    if hasattr(response, "usage") and response.usage:
        return {
            "prompt": response.usage.prompt_tokens,
            "completion": response.usage.completion_tokens,
            "total": response.usage.total_tokens,
        }
    return {"prompt": -1, "completion": -1, "total": -1}

def dump_json(data, output_file):
    with open(output_file, 'a', encoding='utf-8') as f:
        f.write(json.dumps(data, ensure_ascii=False) + '\n')

# ====== Main Execution Function ======
def run_experiment(args):
    client = get_client(args.provider, args.api_key)
    prompt_name = args.prompt
    model_name = args.model
    folder_name = args.folder
    start, end = args.start, args.end
    output_file = f"temp/{folder_name}/{prompt_name}_shard{args.shard_id}.json"

    os.makedirs(f"temp/{folder_name}", exist_ok=True)
    os.makedirs(f"log/{folder_name}", exist_ok=True)

    log_file = create_logger(prompt_name, folder_name, args.shard_id)
    error_log_file = create_logger(f"{prompt_name}_error_index", folder_name, args.shard_id)
    error_indices = []

    ds = load_dataset("cais/mmlu", "all")
    build_prompt = prompt_map[prompt_name]

    # Determine indices to run
    if args.indices:
        indices_to_run = list(map(int, args.indices.split(",")))
        print(f"🎯 Using manually specified indices: {len(indices_to_run)} items")
    elif args.fill_missing:
        with open(f"log/missing_lists/{args.fill_missing}_missing.json", "r", encoding="utf-8") as f:
            missing_data = json.load(f)
        indices_to_run = sorted(set(missing_data.get("index_missing_list", [])) |
                                 set(missing_data.get("response_ans_missing_list", [])) |
                                 set(missing_data.get("response_missing_list", [])))
        print(f"🔄 Filling missing indices for {args.fill_missing}: {len(indices_to_run)} items")
    else:
        indices_to_run = list(range(start, end))


    for idx in tqdm(indices_to_run, desc=f"Running {prompt_name}"):
        question = ds['test'][idx]['question']
        choices = ds['test'][idx]['choices']
        answer = ds['test'][idx]['answer']
        prompt = build_prompt(question, choices)

        log_line(log_file, f"\n--- Sample {idx} ---", args.mode)
        log_line(log_file, f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", args.mode)
        log_line(log_file, f"Prompt: {prompt}", args.mode)

        retry_count = 0
        max_retries = 5
        success = False

        while retry_count < max_retries:
            start_time = timer()
            response, error_message = call_api(client, args.provider, model_name, prompt, args.stream)
            elapsed_time = timer() - start_time

            if response is None:
                log_line(log_file, f"⚠️ API Error: {error_message}", args.mode)
                retry_count += 1
                continue

            try:
                response_text = extract_response_text(response, args.provider, args.stream)
                # print(f"Response: {response_text}")
                response_ans = extract_response_ans(response_text)
                token_usage = extract_token_usage(response)

                if response_text is None:
                    log_line(log_file, f"⚠️ No response text found.", args.mode)
                    retry_count += 1
                    continue
                if response_ans is None:
                    log_line(log_file, f"⚠️ No answer found in response.", args.mode)
                    retry_count += 1
                    continue

                success = True
                break

            except Exception as e:
                log_line(log_file, f"⚠️ Exception parsing response: {e}", args.mode)
                retry_count += 1
                continue

        if not success:
            error_indices.append(idx)
            continue

        correct = response_ans == answer
        data = {
            "index": idx,
            "question": question,
            "choices": choices,
            "answer": answer,
            "response_ans": response_ans,
            "correct": correct,
            "prompt": prompt,
            "response": response_text,
            "time_usage": elapsed_time,
            "token_usage": token_usage,
        }

        if args.mode == "run":
            dump_json(data, output_file)

        log_line(log_file, f"Response: {response_text}", args.mode)
        log_line(log_file, f"Response Answer: {response_ans}", args.mode)
        log_line(log_file, f"Correct: {correct}", args.mode)
        log_line(log_file, f"Timing Info: {elapsed_time:.2f}s", args.mode)
        log_line(log_file, f"Token Usage: {token_usage}", args.mode)

    if error_indices:
        log_line(error_log_file, f"Error Indices: {error_indices}", args.mode)
    else:
        print("No errors encountered.")

    log_file.close()
    error_log_file.close()

# ====== Entry Point ======
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run LLM with different reasoning prompts.")
    parser.add_argument("--prompt", type=str, default="standard", choices=prompt_map.keys())
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-235B-A22B-fp8-tput")
    parser.add_argument("--api_key", type=str)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=14042)
    parser.add_argument("--mode", type=str, default="run", choices=["run", "test"])
    parser.add_argument("--shard_id", type=int, required=True)
    parser.add_argument("--fill_missing", type=str, help="Prompt name to fill missing indices from log/missing_lists")
    parser.add_argument("--folder", type=str, default="20256666")
    parser.add_argument("--stream", action="store_true", help="Use streaming response from model")
    parser.add_argument("--provider", type=str, choices=["together", "gemini", "nvidia"], required=True, help="Choose model provider: together or gemini")
    parser.add_argument("--indices", type=str, help="Comma-separated index list (e.g. 100,102,105)")

    
    args = parser.parse_args()

    run_experiment(args)
