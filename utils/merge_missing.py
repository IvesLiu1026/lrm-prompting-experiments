import os
import json
import argparse

def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]

def write_jsonl(path, data):
    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

def merge_missing(prompt, folder="output", shard_ids=None):
    # === Load original output ===
    output_path = os.path.join(folder, f"{prompt}_output.json")
    if not os.path.exists(output_path):
        raise FileNotFoundError(f"❌ Output file not found: {output_path}")
    print(f"📄 Loaded original: {output_path}")
    original = load_jsonl(output_path)

    # === Load all shards ===
    combined = {item["index"]: item for item in original}

    shard_ids = shard_ids or [0]  # default: only shard 0
    for shard_id in shard_ids:
        shard_path = os.path.join("temp", f"20250503", f"{prompt}_shard{shard_id}.json")
        if not os.path.exists(shard_path):
            print(f"⚠️ Shard not found: {shard_path}, skipping.")
            continue
        print(f"🔄 Merging shard: {shard_path}")
        shard_data = load_jsonl(shard_path)
        for item in shard_data:
            combined[item["index"]] = item

    # === Write back merged output ===
    merged = [combined[k] for k in sorted(combined.keys())]
    write_jsonl(output_path, merged)
    print(f"✅ Merged complete. Total items: {len(merged)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, required=True, help="Prompt name (e.g. fast_confident)")
    parser.add_argument("--folder", type=str, default="output", help="Folder name where output.json lives")
    parser.add_argument("--shards", type=int, nargs="*", help="Shard IDs to merge (default: 0)")
    args = parser.parse_args()

    merge_missing(args.prompt, folder=args.folder, shard_ids=args.shards)
