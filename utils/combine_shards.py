import os
import json
import argparse
from glob import glob

parser = argparse.ArgumentParser()
parser.add_argument("--prompt", type=str, required=True)
parser.add_argument("--date", type=str, required=True)  # e.g. 20250422
parser.add_argument("--outdir", type=str, default="output")
args = parser.parse_args()

shard_files = sorted(glob(f"temp/{args.date}/{args.prompt}_shard*.json"))
combined = []

for fname in shard_files:
    with open(fname, 'r', encoding='utf-8') as f:
        for line in f:
            combined.append(json.loads(line))

# Sorted by index
combined = sorted(combined, key=lambda x: x.get("index", 0))

# Save combined output
os.makedirs(args.outdir, exist_ok=True)
output_path = f"{args.outdir}/{args.prompt}_combined.json"
with open(output_path, 'w', encoding='utf-8') as f:
    for item in combined:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

print(f"✅ Combined {len(shard_files)} shard files → {output_path}")
