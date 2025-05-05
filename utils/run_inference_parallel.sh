#!/bin/bash

PROMPT=""
START_KEY=""
END_KEY=""
KEYS_RAW=""
FOLDER_NAME=""
MISSING_FILE=""
PROVIDER="nvidia"
MODEL="qwen/qwq-32b"  # default

# Parse arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --prompt) PROMPT="$2"; shift ;;
        --start_key) START_KEY="$2"; shift ;;
        --end_key) END_KEY="$2"; shift ;;
        --keys) shift; while [[ "$1" != "" && "$1" != --* ]]; do KEYS_RAW+="$1 "; shift; done ;;
        --folder) FOLDER_NAME="$2"; shift ;;
        --missing_file) MISSING_FILE="$2"; shift ;;
        --provider) PROVIDER="$2"; shift ;;
        --model) MODEL="$2"; shift ;;
        *) echo "❌ Unknown parameter: $1"; exit 1 ;;
    esac
    shift
done

# Validate required args
if [ -z "$PROMPT" ] || [ -z "$FOLDER_NAME" ]; then
    echo "❌ Missing required arguments: --prompt and --folder are required"
    exit 1
fi

# Build KEYS array
KEYS=()
if [ -n "$KEYS_RAW" ]; then
    read -ra KEY_ARRAY <<< "$KEYS_RAW"
    KEYS=("${KEY_ARRAY[@]}")
elif [[ "$START_KEY" =~ ^[0-9]+$ ]] && [[ "$END_KEY" =~ ^[0-9]+$ ]]; then
    for i in $(seq $START_KEY $END_KEY); do
        KEYS+=("${PROVIDER}_${i}")
    done
else
    echo "❌ Must provide either --start_key/--end_key (as numbers) or --keys"
    exit 1
fi

# Load indices
if [ -n "$MISSING_FILE" ]; then
    if [ ! -f "$MISSING_FILE" ]; then
        echo "❌ Missing file '$MISSING_FILE' not found."
        exit 1
    fi
    echo "📥 Loading indices from $MISSING_FILE ..."
    INDICES=($(jq '.index_missing_list // []' "$MISSING_FILE" | jq -r '.[]'))
    TOTAL=${#INDICES[@]}
    if [ $TOTAL -eq 0 ]; then
        echo "⚠️ No missing indices found in file."
        exit 1
    fi
else
    echo "📥 No --missing_file specified. Using full index range (0–14042)"
    TOTAL=14042
fi

# Compute shards
SHARDS=${#KEYS[@]}
if [ "$SHARDS" -eq 0 ]; then
    echo "❌ No API keys found."
    exit 1
fi
CHUNK=$((TOTAL / SHARDS))
SESSION="mmlu_${PROMPT}_${FOLDER_NAME}"

echo "🚀 Launching $SHARDS shards for prompt '$PROMPT' into folder '$FOLDER_NAME'"
echo "🔑 API Keys: ${KEYS[@]}"
echo "📊 Total indices: $TOTAL"
echo "📂 Output folder: temp/${FOLDER_NAME}"

tmux new-session -d -s $SESSION -n shard_0

for i in $(seq 0 $((SHARDS - 1))); do
    START=$((i * CHUNK))
    END=$(( (i + 1) * CHUNK ))
    if [ $i -eq $((SHARDS - 1)) ]; then
        END=$TOTAL
    fi

    if [ -n "$MISSING_FILE" ]; then
        PARTIAL_INDICES=("${INDICES[@]:$START:$((END - START))}")
        INDEX_ARG=$(IFS=, ; echo "${PARTIAL_INDICES[*]}")
        CMD="source .venv/bin/activate && python main.py --prompt $PROMPT --indices $INDEX_ARG --api_key ${KEYS[$i]} --shard_id $i --mode run --model $MODEL --folder $FOLDER_NAME --provider $PROVIDER"
    else
        CMD="source .venv/bin/activate && python main.py --prompt $PROMPT --start $START --end $END --api_key ${KEYS[$i]} --shard_id $i --mode run --model $MODEL --folder $FOLDER_NAME --provider $PROVIDER"
    fi

    if [ $i -eq 0 ]; then
        tmux send-keys -t $SESSION:0 "$CMD" C-m
    else
        tmux split-window -t $SESSION:0
        tmux select-layout -t $SESSION:0 tiled
        tmux send-keys -t $SESSION:0.$i "$CMD" C-m
    fi
    sleep 0.5
done

tmux select-pane -t $SESSION:0.0
tmux attach -t $SESSION
