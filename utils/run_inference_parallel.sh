#!/bin/bash

PROMPT=""
START_KEY=""
END_KEY=""
KEYS_RAW=""
FOLDER_NAME=""

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --prompt) PROMPT="$2"; shift ;;
        --start_key) START_KEY="$2"; shift ;;
        --end_key) END_KEY="$2"; shift ;;
        --keys) shift; while [[ "$1" != "" && "$1" != --* ]]; do KEYS_RAW+="$1 "; shift; done ;;
        --folder) FOLDER_NAME="$2"; shift ;;
        *) echo "❌ Unknown parameter: $1"; exit 1 ;;
    esac
    shift
done

# ====== 驗證參數 ======
if [ -z "$PROMPT" ]; then
    echo "❌ Missing --prompt"
    echo "Usage:"
    echo "  --prompt <name>                  # Required"
    echo "  --start_key <N> --end_key <M>     # OR"
    echo "  --keys together_1 together_2 ...  # Optional custom keys"
    echo "  --folder <name>                   # Required output folder name"
    exit 1
fi

if [ -z "$FOLDER_NAME" ]; then
    echo "❌ Missing --folder"
    exit 1
fi

# ====== buling keys ======
KEYS=()

if [ -n "$KEYS_RAW" ]; then
    read -ra KEY_ARRAY <<< "$KEYS_RAW"
    for key in "${KEY_ARRAY[@]}"; do
        if [[ -n "$key" && ! " ${KEYS[*]} " =~ " $key " ]]; then
            KEYS+=("$key")
        fi
    done
elif [ -n "$START_KEY" ] && [ -n "$END_KEY" ]; then
    for i in $(seq $START_KEY $END_KEY); do
        KEYS+=("together_$i")
    done
else
    echo "❌ Must provide either --start_key and --end_key OR --keys"
    exit 1
fi

# ====== 執行參數計算 ======
TOTAL=14042
SHARDS=${#KEYS[@]}
CHUNK=$((TOTAL / SHARDS))
SESSION="mmlu_${PROMPT}_${FOLDER_NAME}"

echo "🚀 Launching $SHARDS shards for prompt '$PROMPT' into folder '$FOLDER_NAME'"
echo "🔑 API Keys: ${KEYS[@]}"
echo "📂 Output folder: temp/${FOLDER_NAME}"

# 建立 tmux session
tmux new-session -d -s $SESSION

for i in $(seq 0 $((SHARDS - 1))); do
    START=$((i * CHUNK))
    if [ $i -eq $((SHARDS - 1)) ]; then
        END=$TOTAL
    else
        END=$(((i + 1) * CHUNK))
    fi
    KEY=${KEYS[$i]}

    CMD="source .venv/bin/activate && python main.py --prompt $PROMPT --start $START --end $END --api_key $KEY --shard_id $i --mode run --folder $FOLDER_NAME"

    if [ $i -eq 0 ]; then
        tmux send-keys -t $SESSION "$CMD" C-m
    else
        tmux split-window -t $SESSION
        tmux select-layout -t $SESSION tiled
        tmux send-keys -t $SESSION.$i "$CMD" C-m
    fi
done

# 選擇第一個 pane
tmux select-pane -t $SESSION.0
tmux attach -t $SESSION
