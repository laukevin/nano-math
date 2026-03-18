#!/bin/bash
# Training status & management
# Usage:
#   ./scripts/status.sh              # check status (default log: /tmp/pretrain_d2.log)
#   ./scripts/status.sh [logfile]    # check status with custom log
#   ./scripts/status.sh cleanup      # kill zombie/duplicate training processes

set -e

# --- cleanup mode ---
if [ "$1" = "cleanup" ]; then
    PIDS=$(pgrep -f "scripts\.(base_train|chat_sft|chat_rl)" 2>/dev/null || true)
    if [ -z "$PIDS" ]; then
        echo "No training processes found"
        exit 0
    fi
    COUNT=$(echo "$PIDS" | wc -l | tr -d ' ')
    echo "Found $COUNT training process(es):"
    ps -p $PIDS -o pid,pcpu,pmem,etime,command 2>/dev/null
    echo ""
    # Find actual python workers (not uv wrappers)
    WORKERS=$(ps -p $PIDS -o pid,command 2>/dev/null | grep "python.*-m scripts" | grep -v "uv run" | awk '{print $1}')
    WORKER_COUNT=$(echo "$WORKERS" | grep -c . 2>/dev/null || echo 0)
    if [ "$WORKER_COUNT" -gt 1 ]; then
        echo "WARNING: $WORKER_COUNT duplicate workers detected!"
        echo "Killing all..."
        kill $PIDS 2>/dev/null || true
        sleep 2
        # force kill stragglers
        REMAINING=$(pgrep -f "scripts\.(base_train|chat_sft|chat_rl)" 2>/dev/null || true)
        if [ -n "$REMAINING" ]; then
            kill -9 $REMAINING 2>/dev/null || true
        fi
        echo "Done. All training processes killed."
    else
        echo "Single process running — looks healthy. Use 'kill $PIDS' to stop."
    fi
    exit 0
fi

# --- status mode ---
LOG="${1:-/tmp/pretrain_d2.log}"

echo "=== PROCESS ==="
PIDS=$(pgrep -f "scripts\.(base_train|chat_sft|chat_rl)" 2>/dev/null || true)
if [ -z "$PIDS" ]; then
    echo "No training process running"
    RUNNING=false
else
    RUNNING=true
    # Count actual python workers
    WORKER_COUNT=$(ps -p $PIDS -o command 2>/dev/null | grep "python.*-m scripts" | wc -l | tr -d ' ')
    if [ "$WORKER_COUNT" -gt 1 ]; then
        echo "WARNING: $WORKER_COUNT duplicate workers! Run: ./scripts/status.sh cleanup"
    fi
    ps -p $PIDS -o pid,pcpu,pmem,etime,command 2>/dev/null | grep -v "^  PID" | grep "python.*-m scripts" | head -3
fi

echo ""
echo "=== PROGRESS ==="
if [ -f "$LOG" ]; then
    LAST=$(grep -E "^step " "$LOG" | tail -1)
    if [ -n "$LAST" ]; then
        STEP=$(echo "$LAST" | grep -oE "step [0-9]+/[0-9]+")
        PCT=$(echo "$LAST" | grep -oE "[0-9]+\.[0-9]+%")
        LOSS=$(echo "$LAST" | grep -oE "loss: [0-9]+\.[0-9]+" | head -1)
        ETA=$(echo "$LAST" | grep -oE "eta: [0-9.]+m")
        TOKSEC=$(echo "$LAST" | grep -oE "tok/sec: [0-9,]+")
        echo "$STEP ($PCT) | $LOSS | $TOKSEC | $ETA"

        # If process is dead but log exists, check if training finished
        if [ "$RUNNING" = "false" ]; then
            if grep -q "Total training time" "$LOG"; then
                echo "FINISHED"
            else
                echo "DEAD (process gone, training incomplete)"
            fi
        fi
    else
        echo "Initializing..."
        tail -1 "$LOG"
    fi

    # Loss trajectory (first, 25%, 50%, 75%, last)
    TOTAL=$(grep -cE "^step " "$LOG" 2>/dev/null || echo 0)
    if [ "$TOTAL" -gt 5 ]; then
        echo ""
        echo "=== LOSS CURVE ==="
        grep -E "^step " "$LOG" | awk -v t="$TOTAL" '
            NR==1 || NR==int(t*0.25) || NR==int(t*0.5) || NR==int(t*0.75) || NR==t {
                for(i=1;i<=NF;i++) { if($i=="step") s=$(i+1); if($i=="loss:") l=$(i+1) }
                gsub(/\|/,"",s); sub(/,$/,"",l)
                print "step " s " | loss: " l
            }'
    fi

    # Validation results
    VAL=$(grep -E "Validation bpb:" "$LOG" | tail -1)
    if [ -n "$VAL" ]; then
        echo ""
        echo "=== VALIDATION ==="
        echo "$VAL"
    fi
else
    echo "No log file at $LOG"
fi

echo ""
echo "=== CHECKPOINTS ==="
FOUND=false
for d in ~/.cache/nanochat/base_checkpoints/d*/; do
    [ -d "$d" ] || continue
    FOUND=true
    NAME=$(basename "$d")
    FILES=$(ls "$d"/*.pt 2>/dev/null | wc -l | tr -d ' ')
    SIZE=$(du -sh "$d" 2>/dev/null | cut -f1)
    STEPS=$(ls "$d"/model_*.pt 2>/dev/null | sed 's/.*model_0*\([0-9]*\)\.pt/\1/' | sort -n | tr '\n' ',' | sed 's/,$//')
    echo "pretrain $NAME: $FILES files, $SIZE — steps: ${STEPS:-none}"
done
for d in ~/.cache/nanochat/chatsft_checkpoints/d*/; do
    [ -d "$d" ] || continue
    FOUND=true
    NAME=$(basename "$d")
    FILES=$(ls "$d"/*.pt 2>/dev/null | wc -l | tr -d ' ')
    SIZE=$(du -sh "$d" 2>/dev/null | cut -f1)
    echo "sft $NAME: $FILES files, $SIZE"
done
if [ "$FOUND" = "false" ]; then
    echo "No checkpoints found"
fi
