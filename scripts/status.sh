#!/bin/bash
# Training status & management
# Usage:
#   ./scripts/status.sh              # check status (default log: logs/train.log)
#   ./scripts/status.sh [logfile]    # check status with custom log
#   ./scripts/status.sh cleanup      # kill zombie/duplicate training processes
#   ./scripts/status.sh plot [log]   # terminal loss curve plot

set -e

# --- cleanup mode ---
if [ "$1" = "cleanup" ]; then
    PIDS=$(pgrep -f "scripts\.(base_train|chat_sft|chat_rl|base_eval)" 2>/dev/null || true)
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
        REMAINING=$(pgrep -f "scripts\.(base_train|chat_sft|chat_rl|base_eval)" 2>/dev/null || true)
        if [ -n "$REMAINING" ]; then
            kill -9 $REMAINING 2>/dev/null || true
        fi
        echo "Done. All training processes killed."
    else
        echo "Single process running — looks healthy. Use 'kill $PIDS' to stop."
    fi
    exit 0
fi

# --- plot mode ---
if [ "$1" = "plot" ]; then
    PLOG="${2:-logs/train.log}"
    python3 << PYEOF
import re
steps, losses = [], []
with open("$PLOG") as f:
    for line in f:
        m = re.match(r"step (\d+)/\d+.*loss: ([0-9.]+)", line)
        if m:
            steps.append(int(m.group(1)))
            losses.append(float(m.group(2)))
if not steps:
    print("No training data in log"); exit()
W, H = 60, 18
mn, mx = min(losses), max(losses)
grid = [[' ']*W for _ in range(H)]
for s, l in zip(steps, losses):
    x = int((s - steps[0]) / max(steps[-1] - steps[0], 1) * (W - 1))
    y = H - 1 - int((l - mn) / max(mx - mn, 1) * (H - 1))
    grid[y][x] = '\u2588'
BAR = '\u2500' * W
print(f"\n  Loss curve ({len(steps)} steps)")
print(f"  {BAR}")
for i, row in enumerate(grid):
    if i == 0: lab = f"{mx:.1f}"
    elif i == H-1: lab = f"{mn:.1f}"
    elif i == H//2: lab = f"{(mx+mn)/2:.1f}"
    else: lab = ""
    rowstr = ''.join(row)
    print(f"{lab:>6} \u2502{rowstr}\u2502")
BBAR = '\u2500' * W
print(f"       \u2514{BBAR}\u2518")
pad1 = ' ' * (W//2 - 2)
pad2 = ' ' * (W//2 - 5)
print(f"        0{pad1}step{pad2}{steps[-1]}")
delta = losses[0]-losses[-1]
print(f"\n  {losses[0]:.2f} \u2192 {losses[-1]:.2f} (\u0394 {delta:.2f})")
with open("$PLOG") as f:
    vals = [l.strip() for l in f if "Validation bpb:" in l]
if vals:
    print(f"  Val: {vals[-1]}")
PYEOF
    exit 0
fi

# --- status mode ---
LOG="${1:-logs/train.log}"

echo "=== PROCESS ==="
PIDS=$(pgrep -f "scripts\.(base_train|chat_sft|chat_rl|base_eval)" 2>/dev/null || true)
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
    VALS=$(grep -E "Validation bpb:" "$LOG" 2>/dev/null || true)
    if [ -n "$VALS" ]; then
        echo ""
        echo "=== VALIDATION ==="
        echo "$VALS"
    fi

    # Eval results (from inline CORE metric or standalone base_eval)
    EVALS=$(grep -E "^Evaluating:" "$LOG" 2>/dev/null || true)
    if [ -n "$EVALS" ]; then
        echo ""
        EVAL_DONE=$(echo "$EVALS" | wc -l | tr -d ' ')
        EVAL_LAST=$(echo "$EVALS" | tail -1)
        echo "=== EVAL ($EVAL_DONE benchmarks) ==="
        # Show CORE metric if available
        CORE=$(grep "CORE metric:" "$LOG" 2>/dev/null | tail -1 || true)
        if [ -n "$CORE" ]; then
            echo "$CORE"
        fi
        # Show top results (accuracy > random)
        echo "$EVALS" | grep -v "accuracy: 0.0000" | while IFS= read -r line; do
            BENCH=$(echo "$line" | sed 's/Evaluating: \([^ ]*\).*/\1/')
            ACC=$(echo "$line" | grep -oE "accuracy: [0-9.]+" | head -1)
            echo "  $BENCH: $ACC"
        done
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
