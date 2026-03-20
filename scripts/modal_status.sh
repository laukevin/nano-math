#!/usr/bin/env bash
# Unified status check for Modal jobs: active containers, GPU metrics, logs, and experiment results
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

CMD="${1:-all}"  # all | active | logs | gpu | results | checkpoints

show_active() {
    echo "=== Active Modal Jobs ==="
    local containers
    containers=$(uv run modal container list --json 2>/dev/null)
    local count
    count=$(echo "$containers" | python3 -c "import json,sys; print(len(json.load(sys.stdin)))" 2>/dev/null || echo 0)

    if [ "$count" = "0" ]; then
        echo "  No active containers"
        return
    fi

    echo "$containers" | python3 -c "
import json, sys
from datetime import datetime, timezone
containers = json.load(sys.stdin)
now = datetime.now(timezone.utc)
for c in containers:
    cid = c['Container ID']
    app = c['App ID']
    start_str = c['Start Time']
    # Parse start time
    try:
        start = datetime.fromisoformat(start_str)
        elapsed = now - start
        mins = int(elapsed.total_seconds() / 60)
        elapsed_str = f'{mins}m'
    except:
        elapsed_str = '?'
    print(f'  {cid}  app={app}  elapsed={elapsed_str}')
"
}

show_gpu() {
    echo "=== GPU Metrics ==="
    local containers
    containers=$(uv run modal container list --json 2>/dev/null)
    local cids
    cids=$(echo "$containers" | python3 -c "import json,sys; [print(c['Container ID']) for c in json.load(sys.stdin)]" 2>/dev/null)

    if [ -z "$cids" ]; then
        echo "  No active containers"
        return
    fi

    while IFS= read -r cid; do
        echo -n "  $cid: "
        local gpu_info
        gpu_info=$(timeout 10 uv run modal container exec "$cid" nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu,temperature.gpu --format=csv,noheader,nounits 2>/dev/null) || true
        if [ -n "$gpu_info" ]; then
            echo "$gpu_info" | python3 -c "
import sys
line = sys.stdin.readline().strip()
parts = [p.strip() for p in line.split(',')]
if len(parts) >= 4:
    mem_used, mem_total, util, temp = parts[:4]
    pct = int(mem_used) * 100 // int(mem_total)
    print(f'GPU {util}% util | {mem_used}/{mem_total} MiB ({pct}%) | {temp}°C')
else:
    print(line)
"
        else
            echo "(exec failed — container may be busy)"
        fi
    done <<< "$cids"
}

show_logs() {
    echo "=== Recent Logs (last 30 lines per app) ==="
    local apps
    apps=$(uv run modal app list --json 2>/dev/null)
    local app_ids
    app_ids=$(echo "$apps" | python3 -c "
import json, sys
apps = json.load(sys.stdin)
for a in apps:
    state = a.get('State', '')
    if 'ephemeral' in state and a.get('Stopped at') is None:
        print(a['App ID'])
" 2>/dev/null)

    if [ -z "$app_ids" ]; then
        echo "  No active apps"
        return
    fi

    while IFS= read -r app_id; do
        echo "--- $app_id ---"
        timeout 5 uv run modal app logs "$app_id" 2>/dev/null | tail -30 || echo "  (no logs or timed out)"
        echo ""
    done <<< "$app_ids"
}

show_checkpoints() {
    echo "=== Checkpoints ==="
    uv run modal volume ls math-nano-checkpoints / 2>/dev/null | while IFS= read -r dir; do
        dir=$(echo "$dir" | xargs)  # trim whitespace
        [ -z "$dir" ] && continue
        # Count items in each checkpoint dir
        local items
        items=$(uv run modal volume ls math-nano-checkpoints "$dir/" 2>/dev/null | wc -l | xargs)
        # Check if it has a final adapter (= training completed)
        local has_adapter
        has_adapter=$(uv run modal volume ls math-nano-checkpoints "$dir/" 2>/dev/null | grep -c "adapter_model" || true)
        if [ "$has_adapter" -gt 0 ]; then
            echo "  $dir  [$items files, COMPLETE]"
        else
            echo "  $dir  [$items files, in-progress]"
        fi
    done
}

show_results() {
    echo "=== Experiment Registry ==="
    REG_FILE="$PROJECT_DIR/.modal_registry_tmp.jsonl"
    rm -f "$REG_FILE"
    uv run modal volume get math-nano-results experiment_registry.jsonl "$REG_FILE" 2>/dev/null || {
        echo "  (no registry file)"
        return
    }

    python3 -c "
import json, sys

rows = []
with open('$REG_FILE') as f:
    for line in f:
        d = json.loads(line)
        if d['experiment_id'].startswith('smoke'): continue
        rows.append(d)

if not rows:
    print('  (no experiments)')
    sys.exit(0)

# Group by size
rows_10k = [r for r in rows if '10k' in r['experiment_id']]
rows_50k = [r for r in rows if '50k' in r['experiment_id']]
rows_other = [r for r in rows if r not in rows_10k and r not in rows_50k]

header = f'{\"Experiment\":<30s} {\"Size\":>6s} {\"Loss\":>7s} {\"SVAMP\":>6s} {\"GSM8K\":>6s} {\"MATH\":>6s} {\"AIME\":>6s} {\"Time\":>6s}'

def print_rows(label, group):
    if not group: return
    print(f'\n  {label}')
    print(f'  {header}')
    print(f'  ' + '-' * 97)
    for d in group:
        ev = d.get('eval', {})
        eid = d['experiment_id']
        size = str(d.get('data_size', '?'))
        loss = f'{d[\"final_loss\"]:.3f}' if d.get('final_loss') else '  --'
        svamp = f'{ev[\"svamp_greedy\"]*100:.0f}%' if 'svamp_greedy' in ev else '  --'
        gsm8k = f'{ev[\"gsm8k_greedy\"]*100:.0f}%' if 'gsm8k_greedy' in ev else '  --'
        math_ = f'{ev[\"math_greedy\"]*100:.0f}%' if 'math_greedy' in ev else '  --'
        aime  = f'{ev[\"aime_2025_greedy\"]*100:.0f}%' if 'aime_2025_greedy' in ev else '  --'
        mins  = f'{d[\"wall_clock_min\"]:.0f}m' if d.get('wall_clock_min') else '  --'
        print(f'  {eid:<30s} {size:>6s} {loss:>7s} {svamp:>6s} {gsm8k:>6s} {math_:>6s} {aime:>6s} {mins:>6s}')

print_rows('10K Experiments', rows_10k)
print_rows('50K Experiments', rows_50k)
print_rows('Other', rows_other)
"
    rm -f "$REG_FILE"
}

case "$CMD" in
    active)
        show_active
        ;;
    gpu)
        show_active
        echo ""
        show_gpu
        ;;
    logs)
        show_logs
        ;;
    results)
        show_results
        ;;
    checkpoints)
        show_checkpoints
        ;;
    all)
        show_active
        echo ""
        show_gpu
        echo ""
        show_logs
        show_results
        echo ""
        show_checkpoints
        ;;
    *)
        echo "Usage: $0 [all|active|gpu|logs|results|checkpoints]"
        echo ""
        echo "  all          - Everything (default)"
        echo "  active       - Active containers with elapsed time"
        echo "  gpu          - GPU utilization on active containers"
        echo "  logs         - Recent logs from active apps"
        echo "  results      - Experiment registry table"
        echo "  checkpoints  - Checkpoint status on volume"
        exit 1
        ;;
esac
