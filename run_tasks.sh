#!/bin/bash

# Fail fast
set -e

# Ensure predictable environment
export PATH="/usr/local/bin:/usr/bin:/bin:/home/masa/.local/bin"

cd /home/masa/Desktop/SmartBin || exit 1

total=$(grep -cve '^\s*$' tasks.txt | grep -cve '^\s*#' || true)
current=0

while read -r task; do
  [[ -z "$task" || "$task" == \#* ]] && continue
  current=$((current + 1))

  echo "========================================"
  echo "[$current] HANDING TASK TO CLAUDE..."
  echo "Task: $(echo "$task" | cut -c 1-80)..."
  echo "========================================"

  # Use full path if needed (recommended)
  if ! /home/masa/.local/bin/claude -p "$task" --dangerously-skip-permissions; then
    echo "CLAUDE FAILED on task $current. Stopping."
    exit 1
  fi

  echo "Claude finished task $current. Committing..."

  /usr/bin/git add -A -- . ':!.env' ':!*.secret' ':!credentials*'
  /usr/bin/git commit -m "Auto-completed task $current: $(echo "$task" | cut -c 1-50)..."

done < tasks.txt

echo "All tasks complete!"
