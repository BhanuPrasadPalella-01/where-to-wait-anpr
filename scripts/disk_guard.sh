#!/bin/zsh
# Pause the RoundaboutHD download/decoding jobs if free disk space drops below a floor; resume when it recovers.
# Usage: scripts/disk_guard.sh [floor_mb]   (logs to reports/logs/disk_guard.log)
FLOOR_MB=${1:-500}
LOG="$(dirname $0)/../reports/logs/disk_guard.log"
paused=0
while true; do
  free_mb=$(df -m ~ | awk 'NR==2 {print $4}')
  pids=$(pgrep -f "roundabout_(clips|fullpass)\.py")
  if [[ -z "$pids" && $paused -eq 0 ]]; then
    echo "$(date '+%H:%M:%S') no jobs left, guard exiting (free ${free_mb} MB)" >> "$LOG"; exit 0
  fi
  if (( free_mb < FLOOR_MB )) && (( paused == 0 )); then
    echo "$(date '+%H:%M:%S') LOW DISK: ${free_mb} MB free < ${FLOOR_MB} MB -> pausing jobs $pids" >> "$LOG"
    kill -STOP ${=pids} 2>/dev/null; paused=1
  elif (( free_mb >= FLOOR_MB + 300 )) && (( paused == 1 )); then
    echo "$(date '+%H:%M:%S') disk recovered: ${free_mb} MB free -> resuming" >> "$LOG"
    kill -CONT ${=pids} 2>/dev/null; paused=0
  fi
  (( $(date +%s) % 300 < 20 )) && echo "$(date '+%H:%M:%S') ok: ${free_mb} MB free, jobs: $(echo $pids | wc -w | tr -d ' ')" >> "$LOG"
  sleep 20
done
