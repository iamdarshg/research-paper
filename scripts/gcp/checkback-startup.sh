#!/bin/bash
# Read-back startup script: on boot, dump the setup+run logs to the serial console.
echo "=== CHECK-BACK START $(date) ==="
echo "--- setup log ---"
tail -80 /var/log/first-training-setup.log 2>/dev/null || echo "(no setup log)"
echo "--- run log ---"
tail -60 /home/iamdarshg/first_training_run.log 2>/dev/null || echo "(no run log)"
echo "--- processes ---"
ps aux | grep -E 'run_protocol|python|pytest' | grep -v grep || echo "(no python processes)"
echo "--- artifacts ---"
ls -la /home/iamdarshg/research-paper/build/first_training 2>/dev/null || echo "(no build/first_training)"
ls -la /home/iamdarshg/research-paper/checkpoints_first_training 2>/dev/null || echo "(no checkpoints_first_training)"
find /home/iamdarshg/research-paper/build -maxdepth 2 -type f -name '*evidence*' 2>/dev/null || true
echo "=== CHECK-BACK END $(date) ==="