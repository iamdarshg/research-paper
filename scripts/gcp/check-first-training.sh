#!/bin/bash
printf '%s\n' '=== SETUP LOG ==='
tail -60 /var/log/first-training-setup.log 2>/dev/null || true
printf '%s\n' '=== RUN LOG ==='
tail -40 /home/iamdarshg/first_training_run.log 2>/dev/null || true
printf '%s\n' '=== PROCESS ==='
ps aux | grep -E 'run_protocol|python' | grep -v grep || true
printf '%s\n' '=== ARTIFACTS ==='
find /home/iamdarshg/research-paper/build /home/iamdarshg/research-paper/checkpoints_first_training -maxdepth 3 -type f 2>/dev/null | head -30 || true