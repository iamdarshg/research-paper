#!/bin/bash
set -e
exec > /var/log/first-training-setup.log 2>&1
echo "=== START $(date) ==="
export DEBIAN_FRONTEND=noninteractive

sudo apt-get update -qq
sudo apt-get install -y -qq python3-venv python3-pip git

rm -rf /home/iamdarshg/research-paper
git clone --depth 1 https://github.com/iamdarshg/research-paper.git /home/iamdarshg/research-paper
chown -R iamdarshg:iamdarshg /home/iamdarshg/research-paper

cd /home/iamdarshg/research-paper
sudo -u iamdarshg python3 -m venv /home/iamdarshg/.venv-gcp
. /home/iamdarshg/.venv-gcp/bin/activate
pip install -q --upgrade pip
pip install -q -r requirements-dev.txt
pip install -q torch==2.3.1+cpu --index-url https://download.pytorch.org/whl/cpu

echo "=== DEPENDENCIES INSTALLED $(date) ==="
python --version
python -c "import torch; print('torch', torch.__version__)"

echo "=== RUN PROTOCOL ==="
python CLI/run_protocol.py --config CLI/run_protocols/first_training.yaml --mode production > /home/iamdarshg/first_training_run.log 2>&1
RUN_EXIT=$?
echo "=== PROTOCOL EXIT CODE: $RUN_EXIT $(date) ==="
echo "--- last 40 lines of run log ---"
tail -40 /home/iamdarshg/first_training_run.log
echo "=== evidence package ==="
find build -name "final_evidence_package.json" -o -name "gate_readiness.json" 2>/dev/null | head
echo "=== artifacts ==="
ls -la checkpoints_first_training 2>/dev/null || true
ls -la build/first_training 2>/dev/null || true
echo "=== DONE $(date) ==="