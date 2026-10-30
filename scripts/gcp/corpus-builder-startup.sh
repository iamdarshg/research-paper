#!/bin/bash
set -e
exec > /var/log/corpus-build-setup.log 2>&1
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

echo "=== DEPS INSTALLED $(date) ==="
python --version
python -c "import torch; print('torch', torch.__version__)"

echo "=== STEP 1: REGENERATE REAL AIRCRAFTVERSE CORPUS AT 96^3 (downloads ZIPs from Zenodo) ==="
mkdir -p /home/iamdarshg/corpus_build
cd /home/iamdarshg/research-paper
python CLI/build_aircraftverse_corpus.py \
  --output-dir /home/iamdarshg/corpus_build/aircraftverse_g96 \
  --grid-size 96 \
  --target-count 2000 \
  --selection-seed 42 \
  --workers 8 \
  > /home/iamdarshg/corpus_build/aircraftverse_build.log 2>&1
AV_EXIT=$?
echo "=== AIRCRAFTVERSE BUILD EXIT=$AV_EXIT $(date) ==="
tail -15 /home/iamdarshg/corpus_build/aircraftverse_build.log

echo "=== STEP 2: COMPRESS TO 32^3 FINAL CORPUS (originals only; no perturb/procedural) ==="
if [ $AV_EXIT -eq 0 ]; then
  python scripts/gcp/rebuild_with_retry.py \
    /home/iamdarshg/corpus_build/aircraftverse_g96/aircraftverse_manifest.jsonl \
    /home/iamdarshg/corpus_build/final_corpus_g32 \
    32 2000 \
    > /home/iamdarshg/corpus_build/rebuild.log 2>&1
  RB_EXIT=$?
  echo "=== REBUILD EXIT=$RB_EXIT $(date) ==="
  tail -30 /home/iamdarshg/corpus_build/rebuild.log

  echo "=== STEP 3: CLAIM-BEARING VALIDATION ==="
  python CLI/validate_manifest.py \
    --manifest /home/iamdarshg/corpus_build/final_corpus_g32/combined_training_manifest.jsonl \
    --level claim-bearing \
    --unique-geometry-target 600 \
    --output /home/iamdarshg/corpus_build/claim_validation.json \
    > /home/iamdarshg/corpus_build/claim_validation.out 2>&1
  echo "=== VALIDATE EXIT=$? $(date) ==="
  head -60 /home/iamdarshg/corpus_build/claim_validation.out
fi

echo "=== ARTIFACTS ==="
find /home/iamdarshg/corpus_build -maxdepth 2 -type f -name '*.json' -o -name '*.jsonl' 2>/dev/null | head -20
echo "=== DONE $(date) ==="