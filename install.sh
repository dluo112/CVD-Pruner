#!/bin/bash
set -e

echo "==== Upgrade pip tools ===="
pip install --upgrade pip setuptools wheel

echo "==== Install PyTorch (CUDA 12.1) ===="
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121

echo "==== Install Python dependencies ===="
pip install -r requirements.txt

echo "==== Install FlashAttention ===="
pip install flash-attn==2.7.4.post1 --no-build-isolation

echo "==== Install llava project ===="
pip install -e .

echo "==== Install third-party editable packages ===="
pip install -e ./lmms-eval

echo "==== Done ===="