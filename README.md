# CVD-Pruner Environment and Reproduction Guide

## 1. Platform

The experiments were conducted under the following environment:

- OS: Ubuntu 22.04
- Python: 3.10
- CUDA: 12.1
- GPU: NVIDIA RTX 4090
- PyTorch: 2.5.1 + cu121

## 2. Repository Structure

This repository contains:
- the main project code
- the modified `lmms-eval` toolkit

Please make sure the following structure is preserved:

```bash
VidCom2/
├── requirements.txt
├── install.sh
├── reproduce.sh
├── lmms-eval/
└── ...