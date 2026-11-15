# Pure LlamaGen full-loss continuation: epochs 21–26

This resumes the **pure LlamaGen VQ-16 tokenizer** control from the evaluated
20-epoch H200 checkpoint. It keeps the original LlamaGen model, its 256-token
grid, the same MixTok reconstruction loss recipe, the checkpoint's model/EMA,
discriminator, both optimizers and LeCam state. No MixTok router or TiTok
tokenizer is trained here. `train.py` saves `latest.pt` after every epoch.

The first 20 epochs were trained with [`fifichloe/addMixtok`](https://github.com/fifichloe/addMixtok),
under `llamagen_full_loss/`. This repository contains the continuation code.

The source checkpoint is pinned to
[`sophiaa/MoT-1-checkpoints/llamagen_full_loss_h200/epoch_20.pt`](https://huggingface.co/sophiaa/MoT-1-checkpoints/tree/main/llamagen_full_loss_h200),
SHA256 `0911153ac8b0b13db0e528598f618d93a172f91016d97dce5d579db430022e4d`.
It records zero-indexed epoch 19 and step 118620. The script verifies the full
training state before resuming. If an output `latest.pt` is already beyond
epoch 20, it resumes that newer progress.

## Run on H200

Use the same environment and pinned source trees as the original 20-epoch run:
`MixTok/reconstruction` commit `6623ccdc7a48688497f7b1f312548d1f14107e4d`,
`1d-tokenizer` commit `942a96fbdd873780179d1b78d5462911528bf8c8`,
and `LlamaGen` commit `ce98ec41803a74a90ce68c40ababa9eaeffeb4ec`.
The reconstruction environment needs PyTorch, `omegaconf`, `wandb`,
`huggingface_hub`, and the pretrained DINOv2, CLIP, ConvNeXt, LPIPS/VGG and
LlamaGen VQ assets already used in the first 20 epochs.

If the three source trees are absent on H200, clone them into the workspace:

```bash
export MIXTOK_H200_WORK=/absolute/path/to/existing/workspace
mkdir -p "$MIXTOK_H200_WORK"
git clone https://github.com/LCGLiChenge/MixTok.git "$MIXTOK_H200_WORK/MixTok"
git -C "$MIXTOK_H200_WORK/MixTok" checkout 6623ccdc7a48688497f7b1f312548d1f14107e4d
git clone https://github.com/bytedance/1d-tokenizer.git "$MIXTOK_H200_WORK/1d-tokenizer"
git -C "$MIXTOK_H200_WORK/1d-tokenizer" checkout 942a96fbdd873780179d1b78d5462911528bf8c8
git clone https://github.com/FoundationVision/LlamaGen.git "$MIXTOK_H200_WORK/LlamaGen"
git -C "$MIXTOK_H200_WORK/LlamaGen" checkout ce98ec41803a74a90ce68c40ababa9eaeffeb4ec
```

The original setup and asset-download instructions are in
[`addMixtok/llamagen_full_loss/README.md`](https://github.com/fifichloe/addMixtok/blob/main/llamagen_full_loss/README.md).
Use that environment and the same ImageNet training directory.

```bash
git clone https://github.com/fifichloe/mixtok-llama.git
export MIXTOK_H200_WORK=/absolute/path/to/existing/workspace
export MIXTOK_IMAGENET_TRAIN=/absolute/path/to/ImageNet/train
export WANDB_API_KEY=...  # Supply your W&B key on H200; keep it out of Git.
# Optional: use the checkpoint already present on H200.
export MIXTOK_LLAMA_E20_CHECKPOINT=/absolute/path/to/epoch_20.pt
bash mixtok-llama/resume_20_to_26.sh
```

If `MIXTOK_LLAMA_E20_CHECKPOINT` is unset, the launcher downloads the exact
public checkpoint from the pinned Hugging Face revision. It defaults to eight
GPUs and per-GPU batch 27, accumulation 1. This matches the original checkpoint's
5931 updates per epoch (20 × 5931 = 118620); the six additional epochs should
end at step 154206. It deliberately skips a new memory probe to keep the batch
and optimizer-step meaning unchanged. Override source and asset locations with
`MIXTOK_SOURCE_ROOT`, `MIXTOK_TITOK_ROOT`, `MIXTOK_LLAMAGEN_ROOT`,
`MIXTOK_LLAMA_ORIGINAL_CKPT`, `MIXTOK_DINO_REPO`, and `MIXTOK_CLIP_CACHE`.

The launcher reads `WANDB_API_KEY` from the H200 environment and runs
`wandb login --verify`; it does not store the key in Git. W&B logs the same
training metrics as the first run. If the original `wandb_run_id.txt` is copied
into the new output directory, logging resumes that run; otherwise a new run
in project `mixtok-rebuttal` is created.

The learning-rate implementation remains cosine. Extending its total horizon
from 20 to 26 epochs gives a higher learning rate at epoch 21 than the final
step of the original 20-epoch schedule. The script does not claim a single
uninterrupted 26-epoch schedule.
