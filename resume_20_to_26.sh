#!/usr/bin/env bash
set -euo pipefail

# On H200, replace this placeholder with your W&B key if the machine is not
# already logged in. An exported WANDB_API_KEY takes precedence.
WANDB_API_KEY="${WANDB_API_KEY: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}"
if test "$WANDB_API_KEY" = 'PASTE_YOUR_WANDB_API_KEY_HERE'; then
  unset WANDB_API_KEY
fi
export WANDB_PROJECT='mixtok-rebuttal'
if test -n "${WANDB_API_KEY:-}"; then
  wandb login --verify --relogin "$WANDB_API_KEY" >/dev/null
elif python - <<'PY'
import netrc
try:
    auth = netrc.netrc().authenticators('api.wandb.ai')
except (FileNotFoundError, netrc.NetrcParseError):
    auth = None
raise SystemExit(0 if auth and auth[2] else 1)
PY
then
  printf 'Using the existing W&B login on this machine.\n'
else
  printf 'No W&B login found. Set WANDB_API_KEY once on this machine, then rerun.\n' >&2
  exit 1
fi

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
work="${MIXTOK_H200_WORK:?Set MIXTOK_H200_WORK to the H200 workspace root}"
train_data="${MIXTOK_IMAGENET_TRAIN:?Set MIXTOK_IMAGENET_TRAIN to ImageNet/train}"
source_root="${MIXTOK_SOURCE_ROOT:-$work/MixTok/reconstruction}"
titok_root="${MIXTOK_TITOK_ROOT:-$work/1d-tokenizer}"
llamagen_root="${MIXTOK_LLAMAGEN_ROOT:-$work/LlamaGen}"
llamagen_ckpt="${MIXTOK_LLAMA_ORIGINAL_CKPT:-$llamagen_root/pretrained_models/vq_ds16_c2i.pt}"
dino_repo="${MIXTOK_DINO_REPO:-${TORCH_HOME:-$work/.cache/torch}/hub/facebookresearch_dinov2_main}"
clip_cache="${MIXTOK_CLIP_CACHE:-$work/.cache/open_clip}"
output="${MIXTOK_LLAMA_RESUME_OUTPUT:-$work/results/llamagen_full_loss_20_to_26}"

for path in "$train_data" "$source_root/train_titok_llamagen_recon.py" \
  "$titok_root/modeling/modules/discriminator.py" "$llamagen_ckpt" \
  "$dino_repo/hubconf.py"; do
  test -e "$path" || { printf 'Missing dependency: %s\n' "$path" >&2; exit 1; }
done
if test "$(git -C "$source_root" rev-parse HEAD)" != '6623ccdc7a48688497f7b1f312548d1f14107e4d'; then
  printf 'MixTok reconstruction source is not at the pinned training revision.\n' >&2
  exit 1
fi

# Existing local epoch-20 checkpoint is preferred. Otherwise fetch the exact
# evaluated full-state checkpoint from the pinned public Hugging Face revision.
checkpoint="${MIXTOK_LLAMA_E20_CHECKPOINT:-$output/epoch_20.pt}"
if ! test -f "$checkpoint"; then
  mkdir -p "$(dirname "$checkpoint")"
  CHECKPOINT_DEST="$checkpoint" python - <<'PY'
import os
import shutil
from huggingface_hub import hf_hub_download

source=hf_hub_download(repo_id='sophiaa/MoT-1-checkpoints',
    filename='llamagen_full_loss_h200/epoch_20.pt',
    revision='9c39643c2a8dffa5def2df5ec9bac2348b4ec61e')
shutil.copy2(source,os.environ['CHECKPOINT_DEST'])
PY
fi
python "$repo_dir/prepare_resume.py" --checkpoint "$checkpoint" --output "$output"

# The epoch-20 checkpoint has 118620 updates = 20 * 5931. For the full
# 1,281,167-image train set, 8 GPUs x batch 27 x accumulation 1 gives 5931
# updates per epoch. Preserve that batch so the saved optimizer step is aligned.
export CUDA_VISIBLE_DEVICES="${MIXTOK_GPU_IDS:-0,1,2,3,4,5,6,7}"
if test "$(tr ',' '\n' <<< "$CUDA_VISIBLE_DEVICES" | wc -l)" -ne 8; then
  printf 'Exactly eight GPUs are required to preserve the checkpoint batch.\n' >&2
  exit 1
fi
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export PYTHONUNBUFFERED=1
mkdir -p "$output"
python -m torch.distributed.run --standalone --nproc_per_node=8 "$repo_dir/train.py" \
  --data-root "$train_data" \
  --output "$output" \
  --source-root "$source_root" \
  --titok-root "$titok_root" \
  --llamagen-root "$llamagen_root" \
  --llamagen-ckpt "$llamagen_ckpt" \
  --dino-repo "$dino_repo" \
  --clip-cache-dir "$clip_cache" \
  --batch-size 27 --accum 1 --epochs 26 \
  --workers "${MIXTOK_WORKERS:-4}" \
  --wandb-project "$WANDB_PROJECT" --wandb-mode online \
  2>&1 | tee -a "$output/train.log"

python - "$output/latest.pt" <<'PY'
import sys,torch
state=torch.load(sys.argv[1],map_location='cpu',weights_only=True,mmap=True)
assert state['epoch']==25 and state['step']==154206,(state['epoch'],state['step'])
print('Completed 26 epochs and 154206 optimizer steps; latest.pt contains all training state.')
PY
