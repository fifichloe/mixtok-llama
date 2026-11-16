"""Validate the exact 20-epoch full-loss checkpoint and install it as latest.pt."""
from __future__ import annotations

import argparse
import hashlib
import os
import shutil
from pathlib import Path

import torch

EXPECTED_SHA256 = '0911153ac8b0b13db0e528598f618d93a172f91016d97dce5d579db430022e4d'
STAGE_NAME = 'mix_tok_top_p_matched_6epoch_no_router'
REQUIRED = {'model','model_ema','discriminator','optimizer','optimizer_d','lecam_real','lecam_fake'}


def sha256(path):
    digest=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(16*1024*1024),b''):
            digest.update(block)
    return digest.hexdigest()


def inspect(path):
    ckpt=torch.load(path,map_location='cpu',weights_only=True,mmap=True)
    if not isinstance(ckpt,dict) or REQUIRED-set(ckpt):
        raise RuntimeError(f'Incomplete training checkpoint: missing {REQUIRED-set(ckpt)}')
    epoch=int(ckpt['epoch']);step=int(ckpt['step'])
    if epoch<19 or step<118620:
        raise RuntimeError(f'Expected epoch >=19 and step >=118620, got epoch={epoch}, step={step}')
    if not ckpt['optimizer'].get('state') or not ckpt['optimizer_d'].get('state'):
        raise RuntimeError('Optimizer state is empty; this is not a full continuation checkpoint')
    return epoch,step


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    source=a.checkpoint.resolve()
    if not source.is_file():raise FileNotFoundError(source)
    output=a.output.resolve();output.mkdir(parents=True,exist_ok=True)
    latest=output/'latest.pt'
    if latest.exists():
        epoch,step=inspect(latest)
        if epoch>19:
            ckpt=torch.load(latest,map_location='cpu',weights_only=True,mmap=True)
            if ckpt.get('stage')!=STAGE_NAME or ckpt.get('source_step')!=118620:
                raise RuntimeError('Existing continuation used another training recipe; select a new output directory')
            print(f'Resuming existing progress: epoch={epoch+1}, step={step}; keeping {latest}',flush=True)
            return
        if sha256(latest)!=EXPECTED_SHA256:
            raise RuntimeError('Existing epoch-20 latest.pt differs from the evaluated checkpoint')
        print(f'Verified existing epoch-20 checkpoint: {latest}',flush=True)
        return
    digest=sha256(source)
    if digest!=EXPECTED_SHA256:
        raise RuntimeError(f'Epoch-20 SHA256 mismatch: {digest}')
    epoch,step=inspect(source)
    if (epoch,step)!=(19,118620):
        raise RuntimeError(f'Wrong starting point: epoch={epoch}, step={step}')
    temporary=output/'.latest.pt.installing'
    if temporary.exists():temporary.unlink()
    shutil.copy2(source,temporary)
    if sha256(temporary)!=digest:raise RuntimeError('Checkpoint copy checksum mismatch')
    os.replace(temporary,latest)
    print(f'Installed verified epoch-20 full state at {latest}',flush=True)


if __name__=='__main__':main()
