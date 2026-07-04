#!/usr/bin/env python3
"""预下载 Chatterbox 模型权重到本地目录，避免 HF 下载超时。

用法:
    python scripts/download_weights.py [--output /path/to/models]

默认输出: /data/workdir/chatterbox_models
设置环境变量 HF_ENDPOINT 走镜像加速
"""

import os
import sys
import argparse
from pathlib import Path

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

REPO_ID = "ResembleAI/chatterbox"
WEIGHTS = [
    # vc.py 需要的文件
    "s3gen.safetensors",
    "conds.pt",
    # mtl_tts.py 需要的文件
    "ve.pt",
    "t3_mtl23ls_v2.safetensors",
    "s3gen.pt",
    "grapheme_mtl_merged_expanded_v1.json",
    "Cangjie5_TC.json",
]


def main():
    parser = argparse.ArgumentParser(description="下载 Chatterbox 模型权重")
    parser.add_argument("--output", default="/data/workdir/chatterbox_models",
                        help="本地输出目录 (默认: /data/workdir/chatterbox_models)")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[下载] 目标目录: {output_dir}")
    print(f"[下载] HF_ENDPOINT: {os.environ['HF_ENDPOINT']}")

    # 方式 1: 用 snapshot_download (推荐，支持断点续传)
    try:
        from huggingface_hub import snapshot_download
        snapshot_download(
            repo_id=REPO_ID,
            repo_type="model",
            revision="main",
            allow_patterns=WEIGHTS,
            local_dir=str(output_dir),
            local_dir_use_symlinks=False,
            resume_download=True,
        )
        print(f"[下载] 完成！权重已保存到 {output_dir}")
        return
    except ImportError:
        print("[下载] huggingface_hub 未安装，改用 hf_hub_download")

    # 方式 2: 逐个下载
    from huggingface_hub import hf_hub_download
    for fname in WEIGHTS:
        print(f"[下载] {fname} ...")
        hf_hub_download(
            repo_id=REPO_ID,
            filename=fname,
            local_dir=str(output_dir),
            local_dir_use_symlinks=False,
            resume_download=True,
        )
    print(f"[下载] 完成！权重已保存到 {output_dir}")


if __name__ == "__main__":
    main()
