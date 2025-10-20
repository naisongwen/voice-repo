import torch
import torchaudio as ta
import os
import argparse
from chatterbox.vc import ChatterboxVC

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# Automatically detect the best available device
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--src_audio', type=str, required=True)
    parser.add_argument('--target_audio', type=str, required=True)
    parser.add_argument('--result_audio', type=str, required=True)

    args = parser.parse_args()

src_audio = args.src_audio
target_audio = args.target_audio
result_audio = args.result_audio

model = ChatterboxVC.from_pretrained(device)
wav = model.generate(
    audio=src_audio,
    target_voice_path=target_audio,
)
ta.save(result_audio, wav, model.sr)
