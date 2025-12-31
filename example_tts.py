import sys
import os

# Add local src to path to ensure we use the latest code
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import torchaudio as ta
import torch
from chatterbox.tts import ChatterboxTTS
from chatterbox.mtl_tts import ChatterboxMultilingualTTS

# Automatically detect the best available device
device = "cpu"
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    try:
        import torch_npu
        if torch.npu.is_available():
            device = "npu"
    except ImportError:
        pass

print(f"Using device: {device}")

model = ChatterboxTTS.from_pretrained(device=device)

text = "Ezreal and Jinx teamed up with Ahri, Yasuo, and Teemo to take down the enemy's Nexus in an epic late-game pentakill."

print("Generating test-1.wav...")
with torch.no_grad():
    wav = model.generate(text)
ta.save("test-1.wav", wav, model.sr)
print("Saved test-1.wav")

multilingual_model = ChatterboxMultilingualTTS.from_pretrained(device=device)
text = "Bonjour, comment ça va? Ceci est le modèle de synthèse vocale multilingue Chatterbox, il prend en charge 23 langues."

print("Generating test-2.wav...")
with torch.no_grad():
    wav = multilingual_model.generate(text, language_id="fr")
ta.save("test-2.wav", wav, multilingual_model.sr)
print("Saved test-2.wav")


# If you want to synthesize with a different voice, specify the audio prompt
AUDIO_PROMPT_PATH = "YOUR_FILE.wav"
# wav = model.generate(text, audio_prompt_path=AUDIO_PROMPT_PATH)
# ta.save("test-3.wav", wav, model.sr)
