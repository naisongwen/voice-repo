import random
import numpy as np
import torch
import gradio as gr
from chatterbox.tts import ChatterboxTTS


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)


def load_model():
    model = ChatterboxTTS.from_pretrained(DEVICE)
    return model


def generate(model, text, audio_prompt_path, exaggeration, temperature, seed_num, cfgw, min_p, top_p, repetition_penalty):
    if model is None:
        model = ChatterboxTTS.from_pretrained(DEVICE)

    if seed_num != 0:
        set_seed(int(seed_num))

    wav = model.generate(
        text,
        audio_prompt_path=audio_prompt_path,
        exaggeration=exaggeration,
        temperature=temperature,
        cfg_weight=cfgw,
        min_p=min_p,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
    )
    return (model.sr, wav.squeeze(0).numpy())


with gr.Blocks(theme=gr.themes.Soft(primary_hue="blue", secondary_hue="emerald", neutral_hue="gray")) as demo:
    model_state = gr.State(None)  # Loaded once per session/user
    
    gr.Markdown(
        """
        # 🎙️ Chatterbox TTS Demo
        ### High-quality Text-to-Speech with Voice Cloning
        
        Chatterbox is a production-grade open-source TTS model that generates human-like speech from text with optional voice cloning.
        """
    )

    with gr.Tab("🎙️ TTS Generation"):
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 📝 Input Settings")
                with gr.Group():
                    text = gr.Textbox(
                        value="Now let's make my mum's favourite. So three mars bars into the pan. Then we add the tuna and just stir for a bit, just let the chocolate and fish infuse. A sprinkle of olive oil and some tomato ketchup. Now smell that. Oh boy this is going to be incredible.",
                        label="📝 Text to synthesize (max 300 characters)",
                        max_lines=5,
                        info="Enter the text you want to convert to speech"
                    )
                    ref_wav = gr.Audio(
                        sources=["upload", "microphone"], 
                        type="filepath", 
                        label="🎤 Reference Audio File (Optional)", 
                        value=None,
                        info="Upload or record a sample of the voice you want to clone"
                    )
                
                gr.Markdown("### ⚙️ Voice Parameters")
                with gr.Group():
                    exaggeration = gr.Slider(
                        0.25, 2, step=.05, 
                        label="🎭 Exaggeration (Neutral = 0.5, extreme values can be unstable)", 
                        value=.5,
                        info="Controls the expressiveness of the generated speech"
                    )
                    cfg_weight = gr.Slider(
                        0.0, 1, step=.05, 
                        label="🎯 CFG/Pace Weight", 
                        value=0.5,
                        info="Higher values improve voice cloning accuracy but may reduce naturalness"
                    )

                with gr.Accordion("🔬 Advanced Generation Options", open=False):
                    gr.Markdown("#### 🎛️ Sampling Parameters")
                    seed_num = gr.Number(value=0, label="🎲 Random Seed (0 for random)", precision=0)
                    temp = gr.Slider(0.05, 5, step=.05, label="🌡️ Temperature", value=.8, info="Controls randomness in generation")
                    
                    gr.Markdown("#### 🎯 Sampling Algorithms")
                    min_p = gr.Slider(
                        0.00, 1.00, step=0.01, 
                        label="📐 Min-P (Newer sampler. Recommend 0.02-0.1. Better with high temperatures. 0.00 disables)", 
                        value=0.05
                    )
                    top_p = gr.Slider(
                        0.00, 1.00, step=0.01, 
                        label="📊 Top-P (Original sampler. 1.0 disables. Original: 0.8)", 
                        value=1.00
                    )
                    
                    gr.Markdown("#### 🔁 Repetition Control")
                    repetition_penalty = gr.Slider(
                        1.00, 2.00, step=0.1, 
                        label="🔄 Repetition Penalty", 
                        value=1.2,
                        info="Higher values reduce text repetition"
                    )

                run_btn = gr.Button("🔊 Generate Speech", variant="primary", size="lg")

            with gr.Column(scale=1):
                gr.Markdown("### 🎧 Output")
                with gr.Group():
                    audio_output = gr.Audio(label="Generated Audio", interactive=False)
                    
                gr.Markdown("### 💡 Tips")
                gr.Markdown("""
                - Use a clear reference audio for best voice cloning results
                - Adjust exaggeration to control emotional expressiveness
                - Lower CFG/Pace weight for more natural but less cloned voices
                - Use a fixed seed for reproducible results
                """)

    with gr.Tab("📘 About Chatterbox"):
        gr.Markdown("""
        # 📘 About Chatterbox TTS
        
        ## 🎯 What is Chatterbox?
        Chatterbox is a production-grade open-source Text-to-Speech (TTS) model developed by Resemble AI. 
        It generates high-quality, human-like speech from text and can clone voices from reference audio samples.
        
        ## 🔧 Key Features
        - **High-Quality Audio**: Generates natural-sounding speech
        - **Voice Cloning**: Clone voices from short audio samples
        - **Emotion Control**: Adjust the expressiveness of the speech
        - **Zero-Shot Learning**: No training required for new voices
        - **Neural Watermarking**: Built-in watermarking for responsible AI usage
        
        ## 🛠️ Technical Details
        - Based on a 0.5B parameter Llama-based acoustic model (T3)
        - Uses flow matching for stable and efficient generation
        - Alignment-aware inference for consistent results
        - CUDA accelerated for fast inference
        
        ## 📖 How to Use
        1. Enter the text you want to convert to speech
        2. Optionally upload a reference audio file to clone a specific voice
        3. Adjust parameters like exaggeration and CFG weight as needed
        4. Click "Generate Speech" to create the audio
        5. Download or listen to the generated audio
        
        ## ⚠️ Notes
        - Make sure your reference audio is in the same language as the text for best results
        - Extreme parameter values may produce unstable results
        - All generated audio contains an imperceptible watermark for responsible AI usage
        """)

    demo.load(fn=load_model, inputs=[], outputs=model_state)

    run_btn.click(
        fn=generate,
        inputs=[
            model_state,
            text,
            ref_wav,
            exaggeration,
            temp,
            seed_num,
            cfg_weight,
            min_p,
            top_p,
            repetition_penalty,
        ],
        outputs=audio_output,
    )

if __name__ == "__main__":
    demo.queue(
        max_size=50,
        default_concurrency_limit=1,
    ).launch(share=True)