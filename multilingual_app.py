import random
import numpy as np
import torch
from chatterbox.mtl_tts import ChatterboxMultilingualTTS, SUPPORTED_LANGUAGES
import gradio as gr
import torchaudio as ta
import tempfile
import os

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🚀 Running on device: {DEVICE}")

# --- Global Model Initialization ---
MODEL = None

LANGUAGE_CONFIG = {
    "ar": {
        "audio": "mtl_prompts/ar_prompts2.flac",
        "text": "في الشهر الماضي، وصلنا إلى معلم جديد بمليارين من المشاهدات على قناتنا على يوتيوب."
    },
    "da": {
        "audio": "mtl_prompts/da_m1.flac",
        "text": "Sidste måned nåede vi en ny milepæl med to milliarder visninger på vores YouTube-kanal."
    },
    "de": {
        "audio": "mtl_prompts/de_f1.flac",
        "text": "Letzten Monat haben wir einen neuen Meilenstein erreicht: zwei Milliarden Aufrufe auf unserem YouTube-Kanal."
    },
    "el": {
        "audio": "mtl_prompts/el_m.flac",
        "text": "Τον περασμένο μήνα, φτάσαμε σε ένα νέο ορόσημο με δύο δισεκατομμύρια προβολές στο κανάλι μας στο YouTube."
    },
    "en": {
        "audio": "mtl_prompts/en_f1.flac",
        "text": "Last month, we reached a new milestone with two billion views on our YouTube channel."
    },
    "es": {
        "audio": "mtl_prompts/es_f1.flac",
        "text": "El mes pasado alcanzamos un nuevo hito: dos mil millones de visualizaciones en nuestro canal de YouTube."
    },
    "fi": {
        "audio": "mtl_prompts/fi_m.flac",
        "text": "Viime kuussa saavutimme uuden virstanpylvään kahden miljardin katselukerran kanssa YouTube-kanavallamme."
    },
    "fr": {
        "audio": "mtl_prompts/fr_f1.flac",
        "text": "Le mois dernier, nous avons atteint un nouveau jalon avec deux milliards de vues sur notre chaîne YouTube."
    },
    "he": {
        "audio": "mtl_prompts/he_m1.flac",
        "text": "בחודש שעבר הגענו לאבן דרך חדשה עם שני מיליארד צפיות בערוץ היוטיוב שלנו."
    },
    "hi": {
        "audio": "mtl_prompts/hi_f1.flac",
        "text": "पिछले महीने हमने एक नया मील का पत्थर छुआ: हमारे YouTube चैनल पर दो अरब व्यूज़।"
    },
    "it": {
        "audio": "mtl_prompts/it_m1.flac",
        "text": "Il mese scorso abbiamo raggiunto un nuovo traguardo: due miliardi di visualizzazioni sul nostro canale YouTube."
    },
    "ja": {
        "audio": "mtl_prompts/ja_prompts1.flac",
        "text": "先月、私たちのYouTubeチャンネルで二十億回の再生回数という新たなマイルストーンに到達しました。"
    },
    "ko": {
        "audio": "mtl_prompts/ko_f.flac",
        "text": "지난달 우리는 유튜브 채널에서 이십억 조회수라는 새로운 이정표에 도달했습니다."
    },
    "ms": {
        "audio": "mtl_prompts/ms_f.flac",
        "text": "Bulan lepas, kami mencapai pencapaian baru dengan dua bilion tontonan di saluran YouTube kami."
    },
    "nl": {
        "audio": "mtl_prompts/nl_m.flac",
        "text": "Vorige maand bereikten we een nieuwe mijlpaal met twee miljard weergaven op ons YouTube-kanaal."
    },
    "no": {
        "audio": "mtl_prompts/no_f1.flac",
        "text": "Forrige måned nådde vi en ny milepæl med to milliarder visninger på YouTube-kanalen vår."
    },
    "pl": {
        "audio": "mtl_prompts/pl_m.flac",
        "text": "W zeszłym miesiącu osiągnęliśmy nowy kamień milowy z dwoma miliardami wyświetleń na naszym kanale YouTube."
    },
    "pt": {
        "audio": "mtl_prompts/pt_m1.flac",
        "text": "No mês passado, alcançámos um novo marco: dois mil milhões de visualizações no nosso canal do YouTube."
    },
    "ru": {
        "audio": "mtl_prompts/ru_m.flac",
        "text": "В прошлом месяце мы достигли нового рубежа: два миллиарда просмотров на нашем YouTube-канале."
    },
    "sv": {
        "audio": "mtl_prompts/sv_f.flac",
        "text": "Förra månaden nådde vi en ny milstolpe med två miljarder visningar på vår YouTube-kanal."
    },
    "sw": {
        "audio": "mtl_prompts/sw_m.flac",
        "text": "Mwezi uliopita, tulifika hatua mpya ya maoni ya bilioni mbili kweny kituo chetu cha YouTube."
    },
    "tr": {
        "audio": "mtl_prompts/tr_m.flac",
        "text": "Geçen ay YouTube kanalımızda iki milyar görüntüleme ile yeni bir dönüm noktasına ulaştık."
    },
    "zh": {
        "audio": "mtl_prompts/zh_f2.flac",
        "text": "上个月，我们达到了一个新的里程碑. 我们的YouTube频道观看次数达到了二十亿次，这绝对令人难以置信。"
    },
}

# --- UI Helpers ---
def default_audio_for_ui(lang: str) -> str | None:
    return LANGUAGE_CONFIG.get(lang, {}).get("audio")


def default_text_for_ui(lang: str) -> str:
    return LANGUAGE_CONFIG.get(lang, {}).get("text", "")


def get_supported_languages_display() -> str:
    """Generate a formatted display of all supported languages."""
    language_items = []
    for code, name in sorted(SUPPORTED_LANGUAGES.items()):
        language_items.append(f"**{name}** (`{code}`)")
    
    # Split into 2 lines
    mid = len(language_items) // 2
    line1 = " • ".join(language_items[:mid])
    line2 = " • ".join(language_items[mid:])
    
    return f"""
### 🌍 支持语言（共 {len(SUPPORTED_LANGUAGES)} 种）
{line1}

{line2}
"""


def get_or_load_model():
    """Loads the ChatterboxMultilingualTTS model if it hasn't been loaded already,
    and ensures it's on the correct device."""
    global MODEL
    if MODEL is None:
        print("Model not loaded, initializing...")
        try:
            MODEL = ChatterboxMultilingualTTS.from_pretrained(DEVICE)
            if hasattr(MODEL, 'to') and str(MODEL.device) != DEVICE:
                MODEL.to(DEVICE)
            print(f"Model loaded successfully. Internal device: {getattr(MODEL, 'device', 'N/A')}")
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
    return MODEL

# Defer model loading until first inference to speed up UI launch.
# If you want eager loading, set env var 'CHATTERBOX_EAGER_LOAD=1'.
if os.getenv("CHATTERBOX_EAGER_LOAD") == "1":
    try:
        get_or_load_model()
    except Exception as e:
        print(f"CRITICAL: Failed to load model on startup. Application may not function. Error: {e}")

def set_seed(seed: int):
    """Sets the random seed for reproducibility across torch, numpy, and random."""
    if seed is not None and seed > 0:
        torch.manual_seed(seed)
        if DEVICE == "cuda":
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        random.seed(seed)
        np.random.seed(seed)
        print(f"Set random seed to: {seed}")
    
def resolve_audio_prompt(language_id: str, provided_path: str | None) -> str | None:
    """
    Decide which audio prompt to use:
    - If user provided a path (upload/mic/url), use it.
    - Else, fall back to language-specific default (if any).
    """
    if provided_path and str(provided_path).strip():
        return provided_path
    return LANGUAGE_CONFIG.get(language_id, {}).get("audio")


def generate_tts_audio(
    text_input: str,
    language_id: str,
    audio_prompt_path_input: str = None,
    exaggeration_input: float = 0.5,
    temperature_input: float = 0.8,
    seed_num_input: int = 0,
    cfgw_input: float = 0.5
) -> tuple[int, np.ndarray]:
    """
    Generate high-quality speech audio from text using Chatterbox Multilingual model with optional reference audio styling.
    Supported languages: English, French, German, Spanish, Italian, Portuguese, and Hindi.
    
    This tool synthesizes natural-sounding speech from input text. When a reference audio file 
    is provided, it captures the speaker's voice characteristics and speaking style. The generated audio 
    maintains the prosody, tone, and vocal qualities of the reference speaker, or uses default voice if no reference is provided.

    Args:
        text_input (str): The text to synthesize into speech (maximum 300 characters)
        language_id (str): The language code for synthesis (eg. en, fr, de, es, it, pt, hi)
        audio_prompt_path_input (str, optional): File path or URL to the reference audio file that defines the target voice style. Defaults to None.
        exaggeration_input (float, optional): Controls speech expressiveness (0.25-2.0, neutral=0.5, extreme values may be unstable). Defaults to 0.5.
        temperature_input (float, optional): Controls randomness in generation (0.05-5.0, higher=more varied). Defaults to 0.8.
        seed_num_input (int, optional): Random seed for reproducible results (0 for random generation). Defaults to 0.
        cfgw_input (float, optional): CFG/Pace weight controlling generation guidance (0.2-1.0). Defaults to 0.5, 0 for language transfer. 

    Returns:
        tuple[int, np.ndarray]: A tuple containing the sample rate (int) and the generated audio waveform (numpy.ndarray)
    """
    current_model = get_or_load_model()

    if current_model is None:
        raise RuntimeError("TTS model is not loaded.")

    if seed_num_input != 0:
        set_seed(int(seed_num_input))

    print(f"Generating audio for text: '{text_input[:50]}...'")
    
    # Handle optional audio prompt
    chosen_prompt = audio_prompt_path_input or default_audio_for_ui(language_id)

    generate_kwargs = {
        "exaggeration": exaggeration_input,
        "temperature": temperature_input,
        "cfg_weight": cfgw_input,
    }
    if chosen_prompt:
        generate_kwargs["audio_prompt_path"] = chosen_prompt
        print(f"Using audio prompt: {chosen_prompt}")
    else:
        print("No audio prompt provided; using default voice.")
        
    wav = current_model.generate(
        text_input[:300],  # Truncate text to max chars
        language_id=language_id,
        **generate_kwargs
    )
    print("Audio generation complete.")
    return (current_model.sr, wav.squeeze(0).numpy())

def save_audio_to_temp(sr: int, audio: np.ndarray) -> str:
    """Save audio array to a temporary WAV file and return its path."""
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tensor = torch.tensor(audio).unsqueeze(0)
    ta.save(tmp.name, tensor, sr)
    return tmp.name


def generate_and_save(
    text_input: str,
    language_id: str,
    audio_prompt_path_input: str = None,
    exaggeration_input: float = 0.5,
    temperature_input: float = 0.8,
    seed_num_input: int = 0,
    cfgw_input: float = 0.5,
):
    sr, audio = generate_tts_audio(
        text_input,
        language_id,
        audio_prompt_path_input,
        exaggeration_input,
        temperature_input,
        seed_num_input,
        cfgw_input,
    )
    path = save_audio_to_temp(sr, audio)
    return (sr, audio), path


def chat_generate(
    history: list,
    user_text: str,
    language_id: str,
    audio_prompt_path_input: str = None,
    exaggeration_input: float = 0.5,
    temperature_input: float = 0.8,
    seed_num_input: int = 0,
    cfgw_input: float = 0.5,
):
    audio_tuple, file_path = generate_and_save(
        user_text,
        language_id,
        audio_prompt_path_input,
        exaggeration_input,
        temperature_input,
        seed_num_input,
        cfgw_input,
    )
    lang_name = SUPPORTED_LANGUAGES.get(language_id, language_id)
    reply = f"✅ 已生成 {lang_name} 语音。请在右侧试听或下载。"
    new_history = (history or []) + [(user_text, reply)]
    return new_history, audio_tuple, file_path

with gr.Blocks(
    theme=gr.themes.Soft(primary_hue="blue", secondary_hue="emerald", neutral_hue="gray"),
    css="""
    :root { --cb-primary: #3b82f6; --cb-secondary: #10b981; }
    #app-title { padding: 12px 16px; border-radius: 12px; background: linear-gradient(90deg, rgba(59,130,246,.12), rgba(16,185,129,.12)); }
    .section-card { border: 1px solid rgba(255,255,255,.08); border-radius: 12px; padding: 12px; background: rgba(255,255,255,.03); }
    .audio-note { font-size: 13px; opacity: .85; }
    #chatbot { border: 1px solid rgba(255,255,255,.08); }
    .generate-btn { height: 44px; font-weight: 600; }
    """
) as demo:
    gr.Markdown(
        """
        # 🌍 VoiceRepo 支持多语言的声音克隆系统
        ### 面向中文用户的多语言文本转语音，支持参考音频风格与情绪控制。
        
        首个工程级的多语言 TTS 模型，支持 23 种语言，开箱即用。
        """,
        elem_id="app-title"
    )

    with gr.Tabs():
        with gr.Tab("💬 对话模式"):
            initial_lang_chat = "fr"
            with gr.Row():
                with gr.Column(scale=1):
                    chat = gr.Chatbot(label="对话", type="tuples", height=400, elem_id="chatbot")
                    with gr.Group(elem_classes=["section-card"]):
                        chat_input = gr.Textbox(
                            value=default_text_for_ui(initial_lang_chat),
                            label="输入文本（最多 300 字）",
                            max_lines=4,
                            placeholder="输入要合成的文本…"
                        )
                        chat_lang = gr.Dropdown(
                            choices=[(name, code) for code, name in ChatterboxMultilingualTTS.get_supported_languages().items()],
                            value=initial_lang_chat,
                            label="语言"
                        )
                        chat_ref = gr.Audio(
                            sources=["upload","microphone"],
                            type="filepath",
                            label="参考音频（可选）",
                            value=None
                        )
                    gr.Markdown("💡 提示：参考音频语言应与选择的语言一致；若做语言迁移可将 CFG 设为 0。", elem_classes=["audio-note"]) 
                    with gr.Accordion("高级设置", open=False):
                        exag_chat = gr.Slider(0.25, 2, step=.05, label="情绪/强度（Neutral=0.5）", value=.5)
                        cfg_chat = gr.Slider(0.2, 1, step=.05, label="CFG/Pace", value=0.5)
                        seed_chat = gr.Number(value=0, label="随机种子（0 为随机）", precision=0)
                        temp_chat = gr.Slider(0.05, 5, step=.05, label="采样温度（Temperature）", value=.8)
                    with gr.Row():
                        send_btn = gr.Button("发送并生成", variant="primary", elem_classes=["generate-btn"]) 
                        clear_chat_btn = gr.Button("清空对话")
                with gr.Column(scale=1):
                    audio_out_chat = gr.Audio(label="输出音频", interactive=False)
                    download_btn_chat = gr.DownloadButton(label="⬇️ 下载音频", value=None)
            def on_language_change(lang, current_ref, current_text):
                return default_audio_for_ui(lang), default_text_for_ui(lang)
            chat_lang.change(
                fn=on_language_change,
                inputs=[chat_lang, chat_ref, chat_input],
                outputs=[chat_ref, chat_input],
                show_progress=False
            )
            send_btn.click(
                fn=chat_generate,
                inputs=[chat, chat_input, chat_lang, chat_ref, exag_chat, temp_chat, seed_chat, cfg_chat],
                outputs=[chat, audio_out_chat, download_btn_chat]
            )
            def _clear_chat():
                return [], None, None
            clear_chat_btn.click(fn=_clear_chat, inputs=[], outputs=[chat, audio_out_chat, download_btn_chat])

        with gr.Tab("🎙️ 生成模式"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### 输入设置")
                    with gr.Group(elem_classes=["section-card"]):
                        initial_lang = "fr"
                        text = gr.Textbox(
                            value=default_text_for_ui(initial_lang),
                            label="合成文本（最多 300 字）",
                            max_lines=5
                        )
                        language_id = gr.Dropdown(
                            choices=[(name, code) for code, name in ChatterboxMultilingualTTS.get_supported_languages().items()],
                            value=initial_lang,
                            label="语言"
                        )
                        ref_wav = gr.Audio(
                            sources=["upload","microphone"],
                            type="filepath",
                            label="参考音频（可选）",
                            value=None
                        )
                    gr.Markdown("💡 提示：参考音频最好与目标语言相同；做语言迁移时可将 CFG 设为 0。", elem_classes=["audio-note"]) 
                    gr.Markdown("### 语音参数")
                    with gr.Group(elem_classes=["section-card"]):
                        exaggeration = gr.Slider(0.25, 2, step=.05, label="情绪/强度（Neutral = 0.5，过高可能不稳定）", value=.5)
                        cfg_weight = gr.Slider(0.2, 1, step=.05, label="CFG/Pace", value=0.5)
                    with gr.Accordion("高级生成选项", open=False):
                        seed_num = gr.Number(value=0, label="随机种子（0 为随机）", precision=0)
                        temp = gr.Slider(0.05, 5, step=.05, label="采样温度（Temperature）", value=.8)
                    with gr.Row():
                        run_btn = gr.Button("生成语音", variant="primary", elem_classes=["generate-btn"]) 
                        clear_btn = gr.Button("清空参数")
                with gr.Column(scale=1):
                    gr.Markdown("### 输出")
                    with gr.Group(elem_classes=["section-card"]):
                        audio_output = gr.Audio(label="输出音频", interactive=False)
                        download_btn = gr.DownloadButton(label="⬇️ 下载音频", value=None)
                    gr.Markdown("### 使用提示")
                    gr.Markdown("""
                    - 参考音频越清晰，克隆效果越好
                    - 提升 exaggeration 增强情感表达
                    - 降低 CFG/Pace 可能更自然但克隆度更低
                    - 固定随机种子以便复现
                    """)
            def on_language_change(lang, current_ref, current_text):
                return default_audio_for_ui(lang), default_text_for_ui(lang)
            language_id.change(
                fn=on_language_change,
                inputs=[language_id, ref_wav, text],
                outputs=[ref_wav, text],
                show_progress=False
            )
            run_btn.click(
                fn=generate_and_save,
                inputs=[text, language_id, ref_wav, exaggeration, temp, seed_num, cfg_weight],
                outputs=[audio_output, download_btn]
            )
            def _clear_params():
                return default_audio_for_ui(initial_lang), default_text_for_ui(initial_lang), 0, 0.8
            clear_btn.click(fn=_clear_params, inputs=[], outputs=[ref_wav, text, seed_num, temp])

        with gr.Tab("📚 语言与示例"):
            gr.Markdown(get_supported_languages_display())
            examples = [[default_text_for_ui(code), code, None] for code in SUPPORTED_LANGUAGES.keys()]
            gr.Examples(
                examples=examples,
                inputs=[text, language_id, ref_wav],
                label="快速示例：点击填充输入",
                examples_per_page=12
            )

        with gr.Tab("📘 关于"):
            gr.Markdown(
                """
                # 📘 关于 VoiceRepo？
                
                ## 🎯 什么是 VoiceRepo？
                VoiceRepo 基于 Resemble AI 开源的工程级多语言 TTS 模型开发，
                支持 23 种语言。在多项评测中超过其他闭源系统，并在用户主观偏好测试中获得更高认可。
                
                ## 🔧 主要特性
                - 多语言支持：23 种语言，零样本声音克隆
                - 情绪/强度控制：通过 exaggeration 参数调节表达力度
                - 声音转换：可提供参考音频进行风格迁移
                - 责任机制：内置神经水印，支持合规使用
                
                ## 🚀 技术细节
                - 5 亿参数的 Llama 系声学模型（T3）
                - 基于 Flow Matching 的解码器（Matcha）
                - 面向对齐的推理以提升稳定性
                - CUDA 加速推理
                
                ## 📖 使用方法
                1. 从下拉菜单选择语言
                2. 在文本框输入或修改要合成的文本
                3. 可选：上传或录制参考音频以克隆目标声音风格
                4. 根据需要调整高级参数
                5. 点击“生成语音”开始合成
                
                ## ⚠️ 注意事项
                - 参考音频与目标语言一致时效果最佳
                - 过高或极端参数可能导致不稳定
                - 所有生成音频包含不可感知的水印以支持负责任的使用
                """
            )

    demo.queue(max_size=10)
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        debug=True,
        show_api=False,
    )

