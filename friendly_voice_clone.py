import gradio as gr
import os
import numpy as np

def clone_voice_and_speak(reference_audio, text, target_lang):
    if not text.strip():
        raise gr.Error("请输入要合成的文本")
    if reference_audio is None:
        raise gr.Error("请上传参考音频")

    print("✅ 函数已调用！正在模拟生成克隆语音...")  # 调试：看终端是否有输出

    sample_rate = 16000
    duration = 1  # 秒
    silent_wave = np.zeros(sample_rate * duration, dtype=np.float32)

    return (sample_rate, silent_wave)  # ✅ Gradio Audio 支持这种格式

# 多语言选项（带国旗 emoji）
LANGUAGE_OPTIONS = [
    ("🇺🇸 English", "en"),
    ("🇨🇳 简体中文", "zh"),
    ("🇪🇸 Español", "es"),
    ("🇫🇷 Français", "fr"),
    ("🇩🇪 Deutsch", "de"),
    ("🇯🇵 日本語", "ja"),
    ("🇰🇷 한국어", "ko"),
    ("🇮🇳 हिन्दी", "hi"),
    ("🇷🇺 Русский", "ru"),
    ("🇵🇹 Português", "pt"),
]

# 自定义 CSS（美化界面）
custom_css = """
#app-title {
    text-align: center;
    font-family: 'Segoe UI', sans-serif;
    color: #2c3e50;
    font-size: 2.2em;
    margin: 20px 0;
    font-weight: bold;
}
#description {
    text-align: center;
    color: #7f8c8d;
    font-size: 1.1em;
    margin-bottom: 30px;
}
.step-label {
    font-weight: bold;
    color: #2980b9;
    margin-top: 10px;
    font-size: 1.1em;
}
.example-text {
    font-style: italic;
    color: #7f8c8d;
    font-size: 0.9em;
}
footer {
    visibility: hidden;
}
"""

# 示例参考音频（可选）
EXAMPLES = [
    ["examples/female_english.wav", "Hello, this is a cloned voice!", "en"],
    ["examples/male_chinese.wav", "你好，这是克隆的声音。", "zh"],
]

with gr.Blocks(css=custom_css, title="🔊 ChatterBox - 声音克隆助手") as demo:
    gr.HTML("<h1 id='app-title'>🔊 ChatterBox</h1>")
    gr.Markdown("<p id='description'>上传一段音频，选择语言，让 AI 用你的声音说世界语言</p>")

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 📌 1. 上传参考音频")
            reference_audio = gr.Audio(
                label="🗣️ 参考音频（清晰人声，5-10秒）",
                type="filepath",
                sources=["upload"],
                interactive=True
            )

            gr.Markdown("### 🌍 2. 设置生成参数")
            target_lang = gr.Dropdown(
                choices=LANGUAGE_OPTIONS,
                value="en",
                label="🎯 目标语言"
            )
            text_input = gr.Textbox(
                label="📝 输入要说的文本",
                lines=4,
                placeholder="请输入要合成的文本...",
                value="Hello, how are you?"
            )
            gr.Markdown("<p class='example-text'>💡 提示：文本语言不必与参考音频一致，系统会自动翻译或合成</p>")

            # 音色预览按钮（可选）
            preview_btn = gr.Button("👂 预览音色特征（模拟）", variant="secondary")

        with gr.Column(scale=1):
            gr.Markdown("### 🎧 3. 生成克隆语音")
            generate_btn = gr.Button("🚀 生成克隆语音", variant="primary", size="lg")

            output_audio = gr.Audio(label="🔊 生成的语音", autoplay=True)

            # 音色预览输出
            with gr.Accordion("🔍 音色分析（模拟）", open=False):
                gr_audio_preview = gr.Audio(label="参考音频波形", interactive=False)
                gr.Textbox(value="音色特征提取成功（模拟）", label="特征状态")

    # 示例区域
    gr.Markdown("### 🧪 示例演示")
    gr.Examples(
        examples=EXAMPLES,
        inputs=[reference_audio, text_input, target_lang],
        outputs=output_audio,
        fn=clone_voice_and_speak,
        cache_examples=False,
        label="点击加载示例"
    )

    # ============ 事件绑定 ============

    with gr.Row():
        generate_btn = gr.Button("🚀 生成克隆语音", variant="primary", size="lg")
        with gr.Group():
            gr.Markdown("...", visible=False)

    generate_btn.click(
        fn=clone_voice_and_speak,
        inputs=[reference_audio, text_input, target_lang],
        outputs=output_audio,
    )

    preview_btn.click(
        fn=lambda x: x,  # 模拟：回显音频
        inputs=reference_audio,
        outputs=gr_audio_preview
    )

# ============ 启动配置 ============
if __name__ == "__main__":
    demo.queue(max_size=10)
    demo.launch(
        server_name="0.0.0.0",    # 允许局域网访问
        server_port=7860,
        share=False,              # 局域网使用
        debug=True,
        show_api=False,           # 隐藏 API 文档（可选）
    )
