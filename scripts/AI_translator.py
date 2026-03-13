#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
🤖在线AI助手
- Tab1: 快速入口 - 一键打开常用翻译/大模型网站
- Tab2: 提示词模板 - 集成 ASR 后处理、字幕翻译等专用提示词
"""

import gradio as gr
import webbrowser

# ==================== 在线翻译网站 URL ====================
URLS = {
    "DeepL": "https://www.deepl.com/zh",
    "有道翻译": "https://fanyi.youdao.com/",
    "豆包": "https://www.doubao.com/",
    "通义千问": "https://www.qianwen.com/",
    "DeepSeek": "https://www.deepseek.com/",
    "ChatGLM": "https://chatglm.cn",
    "Kimi": "https://kimi.moonshot.cn/",
    "腾讯元宝": "https://yuanbao.tencent.com/",
}

# ==================== 预置翻译提示词 ====================
PROMPTS = {
    "SRT字幕翻译(保留时间码)": """你是一个专业的字幕翻译专家。请将以下SRT字幕内容翻译成中文。
要求：
1. 严格保留原文的时间轴格式 (例如: 00:00:01,000 --> 00:00:03,000)。
2. 保持字幕序号不变。
3. 翻译要自然流畅，适合口语表达，注意联系上下文语境。
4. 对于专有名词或专业术语，请保持一致性。
5. 直接输出翻译后的SRT内容，不要包含解释。

原文内容：
""",

    "语义断句与合并(ASR优化)": """你是专业字幕精修师。
我将给你ASR识别的带时间戳字幕，可能是碎句、断句错误、错别字。
请按以下规则处理：

1. 按语义合并碎句，不要在一句话中间切开。
2. 每条字幕中文字数 ≤ 20 字。
3. 合并多条时，开始时间=第一条，结束时间=最后一条。
4. 自动修正ASR错别字、同音字错误。
5. 去掉口语冗余词：那个、就是、然后、嘛、啊等。
6. 输出严格标准SRT格式，不要任何解释、不要JSON。

输入内容：
""",

    "短视频极速版(10字以内)": """你是短视频字幕专家。
规则：
1. 每一条字幕 ≤ 10 个字。
2. 短句、有力、适合短视频节奏。
3. 自动合并、自动拆分过长句。
4. 保留时间码，输出标准SRT。

输入内容：
""",

    "双语字幕版(中英对照)": """你是专业字幕翻译。
规则：
1. 保留原时间轴。
2. 中文在上，英文在下。
3. 语言自然口语化，不生硬。
4. 输出标准SRT。

输入内容：
""",

    "双语字幕生成(中英对照)": """请将以下文本翻译成中文，并生成中英双语对照格式。
格式要求：
第一行：原文
第二行：译文
(空行分隔不同段落)

请处理以下内容：
""",

    "文本润色与校对": """请对以下文本进行润色和校对。
要求：
1. 修正错别字和标点符号错误。
2. 优化语句通顺度，使其更符合中文阅读习惯。
3. 保持原意不变，不增加或删减关键信息。
4. 如果没有错误，请原样输出。

待处理文本：
""",

    "长文本总结(适合Kimi/元宝)": """请阅读以下长文本，并进行总结。
要求：
1. 提炼出核心观点和关键信息。
2. 使用条理清晰的列表形式输出。
3. 语言简洁明了。

文本内容：
""",

    "ASR校对与纠错(专业版)": """你是一个专业的校对员。以下是 ASR 语音识别的原始结果，可能包含错别字。
请根据上下文修正错别字，并输出修正后的纯文本内容。

要求：
1. 重点修正同音字错误。
2. 保持原意，不要大幅改写。
3. 输出修正后的文本即可。

原文内容：
"""
}

def open_url(url):
    """调用浏览器打开指定链接"""
    webbrowser.open(url)
    return f"✅ 已打开 {url}，请查看浏览器"

def update_prompt(prompt_name):
    """根据选择更新提示词内容"""
    return PROMPTS.get(prompt_name, "")

# ==================== 构建界面 ====================
with gr.Blocks(title="🤖 在线AI助手 Pro", theme=gr.themes.Default()) as demo:
    gr.Markdown("""
    <div style="text-align: center; margin-bottom: 1rem;">
        <h1>🤖 在线AI助手 Pro</h1>
        <p style="color: #666;">集成提示词模板 · 一键唤醒翻译/大模型平台</p>
    </div>
    """)

    # 使用 Tabs 组织两个功能模块
    with gr.Tabs():
        # ---------- Tab1: 快速入口 ----------
        with gr.Tab("快速入口"):
            with gr.Column():
                gr.Markdown("### 一键打开常用翻译/大模型网站")
                gr.Markdown("点击下方按钮，浏览器将自动打开对应网站。")

                # 按钮区域：两行，每行四个按钮，高度一致
                with gr.Row(equal_height=True):
                    btn_deepl = gr.Button("🌍 DeepL", variant="secondary", scale=1, min_width=120)
                    btn_youdao = gr.Button("📚 有道翻译", variant="secondary", scale=1, min_width=120)
                    btn_deepseek = gr.Button("🔍 DeepSeek", variant="secondary", scale=1, min_width=120)
                    btn_doubao = gr.Button("🥟 豆包", variant="secondary", scale=1, min_width=120)

                with gr.Row(equal_height=True):
                    btn_qianwen = gr.Button("⏰ 通义千问", variant="secondary", scale=1, min_width=120)
                    btn_kimi = gr.Button("🌓 Kimi", variant="secondary", scale=1, min_width=120)
                    btn_chatglm = gr.Button("💬 ChatGLM", variant="secondary", scale=1, min_width=120)
                    btn_yuanbao = gr.Button("🎠 腾讯元宝", variant="secondary", scale=1, min_width=120)

                status = gr.Textbox(label="", value="等待操作...", interactive=False)

        # ---------- Tab2: 提示词模板 ----------
        with gr.Tab("提示词模板"):
            with gr.Column():
                gr.Markdown("### 📝 专业提示词模板")
                gr.Markdown("选择模板后复制，然后到上方「快速入口」打开网站粘贴使用。")

                with gr.Row():
                    prompt_selector = gr.Dropdown(
                        label="选择提示词类型",
                        choices=list(PROMPTS.keys()),
                        value="SRT字幕翻译(保留时间码)",
                        scale=2
                    )
                    # 右侧留空保持对称
                    gr.Column(scale=1)

                prompt_display = gr.Textbox(
                    label="提示词内容 (可直接编辑，点击右上角按钮复制)",
                    value=PROMPTS["SRT字幕翻译(保留时间码)"],
                    lines=10,
                    interactive=True,
                    show_copy_button=True,
                )

                gr.Markdown("💡 **使用方法**：选择模板 → 复制提示词 → 切换到「快速入口」打开网站 → 粘贴使用。")

                # 同样放置一组按钮，方便直接点击打开网站（与 Tab1 一致）
                gr.Markdown("---")
                with gr.Row(equal_height=True):
                    btn_deepl_tab2 = gr.Button("🌍 DeepL", variant="secondary", scale=1, min_width=120)
                    btn_youdao_tab2 = gr.Button("📚 有道翻译", variant="secondary", scale=1, min_width=120)
                    btn_deepseek_tab2 = gr.Button("🔍 DeepSeek", variant="secondary", scale=1, min_width=120)
                    btn_doubao_tab2 = gr.Button("🥟 豆包", variant="secondary", scale=1, min_width=120)

                with gr.Row(equal_height=True):
                    btn_qianwen_tab2 = gr.Button("⏰ 通义千问", variant="secondary", scale=1, min_width=120)
                    btn_kimi_tab2 = gr.Button("🌓 Kimi", variant="secondary", scale=1, min_width=120)
                    btn_chatglm_tab2 = gr.Button("💬 ChatGLM", variant="secondary", scale=1, min_width=120)
                    btn_yuanbao_tab2 = gr.Button("🎠 腾讯元宝", variant="secondary", scale=1, min_width=120)

                status2 = gr.Textbox(label="", value="等待操作...", interactive=False)

                # 绑定下拉菜单变化事件
                prompt_selector.change(
                    fn=update_prompt,
                    inputs=[prompt_selector],
                    outputs=[prompt_display]
                )

    # 页脚版权与免责声明（两个 Tab 共用）
    gr.Markdown("---")
    gr.HTML("""
    <div class="notice">
        注意事项：<br>
        • 本工具仅用于个人学习与视频剪辑使用<br>
        • 禁止用于商业用途及侵权行为<br>            
        • 使用前确保模型与依赖环境正常配置
    </div>
    <div style="text-align: center; color: #666; font-size: 0.9em;">
        <p>本软件包不提供任何模型文件，模型由用户自行从官方渠道获取。用户需自行遵守模型的原许可证。</p>
        <p>本软件包按“原样”提供，不提供任何明示或暗示的担保。使用本软件所产生的一切风险由用户自行承担。</p>
        <p>本软件包开发者不对因使用本软件而导致的任何直接或间接损失负责。</p> 
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 15px; border-radius: 8px; margin: 15px auto; max-width: 600px;">
            <p style="color: white; font-weight: bold; margin: 5px 0; font-size: 1em;">🎬 更新请关注B站up主：光影的故事2018</p>
            <p style="color: white; margin: 5px 0; font-size: 0.9em;">
                🔗 <strong>B站主页</strong>: <a href="https://space.bilibili.com/381518712" target="_blank" style="color: #ffdd40; text-decoration: none; font-weight: bold;">
                    space.bilibili.com/381518712
                </a>
            </p>
        </div>
    </div>
    <div style="text-align: center; color: #666; margin-top: 10px; font-size: 0.9em;">
        © 原创 WebUI 代码 © 2026 光影紐扣 版权所有
    </div>
    """)

    # ========== 事件绑定 ==========
    # Tab1 按钮
    btn_deepl.click(fn=lambda: open_url(URLS["DeepL"]), outputs=status)
    btn_youdao.click(fn=lambda: open_url(URLS["有道翻译"]), outputs=status)
    btn_deepseek.click(fn=lambda: open_url(URLS["DeepSeek"]), outputs=status)
    btn_doubao.click(fn=lambda: open_url(URLS["豆包"]), outputs=status)
    btn_qianwen.click(fn=lambda: open_url(URLS["通义千问"]), outputs=status)
    btn_kimi.click(fn=lambda: open_url(URLS["Kimi"]), outputs=status)
    btn_chatglm.click(fn=lambda: open_url(URLS["ChatGLM"]), outputs=status)
    btn_yuanbao.click(fn=lambda: open_url(URLS["腾讯元宝"]), outputs=status)

    # Tab2 按钮（绑定到 status2）
    btn_deepl_tab2.click(fn=lambda: open_url(URLS["DeepL"]), outputs=status2)
    btn_youdao_tab2.click(fn=lambda: open_url(URLS["有道翻译"]), outputs=status2)
    btn_deepseek_tab2.click(fn=lambda: open_url(URLS["DeepSeek"]), outputs=status2)
    btn_doubao_tab2.click(fn=lambda: open_url(URLS["豆包"]), outputs=status2)
    btn_qianwen_tab2.click(fn=lambda: open_url(URLS["通义千问"]), outputs=status2)
    btn_kimi_tab2.click(fn=lambda: open_url(URLS["Kimi"]), outputs=status2)
    btn_chatglm_tab2.click(fn=lambda: open_url(URLS["ChatGLM"]), outputs=status2)
    btn_yuanbao_tab2.click(fn=lambda: open_url(URLS["腾讯元宝"]), outputs=status2)

if __name__ == "__main__":
    demo.launch(
        server_name="127.0.0.1",
        server_port=18009,
        inbrowser=True,
        show_error=True
    )