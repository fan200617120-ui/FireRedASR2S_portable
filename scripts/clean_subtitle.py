#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
字幕清洗工具 - 自定义版
功能：去除语气词、重复词，支持自定义词库（可保存/加载TXT）
输出目录：根目录下的 output/字幕清洗
"""

import os
import sys
import re
import time
import tempfile
from pathlib import Path
import gradio as gr

# ==================== 路径设置 ====================
# 脚本所在目录: F:\FireRedASR2S_portable_win\scripts
SCRIPT_DIR = Path(__file__).parent.absolute()
# 项目根目录: F:\FireRedASR2S_portable_win
ROOT_DIR = SCRIPT_DIR.parent
# 输出目录: 根目录/output/字幕清洗
OUTPUT_DIR = ROOT_DIR / "output" / "字幕清洗"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 默认自定义词库文件路径（放在 scripts 目录下，方便管理）
DEFAULT_CUSTOM_WORDS_PATH = SCRIPT_DIR / "custom_words.txt"

# 内置语气词库
DEFAULT_FILLER_WORDS = {
    "啊", "呀", "呢", "吧", "吗", "嘛", "噢", "哦", "哟", "咳", "唉", 
    "嗨", "嘿", "喂", "嗯", "呃", "哎", "喔", "呵", "哈", "嘻", "哼",
    "哇", "呐", "咯", "哩", "咧", "啦", "啰", "喽",
    "哎呀", "哎哟", "哎呦", "啊呀", "啊哟", "啊哈", "嗯哼", "呃呃", "哎哎",
    "哈哈", "呵呵", "嘿嘿", "嘻嘻", "咳咳", "唉呀", "唉哟", "哦哦", "噢噢",
    "哎哟喂", "我的天", "天呐", "老天", "我的妈", "妈呀",
    "这个", "那个", "然后", "就是", "的话", "那么", "所以", "但是", "而且",
    "不过", "然而", "因此", "因而", "于是", "接下来", "实际上", "其实呢",
    "就是说", "我们说", "可以说", "换句话说", "也就是说", "大家知道",
    "我们知道", "应该说", "老实说", "说实话", "坦率地说",
}

class SubtitleCleaner:
    def __init__(self):
        self.filler_words = DEFAULT_FILLER_WORDS.copy()
        self.custom_words = set()

    def set_custom_words(self, words_str):
        """设置自定义词库（从字符串按行解析）"""
        custom = set()
        for line in words_str.strip().split('\n'):
            w = line.strip()
            if w:
                custom.add(w)
        self.custom_words = custom

    def get_all_fillers(self):
        return self.filler_words.union(self.custom_words)

    def clean_text(self, text, aggressive=True):
        """清洗单行文本"""
        if not text:
            return text
        original = text
        all_fillers = self.get_all_fillers()
        sorted_fillers = sorted(all_fillers, key=len, reverse=True)

        # 去除开头语气词
        for filler in sorted_fillers:
            if original.startswith(filler):
                original = original[len(filler):].lstrip()
                if original and original[0] in "，、；：":
                    original = original[1:].lstrip()
                break

        # 去除结尾语气词
        for filler in sorted_fillers:
            if original.endswith(filler):
                original = original[:-len(filler)].rstrip()
                if original and original[-1] in "，、；：":
                    original = original[:-1].rstrip()
                break

        # 激进模式：去除句中语气词
        if aggressive:
            for filler in sorted_fillers:
                original = re.sub(rf'\s{re.escape(filler)}\s', ' ', original)
                original = re.sub(rf'^({re.escape(filler)})\s', '', original)
                original = re.sub(rf'\s({re.escape(filler)})$', '', original)
        return original.strip()

    def clean_srt(self, content, aggressive=True):
        """清洗SRT格式字幕，保持时间轴"""
        lines = content.split('\n')
        result = []
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            if line.isdigit():  # 序号
                result.append(line)
                i += 1
                if i < len(lines) and '-->' in lines[i]:  # 时间轴
                    result.append(lines[i].strip())
                    i += 1
                    # 文本行
                    text_lines = []
                    while i < len(lines) and lines[i].strip():
                        text_lines.append(lines[i].strip())
                        i += 1
                    if text_lines:
                        combined = ' '.join(text_lines)
                        cleaned = self.clean_text(combined, aggressive)
                        result.append(cleaned)
                    result.append('')  # 空行
                else:
                    # 没有时间轴，直接原样
                    result.append(line)
                    i += 1
            else:
                result.append(line)
                i += 1
        return '\n'.join(result)

    def clean_txt(self, content, aggressive=True):
        """清洗纯文本，按行处理"""
        lines = content.split('\n')
        cleaned = []
        for line in lines:
            if line.strip():
                cleaned.append(self.clean_text(line, aggressive))
            else:
                cleaned.append('')
        return '\n'.join(cleaned)

# ==================== 界面函数 ====================

def load_custom_words_file(file):
    """加载上传的TXT词库文件，返回内容"""
    if file is None:
        return ""
    try:
        with open(file.name, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        return f"读取失败: {e}"

def save_custom_words_file(words_content):
    """将词库内容保存为临时TXT，返回下载链接"""
    if not words_content.strip():
        return None
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as f:
        f.write(words_content)
        return f.name

def process_file(file, aggressive, custom_words, file_type):
    if file is None:
        return "请上传文件", None

    try:
        with open(file.name, 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception as e:
        return f"读取文件失败: {e}", None

    cleaner = SubtitleCleaner()
    cleaner.set_custom_words(custom_words)

    if file_type == "自动检测":
        if '-->' in content:
            cleaned = cleaner.clean_srt(content, aggressive)
            suffix = ".srt"
        else:
            cleaned = cleaner.clean_txt(content, aggressive)
            suffix = ".txt"
    elif file_type == "SRT字幕":
        cleaned = cleaner.clean_srt(content, aggressive)
        suffix = ".srt"
    else:  # TXT文本
        cleaned = cleaner.clean_txt(content, aggressive)
        suffix = ".txt"

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    stem = Path(file.name).stem
    out_filename = f"{stem}_cleaned_{timestamp}{suffix}"
    out_path = OUTPUT_DIR / out_filename
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(cleaned)

    preview = '\n'.join(cleaned.split('\n')[:10]) + ('...' if len(cleaned.split('\n')) > 10 else '')
    return preview, str(out_path)

def process_text(text, aggressive, custom_words):
    if not text.strip():
        return "请输入文本", None
    cleaner = SubtitleCleaner()
    cleaner.set_custom_words(custom_words)
    is_srt = '-->' in text
    if is_srt:
        cleaned = cleaner.clean_srt(text, aggressive)
        suffix = ".srt"
    else:
        cleaned = cleaner.clean_txt(text, aggressive)
        suffix = ".txt"

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    out_filename = f"text_cleaned_{timestamp}{suffix}"
    out_path = OUTPUT_DIR / out_filename
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(cleaned)

    preview = '\n'.join(cleaned.split('\n')[:10]) + ('...' if len(cleaned.split('\n')) > 10 else '')
    return preview, str(out_path)

# ==================== 创建界面 ====================
with gr.Blocks(title="字幕清洗工具", theme=gr.themes.Default()) as demo:
    gr.Markdown("""
    #  字幕清洗工具
    **去除语气词、重复词，支持自定义词库（可保存/加载TXT）**
    """)

    # 初始化自定义词库输入框的内容（如果默认文件存在）
    if DEFAULT_CUSTOM_WORDS_PATH.exists():
        with open(DEFAULT_CUSTOM_WORDS_PATH, 'r', encoding='utf-8') as f:
            default_custom_content = f.read()
    else:
        default_custom_content = ""

    with gr.Tabs():
        with gr.Tab("上传文件"):
            with gr.Row():
                with gr.Column(scale=1):
                    file_input = gr.File(label="选择文件", file_types=[".txt", ".srt"])
                    file_type = gr.Radio(
                        choices=["自动检测", "SRT字幕", "TXT文本"],
                        label="文件类型",
                        value="自动检测"
                    )
                    aggressive_mode = gr.Checkbox(label="激进模式", value=False)
                    
                    # 自定义词库区域（带加载/保存按钮）
                    with gr.Group():
                        gr.Markdown("### 自定义词库")
                        custom_words_input = gr.Textbox(
                            label="要删除的词（每行一个）",
                            placeholder="例如：\n哎哟喂\n我的天\n讲真的",
                            lines=4,
                            value=default_custom_content
                        )
                        with gr.Row():
                            load_btn = gr.Button(" 加载词库文件", variant="secondary")
                            save_btn = gr.Button(" 保存词库文件", variant="secondary")
                        # 隐藏的上传组件（用于加载词库文件）
                        load_file = gr.File(label="选择词库文件", file_types=[".txt"], visible=False)
                        # 用于保存词库的下载链接
                        save_download = gr.File(label="下载词库", visible=False)

                    submit_btn = gr.Button("开始清洗", variant="primary")

                with gr.Column(scale=1):
                    file_preview = gr.Textbox(label="清洗预览", lines=15)
                    file_download = gr.File(label="下载清洗后文件")

            # 加载词库按钮：触发隐藏的文件上传
            load_btn.click(
                lambda: gr.update(visible=True), None, load_file
            )
            load_file.change(
                load_custom_words_file, load_file, custom_words_input
            ).then(
                lambda: gr.update(visible=False), None, load_file
            )

            # 保存词库按钮：将当前词库内容保存为文件供下载
            save_btn.click(
                save_custom_words_file, custom_words_input, save_download
            ).then(
                lambda: gr.update(visible=True), None, save_download
            )

            submit_btn.click(
                process_file,
                inputs=[file_input, aggressive_mode, custom_words_input, file_type],
                outputs=[file_preview, file_download]
            )

        with gr.Tab("直接粘贴文本"):
            with gr.Row():
                with gr.Column(scale=1):
                    text_input = gr.Textbox(label="输入文本", lines=10, placeholder="粘贴要清洗的文本...")
                    text_aggressive = gr.Checkbox(label="激进模式", value=False)
                    
                    # 文本清洗选项卡的自定义词库
                    text_custom = gr.Textbox(
                        label="自定义词库（每行一个词）",
                        placeholder="例如：\n哎哟喂\n我的天\n讲真的",
                        lines=4,
                        value=default_custom_content
                    )
                    # 同样添加加载/保存按钮
                    with gr.Row():
                        text_load_btn = gr.Button(" 加载词库文件", variant="secondary")
                        text_save_btn = gr.Button(" 保存词库文件", variant="secondary")
                    text_load_file = gr.File(label="选择词库文件", file_types=[".txt"], visible=False)
                    text_save_download = gr.File(label="下载词库", visible=False)

                    text_submit = gr.Button("清洗文本", variant="primary")
                with gr.Column(scale=1):
                    text_preview = gr.Textbox(label="清洗预览", lines=15)
                    text_download = gr.File(label="下载清洗后文件")

            # 文本选项卡的加载/保存逻辑
            text_load_btn.click(lambda: gr.update(visible=True), None, text_load_file)
            text_load_file.change(load_custom_words_file, text_load_file, text_custom).then(
                lambda: gr.update(visible=False), None, text_load_file
            )
            text_save_btn.click(save_custom_words_file, text_custom, text_save_download).then(
                lambda: gr.update(visible=True), None, text_save_download
            )

            text_submit.click(
                process_text,
                inputs=[text_input, text_aggressive, text_custom],
                outputs=[text_preview, text_download]
            )

    gr.Markdown(f"**输出目录**: `{OUTPUT_DIR}`")

    # 页脚版权与免责声明
    gr.Markdown("---")
    gr.Markdown(f"""
    <div style="text-align: center; color: #666; font-size: 0.9em;">
    <p>本软件包不提供任何模型文件，模型由用户自行从官方渠道获取。用户需自行遵守模型的原许可证。</p>
    <p>本软件包按“原样”提供，不提供任何明示或暗示的担保。使用本软件所产生的一切风险由用户自行承担。</p>
    <p>本软件包开发者不对因使用本软件而导致的任何直接或间接损失负责。</p>       
    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 15px; border-radius: 8px; margin: 15px auto; max-width: 600px;">
    <p style="color: white; font-weight: bold; margin: 5px 0; font-size: 1em;">🎬 更新请关注B站up主：光影的故事2018</p>
    <p style="color: white; margin: 5px 0; font-size: 0.9em;">
    🔗 <strong>B站主页</strong>: 
    <a href="https://space.bilibili.com/381518712" target="_blank" style="color: #ffdd40; text-decoration: none; font-weight: bold;">
    space.bilibili.com/381518712
    </a>
    </p>
    </div>
    </div>
    """)
    gr.Markdown("""
    <div style="text-align: center; color: #666; margin-top: 10px; font-size: 0.9em;">
    © 原创 WebUI 代码 © 2026 光影紐扣 版权所有 | 基于 FireRedASR2S (Apache 2.0) 二次开发
    </div>
    """)

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=18007, inbrowser=True)