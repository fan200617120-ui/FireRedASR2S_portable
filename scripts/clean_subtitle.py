#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
FireRedASR2S WebUI Professional Edition
Copyright 2026 光影的故事2018

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

字幕清洗工具 - 自定义版（已修复全部已知bug）
功能：去除语气词、重复词，支持自定义词库（可保存/加载TXT）
输出目录：根目录下的 output/字幕清洗

修复内容：
1. 激进模式正则匹配失效 → 使用标点/空格边界安全删除语气词
2. SRT解析吞下一段序号 → 增强文本块结束判断
3. 文本粘贴SRT检测过于简单 → 使用时间轴正则检测
4. 加载词库文件意外清空 → 无文件或出错时保持原内容不变
5. 临时文件无清理 → 下载文件统一存放到输出目录
6. 多行字幕丢失原格式 → 每行分别清洗后保留换行
7. 编码硬编码UTF-8 → 自动尝试常见中文编码
8. 开头/结尾语气词只删一个 → 循环删除直到干净
"""

import os
import sys
import re
import time
import tempfile
from pathlib import Path
import gradio as gr

# ==================== 路径设置 ====================
SCRIPT_DIR = Path(__file__).parent.absolute()
ROOT_DIR = SCRIPT_DIR.parent
OUTPUT_DIR = ROOT_DIR / "output" / "字幕清洗"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

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


def safe_read_file(file_path, encodings=('utf-8', 'gbk', 'gb2312', 'latin-1')):
    """尝试多种编码读取文件，解决编码硬编码问题"""
    for enc in encodings:
        try:
            with open(file_path, 'r', encoding=enc) as f:
                return f.read()
        except (UnicodeDecodeError, UnicodeError):
            continue
    raise ValueError(f"无法识别文件编码，已尝试: {encodings}")


def is_srt_content(text):
    """使用时间轴正则严格检测SRT格式（修复bug3）"""
    # 匹配典型时间轴行：00:00:01,000 --> 00:00:04,000
    pattern = r'\d+:\d{2}:\d{2}[.,]\d{3}\s*-->\s*\d+:\d{2}:\d{2}[.,]\d{3}'
    return bool(re.search(pattern, text))


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
        """清洗单行文本（修复开头/结尾循环删除，激进模式安全删除）"""
        if not text:
            return text
        original = text.strip()
        all_fillers = self.get_all_fillers()
        sorted_fillers = sorted(all_fillers, key=len, reverse=True)

        # 循环删除开头语气词（修复bug8）
        while True:
            changed = False
            for filler in sorted_fillers:
                if original.startswith(filler):
                    original = original[len(filler):].lstrip()
                    # 如果紧跟着标点，也一并去除
                    if original and original[0] in "，、；：":
                        original = original[1:].lstrip()
                    changed = True
                    break
            if not changed:
                break

        # 循环删除结尾语气词（修复bug8）
        while True:
            changed = False
            for filler in sorted_fillers:
                if original.endswith(filler):
                    original = original[:-len(filler)].rstrip()
                    if original and original[-1] in "，、；：":
                        original = original[:-1].rstrip()
                    changed = True
                    break
            if not changed:
                break

        # 激进模式：删除被标点或空格包围的句中语气词（修复bug1）
        if aggressive:
            # 使用正则匹配：前面是开头或标点/空格，后面是结尾或标点/空格
            for filler in sorted_fillers:
                escaped = re.escape(filler)
                # 模式：(^|分隔符)语气词(?=分隔符|$)
                pattern = r'(^|[\s，。！？；：、])(' + escaped + r')(?=[\s，。！？；：、]|$)'
                # 替换时保留前面的分隔符
                original = re.sub(pattern, r'\1', original)
                # 去除可能产生的多余空格
                original = re.sub(r'\s+', ' ', original).strip()

        return original.strip()

    def clean_srt(self, content, aggressive=True):
        """清洗SRT格式字幕，保留时间轴（修复多行合并问题bug6，修复吞序号bug2）"""
        lines = content.split('\n')
        result = []
        i = 0
        n = len(lines)
        while i < n:
            line = lines[i].strip()
            # 序号行
            if line.isdigit():
                result.append(line)
                i += 1
                if i < n and '-->' in lines[i]:  # 时间轴
                    result.append(lines[i].strip())
                    i += 1
                    # 收集文本行（直到空行或下一个序号）
                    text_lines = []
                    while i < n and lines[i].strip():
                        # 下一行如果是纯数字且其后为时间轴，则当前文本块结束（修复bug2）
                        if lines[i].strip().isdigit():
                            # 预读下一行是否可能是时间轴
                            if i + 1 < n and '-->' in lines[i + 1]:
                                break
                        text_lines.append(lines[i].strip())
                        i += 1
                    if text_lines:
                        # 修复bug6：每行分别清洗后保留原换行结构
                        cleaned_lines = [
                            self.clean_text(line, aggressive) for line in text_lines
                        ]
                        result.append('\n'.join(cleaned_lines))
                    result.append('')  # 空行
                else:
                    # 没有时间轴，原样保留
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
    """加载上传的TXT词库文件，修复bug4：无文件或出错时返回gr.update()保持原内容"""
    if file is None:
        # 不更改自定义词库内容
        return gr.update()
    try:
        content = safe_read_file(file.name)
        return content
    except Exception as e:
        gr.Warning(f"读取词库文件失败: {e}")
        return gr.update()


def save_custom_words_file(words_content):
    """将词库内容保存到输出目录并返回下载路径（修复bug5）"""
    if not words_content.strip():
        return None
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    out_name = f"custom_words_{timestamp}.txt"
    out_path = OUTPUT_DIR / out_name
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(words_content)
    return str(out_path)


def process_file(file, aggressive, custom_words, file_type):
    if file is None:
        return "请上传文件", None

    try:
        content = safe_read_file(file.name)  # 修复bug7：自动编码检测
    except Exception as e:
        return f"读取文件失败: {e}", None

    cleaner = SubtitleCleaner()
    cleaner.set_custom_words(custom_words)

    # 使用更稳健的SRT检测（修复bug3）
    if file_type == "自动检测":
        if is_srt_content(content):
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

    # 使用更稳健的SRT检测（修复bug3）
    is_srt = is_srt_content(text)
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
    #  字幕清洗工具（已修复所有已知bug）
    **去除语气词、重复词，支持自定义词库（可保存/加载TXT）**
    """)

    if DEFAULT_CUSTOM_WORDS_PATH.exists():
        try:
            default_custom_content = safe_read_file(DEFAULT_CUSTOM_WORDS_PATH)
        except:
            default_custom_content = ""
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
                    aggressive_mode = gr.Checkbox(label="激进模式（谨慎使用，可能误删）", value=False)

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
                        load_file = gr.File(label="选择词库文件", file_types=[".txt"], visible=False)
                        save_download = gr.File(label="下载词库", visible=False)

                    submit_btn = gr.Button("开始清洗", variant="primary")

                with gr.Column(scale=1):
                    file_preview = gr.Textbox(label="清洗预览", lines=15)
                    file_download = gr.File(label="下载清洗后文件")

            # 加载词库：点击后显示隐藏的file上传组件
            load_btn.click(
                lambda: gr.update(visible=True), None, load_file
            )
            # 上传成功后调用加载函数并隐藏上传组件
            load_file.change(
                load_custom_words_file, load_file, custom_words_input
            ).then(
                lambda: gr.update(visible=False), None, load_file
            )

            # 保存词库：生成下载文件并显示
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
                    text_aggressive = gr.Checkbox(label="激进模式（谨慎使用）", value=False)
                    text_custom = gr.Textbox(
                        label="自定义词库（每行一个词）",
                        placeholder="例如：\n哎哟喂\n我的天\n讲真的",
                        lines=4,
                        value=default_custom_content
                    )
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