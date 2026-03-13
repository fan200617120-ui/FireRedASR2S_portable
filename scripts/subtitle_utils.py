#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
字幕处理工具箱
功能：双语合并 / SRT转TXT / 中文字幕添加拼音 / 纯文本转字幕 / LRC转SRT
输出目录: ../output/字幕处理
"""

import os
import re
import time
from pathlib import Path

# 尝试导入依赖
try:
    import gradio as gr
except ImportError:
    print("请安装 gradio: pip install gradio")
    exit(1)

try:
    from pypinyin import pinyin, Style
    PYPINYIN_AVAILABLE = True
except ImportError:
    PYPINYIN_AVAILABLE = False   
    print("警告: 未安装 pypinyin，拼音功能将不可用。请在项目根目录下运行以下命令安装：")
    print("   .\\python_embeded\\python.exe -m pip install pypinyin")

# 项目路径
SCRIPT_DIR = Path(__file__).parent.absolute()
ROOT_DIR = SCRIPT_DIR.parent
OUTPUT_DIR = ROOT_DIR / "output" / "字幕处理"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ==================== 通用函数 ====================
def parse_srt(content):
    """解析SRT内容，返回条目列表"""
    entries = []
    blocks = re.split(r'\n\n+', content.strip())
    for block in blocks:
        lines = block.strip().split('\n')
        if len(lines) >= 3:
            entries.append({
                'index': lines[0].strip(),
                'timecode': lines[1].strip(),
                'text': '\n'.join(lines[2:]).strip()
            })
    return entries

def build_srt(entries):
    """从条目列表重建SRT"""
    return '\n\n'.join([f"{e['index']}\n{e['timecode']}\n{e['text']}" for e in entries])

def parse_lrc(content):
    """解析LRC格式，返回 (时间戳秒, 文本) 列表，时间戳为秒"""
    pattern = re.compile(r'\[(\d{2}):(\d{2})\.(\d{2,3})\](.*)')
    entries = []
    for line in content.split('\n'):
        line = line.strip()
        if not line:
            continue
        m = pattern.match(line)
        if m:
            minutes = int(m.group(1))
            seconds = int(m.group(2))
            millis = int(m.group(3).ljust(3, '0')[:3])  # 补全3位毫秒
            total_seconds = minutes * 60 + seconds + millis / 1000.0
            text = m.group(4).strip()
            entries.append((total_seconds, text))
    return entries

def seconds_to_srt_time(seconds):
    """将秒数转换为SRT时间格式 (HH:MM:SS,mmm)"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds - int(seconds)) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"

# ==================== 功能1：双语合并 ====================
def merge_bilingual(zh_file, en_file):
    if zh_file is None or en_file is None:
        return None, "请上传中文和英文字幕文件"

    try:
        zh_content = Path(zh_file.name).read_text(encoding='utf-8')
        en_content = Path(en_file.name).read_text(encoding='utf-8')
    except Exception as e:
        return None, f"读取文件失败: {e}"

    zh_entries = parse_srt(zh_content)
    en_entries = parse_srt(en_content)

    if len(zh_entries) != len(en_entries):
        return None, "中文和英文字幕条数不一致，请检查"

    merged = []
    for zh, en in zip(zh_entries, en_entries):
        merged.append({
            'index': zh['index'],
            'timecode': zh['timecode'],
            'text': zh['text'] + '\n' + en['text']
        })

    result = build_srt(merged)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    out_path = OUTPUT_DIR / f"bilingual_{timestamp}.srt"
    out_path.write_text(result, encoding='utf-8')
    return str(out_path), f"✅ 合并成功，文件保存在 {out_path}"

# ==================== 功能2：SRT转TXT ====================
def srt_to_txt(srt_file):
    if srt_file is None:
        return None, "请上传SRT文件"

    try:
        content = Path(srt_file.name).read_text(encoding='utf-8')
    except Exception as e:
        return None, f"读取文件失败: {e}"

    entries = parse_srt(content)
    lines = [e['text'] for e in entries]
    result = '\n'.join(lines)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    out_path = OUTPUT_DIR / f"text_{timestamp}.txt"
    out_path.write_text(result, encoding='utf-8')
    return str(out_path), f"✅ 转换成功，文件保存在 {out_path}"

# ==================== 功能3：中文字幕添加拼音 ====================
def add_pinyin_to_srt(srt_file, tone_style):
    if not PYPINYIN_AVAILABLE:
        return None, "❌ 未安装 pypinyin，无法使用拼音功能。请运行: .\\python_embeded\\python.exe -m pip install pypinyin"

    if srt_file is None:
        return None, "请上传SRT文件"

    try:
        content = Path(srt_file.name).read_text(encoding='utf-8')
    except Exception as e:
        return None, f"读取文件失败: {e}"

    # 选择拼音风格
    if tone_style == "带声调":
        style = Style.TONE
    elif tone_style == "不带声调":
        style = Style.NORMAL
    else:  # 数字声调
        style = Style.TONE3

    entries = parse_srt(content)
    new_entries = []
    for entry in entries:
        text = entry['text']
        pinyin_list = pinyin(text, style=style)
        pinyin_text = ' '.join([item[0] for item in pinyin_list])
        # 拼音在上，原文在下
        new_text = f"{pinyin_text}\n{text}"
        new_entries.append({
            'index': entry['index'],
            'timecode': entry['timecode'],
            'text': new_text
        })
    result = build_srt(new_entries)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    out_path = OUTPUT_DIR / f"pinyin_{timestamp}.srt"
    out_path.write_text(result, encoding='utf-8')
    return str(out_path), f"✅ 拼音添加成功，文件保存在 {out_path}"

# ==================== 功能4：纯文本转字幕 ====================
def text_to_srt(text_file, default_duration):
    """
    解析纯文本格式：
    时间码（如 0:00）后跟文本，可空格/换行/tab分隔。
    示例：
    0:00
    第一行字幕
    0:22
    第二行字幕
    """
    if text_file is None:
        return None, "请上传文本文件"

    try:
        content = Path(text_file.name).read_text(encoding='utf-8')
    except Exception as e:
        return None, f"读取文件失败: {e}"

    lines = content.strip().split('\n')
    entries = []
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line:
            i += 1
            continue
        # 尝试匹配时间码格式：数字:数字（如 0:00、12:34）
        time_match = re.match(r'^(\d+):(\d+)$', line)
        if time_match:
            minutes = int(time_match.group(1))
            seconds = int(time_match.group(2))
            start_seconds = minutes * 60 + seconds
            # 寻找后续文本行（直到下一个时间码或结尾）
            i += 1
            text_lines = []
            while i < len(lines) and not re.match(r'^\d+:\d+$', lines[i].strip()):
                text_lines.append(lines[i].strip())
                i += 1
            text = ' '.join(text_lines).strip()
            if text:
                entries.append((start_seconds, text))
        else:
            # 没有时间码，跳过
            i += 1

    if not entries:
        return None, "未找到有效的时间码和字幕内容"

    # 生成SRT条目
    srt_entries = []
    for idx, (start_sec, text) in enumerate(entries):
        # 结束时间：下一条的开始时间，或当前时间+默认时长
        if idx < len(entries) - 1:
            end_sec = entries[idx+1][0]
        else:
            end_sec = start_sec + default_duration
        # 如果结束时间小于等于开始时间，强制增加默认时长
        if end_sec <= start_sec:
            end_sec = start_sec + default_duration

        start_srt = seconds_to_srt_time(start_sec)
        end_srt = seconds_to_srt_time(end_sec)
        srt_entries.append({
            'index': str(idx + 1),
            'timecode': f"{start_srt} --> {end_srt}",
            'text': text
        })

    result = build_srt(srt_entries)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    out_path = OUTPUT_DIR / f"from_text_{timestamp}.srt"
    out_path.write_text(result, encoding='utf-8')
    return str(out_path), f"✅ 转换成功，文件保存在 {out_path}"

# ==================== 功能5：LRC转SRT ====================
def lrc_to_srt(lrc_file, default_duration):
    if lrc_file is None:
        return None, "请上传LRC文件"

    try:
        content = Path(lrc_file.name).read_text(encoding='utf-8')
    except Exception as e:
        return None, f"读取文件失败: {e}"

    lrc_entries = parse_lrc(content)
    if not lrc_entries:
        return None, "未找到有效的LRC歌词"

    srt_entries = []
    for idx, (start_sec, text) in enumerate(lrc_entries):
        if idx < len(lrc_entries) - 1:
            end_sec = lrc_entries[idx+1][0]
        else:
            end_sec = start_sec + default_duration
        if end_sec <= start_sec:
            end_sec = start_sec + default_duration

        start_srt = seconds_to_srt_time(start_sec)
        end_srt = seconds_to_srt_time(end_sec)
        srt_entries.append({
            'index': str(idx + 1),
            'timecode': f"{start_srt} --> {end_srt}",
            'text': text
        })

    result = build_srt(srt_entries)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    out_path = OUTPUT_DIR / f"from_lrc_{timestamp}.srt"
    out_path.write_text(result, encoding='utf-8')
    return str(out_path), f"✅ 转换成功，文件保存在 {out_path}"

# ==================== Gradio界面 ====================
with gr.Blocks(title="字幕处理工具箱", theme=gr.themes.Default()) as demo:
    gr.Markdown("# 字幕处理工具箱\n输出目录: `output/字幕处理`")

    with gr.Tabs():
        # 标签页1：双语合并
        with gr.Tab("双语合并"):
            with gr.Row():
                with gr.Column():
                    zh_file = gr.File(label="上行SRT", file_types=[".srt"])
                    en_file = gr.File(label="下行SRT", file_types=[".srt"])
                    merge_btn = gr.Button("合并字幕", variant="primary")
                with gr.Column():
                    merge_status = gr.Textbox(label="状态", interactive=False)
                    merge_download = gr.File(label="下载双语字幕")

            merge_btn.click(
                fn=merge_bilingual,
                inputs=[zh_file, en_file],
                outputs=[merge_download, merge_status]
            )

        # 标签页2：SRT转TXT
        with gr.Tab("SRT转TXT"):
            with gr.Row():
                with gr.Column():
                    srt_file_txt = gr.File(label="上传SRT文件", file_types=[".srt"])
                    convert_btn = gr.Button("转换为TXT", variant="primary")
                with gr.Column():
                    convert_status = gr.Textbox(label="状态", interactive=False)
                    convert_download = gr.File(label="下载TXT文件")

            convert_btn.click(
                fn=srt_to_txt,
                inputs=[srt_file_txt],
                outputs=[convert_download, convert_status]
            )

        # 标签页3：中文字幕添加拼音
        with gr.Tab("添加拼音"):
            with gr.Row():
                with gr.Column():
                    pinyin_file = gr.File(label="上传中文SRT", file_types=[".srt"])
                    tone_style = gr.Radio(
                        choices=["带声调", "不带声调", "数字声调"],
                        value="带声调",
                        label="拼音风格"
                    )
                    pinyin_btn = gr.Button("添加拼音", variant="primary")
                with gr.Column():
                    pinyin_status = gr.Textbox(label="状态", interactive=False)
                    pinyin_download = gr.File(label="下载带拼音的字幕")

            pinyin_btn.click(
                fn=add_pinyin_to_srt,
                inputs=[pinyin_file, tone_style],
                outputs=[pinyin_download, pinyin_status]
            )

        # 标签页4：纯文本转字幕
        with gr.Tab("文本转字幕"):
            with gr.Row():
                with gr.Column():
                    text_file = gr.File(label="上传文本文件", file_types=[".txt"])
                    default_duration_text = gr.Number(
                        label="默认每句时长(秒)",
                        value=2.0,
                        minimum=0.5,
                        maximum=10.0,
                        step=0.5
                    )
                    text_to_srt_btn = gr.Button("转换为SRT", variant="primary")
                with gr.Column():
                    text_status = gr.Textbox(label="状态", interactive=False)
                    text_download = gr.File(label="下载SRT字幕")

            text_to_srt_btn.click(
                fn=text_to_srt,
                inputs=[text_file, default_duration_text],
                outputs=[text_download, text_status]
            )

        # 标签页5：LRC转SRT
        with gr.Tab("LRC转SRT"):
            with gr.Row():
                with gr.Column():
                    lrc_file = gr.File(label="上传LRC文件", file_types=[".lrc"])
                    default_duration_lrc = gr.Number(
                        label="默认每句时长(秒)",
                        value=2.0,
                        minimum=0.5,
                        maximum=10.0,
                        step=0.5
                    )
                    lrc_to_srt_btn = gr.Button("转换为SRT", variant="primary")
                with gr.Column():
                    lrc_status = gr.Textbox(label="状态", interactive=False)
                    lrc_download = gr.File(label="下载SRT字幕")

            lrc_to_srt_btn.click(
                fn=lrc_to_srt,
                inputs=[lrc_file, default_duration_lrc],
                outputs=[lrc_download, lrc_status]
            )

    # 页脚（整合提供的HTML）
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
            🔗 <strong>B站主页</strong>: 
            <a href="https://space.bilibili.com/381518712" target="_blank" style="color: #ffdd40; text-decoration: none; font-weight: bold;">
            space.bilibili.com/381518712
            </a>
            </p>
        </div>
    </div>
    <div style="text-align: center; color: #666; margin-top: 10px; font-size: 0.9em;">
        © 原创 WebUI 代码 © 2026 光影紐扣 版权所有 
    </div>
    """)

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7871, inbrowser=True)