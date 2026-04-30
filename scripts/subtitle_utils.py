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

字幕处理工具箱
功能：双语合并 / SRT转TXT / 中文字幕添加拼音 / 纯文本转字幕 / LRC转SRT / 格式转换（SRT<->ASS, TXT转SRT简易版）+ 繁简转换
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

# ==================== 新增：繁简转换依赖 ====================
try:
    import opencc
    OPENCC_AVAILABLE = True
except ImportError:
    OPENCC_AVAILABLE = False
    print("警告: 未安装 opencc-python-reimplemented，繁简转换功能将不可用。")
    print("请运行: .\\python_embeded\\python.exe -m pip install opencc-python-reimplemented")

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

def srt_time_to_ass(time_str):
    """将SRT时间格式 (HH:MM:SS,mmm) 转换为ASS时间格式 (H:MM:SS.cc)"""
    time_str = time_str.replace(',', '.')
    parts = time_str.split(':')
    if len(parts) == 3:
        hours = int(parts[0])
        minutes = int(parts[1])
        sec_frac = parts[2].split('.')
        seconds = int(sec_frac[0])
        centiseconds = int(float('0.' + sec_frac[1]) * 100) if len(sec_frac) > 1 else 0
        return f"{hours}:{minutes:02d}:{seconds:02d}.{centiseconds:02d}"
    return time_str

def ass_time_to_srt(time_str):
    """将ASS时间格式 (H:MM:SS.cc) 转换为SRT时间格式 (HH:MM:SS,mmm)"""
    parts = time_str.split(':')
    if len(parts) == 3:
        hours = int(parts[0])
        minutes = int(parts[1])
        sec_frac = parts[2].split('.')
        seconds = int(sec_frac[0])
        centiseconds = int(sec_frac[1]) if len(sec_frac) > 1 else 0
        millis = centiseconds * 10
        return f"{hours:02d}:{minutes:02d}:{seconds:02d},{millis:03d}"
    return time_str

def parse_ass(content):
    """解析ASS文件，返回条目列表 (start_sec, end_sec, text)"""
    entries = []
    in_events = False
    for line in content.split('\n'):
        line = line.strip()
        if line.startswith('[Events]'):
            in_events = True
            continue
        if in_events and line.startswith('Dialogue:'):
            parts = line.split(',', 9)
            if len(parts) >= 10:
                start_ass = parts[1].strip()
                end_ass = parts[2].strip()
                text = parts[9].strip()
                text = text.replace('\\N', '\n')
                start_sec = ass_time_to_seconds(start_ass)
                end_sec = ass_time_to_seconds(end_ass)
                if start_sec is not None and end_sec is not None:
                    entries.append({
                        'start': start_sec,
                        'end': end_sec,
                        'text': text
                    })
    return entries

def ass_time_to_seconds(time_str):
    """将ASS时间字符串 (H:MM:SS.cc) 转换为秒数"""
    try:
        parts = time_str.split(':')
        if len(parts) == 3:
            hours = int(parts[0])
            minutes = int(parts[1])
            sec_frac = parts[2].split('.')
            seconds = int(sec_frac[0])
            centiseconds = int(sec_frac[1]) if len(sec_frac) > 1 else 0
            return hours * 3600 + minutes * 60 + seconds + centiseconds / 100.0
    except:
        return None
    return None

def build_ass(entries, header_template=None):
    """从条目列表构建ASS内容，entries: list of {'start':秒, 'end':秒, 'text':str}"""
    if header_template is None:
        header = """[Script Info]
ScriptType: v4.00+
Collisions: Normal
PlayDepth: 0
Timer: 100.0000

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,Microsoft YaHei,24,&H00FFFFFF,&H00000000,&H00000000,&H00000000,0,0,0,0,100,100,0,0,1,2,0,2,20,20,20,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    else:
        header = header_template

    lines = [header]
    for idx, item in enumerate(entries, start=1):
        start_ass = seconds_to_ass_time(item['start'])
        end_ass = seconds_to_ass_time(item['end'])
        text = item['text'].replace('\n', '\\N')
        dialogue = f"Dialogue: 0,{start_ass},{end_ass},Default,,0,0,0,,{text}"
        lines.append(dialogue)
    return '\n'.join(lines)

def seconds_to_ass_time(seconds):
    """将秒数转换为ASS时间格式 (H:MM:SS.cc)"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    centiseconds = int((seconds - int(seconds)) * 100)
    return f"{hours}:{minutes:02d}:{secs:02d}.{centiseconds:02d}"

def srt_time_to_seconds(time_str):
    """将SRT时间字符串 (HH:MM:SS,mmm) 转换为秒数"""
    try:
        time_str = time_str.replace(',', '.')
        parts = time_str.split(':')
        if len(parts) == 3:
            hours = int(parts[0])
            minutes = int(parts[1])
            sec_frac = parts[2].split('.')
            seconds = int(sec_frac[0])
            millis = int(sec_frac[1]) if len(sec_frac) > 1 else 0
            return hours * 3600 + minutes * 60 + seconds + millis / 1000.0
    except:
        return None
    return None

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

    if tone_style == "带声调":
        style = Style.TONE
    elif tone_style == "不带声调":
        style = Style.NORMAL
    else:
        style = Style.TONE3

    entries = parse_srt(content)
    new_entries = []
    for entry in entries:
        text = entry['text']
        pinyin_list = pinyin(text, style=style)
        pinyin_text = ' '.join([item[0] for item in pinyin_list])
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

# ==================== 功能4：纯文本转字幕（带时间码格式） ====================
def text_to_srt(text_file, default_duration):
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
        time_match = re.match(r'^(\d+):(\d+)$', line)
        if time_match:
            minutes = int(time_match.group(1))
            seconds = int(time_match.group(2))
            start_seconds = minutes * 60 + seconds
            i += 1
            text_lines = []
            while i < len(lines) and not re.match(r'^\d+:\d+$', lines[i].strip()):
                text_lines.append(lines[i].strip())
                i += 1
            text = ' '.join(text_lines).strip()
            if text:
                entries.append((start_seconds, text))
        else:
            i += 1

    if not entries:
        return None, "未找到有效的时间码和字幕内容"

    srt_entries = []
    for idx, (start_sec, text) in enumerate(entries):
        if idx < len(entries) - 1:
            end_sec = entries[idx+1][0]
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

# ==================== 功能：格式转换 ====================
def srt_to_ass(srt_file):
    if srt_file is None:
        return None, "请上传SRT文件"
    try:
        content = Path(srt_file.name).read_text(encoding='utf-8')
    except Exception as e:
        return None, f"读取文件失败: {e}"
    entries_srt = parse_srt(content)
    ass_entries = []
    for e in entries_srt:
        time_part = e['timecode'].split(' --> ')
        if len(time_part) == 2:
            start_srt = time_part[0].strip()
            end_srt = time_part[1].strip()
            start_sec = srt_time_to_seconds(start_srt)
            end_sec = srt_time_to_seconds(end_srt)
            if start_sec is not None and end_sec is not None:
                ass_entries.append({
                    'start': start_sec,
                    'end': end_sec,
                    'text': e['text']
                })
    if not ass_entries:
        return None, "解析SRT失败"
    ass_content = build_ass(ass_entries)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    out_path = OUTPUT_DIR / f"converted_{timestamp}.ass"
    out_path.write_text(ass_content, encoding='utf-8')
    return str(out_path), f"✅ SRT转ASS成功，文件保存在 {out_path}"

def ass_to_srt(ass_file):
    if ass_file is None:
        return None, "请上传ASS文件"
    try:
        content = Path(ass_file.name).read_text(encoding='utf-8')
    except Exception as e:
        return None, f"读取文件失败: {e}"
    ass_entries = parse_ass(content)
    if not ass_entries:
        return None, "未找到有效的ASS对话行"
    srt_entries = []
    for idx, item in enumerate(ass_entries, start=1):
        start_srt = seconds_to_srt_time(item['start'])
        end_srt = seconds_to_srt_time(item['end'])
        srt_entries.append({
            'index': str(idx),
            'timecode': f"{start_srt} --> {end_srt}",
            'text': item['text']
        })
    result = build_srt(srt_entries)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    out_path = OUTPUT_DIR / f"converted_{timestamp}.srt"
    out_path.write_text(result, encoding='utf-8')
    return str(out_path), f"✅ ASS转SRT成功，文件保存在 {out_path}"

def ass_to_txt(ass_file):
    if ass_file is None:
        return None, "请上传ASS文件"
    try:
        content = Path(ass_file.name).read_text(encoding='utf-8')
    except Exception as e:
        return None, f"读取文件失败: {e}"
    ass_entries = parse_ass(content)
    if not ass_entries:
        return None, "未找到有效的ASS对话行"
    text_lines = [item['text'] for item in ass_entries]
    result = '\n'.join(text_lines)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    out_path = OUTPUT_DIR / f"ass_text_{timestamp}.txt"
    out_path.write_text(result, encoding='utf-8')
    return str(out_path), f"✅ ASS转TXT成功，文件保存在 {out_path}"

def txt_to_srt_simple(txt_file, duration_mode, fixed_duration, chars_per_second):
    if txt_file is None:
        return None, "请上传TXT文件"
    try:
        content = Path(txt_file.name).read_text(encoding='utf-8')
    except Exception as e:
        return None, f"读取文件失败: {e}"
    lines = [line.strip() for line in content.split('\n') if line.strip()]
    if not lines:
        return None, "文本文件为空"

    entries = []
    current_time = 0.0
    for line in lines:
        if duration_mode == 'fixed':
            duration = fixed_duration
        else:
            char_count = len(line)
            duration = max(1.0, char_count / chars_per_second)
        start_sec = current_time
        end_sec = current_time + duration
        entries.append({
            'start': start_sec,
            'end': end_sec,
            'text': line
        })
        current_time = end_sec

    srt_entries = []
    for idx, item in enumerate(entries, start=1):
        start_srt = seconds_to_srt_time(item['start'])
        end_srt = seconds_to_srt_time(item['end'])
        srt_entries.append({
            'index': str(idx),
            'timecode': f"{start_srt} --> {end_srt}",
            'text': item['text']
        })
    result = build_srt(srt_entries)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    out_path = OUTPUT_DIR / f"simple_txt_{timestamp}.srt"
    out_path.write_text(result, encoding='utf-8')
    return str(out_path), f"✅ TXT转SRT成功，文件保存在 {out_path}"

# ==================== 新增：繁简转换 ====================
_opencc_converters = {}

def get_opencc_converter(mode):
    """获取 OpenCC 转换器（带缓存）"""
    if mode not in _opencc_converters:
        _opencc_converters[mode] = opencc.OpenCC(mode)
    return _opencc_converters[mode]

def convert_subtitle_file(input_file, convert_mode, output_format):
    """
    繁简转换主函数
    convert_mode: "t2s" / "s2t" / "t2tw" / "t2hk"
    output_format: "same" / "srt" / "txt"
    """
    if not OPENCC_AVAILABLE:
        return None, "❌ 未安装 opencc-python-reimplemented，请运行: .\\python_embeded\\python.exe -m pip install opencc-python-reimplemented"
    if input_file is None:
        return None, "请上传字幕文件"

    try:
        content = Path(input_file.name).read_text(encoding='utf-8')
    except Exception as e:
        return None, f"读取文件失败: {e}"

    ext = Path(input_file.name).suffix.lower()
    converter = get_opencc_converter(convert_mode)
    converted_content = converter.convert(content)

    # 根据输出格式转换
    if ext == '.srt' and output_format == 'txt':
        entries = parse_srt(converted_content)
        result = '\n'.join([e['text'] for e in entries])
    elif ext == '.ass' and output_format == 'srt':
        entries = parse_ass(converted_content)
        if not entries:
            return None, "解析 ASS 文件失败"
        srt_entries = []
        for idx, item in enumerate(entries, start=1):
            srt_entries.append({
                'index': str(idx),
                'timecode': f"{seconds_to_srt_time(item['start'])} --> {seconds_to_srt_time(item['end'])}",
                'text': item['text']
            })
        result = build_srt(srt_entries)
    elif ext == '.ass' and output_format == 'txt':
        entries = parse_ass(converted_content)
        if not entries:
            return None, "解析 ASS 文件失败"
        result = '\n'.join([item['text'] for item in entries])
    elif ext == '.txt' and output_format == 'srt':
        lines = [l.strip() for l in converted_content.split('\n') if l.strip()]
        srt_entries = []
        current_time = 0.0
        for idx, line in enumerate(lines, start=1):
            duration = max(1.0, len(line) / 3.0)
            srt_entries.append({
                'index': str(idx),
                'timecode': f"{seconds_to_srt_time(current_time)} --> {seconds_to_srt_time(current_time + duration)}",
                'text': line
            })
            current_time += duration
        result = build_srt(srt_entries)
    else:
        result = converted_content

    mode_labels = {
        "t2s": "繁转简", "s2t": "简转繁",
        "t2tw": "繁转台", "t2hk": "繁转港"
    }
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    if output_format == "same":
        suffix = ext.lstrip('.')
    else:
        suffix = output_format
    out_path = OUTPUT_DIR / f"{mode_labels.get(convert_mode, 'convert')}_{timestamp}.{suffix}"
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

        # 标签页4：纯文本转字幕（带时间码格式）
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

        # ==================== 新增标签页：格式转换 ====================
        with gr.Tab("格式转换"):
            gr.Markdown("### SRT ↔ ASS 互转及TXT转换")
            with gr.Tabs():
                # 子标签页1：SRT转ASS
                with gr.Tab("SRT → ASS"):
                    with gr.Row():
                        with gr.Column():
                            srt_for_ass = gr.File(label="上传SRT文件", file_types=[".srt"])
                            srt2ass_btn = gr.Button("转换为ASS", variant="primary")
                        with gr.Column():
                            srt2ass_status = gr.Textbox(label="状态", interactive=False)
                            srt2ass_download = gr.File(label="下载ASS文件")
                    srt2ass_btn.click(
                        fn=srt_to_ass,
                        inputs=[srt_for_ass],
                        outputs=[srt2ass_download, srt2ass_status]
                    )

                # 子标签页2：ASS转SRT
                with gr.Tab("ASS → SRT"):
                    with gr.Row():
                        with gr.Column():
                            ass_for_srt = gr.File(label="上传ASS文件", file_types=[".ass"])
                            ass2srt_btn = gr.Button("转换为SRT", variant="primary")
                        with gr.Column():
                            ass2srt_status = gr.Textbox(label="状态", interactive=False)
                            ass2srt_download = gr.File(label="下载SRT文件")
                    ass2srt_btn.click(
                        fn=ass_to_srt,
                        inputs=[ass_for_srt],
                        outputs=[ass2srt_download, ass2srt_status]
                    )

                # 子标签页3：ASS转TXT
                with gr.Tab("ASS → TXT"):
                    with gr.Row():
                        with gr.Column():
                            ass_for_txt = gr.File(label="上传ASS文件", file_types=[".ass"])
                            ass2txt_btn = gr.Button("提取纯文本", variant="primary")
                        with gr.Column():
                            ass2txt_status = gr.Textbox(label="状态", interactive=False)
                            ass2txt_download = gr.File(label="下载TXT文件")
                    ass2txt_btn.click(
                        fn=ass_to_txt,
                        inputs=[ass_for_txt],
                        outputs=[ass2txt_download, ass2txt_status]
                    )

                # 子标签页4：TXT转SRT（简易版）
                with gr.Tab("TXT → SRT (简易)"):
                    with gr.Row():
                        with gr.Column():
                            txt_for_srt = gr.File(label="上传TXT文件", file_types=[".txt"])
                            duration_mode = gr.Radio(
                                choices=[("固定时长", "fixed"), ("自动（根据字数）", "auto")],
                                label="时长模式",
                                value="fixed"
                            )
                            fixed_duration = gr.Number(label="固定时长(秒)", value=2.0, minimum=0.5, maximum=10.0, step=0.5)
                            chars_per_second = gr.Number(label="自动模式：每秒字数", value=3.0, minimum=1.0, maximum=10.0, step=0.5)
                            txt2srt_btn = gr.Button("转换为SRT", variant="primary")
                        with gr.Column():
                            txt2srt_status = gr.Textbox(label="状态", interactive=False)
                            txt2srt_download = gr.File(label="下载SRT文件")

                    def update_visibility(mode):
                        if mode == "fixed":
                            return gr.update(visible=True), gr.update(visible=False)
                        else:
                            return gr.update(visible=False), gr.update(visible=True)
                    duration_mode.change(
                        fn=update_visibility,
                        inputs=[duration_mode],
                        outputs=[fixed_duration, chars_per_second]
                    )
                    txt2srt_btn.click(
                        fn=txt_to_srt_simple,
                        inputs=[txt_for_srt, duration_mode, fixed_duration, chars_per_second],
                        outputs=[txt2srt_download, txt2srt_status]
                    )

                # ==================== 新增子标签页5：繁简转换 ====================
                with gr.Tab("繁简转换"):
                    with gr.Row():
                        with gr.Column():
                            convert_input = gr.File(
                                label="上传字幕文件",
                                file_types=[".srt", ".txt", ".ass"]
                            )
                            convert_mode = gr.Dropdown(
                                label="转换模式",
                                choices=[
                                    ("繁体 → 简体", "t2s"),
                                    ("简体 → 繁体", "s2t"),
                                    ("繁体 → 台湾正体", "t2tw"),
                                    ("繁体 → 香港繁体", "t2hk"),
                                ],
                                value="t2s"
                            )
                            output_format = gr.Dropdown(
                                label="输出格式",
                                choices=[
                                    ("与输入相同", "same"),
                                    ("SRT", "srt"),
                                    ("纯文本 TXT", "txt"),
                                ],
                                value="same"
                            )
                            convert_btn = gr.Button("开始转换", variant="primary")
                        with gr.Column():
                            convert_status = gr.Textbox(label="状态", interactive=False)
                            convert_download = gr.File(label="下载转换后的文件")

                    convert_btn.click(
                        fn=convert_subtitle_file,
                        inputs=[convert_input, convert_mode, output_format],
                        outputs=[convert_download, convert_status]
                    )

    # 页脚
    gr.HTML("""
    <div class="notice" style="margin: 10px 0; padding: 10px; background: transparent; border-left: 4px solid #ff9800; font-size: 0.9em;">
        注意事项：<br>
        • 本工具仅用于个人学习与视频剪辑使用<br>
        • 禁止用于商业用途及侵权行为<br>            
        • 使用前确保模型与依赖环境正常配置
    </div>
    <div style="text-align: center; color: #666; font-size: 0.9em; margin-top: 15px;">
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
# ==================== 结尾 ====================

demo.queue().launch(
    server_name="127.0.0.1",
    server_port=18009,
    inbrowser=True,
    show_error=True,
    allowed_paths=[str(OUTPUT_DIR)]   
)
