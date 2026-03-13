#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
双语字幕翻译工具 Pro (在线API版)
功能：上下文感知翻译 + 批量加速 + 多种输出格式
支持：所有兼容 OpenAI 格式的 API (DeepSeek, Qwen, OpenAI, Moonshot 等)
本工具需自行配置 API Key，适合有一定技术基础的专业用户。
"""

import os
import re
import json
import time
import requests
from datetime import datetime
from typing import List, Dict, Optional
from pathlib import Path
import gradio as gr

class OnlineTranslator:
    """在线大模型翻译器"""
    
    def __init__(self):
        self.output_dir = self._get_output_dir()
        self.translation_cache = {}
        
    def _get_output_dir(self) -> str:
        """统一输出目录：项目根目录/output/双语字幕输出"""
        SCRIPT_DIR = Path(__file__).parent.absolute()
        ROOT_DIR = SCRIPT_DIR.parent
        output_dir = ROOT_DIR / "output" / "双语字幕输出"
        output_dir.mkdir(parents=True, exist_ok=True)
        return str(output_dir)

    def parse_srt(self, srt_content: str) -> List[Dict]:
        """解析SRT字幕格式"""
        subtitles = []
        srt_content = srt_content.replace('\r\n', '\n').replace('\r', '\n')
        blocks = re.split(r'\n\s*\n', srt_content.strip())
        
        for block in blocks:
            lines = block.strip().split('\n')
            if len(lines) >= 3:
                try:
                    if lines[0].strip().isdigit():
                        index = int(lines[0])
                        timecode_line = lines[1]
                        text_lines = lines[2:]
                    else:
                        index = len(subtitles) + 1
                        timecode_line = lines[0]
                        text_lines = lines[1:]

                    time_match = re.search(r'(\d{2}:\d{2}:\d{2}[,.]\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2}[,.]\d{3})', timecode_line)
                    if time_match:
                        start_time = time_match.group(1).replace(',', '.')
                        end_time = time_match.group(2).replace(',', '.')
                    else:
                        continue

                    text = '\n'.join(text_lines).strip()
                    if text:
                        subtitles.append({
                            'index': index,
                            'start_time': start_time,
                            'end_time': end_time,
                            'original_text': text,
                            'translated_text': '',
                        })
                except Exception as e:
                    print(f"解析SRT块失败: {e}")
                    continue
        return subtitles

    def parse_txt(self, txt_content: str) -> List[Dict]:
        """解析TXT字幕格式"""
        subtitles = []
        lines = txt_content.strip().split('\n')
        for i, line in enumerate(lines, 1):
            line = line.strip()
            if line:
                subtitles.append({
                    'index': i,
                    'original_text': line,
                    'translated_text': '',
                })
        return subtitles

    def call_api(self, messages: List[Dict], api_key: str, base_url: str, model: str, temperature: float) -> Optional[str]:
        """通用 API 调用函数"""
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }
        
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            # "max_tokens": 4096 # 根据需要开启
        }
        
        # 兼容不同的 base_url 格式
        endpoint = base_url.strip()
        if not endpoint.endswith("/chat/completions"):
            if endpoint.endswith("/"):
                endpoint += "v1/chat/completions"
            else:
                endpoint += "/v1/chat/completions"

        try:
            response = requests.post(endpoint, headers=headers, json=payload, timeout=120)
            if response.status_code == 200:
                result = response.json()
                return result["choices"][0]["message"]["content"].strip()
            else:
                print(f"API 错误 {response.status_code}: {response.text}")
                return None
        except Exception as e:
            print(f"API 连接异常: {e}")
            return None

    def translate_batch_with_context(self, subtitles: List[Dict], start_idx: int, batch_size: int, 
                                     window_size: int, target_lang: str, api_key: str, 
                                     base_url: str, model: str, temperature: float) -> Dict[int, str]:
        """
        带上下文窗口的批量翻译
        window_size: 向前看几句（上下文）
        """
        # 1. 提取当前批次
        end_idx = min(start_idx + batch_size, len(subtitles))
        batch = subtitles[start_idx:end_idx]
        
        # 2. 构建带上下文的 Prompt
        # 提取上下文（前 window_size 句，如果存在）
        context_before = []
        if start_idx > 0:
            context_start = max(0, start_idx - window_size)
            context_before = [s['original_text'] for s in subtitles[context_start:start_idx]]
        
        # 提取后文（后 window_size 句，帮助模型理解语气）
        context_after = []
        if end_idx < len(subtitles):
            context_end = min(len(subtitles), end_idx + window_size)
            context_after = [s['original_text'] for s in subtitles[end_idx:context_end]]

        # 构建文本块
        source_texts = [s['original_text'] for s in batch]
        
        # 3. 构造消息
        prompt_content = f"请将以下【待翻译文本】翻译成{target_lang}。\n"
        
        if context_before:
            prompt_content += f"\n[前文参考]:\n" + "\n".join(context_before) + "\n"
        
        prompt_content += f"\n[待翻译文本] (共{len(source_texts)}条，请保持行数一致，不要合并):\n" + "\n".join(f"{i+1}. {t}" for i, t in enumerate(source_texts))
        
        if context_after:
            prompt_content += f"\n[后文参考]:\n" + "\n".join(context_after) + "\n"
            
        prompt_content += "\n\n请直接输出翻译结果，每行一条，不要包含序号。"

        messages = [
            {"role": "system", "content": f"你是一个专业的字幕翻译专家。请根据上下文语境，将用户提供的文本翻译成{target_lang}。要求翻译准确、口语化，适合字幕阅读。请直接输出翻译结果，不要有多余解释。"},
            {"role": "user", "content": prompt_content}
        ]

        # 4. 调用 API
        response_text = self.call_api(messages, api_key, base_url, model, temperature)
        
        # 5. 解析结果
        results = {}
        if response_text:
            # 按行分割，去除可能存在的序号（如 "1. "）
            lines = response_text.strip().split('\n')
            cleaned_lines = []
            for line in lines:
                # 正则去除 "1. ", "1、" 等开头
                clean_line = re.sub(r'^\s*\d+[\.、\s]\s*', '', line).strip()
                if clean_line:
                    cleaned_lines.append(clean_line)
            
            # 匹配结果
            for i, text in enumerate(cleaned_lines):
                if i < len(batch):
                    original_idx = start_idx + i
                    results[original_idx] = text
                else:
                    break # 模型输出行数过多，忽略
        
        return results

    def translate_subtitles(self, subtitles: List[Dict], target_lang: str, api_key: str, 
                           base_url: str, model: str, temperature: float, batch_size: int = 10, 
                           context_window: int = 2, progress=None) -> List[Dict]:
        """全量翻译主逻辑"""
        total = len(subtitles)
        
        # 用于存储翻译结果
        translations = {}
        
        # 分批处理
        idx = 0
        while idx < total:
            # 更新进度
            if progress:
                progress(idx / total, desc=f"翻译中 {idx}/{total}...")
            
            # 尝试批量翻译
            batch_results = self.translate_batch_with_context(
                subtitles, idx, batch_size, context_window, 
                target_lang, api_key, base_url, model, temperature
            )
            
            # 如果批量翻译成功且数量匹配
            if len(batch_results) == min(batch_size, total - idx):
                translations.update(batch_results)
                idx += batch_size
            else:
                # 批量失败或数量不匹配，回退到单句翻译（带上下文）
                print(f"批量翻译结果异常，回退单句处理: Index {idx}")
                for i in range(idx, min(idx + batch_size, total)):
                    # 单句翻译实际上就是 batch_size=1
                    single_res = self.translate_batch_with_context(
                        subtitles, i, 1, context_window, 
                        target_lang, api_key, base_url, model, temperature
                    )
                    if i in single_res:
                        translations[i] = single_res[i]
                    else:
                        translations[i] = f"[翻译失败] {subtitles[i]['original_text']}"
                idx += batch_size

        # 填充结果
        for i, sub in enumerate(subtitles):
            sub['translated_text'] = translations.get(i, sub['original_text'])
            
        return subtitles

    def generate_bilingual_srt(self, subtitles: List[Dict], style: str = "上下对照") -> str:
        """生成双语SRT"""
        srt_lines = []
        for sub in subtitles:
            srt_lines.append(str(sub['index']))
            if 'start_time' in sub:
                srt_lines.append(f"{sub['start_time']} --> {sub['end_time']}")
            
            orig = sub['original_text']
            trans = sub['translated_text']
            
            if style == "上下对照":
                srt_lines.append(f"{orig}\n{trans}")
            elif style == "原文优先":
                srt_lines.append(f"{orig}\n({trans})")
            else: # 仅译文
                srt_lines.append(trans)
            srt_lines.append("")
            
        return '\n'.join(srt_lines)

    def save_results(self, content: str, filename: str) -> str:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = os.path.join(self.output_dir, f"{filename}_{timestamp}.srt")
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        return filepath

class TranslatorUI:
    def __init__(self):
        self.translator = OnlineTranslator()
        
    def create_interface(self):
        with gr.Blocks(title="双语字幕翻译 Pro", theme=gr.themes.Default()) as demo:
            # 顶部注意事项（专业从业者提示）
            gr.Markdown("""
            <div style="background: #721c24; border-left: 5px solid #dc3545; color: #fff; padding: 15px; border-radius: 8px; margin-bottom: 20px;">
            <strong>⚠️ 注意事项：</strong> 本工具需要自行配置 API Key 和模型接口，适合有一定技术基础的专业用户。<br>
            如只想快速翻译几句，建议使用「在线翻译快速入口」工具（本工具箱内提供）。
            </div>
            """)

            gr.Markdown("""#  双语字幕翻译工具 Pro (在线版)
            **支持上下文理解** | **支持 DeepSeek / Qwen / OpenAI / Moonshot**
            """)
            
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### 📥 输入")
                    input_file = gr.File(label="上传 SRT/TXT", file_types=[".srt", ".txt"], type="filepath")
                    input_text = gr.Textbox(label="或粘贴内容", lines=10, placeholder="支持标准SRT格式...")
                    
                    gr.Markdown("### ⚙️ 模型配置")
                    api_key = gr.Textbox(label="API Key", type="password", placeholder="sk-...")
                    
                    with gr.Row():
                        # 预设常用 API 地址
                        base_url = gr.Dropdown(
                            label="API Base URL", 
                            choices=[
                                "https://api.deepseek.com",
                                "https://api.openai.com",
                                "https://dashscope.aliyuncs.com/compatible-mode/v1", # 通义千问
                                "https://api.moonshot.cn/v1",
                                "http://localhost:11434/v1" # Ollama 兼容接口
                            ],
                            value="https://api.deepseek.com",
                            allow_custom_value=True
                        )
                    
                    model_name = gr.Textbox(label="模型名称", value="deepseek-chat", placeholder="gpt-4o, qwen-plus, deepseek-chat...")
                    
                    with gr.Accordion("高级参数", open=False):
                        target_lang = gr.Textbox(label="目标语言", value="简体中文")
                        temperature = gr.Slider(label="温度", minimum=0.1, maximum=1.0, value=0.3)
                        batch_size = gr.Slider(label="批量大小", minimum=1, maximum=20, value=5, step=1, 
                                               info="一次发送几句翻译，越大速度越快，但可能越不稳定")
                        context_window = gr.Slider(label="上下文窗口", minimum=0, maximum=5, value=2, step=1,
                                                   info="翻译时携带前后几句作为参考，0为无上下文")

                    translate_btn = gr.Button(" 开始翻译", variant="primary")
                    
                with gr.Column(scale=1):
                    gr.Markdown("### 📤 输出")
                    output_style = gr.Radio(["上下对照", "原文优先", "仅译文"], value="上下对照", label="SRT样式")
                    output_preview = gr.Textbox(label="预览", lines=20)
                    output_file = gr.File(label="下载文件")
                    status_bar = gr.Textbox(label="状态", interactive=False)

            # 事件绑定
            input_file.change(fn=self.load_file, inputs=[input_file], outputs=[input_text])
            
            translate_btn.click(
                fn=self.run_translation,
                inputs=[input_text, api_key, base_url, model_name, target_lang, temperature, batch_size, context_window, output_style],
                outputs=[output_preview, output_file, status_bar]
            )
            
            # 页脚版权与免责声明（整合您提供的HTML）
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
            
        return demo

    def load_file(self, file):
        if not file: return ""
        try:
            with open(file, 'r', encoding='utf-8') as f: return f.read()
        except: return "读取失败"

    def run_translation(self, content, api_key, base_url, model, target_lang, temp, batch_size, ctx_window, style, progress=gr.Progress()):
        if not content.strip(): return "请输入内容", None, "错误"
        if not api_key.strip(): return "请输入 API Key", None, "错误"
        
        progress(0, desc="解析字幕...")
        subtitles = self.translator.parse_srt(content)
        if not subtitles: subtitles = self.translator.parse_txt(content)
        
        if not subtitles: return "未解析到字幕", None, "错误"
        
        try:
            # 执行翻译
            translated_subs = self.translator.translate_subtitles(
                subtitles, target_lang, api_key, base_url, model, temp, 
                int(batch_size), int(ctx_window), progress
            )
            
            progress(0.9, desc="生成文件...")
            srt_content = self.translator.generate_bilingual_srt(translated_subs, style)
            file_path = self.translator.save_results(srt_content, "translated")
            
            preview = "\n\n".join([f"【原】{s['original_text']}\n【译】{s['translated_text']}" for s in translated_subs[:5]])
            
            return preview, file_path, f"✅ 翻译完成！共 {len(translated_subs)} 条"
            
        except Exception as e:
            return "", None, f"❌ 翻译出错: {str(e)}"

if __name__ == "__main__":
    ui = TranslatorUI()
    demo = ui.create_interface()
    demo.launch(server_name="127.0.0.1", server_port=7869, inbrowser=True)