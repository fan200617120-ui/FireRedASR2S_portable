#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
FireRedASR2S WebUI 专业版 (最终增强版) - 含强制对齐与智能合并
- 保留原有所有功能
- 新增字幕预览标签页，支持音频播放与字幕高亮同步
- 新增视频字幕处理：从视频提取音频→识别→嵌入字幕（软/硬）
- 新增强制对齐功能：用稿子生成精准字幕，支持自定义合并规则（含空行断句、字数限制、时长限制）
- 新增自定义输出目录、错误日志、配置保存/加载功能
- 改进：空行断句恢复为精确 token 匹配，确保分段稳定
- 改进：FFmpeg 自动配置便携版，无需用户手动添加 PATH
- 改进：字幕预览对大文件进行保护，阈值可配置
- 改进：LLM 模型选项仅在目录存在时显示，避免误导
- 改进：视频处理临时文件确保清理
- 改进：日志自动清理旧文件
- 改进：统一所有输出文件名格式（原文件名+时间戳+描述符）
- 基于 fireredasr2s 模块
- 增加高级参数折叠面板，支持解码、VAD、Punc 微调
本软件包 (FireRedASR2S 便携版) 包含以下部分：
- FireRedASR2S 项目，遵循 Apache 2.0 许可证。
- WebUI 界面、启动脚本及整合编排 © 2026 [B站up主：光影的故事2018]。保留所有权利。
本软件包仅为原项目的封装与界面增强，不包含任何模型文件。模型需用户自行下载，其版权归原项目所有。
"""

import sys
import os
import json
import logging
import traceback
import time
import gc
import threading
import atexit
import tempfile
import hashlib
import re
import base64
import subprocess
import shutil
from pathlib import Path
from datetime import timedelta, datetime
from typing import List, Dict, Optional, Tuple

# ==================== 日志设置 ====================
LOG_DIR = Path(__file__).parent.parent / "logs"
LOG_DIR.mkdir(exist_ok=True)
def clean_old_logs(days=7):
    cutoff = time.time() - days * 24 * 3600
    for f in LOG_DIR.glob("error_*.log"):
        if f.stat().st_mtime < cutoff:
            try:
                f.unlink()
            except:
                pass
clean_old_logs()
log_file = LOG_DIR / f"error_{time.strftime('%Y%m%d')}.log"
logging.basicConfig(filename=log_file, level=logging.ERROR,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# ==================== 路径设置 ====================
CURRENT_DIR = Path(__file__).parent.absolute()
PROJECT_ROOT = CURRENT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT / "FireRedASR2S"))

# 导入 FireRedASR2S 模块
from fireredasr2s import FireRedAsr2System, FireRedAsr2SystemConfig
from fireredasr2s.fireredasr2 import FireRedAsr2Config
from fireredasr2s.fireredvad import FireRedVadConfig
from fireredasr2s.fireredlid import FireRedLidConfig
from fireredasr2s.fireredpunc import FireRedPuncConfig

# ==================== 基础路径 ====================
BASE_DIR = Path(__file__).parent.absolute()
ROOT_DIR = BASE_DIR.parent
DEFAULT_OUTPUT_DIR = ROOT_DIR / "output"
OUTPUT_DIR = DEFAULT_OUTPUT_DIR
ALIGN_OUTPUT_DIR = OUTPUT_DIR / "字幕自动打轴"
ALIGN_OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

# 配置文件目录
PRESET_DIR = ROOT_DIR / "preset"
PRESET_DIR.mkdir(exist_ok=True)
CONFIG_FILE = PRESET_DIR / "settings.json"

# 线程锁，用于保护全局目录修改
config_lock = threading.RLock()

# ==================== 自动配置 FFmpeg ====================
PORTABLE_FFMPEG_DIR = ROOT_DIR / "ffmpeg" / "bin"
PORTABLE_FFMPEG_EXE = PORTABLE_FFMPEG_DIR / "ffmpeg.exe"
if PORTABLE_FFMPEG_EXE.exists():
    os.environ["PATH"] = str(PORTABLE_FFMPEG_DIR) + os.pathsep + os.environ.get("PATH", "")
    FFMPEG_PATH = str(PORTABLE_FFMPEG_EXE)
    print(f"✅ 已自动加载内置 FFmpeg: {FFMPEG_PATH}")
else:
    system_ffmpeg = shutil.which("ffmpeg")
    if system_ffmpeg:
        FFMPEG_PATH = system_ffmpeg
        print(f"✅ 使用系统已安装的 FFmpeg: {FFMPEG_PATH}")
    else:
        FFMPEG_PATH = "ffmpeg"
        print("⚠️ 警告：未找到内置 FFmpeg，视频处理可能失败，请将 ffmpeg.exe 放入 ffmpeg/bin 目录。")

# ==================== 加载/保存配置 ====================
def load_settings():
    if CONFIG_FILE.exists():
        try:
            with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except:
            return {}
    return {}

def save_settings(settings):
    try:
        with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
            json.dump(settings, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"保存配置失败: {e}")

# ==================== 导入检查 ====================
try:
    FIRERED_AVAILABLE = True
    print("FireRedASR2S 模块导入成功")
except ImportError as e:
    FIRERED_AVAILABLE = False
    print(f"导入 FireRedASR2S 失败: {e}")

try:
    import gradio as gr
    import torch
    import numpy as np
    import librosa
    import soundfile as sf
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA可用: {torch.cuda.is_available()}")
except ImportError as e:
    print(f"基础依赖缺失: {e}")
    sys.exit(1)

# ==================== 全局模型管理器 ====================
class FireRedASR2SManager:
    def __init__(self):
        self.asr_system = None
        self.config = None
        self.lock = threading.RLock()
        self.temp_files = []
        self.settings = load_settings()

    def load_system(self, config_dict=None, advanced_params=None):
        with self.lock:
            if self.asr_system is not None:
                return True, "系统已加载"
            try:
                default_config = {
                    "use_gpu": torch.cuda.is_available(),
                    "use_half": False,
                    "enable_vad": True,
                    "enable_lid": True,
                    "enable_punc": True,
                    "asr_model_type": "aed",
                }
                if config_dict:
                    default_config.update(config_dict)

                model_dir = ROOT_DIR / "pretrained_models" / f"FireRedASR2-{default_config['asr_model_type'].upper()}"
                if not model_dir.exists():
                    return False, f"模型目录不存在: {model_dir}，请先下载对应模型"

                vad_config = FireRedVadConfig(use_gpu=default_config["use_gpu"])
                lid_config = FireRedLidConfig(use_gpu=default_config["use_gpu"])
                asr_config = FireRedAsr2Config(
                    use_gpu=default_config["use_gpu"],
                    use_half=default_config["use_half"],
                    return_timestamp=True
                )
                punc_config = FireRedPuncConfig(use_gpu=default_config["use_gpu"])

                if advanced_params:
                    if "beam_size" in advanced_params:
                        asr_config.beam_size = advanced_params["beam_size"]
                    if "nbest" in advanced_params:
                        asr_config.nbest = advanced_params["nbest"]
                    if "decode_max_len" in advanced_params:
                        asr_config.decode_max_len = advanced_params["decode_max_len"]
                    if "softmax_smoothing" in advanced_params:
                        asr_config.softmax_smoothing = advanced_params["softmax_smoothing"]
                    if "aed_length_penalty" in advanced_params:
                        asr_config.aed_length_penalty = advanced_params["aed_length_penalty"]
                    if "eos_penalty" in advanced_params:
                        asr_config.eos_penalty = advanced_params["eos_penalty"]
                    if "elm_weight" in advanced_params:
                        asr_config.elm_weight = advanced_params["elm_weight"]
                    if "vad_min_speech_frame" in advanced_params:
                        vad_config.min_speech_frame = advanced_params["vad_min_speech_frame"]
                    if "vad_max_speech_frame" in advanced_params:
                        vad_config.max_speech_frame = advanced_params["vad_max_speech_frame"]
                    if "vad_min_silence_frame" in advanced_params:
                        vad_config.min_silence_frame = advanced_params["vad_min_silence_frame"]
                    if "vad_speech_threshold" in advanced_params:
                        vad_config.speech_threshold = advanced_params["vad_speech_threshold"]
                    if "vad_smooth_window_size" in advanced_params:
                        vad_config.smooth_window_size = advanced_params["vad_smooth_window_size"]
                    # punc_threshold 可能不存在于 FireRedPuncConfig 中，但用户要求保留界面显示，故此处保留但可能无效
                    if "punc_threshold" in advanced_params:
                        punc_config.threshold = advanced_params["punc_threshold"]

                system_config = FireRedAsr2SystemConfig(
                    vad_model_dir=str(ROOT_DIR / "pretrained_models" / "FireRedVAD" / "VAD"),
                    lid_model_dir=str(ROOT_DIR / "pretrained_models" / "FireRedLID"),
                    asr_model_dir=str(model_dir),
                    punc_model_dir=str(ROOT_DIR / "pretrained_models" / "FireRedPunc"),
                    vad_config=vad_config,
                    lid_config=lid_config,
                    asr_config=asr_config,
                    punc_config=punc_config,
                    enable_vad=int(default_config["enable_vad"]),
                    enable_lid=int(default_config["enable_lid"]),
                    enable_punc=int(default_config["enable_punc"])
                )

                self.asr_system = FireRedAsr2System(system_config)
                self.config = default_config
                return True, f"系统加载成功 (ASR: {default_config['asr_model_type']})"
            except Exception as e:
                logging.error(traceback.format_exc())
                return False, f"加载失败: {str(e)}"

    def unload_system(self):
        with self.lock:
            self.asr_system = None
            self.config = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return True, "系统已卸载"

    def transcribe(self, audio_input, **kwargs):
        if self.asr_system is None:
            return None, None, "系统未加载，请先加载模型"
        audio_path = self._prepare_audio(audio_input)
        if audio_path is None:
            return None, None, "音频处理失败"
        try:
            result = self.asr_system.process(audio_path)
            return result, audio_path, None
        except Exception as e:
            logging.error(traceback.format_exc())
            return None, None, f"识别失败: {str(e)}"

    def _prepare_audio(self, audio_input):
        if isinstance(audio_input, str) and os.path.exists(audio_input):
            try:
                data, sr = librosa.load(audio_input, sr=None)
                if sr != 16000:
                    data = librosa.resample(data.astype(np.float32), orig_sr=sr, target_sr=16000)
                    sr = 16000
                temp_hash = hashlib.md5(data.tobytes() + str(time.time()).encode()).hexdigest()[:8]
                temp_path = os.path.join(tempfile.gettempdir(), f"firered_temp_{temp_hash}.wav")
                sf.write(temp_path, data, sr)
                self.temp_files.append(temp_path)
                return temp_path
            except Exception as e:
                print(f"音频文件转换失败: {e}")
                return None
        if isinstance(audio_input, tuple):
            try:
                sr, data = audio_input
                if data.ndim > 1:
                    data = np.mean(data, axis=1)
                if sr != 16000:
                    data = librosa.resample(data.astype(np.float32), orig_sr=sr, target_sr=16000)
                    sr = 16000
                temp_hash = hashlib.md5(data.tobytes() + str(time.time()).encode()).hexdigest()[:8]
                temp_path = os.path.join(tempfile.gettempdir(), f"firered_temp_{temp_hash}.wav")
                sf.write(temp_path, data, sr)
                self.temp_files.append(temp_path)
                return temp_path
            except Exception as e:
                print(f"音频转换失败: {e}")
                return None
        return None

    def cleanup_temp(self):
        cleaned = 0
        for f in self.temp_files[:]:
            try:
                os.unlink(f)
                self.temp_files.remove(f)
                cleaned += 1
            except:
                pass
        return cleaned

    # ==================== 强制对齐方法 ====================
    def _seconds_to_srt_time(self, seconds):
        td = timedelta(seconds=seconds)
        total_seconds = int(td.total_seconds())
        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60
        secs = total_seconds % 60
        ms = int((td.total_seconds() - total_seconds) * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{ms:03d}"

    def force_align(self, audio_input, reference_text, progress=None):
        if self.asr_system is None:
            return None, None, None, None, "模型未加载，请先加载 AED 模型"
        if self.config['asr_model_type'] != 'aed':
            return None, None, None, None, "强制对齐仅支持 AED 模型，当前为 " + self.config['asr_model_type']

        audio_path = self._prepare_audio(audio_input)
        if audio_path is None:
            return None, None, None, None, "音频处理失败"

        try:
            waveform, sr = librosa.load(audio_path, sr=16000, mono=True)
            duration = len(waveform) / 16000

            asr = self.asr_system.asr

            feats, lengths, durs, _, _ = asr.feat_extractor([(16000, waveform)], ["tmp"])
            if not isinstance(lengths, torch.Tensor):
                lengths = torch.tensor(lengths, dtype=torch.long)
            else:
                lengths = lengths.long()
            if self.config['use_gpu']:
                feats = feats.cuda()
                lengths = lengths.cuda()
                if asr.config.use_half:
                    feats = feats.half()

            asr.model.eval()
            with torch.no_grad():
                enc_outputs, enc_lengths, _ = asr.model.encoder(feats, lengths)
                T = enc_outputs.size(1)
                frame_shift = duration / T

            tokens, token_ids = asr.tokenizer.tokenize(reference_text)
            if len(token_ids) == 0:
                return None, None, None, None, "参考文本分词后为空"

            yseq = torch.tensor(token_ids, device=enc_outputs.device)
            hyps = [[{"yseq": yseq}]]

            nbest_hyps = asr.model.get_token_timestamp_torchaudio(enc_outputs, enc_lengths, hyps)
            timestamp = nbest_hyps[0][0].get("timestamp")
            if timestamp is None:
                return None, None, None, None, "模型未返回时间戳"

            starts, ends = timestamp
            if len(starts) == 0:
                return None, None, None, None, "时间戳为空"

            # 时间戳单位检测
            print("\n" + "="*50)
            print("[调试] 开始时间戳单位检测")
            print(f"[调试] 音频时长 duration = {duration:.3f}s")
            print(f"[调试] 编码器输出帧数 T = {T}")
            print(f"[调试] frame_shift = {frame_shift:.6f}s")

            max_start = max(starts)
            min_start = min(starts)
            print(f"[调试] 原始时间戳范围: {min_start:.3f} - {max_start:.3f}")

            if max_start <= duration * 1.2:
                print("[调试] ✅ 检测到原始值接近音频时长，按秒单位处理")
                timestamps_sec = list(zip(starts, ends))
            else:
                hypothetical_max_sec = max_start * frame_shift
                print(f"[调试] 假设为帧编号 -> 理论最大时间 = {hypothetical_max_sec:.3f}s")
                if abs(hypothetical_max_sec - duration) < 0.2 * duration:
                    print("[调试] ✅ 按帧编号处理")
                    timestamps_sec = [(s * frame_shift, e * frame_shift) for s, e in zip(starts, ends)]
                else:
                    scale = duration / max_start * 0.95
                    print(f"[调试] ✅ 按未知单位缩放，系数 = {scale:.6f}")
                    timestamps_sec = [(s * scale, e * scale) for s, e in zip(starts, ends)]

            if timestamps_sec:
                converted_max = max(e for _, e in timestamps_sec)
                print(f"[调试] 转换后最大时间 = {converted_max:.3f}s")
                print(f"[调试] 与音频时长比值 = {converted_max/duration:.2f}")
            print("="*50 + "\n")

            min_len = min(len(timestamps_sec), len(token_ids))
            timestamps_sec = timestamps_sec[:min_len]
            token_ids = token_ids[:min_len]
            tokens = tokens[:min_len]

            token_texts = [asr.tokenizer.detokenize([tid]) for tid in token_ids]

            word_srt = []
            for i, ((start, end), txt) in enumerate(zip(timestamps_sec, token_texts), 1):
                word_srt.append(str(i))
                word_srt.append(self._seconds_to_srt_time(start) + " --> " + self._seconds_to_srt_time(end))
                word_srt.append(txt)
                word_srt.append("")
            word_srt_str = "\n".join(word_srt)

            if timestamps_sec:
                start_all = timestamps_sec[0][0]
                end_all = timestamps_sec[-1][1]
                full_text = asr.tokenizer.detokenize(token_ids)
                sentence_srt = f"1\n{self._seconds_to_srt_time(start_all)} --> {self._seconds_to_srt_time(end_all)}\n{full_text}\n"
            else:
                sentence_srt = ""

            # ---------- 修改：使用统一命名函数生成文件名 ----------
            timestamp_str = time.strftime("%Y%m%d_%H%M%S")
            prefix = generate_output_filename(audio_input, timestamp_str, default_name="align")
            word_path = ALIGN_OUTPUT_DIR / f"{prefix}_words.srt"
            sent_path = ALIGN_OUTPUT_DIR / f"{prefix}_sentence.srt"
            with open(word_path, "w", encoding="utf-8") as f:
                f.write(word_srt_str)
            with open(sent_path, "w", encoding="utf-8") as f:
                f.write(sentence_srt)

            return word_srt_str, sentence_srt, timestamps_sec, token_texts, None

        except Exception as e:
            logging.error(traceback.format_exc())
            return None, None, None, None, f"强制对齐失败: {str(e)}"
        finally:
            if audio_path in self.temp_files:
                self.cleanup_temp()

manager = FireRedASR2SManager()

# ==================== 工具函数 ====================
def seconds_to_srt_time(seconds):
    td = timedelta(seconds=seconds)
    total_seconds = int(td.total_seconds())
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    secs = total_seconds % 60
    ms = int((td.total_seconds() - total_seconds) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{ms:03d}"

def format_result_to_outputs(result):
    if not result or not isinstance(result, dict):
        return "无结果", "{}", "", []

    text = result.get("text", "")
    sentences = result.get("sentences", [])
    words = result.get("words", [])
    vad_segments = result.get("vad_segments_ms", [])

    segments = []
    if sentences:
        for s in sentences:
            segments.append({
                "start": s.get("start_ms", 0) / 1000.0,
                "end": s.get("end_ms", 0) / 1000.0,
                "text": s.get("text", "")
            })
    elif words:
        for w in words:
            segments.append({
                "start": w.get("start_ms", 0) / 1000.0,
                "end": w.get("end_ms", 0) / 1000.0,
                "text": w.get("text", "")
            })

    timestamps_json = json.dumps(segments, ensure_ascii=False, indent=2)

    srt_lines = []
    for i, seg in enumerate(segments, 1):
        start = seconds_to_srt_time(seg["start"])
        end = seconds_to_srt_time(seg["end"])
        srt_lines.append(str(i))
        srt_lines.append(f"{start} --> {end}")
        srt_lines.append(seg["text"])
        srt_lines.append("")
    srt_text = "\n".join(srt_lines)

    extra = f"VAD段: {len(vad_segments)}"
    if sentences and "asr_confidence" in sentences[0]:
        extra += f" | 置信度: {sentences[0]['asr_confidence']:.3f}"
    full_text = f"{text}\n\n[元数据] {extra}"

    return full_text, timestamps_json, srt_text, segments

def generate_subtitle_html(segments, audio_path, max_size_mb=None):
    if not audio_path or not os.path.exists(audio_path) or not segments:
        return '<div style="padding:20px; text-align:center; color:#999;">暂无字幕预览</div>'

    # 从设置中读取阈值，若未传入则使用默认值
    if max_size_mb is None:
        max_size_mb = manager.settings.get("preview_max_size_mb", 5)

    try:
        size = os.path.getsize(audio_path) / (1024 * 1024)
        if size > max_size_mb:
            return f'<div style="padding:20px; text-align:center; color:#666;">⚠️ 音频文件过大 ({size:.1f} MB)，超过 {max_size_mb} MB 预览限制。请直接使用 SRT 文件。</div>'
    except:
        pass

    mime_map = {
        '.wav': 'audio/wav', '.mp3': 'audio/mpeg', '.m4a': 'audio/mp4',
        '.flac': 'audio/flac', '.ogg': 'audio/ogg', '.aac': 'audio/aac',
    }
    ext = os.path.splitext(audio_path)[1].lower()
    mime_type = mime_map.get(ext, 'audio/wav')
    try:
        with open(audio_path, "rb") as f:
            audio_data = f.read()
            b64_data = base64.b64encode(audio_data).decode('utf-8')
    except Exception as e:
        return f'<div style="color:red;">音频加载失败: {e}</div>'
    audio_html = f'''
    <div style="margin-bottom:15px; padding:15px; background:#f8f9fa; border-radius:8px; border:1px solid #ddd;">
        <div style="margin-bottom:8px; font-weight:bold; color:#333;">▶ 音频回放</div>
        <audio id="preview-audio" controls style="width:100%;">
            <source src="data:{mime_type};base64,{b64_data}" type="{mime_type}">
            您的浏览器不支持 audio 元素。
        </audio>
    </div>
    '''
    cues_html = '<div id="subtitle-container" style="height:500px; overflow-y:auto; border:1px solid #ccc; padding:10px; border-radius:5px; background-color:#fff;">'
    for i, seg in enumerate(segments):
        cues_html += f'''
        <div class="subtitle-row" data-index="{i}" data-start="{seg['start']}" data-end="{seg['end']}"
             style="padding:10px; margin:5px 0; border-radius:5px; border-bottom:1px solid #eee; cursor:pointer; display:flex; align-items:flex-start;">
            <span style="color:#666; font-size:0.85em; margin-right:15px; font-family:monospace; background:#eee; padding:2px 6px; border-radius:4px; min-width:60px; text-align:center;">
                {seg['start']:.2f}s
            </span>
            <span class="text-content" style="font-size:1.1rem; color:#333; line-height:1.5;">{seg['text']}</span>
        </div>
        '''
    cues_html += '</div>'
    js_script = '''
    <script>
    (function() {
        let audio = document.getElementById("preview-audio");
        let container = document.getElementById("subtitle-container");
        if (!audio || !container) return;
        let rows = container.querySelectorAll(".subtitle-row");
        let clickBound = false;
        function updateHighlight() {
            let currentTime = audio.currentTime;
            rows.forEach(row => {
                let start = parseFloat(row.getAttribute("data-start"));
                let end = parseFloat(row.getAttribute("data-end"));
                if (currentTime >= start && currentTime < end) {
                    if (!row.classList.contains("active")) {
                        row.classList.add("active");
                        row.style.backgroundColor = "#e3f2fd";
                        row.style.borderLeft = "5px solid #2196f3";
                        row.style.fontWeight = "bold";
                        row.style.transform = "scale(1.01)";
                        row.style.boxShadow = "0 2px 5px rgba(0,0,0,0.1)";
                        row.scrollIntoView({ behavior: "smooth", block: "center" });
                    }
                } else {
                    if (row.classList.contains("active")) {
                        row.classList.remove("active");
                        row.style.backgroundColor = "";
                        row.style.borderLeft = "";
                        row.style.fontWeight = "";
                        row.style.transform = "";
                        row.style.boxShadow = "";
                    }
                }
            });
        }
        if (!clickBound) {
            rows.forEach(row => {
                row.onclick = function() {
                    let start = parseFloat(row.getAttribute("data-start"));
                    audio.currentTime = start;
                    audio.play();
                };
            });
            clickBound = true;
        }
        audio.addEventListener("timeupdate", updateHighlight);
        setTimeout(updateHighlight, 100);
    })();
    </script>
    '''
    return audio_html + cues_html + js_script

def save_outputs(base_name, full_text, timestamps_json, srt_text, language, model_info):
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    if base_name:
        safe = re.sub(r'[^\w\u4e00-\u9fff\-\.]', '', Path(base_name).stem)
        prefix = f"{safe}_{timestamp}"
    else:
        prefix = f"firered_{timestamp}"
    saved = {}
    txt_path = OUTPUT_DIR / f"{prefix}.txt"
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write(full_text)
    saved['txt'] = str(txt_path)
    if timestamps_json and timestamps_json != "{}":
        json_path = OUTPUT_DIR / f"{prefix}.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            f.write(timestamps_json)
        saved['json'] = str(json_path)
    if srt_text.strip():
        srt_path = OUTPUT_DIR / f"{prefix}.srt"
        with open(srt_path, 'w', encoding='utf-8') as f:
            f.write(srt_text)
        saved['srt'] = str(srt_path)
    return saved

def get_system_info():
    info = []
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        total = torch.cuda.get_device_properties(0).total_memory / 1e9
        allocated = torch.cuda.memory_allocated(0) / 1e9
        info.append(f"显卡: {gpu_name}")
        info.append(f"总显存: {total:.1f} GB")
        info.append(f"已分配: {allocated:.1f} GB ({allocated/total*100:.1f}%)")
    else:
        info.append("设备: CPU模式")
    with manager.lock:
        if manager.asr_system is not None:
            info.append(f"ASR系统: 已加载 ({manager.config['asr_model_type']})")
            info.append(f"  VAD: {'启用' if manager.config['enable_vad'] else '禁用'}")
            info.append(f"  LID: {'启用' if manager.config['enable_lid'] else '禁用'}")
            info.append(f"  Punc: {'启用' if manager.config['enable_punc'] else '禁用'}")
        else:
            info.append("ASR系统: 未加载")
    info.append(f"输出目录: {OUTPUT_DIR}")
    info.append(f"字幕自动打轴输出: {ALIGN_OUTPUT_DIR}")
    info.append(f"日志文件: {log_file}")
    return "\n".join(info)

# ==================== 字幕合并函数（增强版）====================
def merge_timestamps_to_sentences(timestamps, words,
                                   sentence_endings="。！？.!?",
                                   max_words=20,
                                   max_chars=50,
                                   max_duration=10.0,
                                   silence_threshold=0.3,
                                   merge_by_punc=True,
                                   merge_by_silence=True,
                                   merge_by_wordcount=True,
                                   merge_by_charcount=True,
                                   merge_by_duration=True,
                                   force_break_indices=None):
    if len(timestamps) == 0:
        return []
    
    sentences = []
    current_start = timestamps[0][0]
    current_words = []
    last_end = timestamps[0][1]
    current_text = ""
    
    for i, ((start, end), word) in enumerate(zip(timestamps, words)):
        should_break = False
        
        if force_break_indices and i < len(force_break_indices) and force_break_indices[i]:
            should_break = True
        else:
            if merge_by_punc and any(word.endswith(p) for p in sentence_endings):
                should_break = True
            if not should_break and merge_by_silence and i > 0:
                gap = start - last_end
                if gap > silence_threshold:
                    should_break = True
            if not should_break and merge_by_wordcount and len(current_words) + 1 >= max_words:
                should_break = True
            if not should_break and merge_by_charcount and current_words:
                new_text = current_text + word
                if len(new_text) >= max_chars:
                    should_break = True
            if not should_break and merge_by_duration and current_words:
                current_duration = last_end - current_start
                if current_duration + (end - start) >= max_duration:
                    should_break = True
        
        if not current_words:
            current_start = start
        current_words.append(word)
        current_text = (current_text + word) if current_words else word
        last_end = end
        
        if should_break:
            sentences.append({
                "start": current_start,
                "end": last_end,
                "text": "".join(current_words).strip()
            })
            current_start = None
            current_words = []
            current_text = ""
    
    if current_words:
        sentences.append({
            "start": current_start,
            "end": last_end,
            "text": "".join(current_words).strip()
        })
    return sentences

def sentences_to_srt(sentences):
    srt_lines = []
    for i, sent in enumerate(sentences, 1):
        start_time = seconds_to_srt_time(sent["start"])
        end_time = seconds_to_srt_time(sent["end"])
        srt_lines.append(str(i))
        srt_lines.append(f"{start_time} --> {end_time}")
        srt_lines.append(sent["text"])
        srt_lines.append("")
    return "\n".join(srt_lines)

# ==================== 统一文件名生成函数 ====================
def generate_output_filename(base_input, timestamp_str, custom_suffix="", default_name="recording"):
    """
    根据输入（文件路径、字典或 None）生成安全的文件名前缀（不含扩展名）
    - base_input: 原始输入，可能是文件路径字符串、Gradio 音频字典或 None
    - timestamp_str: 时间戳字符串，如 '20250313_153045'
    - custom_suffix: 自定义后缀，如 'soft' 或 'hard'，可为空
    - default_name: 当无法提取原文件名时使用的默认名称
    返回: 安全的前缀字符串，例如 'meeting_20250313_153045_soft'
    """
    # 尝试提取原始文件名
    original_name = None
    if isinstance(base_input, str) and os.path.exists(base_input):
        original_name = Path(base_input).stem
    elif isinstance(base_input, dict) and base_input.get('path') and os.path.exists(base_input['path']):
        original_name = Path(base_input['path']).stem
    elif isinstance(base_input, tuple):
        # 麦克风录制的音频，使用默认名称
        original_name = default_name

    if not original_name:
        original_name = default_name

    # 去除非法字符（允许中文、字母、数字、下划线、连字符）
    safe_name = re.sub(r'[^\w\u4e00-\u9fff\-]', '', original_name)
    if not safe_name:
        safe_name = default_name

    # 组装前缀
    parts = [safe_name, timestamp_str]
    if custom_suffix:
        parts.append(custom_suffix)
    return "_".join(parts)

# ==================== 公共模型加载辅助函数 ====================
def ensure_model_loaded(asr_model_type, use_gpu, use_half,
                        enable_vad, enable_lid, enable_punc,
                        beam_size, nbest, decode_max_len,
                        softmax_smoothing, aed_length_penalty,
                        eos_penalty, elm_weight,
                        vad_min_speech_frame, vad_max_speech_frame,
                        vad_min_silence_frame, vad_speech_threshold,
                        vad_smooth_window_size, punc_threshold,
                        progress=None):
    config = {
        "use_gpu": use_gpu,
        "use_half": use_half,
        "enable_vad": enable_vad,
        "enable_lid": enable_lid,
        "enable_punc": enable_punc,
        "asr_model_type": asr_model_type
    }
    advanced = {
        "beam_size": beam_size,
        "nbest": nbest,
        "decode_max_len": decode_max_len,
        "softmax_smoothing": softmax_smoothing,
        "aed_length_penalty": aed_length_penalty,
        "eos_penalty": eos_penalty,
        "elm_weight": elm_weight,
        "vad_min_speech_frame": vad_min_speech_frame,
        "vad_max_speech_frame": vad_max_speech_frame,
        "vad_min_silence_frame": vad_min_silence_frame,
        "vad_speech_threshold": vad_speech_threshold,
        "vad_smooth_window_size": vad_smooth_window_size,
        "punc_threshold": punc_threshold,
    }
    with manager.lock:
        if manager.asr_system is None or manager.config != config:
            if manager.asr_system is not None:
                manager.unload_system()
            if progress:
                progress(0.1, desc="加载模型...")
            success, msg = manager.load_system(config, advanced)
            if not success:
                raise RuntimeError(f"加载失败: {msg}")
    return manager

# ==================== 音频识别函数 ====================
def transcribe_audio(audio, asr_model_type, use_gpu, use_half, enable_vad, enable_lid, enable_punc,
                     beam_size, nbest, decode_max_len, softmax_smoothing, aed_length_penalty,
                     eos_penalty, elm_weight,
                     vad_min_speech_frame, vad_max_speech_frame, vad_min_silence_frame,
                     vad_speech_threshold, vad_smooth_window_size,
                     punc_threshold,
                     progress=gr.Progress()):
    if not FIRERED_AVAILABLE:
        return "错误: FireRedASR2S 模块不可用", "", "", ""

    if audio is None:
        return "请上传或录制音频", "", "", ""

    progress(0, desc="初始化...")
    try:
        ensure_model_loaded(asr_model_type, use_gpu, use_half,
                            enable_vad, enable_lid, enable_punc,
                            beam_size, nbest, decode_max_len,
                            softmax_smoothing, aed_length_penalty,
                            eos_penalty, elm_weight,
                            vad_min_speech_frame, vad_max_speech_frame,
                            vad_min_silence_frame, vad_speech_threshold,
                            vad_smooth_window_size, punc_threshold,
                            progress)
    except RuntimeError as e:
        return str(e), "", "", ""

    progress(0.3, desc="识别中...")
    result, audio_path, error = manager.transcribe(audio)
    if error:
        return f"错误: {error}", "", "", ""

    progress(0.7, desc="生成输出...")
    full_text, timestamps_json, srt_text, segments = format_result_to_outputs(result)

    base_name = None
    if isinstance(audio, str) and os.path.exists(audio):
        base_name = audio
    saved = save_outputs(base_name, full_text, timestamps_json, srt_text,
                         language="自动检测", model_info=asr_model_type)

    # 使用配置的预览阈值
    preview_max_size = manager.settings.get("preview_max_size_mb", 5)
    subtitle_html = generate_subtitle_html(segments, audio_path, preview_max_size)

    save_info = "文件已保存:\n"
    if saved.get('txt'):
        save_info += f"  {Path(saved['txt']).name}\n"
    if saved.get('json'):
        save_info += f"  {Path(saved['json']).name}\n"
    if saved.get('srt'):
        save_info += f"  {Path(saved['srt']).name}\n"
    full_text = save_info + "\n" + full_text

    progress(0.9, desc="清理...")
    manager.cleanup_temp()

    progress(1.0, desc="完成")
    return full_text, timestamps_json, srt_text, subtitle_html

# ==================== 视频字幕处理函数 ====================
def transcribe_video(video, asr_model_type, use_gpu, use_half, enable_vad, enable_lid, enable_punc,
                     beam_size, nbest, decode_max_len, softmax_smoothing, aed_length_penalty,
                     eos_penalty, elm_weight,
                     vad_min_speech_frame, vad_max_speech_frame, vad_min_silence_frame,
                     vad_speech_threshold, vad_smooth_window_size,
                     punc_threshold, subtitle_mode,
                     progress=gr.Progress()):
    temp_audio_path = None
    try:
        if not FIRERED_AVAILABLE:
            return "错误: FireRedASR2S 模块不可用", "", ""

        if video is None:
            return "请上传视频文件", "", ""

        progress(0, desc="初始化...")

        try:
            ensure_model_loaded(asr_model_type, use_gpu, use_half,
                                enable_vad, enable_lid, enable_punc,
                                beam_size, nbest, decode_max_len,
                                softmax_smoothing, aed_length_penalty,
                                eos_penalty, elm_weight,
                                vad_min_speech_frame, vad_max_speech_frame,
                                vad_min_silence_frame, vad_speech_threshold,
                                vad_smooth_window_size, punc_threshold,
                                progress)
        except RuntimeError as e:
            return str(e), "", ""

        progress(0.2, desc="提取视频音频...")
        temp_audio = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        temp_audio.close()
        audio_path = temp_audio.name
        temp_audio_path = audio_path

        cmd = [
            str(FFMPEG_PATH), "-i", video,
            "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
            "-y", audio_path
        ]
        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as e:
            return f"音频提取失败: {e.stderr}", "", ""

        progress(0.4, desc="识别音频...")
        result, _, error = manager.transcribe(audio_path)
        if error:
            return f"识别失败: {error}", "", ""

        progress(0.6, desc="生成字幕...")
        full_text, timestamps_json, srt_text, segments = format_result_to_outputs(result)

        base_name = video if isinstance(video, str) and os.path.exists(video) else None
        saved = save_outputs(base_name, full_text, timestamps_json, srt_text,
                             language="自动检测", model_info=asr_model_type)

        save_info = "文件已保存:\n"
        if saved.get('txt'):
            save_info += f"  {Path(saved['txt']).name}\n"
        if saved.get('json'):
            save_info += f"  {Path(saved['json']).name}\n"
        if saved.get('srt'):
            save_info += f"  {Path(saved['srt']).name}\n"

        progress(0.8, desc="嵌入字幕...")
        srt_path = saved.get('srt')
        if srt_path is None:
            result_msg = f"处理完成，但未生成字幕（可能音频无语音）。\n{save_info}"
            progress(0.9, desc="清理...")
            manager.cleanup_temp()
            combined_text = f"{result_msg}\n\n【识别文本】\n{full_text}"
            progress(1.0, desc="完成")
            return combined_text, timestamps_json, srt_text

        # ---------- 修改：使用统一命名函数生成视频文件名 ----------
        timestamp_str = time.strftime("%Y%m%d_%H%M%S")
        prefix = generate_output_filename(video, timestamp_str, subtitle_mode, default_name="video")
        out_filename = f"{prefix}.mp4"
        out_path = OUTPUT_DIR / out_filename

        srt_path_str = str(srt_path).replace('\\', '/')
        video_path_str = str(video).replace('\\', '/')
        out_path_str = str(out_path).replace('\\', '/')

        if subtitle_mode == "soft":
            cmd = [
                str(FFMPEG_PATH), "-i", video_path_str,
                "-i", srt_path_str,
                "-c", "copy", "-c:s", "mov_text",
                "-metadata:s:s:0", "language=chi",
                "-y", out_path_str
            ]
        else:
            cmd = [
                str(FFMPEG_PATH), "-i", video_path_str,
                "-vf", f"subtitles='{srt_path_str}':force_style='FontName=Microsoft YaHei,FontSize=24,PrimaryColour=&HFFFFFF,OutlineColour=&H000000,BorderStyle=3'",
                "-c:a", "copy",
                "-y", out_path_str
            ]

        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
            result_msg = f"处理完成！\n输出视频: {out_filename}\n{save_info}"
        except subprocess.CalledProcessError as e:
            result_msg = f"字幕嵌入失败: {e.stderr}\n{save_info}"

        combined_text = f"{result_msg}\n\n【识别文本】\n{full_text}"
        progress(0.9, desc="清理...")
        manager.cleanup_temp()
        progress(1.0, desc="完成")
        return combined_text, timestamps_json, srt_text
    except Exception as e:
        import traceback
        traceback.print_exc()
        error_msg = f"处理视频时发生未知错误: {str(e)}"
        return error_msg, "", ""
    finally:
        if temp_audio_path and os.path.exists(temp_audio_path):
            try:
                os.unlink(temp_audio_path)
            except:
                pass

# ==================== 批量处理函数 ====================
def transcribe_batch(files, asr_model_type, use_gpu, use_half, enable_vad, enable_lid, enable_punc,
                     beam_size, nbest, decode_max_len, softmax_smoothing, aed_length_penalty,
                     eos_penalty, elm_weight,
                     vad_min_speech_frame, vad_max_speech_frame, vad_min_silence_frame,
                     vad_speech_threshold, vad_smooth_window_size,
                     punc_threshold,
                     progress=gr.Progress()):
    if not files:
        return "请选择音频文件", "", ""

    try:
        ensure_model_loaded(asr_model_type, use_gpu, use_half,
                            enable_vad, enable_lid, enable_punc,
                            beam_size, nbest, decode_max_len,
                            softmax_smoothing, aed_length_penalty,
                            eos_penalty, elm_weight,
                            vad_min_speech_frame, vad_max_speech_frame,
                            vad_min_silence_frame, vad_speech_threshold,
                            vad_smooth_window_size, punc_threshold,
                            progress)
    except RuntimeError as e:
        return str(e), "", ""

    results_text = []
    total = len(files)
    for i, file_obj in enumerate(files, 1):
        file_path = file_obj.name if hasattr(file_obj, 'name') else str(file_obj)
        progress(i/total, desc=f"处理 {i}/{total}: {os.path.basename(file_path)}")
        result, audio_path, error = manager.transcribe(file_path)
        if error:
            results_text.append(f"【{os.path.basename(file_path)}】\n错误: {error}\n")
        else:
            full_text, timestamps_json, srt_text, _ = format_result_to_outputs(result)
            saved = save_outputs(file_path, full_text, timestamps_json, srt_text,
                                 language="自动检测", model_info=asr_model_type)
            saved_files = []
            if saved.get('txt'):
                saved_files.append(f"{Path(saved['txt']).name}")
            if saved.get('json'):
                saved_files.append(f"{Path(saved['json']).name}")
            if saved.get('srt'):
                saved_files.append(f"{Path(saved['srt']).name}")
            file_list = "\n    ".join(saved_files) if saved_files else "无文件保存"
            results_text.append(f"【{os.path.basename(file_path)}】\n已保存:\n    {file_list}\n")
    manager.cleanup_temp()
    return "\n".join(results_text), "", ""

def load_model_click(asr_model_type, use_gpu, use_half, enable_vad, enable_lid, enable_punc,
                     beam_size, nbest, decode_max_len, softmax_smoothing, aed_length_penalty,
                     eos_penalty, elm_weight,
                     vad_min_speech_frame, vad_max_speech_frame, vad_min_silence_frame,
                     vad_speech_threshold, vad_smooth_window_size,
                     punc_threshold):
    config = {
        "use_gpu": use_gpu,
        "use_half": use_half,
        "enable_vad": enable_vad,
        "enable_lid": enable_lid,
        "enable_punc": enable_punc,
        "asr_model_type": asr_model_type
    }
    advanced = {
        "beam_size": beam_size,
        "nbest": nbest,
        "decode_max_len": decode_max_len,
        "softmax_smoothing": softmax_smoothing,
        "aed_length_penalty": aed_length_penalty,
        "eos_penalty": eos_penalty,
        "elm_weight": elm_weight,
        "vad_min_speech_frame": vad_min_speech_frame,
        "vad_max_speech_frame": vad_max_speech_frame,
        "vad_min_silence_frame": vad_min_silence_frame,
        "vad_speech_threshold": vad_speech_threshold,
        "vad_smooth_window_size": vad_smooth_window_size,
        "punc_threshold": punc_threshold,
    }
    with manager.lock:
        if manager.asr_system is not None:
            manager.unload_system()
        success, msg = manager.load_system(config, advanced)
    return msg, get_system_info()

def unload_model_click():
    success, msg = manager.unload_system()
    return msg, get_system_info()

def refresh_status():
    return get_system_info()

# ==================== 强制对齐包装函数（精确空行匹配）====================
def force_align_wrapper(audio, text, asr_model_type, use_gpu, use_half,
                        enable_vad, enable_lid, enable_punc,
                        beam_size, nbest, decode_max_len,
                        softmax_smoothing, aed_length_penalty,
                        eos_penalty, elm_weight,
                        vad_min_speech_frame, vad_max_speech_frame,
                        vad_min_silence_frame, vad_speech_threshold,
                        vad_smooth_window_size, punc_threshold,
                        merge_punctuations, merge_max_words, merge_max_chars, merge_max_duration, merge_silence_threshold,
                        merge_by_punc, merge_by_silence, merge_by_wordcount, merge_by_charcount, merge_by_duration,
                        merge_by_newline,
                        progress=gr.Progress()):
    if not FIRERED_AVAILABLE:
        return "错误: FireRedASR2S 模块不可用", "", ""
    if audio is None:
        return "请上传音频文件", "", ""
    if not text.strip():
        return "请粘贴参考文本", "", ""

    progress(0, desc="初始化...")
    try:
        ensure_model_loaded(asr_model_type, use_gpu, use_half,
                            enable_vad, enable_lid, enable_punc,
                            beam_size, nbest, decode_max_len,
                            softmax_smoothing, aed_length_penalty,
                            eos_penalty, elm_weight,
                            vad_min_speech_frame, vad_max_speech_frame,
                            vad_min_silence_frame, vad_speech_threshold,
                            vad_smooth_window_size, punc_threshold,
                            progress)
    except RuntimeError as e:
        return str(e), "", ""

    progress(0.3, desc="强制对齐中...")
    word_srt, sent_srt, timestamps, words, error = manager.force_align(audio, text, progress)
    if error:
        return f"错误: {error}", "", ""

    # 空行断句（精确 token 匹配）
    force_break = None
    if merge_by_newline and words and timestamps:
        asr = manager.asr_system.asr
        paragraphs = [p.strip() for p in text.split('\n') if p.strip()]
        if len(paragraphs) > 1:
            force_break = [False] * len(words)
            current_pos = 0
            for para in paragraphs:
                para_tokens, _ = asr.tokenizer.tokenize(para)
                if len(para_tokens) == 0:
                    continue
                found = -1
                for start in range(current_pos, len(words) - len(para_tokens) + 1):
                    if words[start:start+len(para_tokens)] == para_tokens:
                        found = start
                        break
                if found >= 0:
                    end_idx = found + len(para_tokens) - 1
                    if end_idx < len(words) - 1:
                        force_break[end_idx] = True
                    current_pos = end_idx + 1
                else:
                    print(f"警告：段落 '{para[:30]}...' 无法与词序列匹配，该段落将不按空行断句")

    # 标点断句索引
    force_break_punc = None
    if merge_by_punc and words and timestamps:
        asr = manager.asr_system.asr
        punc_positions = [idx for idx, ch in enumerate(text) if ch in merge_punctuations]
        if punc_positions:
            tokens, token_ids = asr.tokenizer.tokenize(text)
            char_to_token = [-1] * len(text)
            cur = 0
            for token_idx, token in enumerate(tokens):
                token_len = len(token)
                for i in range(token_len):
                    if cur + i < len(text):
                        char_to_token[cur + i] = token_idx
                cur += token_len
            break_indices = []
            for pos in punc_positions:
                tidx = char_to_token[pos]
                if tidx >= 0 and tidx < len(words) - 1:
                    break_indices.append(tidx)
            if break_indices:
                force_break_punc = [False] * len(words)
                for idx in break_indices:
                    force_break_punc[idx] = True

    final_force_break = None
    if force_break is not None or force_break_punc is not None:
        final_force_break = [False] * len(words)
        if force_break:
            for i, v in enumerate(force_break):
                if v:
                    final_force_break[i] = True
        if force_break_punc:
            for i, v in enumerate(force_break_punc):
                if v:
                    final_force_break[i] = True

    merged_srt = ""
    if timestamps and words:
        sentences = merge_timestamps_to_sentences(
            timestamps, words,
            sentence_endings=merge_punctuations,
            max_words=merge_max_words,
            max_chars=merge_max_chars,
            max_duration=merge_max_duration,
            silence_threshold=merge_silence_threshold,
            merge_by_punc=False,
            merge_by_silence=merge_by_silence,
            merge_by_wordcount=merge_by_wordcount,
            merge_by_charcount=merge_by_charcount,
            merge_by_duration=merge_by_duration,
            force_break_indices=final_force_break
        )
        merged_srt = sentences_to_srt(sentences)
        
        # ---------- 修改：使用统一命名函数生成合并字幕文件名 ----------
        timestamp_str = time.strftime("%Y%m%d_%H%M%S")
        prefix = generate_output_filename(audio, timestamp_str, default_name="align")
        merged_path = ALIGN_OUTPUT_DIR / f"{prefix}_merged_custom.srt"
        with open(merged_path, "w", encoding="utf-8") as f:
            f.write(merged_srt)

    progress(0.9, desc="清理...")
    manager.cleanup_temp()
    progress(1.0, desc="完成")
    return word_srt, sent_srt, merged_srt

# ==================== 创建 Gradio 界面 ====================
def create_interface():
    settings = manager.settings
    default_output_dir = settings.get("output_dir", str(DEFAULT_OUTPUT_DIR))
    global OUTPUT_DIR, ALIGN_OUTPUT_DIR
    with config_lock:
        OUTPUT_DIR = Path(default_output_dir)
        ALIGN_OUTPUT_DIR = OUTPUT_DIR / "字幕自动打轴"
        ALIGN_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    llm_dir = ROOT_DIR / "pretrained_models" / "FireRedASR2-LLM"
    model_choices = ["aed"]
    if llm_dir.exists():
        model_choices.append("llm")

    with gr.Blocks(title="FireRedASR2S WebUI 专业版", theme=gr.themes.Default()) as demo:
        gr.Markdown(f"""
        # FireRedASR2S 语音识别系统 专业版
        **支持 VAD、LID、标点恢复、时间戳、SRT字幕生成，以及字幕预览与同步高亮**
        输出目录: `{OUTPUT_DIR}`
        字幕自动打轴输出: `{ALIGN_OUTPUT_DIR}`
        """)

        # 系统状态折叠面板
        with gr.Accordion("系统状态信息 (点击展开/折叠)", open=False):
            with gr.Row():
                status_display = gr.Textbox(label="系统状态", value=get_system_info(), lines=6, interactive=False, scale=4)
                with gr.Column(scale=1):
                    refresh_btn = gr.Button("刷新状态", variant="secondary")
                    health_btn = gr.Button("健康检查", variant="secondary")

        def health_check():
            info = get_system_info()
            with manager.lock:
                if manager.asr_system is None:
                    info += "\n\n⚠️ 系统未加载，请先加载模型。"
                else:
                    info += "\n\n✅ 系统已就绪。"
            return info
        health_btn.click(health_check, outputs=[status_display])

        # 模型配置区域
        with gr.Row():
            with gr.Column(scale=1):
                asr_model_type = gr.Dropdown(
                    label="ASR 模型类型",
                    choices=model_choices,
                    value="aed",
                    info="aed: 平衡性能与效率；llm: 追求极致准确率，硬件配置要求很高"
                )
            with gr.Column(scale=1):
                use_gpu = gr.Checkbox(label="使用 GPU (如果可用)", value=torch.cuda.is_available())
                default_half = torch.cuda.is_available() and torch.cuda.get_device_properties(0).total_memory < 10e9
                use_half = gr.Checkbox(
                    label="使用半精度 (FP16)",
                    value=default_half,
                    info="开启后显存占用减半，速度更快，适合8GB左右显存的显卡"
                )

        with gr.Row():
            with gr.Column(scale=1):
                with gr.Row():
                    load_btn = gr.Button("加载模型", variant="primary")
                    unload_btn = gr.Button("卸载模型", variant="stop")
            with gr.Column(scale=1):
                with gr.Row():
                    enable_vad = gr.Checkbox(label="启用 VAD", value=True)
                    enable_lid = gr.Checkbox(label="启用 LID", value=True)
                    enable_punc = gr.Checkbox(label="启用 标点恢复", value=True)

        # 高级参数折叠面板
        with gr.Accordion("高级参数 (点击展开/折叠，非专业人士请保持默认)", open=False):
            gr.Markdown("### 解码参数 (ASR)")
            with gr.Row():
                with gr.Column():
                    beam_size = gr.Slider(label="Beam 大小", minimum=1, maximum=10, value=3, step=1)
                    nbest = gr.Slider(label="候选结果数", minimum=1, maximum=5, value=1, step=1)
                    decode_max_len = gr.Slider(label="最大解码长度", minimum=0, maximum=500, value=0, step=10)
                with gr.Column():
                    softmax_smoothing = gr.Slider(label="Softmax 平滑", minimum=0.5, maximum=2.0, value=1.25, step=0.05)
                    aed_length_penalty = gr.Slider(label="长度惩罚", minimum=-2.0, maximum=2.0, value=0.6, step=0.1)
                    eos_penalty = gr.Slider(label="结束符惩罚", minimum=0.5, maximum=2.0, value=1.0, step=0.1)
            with gr.Row():
                elm_weight = gr.Slider(label="外部语言模型权重", minimum=0.0, maximum=1.0, value=0.0, step=0.05)

            gr.Markdown("### VAD 参数")
            with gr.Row():
                with gr.Column():
                    vad_speech_threshold = gr.Slider(label="语音阈值", minimum=0.1, maximum=0.9, value=0.4, step=0.05)
                    vad_min_speech_frame = gr.Slider(label="最小语音帧数", minimum=1, maximum=50, value=20, step=1)
                with gr.Column():
                    vad_max_speech_frame = gr.Slider(label="最大语音帧数", minimum=100, maximum=3000, value=2000, step=50)
                    vad_min_silence_frame = gr.Slider(label="最小静音帧数", minimum=5, maximum=50, value=20, step=1)
            with gr.Row():
                vad_smooth_window_size = gr.Slider(label="平滑窗口大小", minimum=1, maximum=20, value=5, step=1)

            gr.Markdown("### Punc 参数")
            punc_threshold = gr.Slider(label="标点阈值", minimum=0.1, maximum=0.9, value=0.45, step=0.05, visible=True)

        # 基础按钮绑定
        load_btn.click(
            load_model_click,
            inputs=[asr_model_type, use_gpu, use_half, enable_vad, enable_lid, enable_punc,
                    beam_size, nbest, decode_max_len, softmax_smoothing, aed_length_penalty,
                    eos_penalty, elm_weight,
                    vad_min_speech_frame, vad_max_speech_frame, vad_min_silence_frame,
                    vad_speech_threshold, vad_smooth_window_size,
                    punc_threshold],
            outputs=[status_display, status_display]
        )
        unload_btn.click(unload_model_click, outputs=[status_display, status_display])
        refresh_btn.click(refresh_status, outputs=[status_display])

        gr.Markdown("---")

        # ========== 主标签页 ==========
        with gr.Tabs():
            # ---------- 音频识别 ----------
            with gr.Tab("音频识别"):
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### 上传音频")
                        audio_input = gr.Audio(
                            label="选择或录制音频",
                            type="numpy",
                            sources=["upload", "microphone"]
                        )
                        with gr.Row():
                            transcribe_btn = gr.Button("开始识别", variant="primary")
                            clear_btn = gr.Button("清空", variant="secondary")
                    with gr.Column(scale=2):
                        with gr.Tabs():
                            with gr.Tab("识别文本"):
                                text_output = gr.Textbox(label="结果", lines=15, show_copy_button=True)
                            with gr.Tab("时间戳 (JSON)"):
                                json_output = gr.Textbox(label="时间戳数据", lines=15, show_copy_button=True)
                            with gr.Tab("SRT字幕"):
                                srt_output = gr.Textbox(label="SRT字幕", lines=15, show_copy_button=True)
                            with gr.Tab("字幕预览"):
                                preview_output = gr.HTML(label="字幕预览", value=generate_subtitle_html([], None))

                transcribe_btn.click(
                    transcribe_audio,
                    inputs=[audio_input, asr_model_type, use_gpu, use_half, enable_vad, enable_lid, enable_punc,
                            beam_size, nbest, decode_max_len, softmax_smoothing, aed_length_penalty,
                            eos_penalty, elm_weight,
                            vad_min_speech_frame, vad_max_speech_frame, vad_min_silence_frame,
                            vad_speech_threshold, vad_smooth_window_size,
                            punc_threshold],
                    outputs=[text_output, json_output, srt_output, preview_output]
                ).then(refresh_status, outputs=[status_display])

                clear_btn.click(
                    lambda: [None, "", "", "", generate_subtitle_html([], None)],
                    outputs=[audio_input, text_output, json_output, srt_output, preview_output]
                )

            # ---------- 视频字幕 ----------
            with gr.Tab("视频字幕"):
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### 上传视频")
                        video_input = gr.Video(
                            label="选择视频文件",
                            sources=["upload"],
                            interactive=True
                        )
                        subtitle_mode = gr.Radio(
                            label="字幕嵌入模式",
                            choices=["soft", "hard"],
                            value="soft",
                            info="soft: 外挂字幕（可开关）| hard: 永久烧录到画面"
                        )
                        with gr.Row():
                            video_transcribe_btn = gr.Button("开始处理", variant="primary")
                            video_clear_btn = gr.Button("清空", variant="secondary")
                    with gr.Column(scale=2):
                        with gr.Tabs():
                            with gr.Tab("识别文本"):
                                video_text_output = gr.Textbox(label="结果", lines=15, show_copy_button=True)
                            with gr.Tab("时间戳 (JSON)"):
                                video_json_output = gr.Textbox(label="时间戳数据", lines=15, show_copy_button=True)
                            with gr.Tab("SRT字幕"):
                                video_srt_output = gr.Textbox(label="SRT字幕", lines=15, show_copy_button=True)

                video_transcribe_btn.click(
                    transcribe_video,
                    inputs=[video_input, asr_model_type, use_gpu, use_half, enable_vad, enable_lid, enable_punc,
                            beam_size, nbest, decode_max_len, softmax_smoothing, aed_length_penalty,
                            eos_penalty, elm_weight,
                            vad_min_speech_frame, vad_max_speech_frame, vad_min_silence_frame,
                            vad_speech_threshold, vad_smooth_window_size,
                            punc_threshold, subtitle_mode],
                    outputs=[video_text_output, video_json_output, video_srt_output]
                ).then(refresh_status, outputs=[status_display])

                video_clear_btn.click(
                    lambda: [None, "", "", ""],
                    outputs=[video_input, video_text_output, video_json_output, video_srt_output]
                )

            # ---------- 批量处理 ----------
            with gr.Tab("批量处理"):
                with gr.Row():
                    with gr.Column(scale=1):
                        file_input = gr.Files(
                            label="上传多个音频文件",
                            file_types=[".wav", ".mp3", ".m4a", ".flac", ".ogg"],
                            file_count="multiple"
                        )
                        batch_transcribe_btn = gr.Button("批量识别", variant="primary")
                        batch_clear = gr.Button("清空", variant="secondary")
                    with gr.Column(scale=2):
                        batch_output = gr.Textbox(label="批量结果", lines=20, show_copy_button=True)

                batch_transcribe_btn.click(
                    transcribe_batch,
                    inputs=[file_input, asr_model_type, use_gpu, use_half, enable_vad, enable_lid, enable_punc,
                            beam_size, nbest, decode_max_len, softmax_smoothing, aed_length_penalty,
                            eos_penalty, elm_weight,
                            vad_min_speech_frame, vad_max_speech_frame, vad_min_silence_frame,
                            vad_speech_threshold, vad_smooth_window_size,
                            punc_threshold],
                    outputs=[batch_output, gr.State(), gr.State()]
                ).then(refresh_status, outputs=[status_display])

                batch_clear.click(
                    lambda: [None, ""],
                    outputs=[file_input, batch_output]
                )

            # ---------- 强制对齐 ----------
            with gr.Tab("字幕自动打轴（文稿生字幕）"):
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### 上传配音音频")
                        align_audio = gr.Audio(
                            label="选择音频文件",
                            type="filepath",
                            sources=["upload"]
                        )
                        align_text = gr.Textbox(
                            label="粘贴稿子文本",
                            lines=8,
                            placeholder="将稿子文本粘贴到这里，确保与音频内容一致...\n用空行分隔段落可实现强制分段。"
                        )
                        
                        # 合并参数区域（可折叠）
                        with gr.Accordion("字幕合并参数", open=True):
                            merge_punctuations = gr.Textbox(
                                label="句末标点符号", value="。！？.!?",
                                info="遇到这些符号时强制断句"
                            )
                            with gr.Row():
                                merge_max_words = gr.Slider(
                                    label="最大词数", minimum=5, maximum=50, value=20, step=1,
                                    info="单条字幕最多包含多少个词"
                                )
                                merge_max_chars = gr.Slider(
                                    label="最大字符数", minimum=5, maximum=100, value=30, step=5,
                                    info="单条字幕最多包含多少个字符（中文按字数）"
                                )
                            with gr.Row():
                                merge_max_duration = gr.Slider(
                                    label="最大时长 (秒)", minimum=1.0, maximum=20.0, value=10.0, step=0.5,
                                    info="单条字幕最大时长"
                                )
                                merge_silence_threshold = gr.Slider(
                                    label="静音阈值 (秒)", minimum=0.1, maximum=1.0, value=0.3, step=0.05,
                                    info="词间静音超过此值则断句"
                                )
                            with gr.Row():
                                merge_by_punc = gr.Checkbox(label="根据标点断句", value=True)
                                merge_by_silence = gr.Checkbox(label="根据静音断句", value=True)
                                merge_by_wordcount = gr.Checkbox(label="根据词数断句", value=True)
                                merge_by_charcount = gr.Checkbox(label="根据字符数断句", value=True)
                                merge_by_duration = gr.Checkbox(label="根据时长断句", value=True)
                                merge_by_newline = gr.Checkbox(
                                    label="根据空行断句", value=False,
                                    info="按文本中的空行强制分段（精确 token 匹配）"
                                )
                        
                        with gr.Row():
                            align_btn = gr.Button("生成精准字幕", variant="primary")
                            align_clear = gr.Button("清空", variant="secondary")
                    
                    with gr.Column(scale=2):
                        with gr.Tabs():
                            with gr.Tab("逐词 SRT"):
                                align_word_output = gr.Textbox(label="逐词字幕", lines=42, show_copy_button=True)
                            with gr.Tab("整句 SRT"):
                                align_sent_output = gr.Textbox(label="整句子幕", lines=42, show_copy_button=True)
                            with gr.Tab("合并字幕（自定义）"):
                                align_merged_output = gr.Textbox(label="合并后的字幕", lines=42, show_copy_button=True)

                align_btn.click(
                    force_align_wrapper,
                    inputs=[align_audio, align_text,
                            asr_model_type, use_gpu, use_half,
                            enable_vad, enable_lid, enable_punc,
                            beam_size, nbest, decode_max_len,
                            softmax_smoothing, aed_length_penalty,
                            eos_penalty, elm_weight,
                            vad_min_speech_frame, vad_max_speech_frame,
                            vad_min_silence_frame, vad_speech_threshold,
                            vad_smooth_window_size, punc_threshold,
                            merge_punctuations, merge_max_words, merge_max_chars, merge_max_duration, merge_silence_threshold,
                            merge_by_punc, merge_by_silence, merge_by_wordcount, merge_by_charcount, merge_by_duration,
                            merge_by_newline],
                    outputs=[align_word_output, align_sent_output, align_merged_output]
                ).then(refresh_status, outputs=[status_display])

                align_clear.click(
                    lambda: [None, "", "", "", ""],
                    outputs=[align_audio, align_text, align_word_output, align_sent_output, align_merged_output]
                )

            # ---------- 系统信息 ----------
            with gr.Tab("系统信息"):
                with gr.Column():
                    system_info_text = gr.Textbox(label="详细信息", value=get_system_info(), lines=20, show_copy_button=True)
                    with gr.Row():
                        output_dir_input = gr.Textbox(label="输出目录", value=str(OUTPUT_DIR), interactive=True, scale=3)
                        update_output_btn = gr.Button("更新输出目录", variant="secondary", scale=1)
                    with gr.Row():
                        # 字幕预览大小阈值滑块
                        preview_max_size = gr.Slider(
                            label="字幕预览最大文件大小 (MB)", minimum=1, maximum=100, value=manager.settings.get("preview_max_size_mb", 5), step=1,
                            info="超过此大小的音频将不会在预览中加载，避免浏览器卡顿"
                        )
                    with gr.Row():
                        open_output_btn = gr.Button("打开输出目录")
                        open_log_btn = gr.Button("打开日志文件夹")
                        clear_cache_btn = gr.Button("清理临时文件")
                    # 配置保存/加载
                    with gr.Row():
                        save_config_btn = gr.Button("保存当前配置", variant="primary")
                        preset_files = sorted([f.name for f in PRESET_DIR.glob("preset_*.json")], reverse=True)
                        preset_selector = gr.Dropdown(label="选择预设文件", choices=preset_files, value=None, interactive=True)
                        load_config_btn = gr.Button("加载所选配置", variant="secondary")
                        refresh_preset_btn = gr.Button("刷新列表", variant="secondary", size="sm")
                    config_status = gr.Textbox(label="配置状态", interactive=False)

                # 更新输出目录
                def update_output_dir(new_dir, new_preview_size):
                    global OUTPUT_DIR, ALIGN_OUTPUT_DIR
                    try:
                        p = Path(new_dir)
                        p.mkdir(parents=True, exist_ok=True)
                        with config_lock:
                            OUTPUT_DIR = p
                            ALIGN_OUTPUT_DIR = p / "字幕自动打轴"
                            ALIGN_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
                        manager.settings["output_dir"] = str(p)
                        manager.settings["preview_max_size_mb"] = new_preview_size
                        save_settings(manager.settings)
                        return f"输出目录已更新为 {p}，预览阈值已设为 {new_preview_size} MB", get_system_info()
                    except Exception as e:
                        return f"更新失败: {e}", get_system_info()
                update_output_btn.click(
                    update_output_dir,
                    inputs=[output_dir_input, preview_max_size],
                    outputs=[config_status, system_info_text]
                )

                def open_output():
                    os.startfile(str(OUTPUT_DIR))
                    return "已打开输出目录"
                open_output_btn.click(open_output, outputs=[config_status])

                def open_log():
                    os.startfile(str(LOG_DIR))
                    return "已打开日志文件夹"
                open_log_btn.click(open_log, outputs=[config_status])

                def clear_cache():
                    cleaned = manager.cleanup_temp()
                    return f"清理了 {cleaned} 个临时文件"
                clear_cache_btn.click(clear_cache, outputs=[config_status])

                # 保存当前配置
                def save_current_config():
                    config = {
                        "asr_model_type": asr_model_type.value,
                        "use_gpu": use_gpu.value,
                        "use_half": use_half.value,
                        "enable_vad": enable_vad.value,
                        "enable_lid": enable_lid.value,
                        "enable_punc": enable_punc.value,
                        "beam_size": beam_size.value,
                        "nbest": nbest.value,
                        "decode_max_len": decode_max_len.value,
                        "softmax_smoothing": softmax_smoothing.value,
                        "aed_length_penalty": aed_length_penalty.value,
                        "eos_penalty": eos_penalty.value,
                        "elm_weight": elm_weight.value,
                        "vad_min_speech_frame": vad_min_speech_frame.value,
                        "vad_max_speech_frame": vad_max_speech_frame.value,
                        "vad_min_silence_frame": vad_min_silence_frame.value,
                        "vad_speech_threshold": vad_speech_threshold.value,
                        "vad_smooth_window_size": vad_smooth_window_size.value,
                        "punc_threshold": punc_threshold.value,
                        # 合并参数
                        "merge_punctuations": merge_punctuations.value,
                        "merge_max_words": merge_max_words.value,
                        "merge_max_chars": merge_max_chars.value,
                        "merge_max_duration": merge_max_duration.value,
                        "merge_silence_threshold": merge_silence_threshold.value,
                        "merge_by_punc": merge_by_punc.value,
                        "merge_by_silence": merge_by_silence.value,
                        "merge_by_wordcount": merge_by_wordcount.value,
                        "merge_by_charcount": merge_by_charcount.value,
                        "merge_by_duration": merge_by_duration.value,
                        "merge_by_newline": merge_by_newline.value,
                        "preview_max_size_mb": preview_max_size.value,
                    }
                    timestamp = time.strftime("%Y%m%d_%H%M%S")
                    preset_path = PRESET_DIR / f"preset_{timestamp}.json"
                    with open(preset_path, "w", encoding="utf-8") as f:
                        json.dump(config, f, ensure_ascii=False, indent=2)
                    # 更新下拉列表
                    new_choices = sorted([f.name for f in PRESET_DIR.glob("preset_*.json")], reverse=True)
                    return f"配置已保存到 {preset_path}", gr.update(choices=new_choices)
                save_config_btn.click(
                    save_current_config,
                    outputs=[config_status, preset_selector]
                )

                # 刷新预设列表
                def refresh_preset_list():
                    new_choices = sorted([f.name for f in PRESET_DIR.glob("preset_*.json")], reverse=True)
                    return gr.update(choices=new_choices)
                refresh_preset_btn.click(refresh_preset_list, outputs=[preset_selector])

                # 加载所选配置
                def load_selected_config(filename):
                    if not filename:
                        return ["请先选择一个预设文件"] + [gr.update() for _ in range(30)]  # 30个更新
                    file_path = PRESET_DIR / filename
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            cfg = json.load(f)
                    except Exception as e:
                        return [f"加载失败: {e}"] + [gr.update() for _ in range(30)]

                    updates = [
                        gr.update(value=cfg.get("asr_model_type", "aed")),
                        gr.update(value=cfg.get("use_gpu", True)),
                        gr.update(value=cfg.get("use_half", False)),
                        gr.update(value=cfg.get("enable_vad", True)),
                        gr.update(value=cfg.get("enable_lid", True)),
                        gr.update(value=cfg.get("enable_punc", True)),
                        gr.update(value=cfg.get("beam_size", 3)),
                        gr.update(value=cfg.get("nbest", 1)),
                        gr.update(value=cfg.get("decode_max_len", 0)),
                        gr.update(value=cfg.get("softmax_smoothing", 1.25)),
                        gr.update(value=cfg.get("aed_length_penalty", 0.6)),
                        gr.update(value=cfg.get("eos_penalty", 1.0)),
                        gr.update(value=cfg.get("elm_weight", 0.0)),
                        gr.update(value=cfg.get("vad_min_speech_frame", 20)),
                        gr.update(value=cfg.get("vad_max_speech_frame", 2000)),
                        gr.update(value=cfg.get("vad_min_silence_frame", 20)),
                        gr.update(value=cfg.get("vad_speech_threshold", 0.4)),
                        gr.update(value=cfg.get("vad_smooth_window_size", 5)),
                        gr.update(value=cfg.get("punc_threshold", 0.45)),
                        gr.update(value=cfg.get("merge_punctuations", "。！？.!?")),
                        gr.update(value=cfg.get("merge_max_words", 20)),
                        gr.update(value=cfg.get("merge_max_chars", 30)),
                        gr.update(value=cfg.get("merge_max_duration", 10.0)),
                        gr.update(value=cfg.get("merge_silence_threshold", 0.3)),
                        gr.update(value=cfg.get("merge_by_punc", True)),
                        gr.update(value=cfg.get("merge_by_silence", True)),
                        gr.update(value=cfg.get("merge_by_wordcount", True)),
                        gr.update(value=cfg.get("merge_by_charcount", True)),
                        gr.update(value=cfg.get("merge_by_duration", True)),
                        gr.update(value=cfg.get("merge_by_newline", False)),
                        gr.update(value=cfg.get("preview_max_size_mb", 5)),
                    ]
                    # 返回列表：第一个是状态信息，后面是30个更新
                    return [f"配置已加载: {filename}"] + updates

                load_config_btn.click(
                    load_selected_config,
                    inputs=[preset_selector],
                    outputs=[config_status,
                             asr_model_type, use_gpu, use_half, enable_vad, enable_lid, enable_punc,
                             beam_size, nbest, decode_max_len, softmax_smoothing, aed_length_penalty, eos_penalty, elm_weight,
                             vad_min_speech_frame, vad_max_speech_frame, vad_min_silence_frame, vad_speech_threshold, vad_smooth_window_size, punc_threshold,
                             merge_punctuations, merge_max_words, merge_max_chars, merge_max_duration, merge_silence_threshold,
                             merge_by_punc, merge_by_silence, merge_by_wordcount, merge_by_charcount, merge_by_duration, merge_by_newline,
                             preview_max_size]
                )

        # 页脚版权
        gr.Markdown("---")
        gr.Markdown(f"""
        <div style="text-align: center; color: #666; font-size: 0.9em;">
        <p>本软件包不提供任何模型文件，模型由用户自行从官方渠道获取。用户需自行遵守模型的原许可证。</p>
        <p>本软件包按“原样”提供，不提供任何明示或暗示的担保。使用本软件所产生的一切风险由用户自行承担。</p>
        <p>本软件包开发者不对因使用本软件而导致的任何直接或间接损失负责。</p>       
        <p><strong>更新请关注B站up主：光影的故事2018</strong></p>
        <p>🔗 <strong>B站主页</strong>: <a href="https://space.bilibili.com/381518712" target="_blank">space.bilibili.com/381518712</a></p>
        </div>
        """)
        gr.Markdown("""
        <div style="text-align: center; color: #666; margin-top: 10px; font-size: 0.9em;">
        © 原创 WebUI 代码 © 2026 光影紐扣 版权所有 | 基于 FireRedASR2S (Apache 2.0) 二次开发
        </div>
        """)

        demo.load(refresh_status, outputs=[status_display])

    return demo

# ==================== 退出清理 ====================
@atexit.register
def cleanup():
    print("正在退出，清理资源...")
    manager.unload_system()
    manager.cleanup_temp()
    clean_old_logs()
    print("清理完成")

# ==================== 主函数 ====================
def main():
    if not FIRERED_AVAILABLE:
        print("错误: FireRedASR2S 模块不可用，请检查环境。")
        return

    model_root = ROOT_DIR / "pretrained_models"
    if not model_root.exists():
        print(f"警告: 模型目录 {model_root} 不存在，请确保模型已下载。")
        print("请创建符号链接或将模型文件夹放在正确位置。")

    demo = create_interface()
    demo.queue().launch(
        server_name="127.0.0.1",
        server_port=18006,
        inbrowser=True,
        show_error=True
    )

if __name__ == "__main__":
    main()