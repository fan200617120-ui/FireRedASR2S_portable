#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
FireRedASR2S WebUI Professional Edition (增强融合版)
- 基于 firered_webui_pro - 副本.py 的完整 UI 与功能
- 融入修复版 (firered_webui_pro.py) 的核心逻辑改进
- 新增：音频预处理开关（FFmpeg 16k mono 可选）
- 新增：视频字幕合并参数面板
- 优化：英文空格连接、强制对齐 token ID 匹配、去除调试打印
- 扩展文件大小上限至 500MB
Copyright 2026 光影的故事2018
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
LOG_DIR = Path(__file__).parent.parent / "logs"          # 保持副本的日志位置
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

# ==================== 路径设置（融合版：智能根目录）====================
CURRENT_DIR = Path(__file__).parent.absolute()
# 如果当前目录的父目录包含 pretrained_models 或 preset，说明是标准项目结构，否则可能脚本在项目根目录
if (CURRENT_DIR.parent / "pretrained_models").exists() or (CURRENT_DIR.parent / "preset").exists():
    PROJECT_ROOT = CURRENT_DIR.parent
else:
    PROJECT_ROOT = CURRENT_DIR
sys.path.insert(0, str(PROJECT_ROOT / "FireRedASR2S"))

# ==================== 导入检查 ====================
FIRERED_AVAILABLE = False
try:
    from fireredasr2s import FireRedAsr2System, FireRedAsr2SystemConfig
    from fireredasr2s.fireredasr2 import FireRedAsr2Config
    from fireredasr2s.fireredvad import FireRedVadConfig
    from fireredasr2s.fireredlid import FireRedLidConfig
    from fireredasr2s.fireredpunc import FireRedPuncConfig
    FIRERED_AVAILABLE = True
    print("FireRedASR2S 模块导入成功")
except ImportError as e:
    print(f"导入 FireRedASR2S 失败: {e}")

# ==================== 基础路径 ====================
BASE_DIR = Path(__file__).parent.absolute()
ROOT_DIR = PROJECT_ROOT
DEFAULT_OUTPUT_DIR = ROOT_DIR / "output"
OUTPUT_DIR = DEFAULT_OUTPUT_DIR
ALIGN_OUTPUT_DIR = OUTPUT_DIR / "字幕自动打轴"
ALIGN_OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

# 配置文件目录
PRESET_DIR = ROOT_DIR / "preset"
PRESET_DIR.mkdir(exist_ok=True)
CONFIG_FILE = PRESET_DIR / "settings.json"

# 线程锁
config_lock = threading.RLock()

# ==================== 自动配置 FFmpeg ====================
PORTABLE_FFMPEG_DIR = ROOT_DIR / "ffmpeg" / "bin"
if sys.platform == "win32":
    PORTABLE_FFMPEG_EXE = PORTABLE_FFMPEG_DIR / "ffmpeg.exe"
else:
    PORTABLE_FFMPEG_EXE = PORTABLE_FFMPEG_DIR / "ffmpeg"
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
        print("⚠️ 警告：未找到内置 FFmpeg，视频处理可能失败，请将 ffmpeg 放入 ffmpeg/bin 目录。")

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

    def _find_model_dir(self, model_type="AED"):
        """自动查找模型目录"""
        candidates = [
            ROOT_DIR / "pretrained_models" / f"FireRedASR2-{model_type}",
            ROOT_DIR / "pretrained_models" / f"FireRedASR2-{model_type}-2025",
            ROOT_DIR / "pretrained_models" / "FireRedASR2-AED",
            ROOT_DIR / "pretrained_models" / "FireRedASR2-AED-2025",
        ]
        for p in candidates:
            if p.exists():
                print(f"✅ 自动检测到模型目录: {p}")
                return str(p)
        return None

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

                # 自动查找模型目录
                model_dir_override = config_dict.get("model_dir", None) if config_dict else None
                if model_dir_override and Path(model_dir_override).exists():
                    model_dir = model_dir_override
                else:
                    model_dir = self._find_model_dir(default_config['asr_model_type'].upper())
                if not model_dir or not Path(model_dir).exists():
                    return False, f"模型目录不存在，请将模型放置在 pretrained_models/FireRedASR2-AED 等目录"

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
                    if "punc_threshold" in advanced_params:
                        try:
                            punc_config.threshold = advanced_params["punc_threshold"]
                        except:
                            pass

                vad_model_dir = str(ROOT_DIR / "pretrained_models" / "FireRedVAD" / "vad")
                if not os.path.exists(vad_model_dir):
                    alt_vad_dir = str(ROOT_DIR / "pretrained_models" / "FireRedVAD" / "VAD")
                    if os.path.exists(alt_vad_dir):
                        vad_model_dir = alt_vad_dir

                system_config = FireRedAsr2SystemConfig(
                    vad_model_dir=vad_model_dir,
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
            if self.asr_system is not None:
                del self.asr_system
            self.asr_system = None
            self.config = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
            return True, "系统已卸载"

    def transcribe(self, audio_input, force_preprocess=True):
        """识别音频，force_preprocess 控制是否强制 16k mono"""
        if self.asr_system is None:
            return None, None, "系统未加载，请先加载模型"
        audio_path = self._prepare_audio(audio_input, force_preprocess=force_preprocess)
        if audio_path is None:
            return None, None, "音频处理失败"
        try:
            result = self.asr_system.process(audio_path)
            return result, audio_path, None
        except Exception as e:
            logging.error(traceback.format_exc())
            if audio_path in self.temp_files:
                self.temp_files.remove(audio_path)
                try:
                    os.unlink(audio_path)
                except:
                    pass
            return None, None, f"识别失败: {str(e)}"

    # 修复 Bug 4：异常时根据 return_waveform 返回正确数量的 None
    def _prepare_audio(self, audio_input, force_preprocess=True, return_waveform=False):
        """使用 FFmpeg 预处理为 16k 单声道 wav，可选关闭强制预处理"""
        try:
            if isinstance(audio_input, str) and os.path.exists(audio_input):
                input_path = audio_input
            elif isinstance(audio_input, tuple):
                sr, data = audio_input
                if data.ndim > 1:
                    data = np.mean(data, axis=1)
                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                    input_path = tmp.name
                sf.write(input_path, data.astype(np.float32), sr)
                self.temp_files.append(input_path)
            else:
                if return_waveform:
                    return None, None, None
                return None

            if not force_preprocess:
                if return_waveform:
                    data, sr = sf.read(input_path, dtype='float32')
                    return input_path, data, sr
                return input_path

            # 用 FFmpeg 生成标准化的临时文件
            with tempfile.NamedTemporaryFile(delete=False, suffix="_16k_mono.wav") as tmp_out:
                out_path = tmp_out.name
            cmd = [
                FFMPEG_PATH, "-y", "-i", input_path,
                "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", out_path
            ]
            subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            self.temp_files.append(out_path)

            if return_waveform:
                data, sr = sf.read(out_path, dtype='float32')
                return out_path, data, sr
            else:
                return out_path
        except Exception as e:
            logging.error(f"音频预处理失败: {e}")
            if return_waveform:
                return None, None, None
            return None

    def cleanup_temp(self):
        cleaned = 0
        for f in self.temp_files[:]:
            try:
                if os.path.exists(f):
                    os.unlink(f)
                    cleaned += 1
            except:
                pass
        self.temp_files = []
        return cleaned

    # ==================== 强制对齐方法（融合改进） ====================
    def _seconds_to_srt_time(self, seconds):
        seconds = max(0.0, float(seconds))
        td = timedelta(seconds=seconds)
        total_seconds = int(td.total_seconds())
        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60
        secs = total_seconds % 60
        ms = int((td.total_seconds() - total_seconds) * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{ms:03d}"

    def force_align(self, audio_input, reference_text, progress=None, force_preprocess=True):
        if self.asr_system is None:
            return None, None, None, None, None, "模型未加载，请先加载 AED 模型"
        if self.config['asr_model_type'] != 'aed':
            return None, None, None, None, None, "强制对齐仅支持 AED 模型，当前为 " + self.config['asr_model_type']

        audio_path, waveform, sr = self._prepare_audio(audio_input, force_preprocess=force_preprocess, return_waveform=True)
        if audio_path is None or waveform is None:
            return None, None, None, None, None, "音频处理失败"

        try:
            duration = len(waveform) / 16000

            asr = self.asr_system.asr

            feats, lengths, _, _, _ = asr.feat_extractor([(16000, waveform)], ["tmp"])
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
                if T == 0:
                    return None, None, None, None, None, "音频过短，无法对齐（编码器输出长度为0）"
                frame_shift = duration / T

            tokens, token_ids = asr.tokenizer.tokenize(reference_text)
            if len(token_ids) == 0:
                return None, None, None, None, None, "参考文本分词后为空"

            yseq = torch.tensor(token_ids, device=enc_outputs.device)
            hyps = [[{"yseq": yseq}]]

            nbest_hyps = asr.model.get_token_timestamp_torchaudio(enc_outputs, enc_lengths, hyps)
            timestamp = nbest_hyps[0][0].get("timestamp")
            if timestamp is None:
                return None, None, None, None, None, "模型未返回时间戳"

            starts, ends = timestamp
            if len(starts) == 0:
                return None, None, None, None, None, "时间戳为空"

            # 时间戳单位检测（无调试打印）
            max_start = max(starts)
            min_start = min(starts)
            if max_start <= duration * 1.5 and max_start > 0.01 * duration:
                timestamps_sec = list(zip(starts, ends))
            else:
                hypothetical_max_sec = max_start * frame_shift
                if abs(hypothetical_max_sec - duration) < 0.2 * duration:
                    timestamps_sec = [(s * frame_shift, e * frame_shift) for s, e in zip(starts, ends)]
                elif max_start > duration * 1000:
                    scale = duration / max_start * 0.99
                    timestamps_sec = [(s * scale, e * scale) for s, e in zip(starts, ends)]
                else:
                    timestamps_sec = [(s * frame_shift, e * frame_shift) for s, e in zip(starts, ends)]

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

            # 显存清理
            del feats, enc_outputs, enc_lengths, yseq
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            return word_srt_str, sentence_srt, timestamps_sec, token_texts, token_ids, None

        except Exception as e:
            logging.error(traceback.format_exc())
            return None, None, None, None, None, f"强制对齐失败: {str(e)}"
        finally:
            if audio_path in self.temp_files:
                self.temp_files.remove(audio_path)
                try:
                    os.unlink(audio_path)
                except:
                    pass

manager = FireRedASR2SManager()

# ==================== 工具函数 ====================
def seconds_to_srt_time(seconds):
    seconds = max(0.0, float(seconds))
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
    if sentences and sentences[0].get("asr_confidence") is not None:
        extra += f" | 置信度: {sentences[0]['asr_confidence']:.3f}"
    full_text = f"{text}\n\n[元数据] {extra}"

    return full_text, timestamps_json, srt_text, segments

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

# ==================== 字幕合并函数（增强版，加入英文空格检测） ====================
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

    # 检测是否含有中日韩文字，无则英文空格连接
    has_cjk = any(re.search(r'[\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af]', w) for w in words)
    join_str = "" if has_cjk else " "

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
                new_text = join_str.join(current_words + [word])
                if len(new_text) >= max_chars:
                    should_break = True
            if not should_break and merge_by_duration and current_words:
                current_duration = last_end - current_start
                if current_duration + (end - start) >= max_duration:
                    should_break = True

        if not current_words:
            current_start = start
        current_words.append(word)
        last_end = end

        if should_break:
            sentences.append({
                "start": current_start,
                "end": last_end,
                "text": join_str.join(current_words).strip()
            })
            current_start = None
            current_words = []

    if current_words:
        sentences.append({
            "start": current_start,
            "end": last_end,
            "text": join_str.join(current_words).strip()
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
    original_name = None
    if isinstance(base_input, str) and os.path.exists(base_input):
        original_name = Path(base_input).stem
    elif isinstance(base_input, dict) and base_input.get('path') and os.path.exists(base_input['path']):
        original_name = Path(base_input['path']).stem
    elif isinstance(base_input, tuple):
        original_name = default_name

    if not original_name:
        original_name = default_name

    safe_name = re.sub(r'[^\w\u4e00-\u9fff\-]', '', original_name)
    if not safe_name:
        safe_name = default_name

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
        need_reload = manager.asr_system is None
        if not need_reload:
            for k, v in config.items():
                if manager.config.get(k) != v:
                    need_reload = True
                    break
        if need_reload:
            if manager.asr_system is not None:
                manager.unload_system()
            if progress is not None and getattr(progress, 'tqdm', None) is not None:
                progress(0.1, desc="加载模型...")
            success, msg = manager.load_system(config, advanced)
            if not success:
                raise RuntimeError(f"加载失败: {msg}")
    return manager

# ==================== 音频识别函数（增加预处理开关） ====================
def transcribe_audio(audio, asr_model_type, use_gpu, use_half, enable_vad, enable_lid, enable_punc,
                     beam_size, nbest, decode_max_len, softmax_smoothing, aed_length_penalty,
                     eos_penalty, elm_weight,
                     vad_min_speech_frame, vad_max_speech_frame, vad_min_silence_frame,
                     vad_speech_threshold, vad_smooth_window_size,
                     punc_threshold,
                     merge_max_duration, merge_max_chars, merge_punctuations, merge_silence_threshold,
                     force_preprocess,
                     progress=gr.Progress()):
    if not FIRERED_AVAILABLE:
        return "错误: FireRedASR2S 模块不可用", "", ""

    if audio is None:
        return "请上传或录制音频", "", ""

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

    progress(0.3, desc="识别中...")
    result, audio_path, error = manager.transcribe(audio, force_preprocess=force_preprocess)
    if error:
        return f"错误: {error}", "", ""

    progress(0.7, desc="生成输出...")
    full_text, timestamps_json, srt_text, segments = format_result_to_outputs(result)

    # 应用合并参数
    if segments:
        ts_data = [(s["start"], s["end"]) for s in segments]
        texts = [s["text"] for s in segments]
        merged_sentences = merge_timestamps_to_sentences(
            ts_data, texts,
            sentence_endings=merge_punctuations,
            max_words=50,  # 不使用词数限制
            max_chars=merge_max_chars,
            max_duration=merge_max_duration,
            silence_threshold=merge_silence_threshold,
            merge_by_punc=True,
            merge_by_silence=True,
            merge_by_wordcount=False,
            merge_by_charcount=True,
            merge_by_duration=True,
            force_break_indices=None
        )
        srt_text = sentences_to_srt(merged_sentences)

    base_name = None
    if isinstance(audio, str) and os.path.exists(audio):
        base_name = audio
    saved = save_outputs(base_name, full_text, timestamps_json, srt_text,
                         language="自动检测", model_info=asr_model_type)

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
    return full_text, timestamps_json, srt_text

# ==================== 视频字幕处理函数（移除 subtitle_mode 参数） ====================
def transcribe_video(video, asr_model_type, use_gpu, use_half, enable_vad, enable_lid, enable_punc,
                     beam_size, nbest, decode_max_len, softmax_smoothing, aed_length_penalty,
                     eos_penalty, elm_weight,
                     vad_min_speech_frame, vad_max_speech_frame, vad_min_silence_frame,
                     vad_speech_threshold, vad_smooth_window_size,
                     punc_threshold,
                     merge_max_duration, merge_max_chars, merge_punctuations, merge_silence_threshold,
                     force_preprocess,
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
        result, _, error = manager.transcribe(audio_path, force_preprocess=force_preprocess)
        if error:
            return f"识别失败: {error}", "", ""

        progress(0.6, desc="生成字幕...")
        full_text, timestamps_json, srt_text, segments = format_result_to_outputs(result)

        # 应用合并参数
        if segments:
            ts_data = [(s["start"], s["end"]) for s in segments]
            texts = [s["text"] for s in segments]
            merged_sentences = merge_timestamps_to_sentences(
                ts_data, texts,
                sentence_endings=merge_punctuations,
                max_words=50,
                max_chars=merge_max_chars,
                max_duration=merge_max_duration,
                silence_threshold=merge_silence_threshold,
                merge_by_punc=True,
                merge_by_silence=True,
                merge_by_wordcount=False,
                merge_by_charcount=True,
                merge_by_duration=True
            )
            srt_text = sentences_to_srt(merged_sentences)

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

        result_msg = f"音频识别完成！字幕文件已生成。\n{save_info}"
        combined_text = f"{result_msg}\n\n【识别文本】\n{full_text}"

        progress(0.9, desc="清理...")
        manager.cleanup_temp()
        progress(1.0, desc="完成")
        return combined_text, timestamps_json, srt_text
    except Exception as e:
        logging.error(traceback.format_exc())
        error_msg = f"处理视频时发生未知错误: {str(e)}"
        return error_msg, "", ""
    finally:
        if temp_audio_path and os.path.exists(temp_audio_path):
            try:
                os.unlink(temp_audio_path)
            except:
                pass

# ==================== 批量处理函数（增加预处理开关） ====================
def transcribe_batch(files, asr_model_type, use_gpu, use_half, enable_vad, enable_lid, enable_punc,
                     beam_size, nbest, decode_max_len, softmax_smoothing, aed_length_penalty,
                     eos_penalty, elm_weight,
                     vad_min_speech_frame, vad_max_speech_frame, vad_min_silence_frame,
                     vad_speech_threshold, vad_smooth_window_size,
                     punc_threshold,
                     force_preprocess,
                     progress=gr.Progress()):
    if not files:
        return "请选择音频文件"

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
        return str(e)

    results_text = []
    total = len(files)
    for i, file_obj in enumerate(files, 1):
        file_path = file_obj.name if hasattr(file_obj, 'name') else str(file_obj)
        progress(i/total, desc=f"处理 {i}/{total}: {os.path.basename(file_path)}")
        result, audio_path, error = manager.transcribe(file_path, force_preprocess=force_preprocess)
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
    progress(1.0, desc="完成")
    manager.cleanup_temp()
    return "\n".join(results_text)

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

# ==================== 强制对齐包装函数（token ID 精确空行 + 预处理开关） ====================
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
                        force_preprocess,
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
    word_srt, sent_srt, timestamps, words, token_ids_out, error = manager.force_align(audio, text, progress,
                                                                                      force_preprocess=force_preprocess)
    if error:
        return f"错误: {error}", "", ""

    # 空行断句（精确 token ID 匹配）
    force_break = None
    merge_warnings = []
    if merge_by_newline and words and timestamps and token_ids_out:
        asr = manager.asr_system.asr
        paragraphs = [p.strip() for p in text.split('\n') if p.strip()]
        if len(paragraphs) > 1:
            force_break = [False] * len(words)
            current_pos = 0
            for para in paragraphs:
                para_tokens, para_ids = asr.tokenizer.tokenize(para)
                if len(para_ids) == 0:
                    continue
                found = -1
                for start in range(current_pos, len(token_ids_out) - len(para_ids) + 1):
                    if token_ids_out[start:start+len(para_ids)] == para_ids:
                        found = start
                        break
                if found >= 0:
                    end_idx = found + len(para_ids) - 1
                    if end_idx < len(words) - 1:
                        force_break[end_idx] = True
                    current_pos = end_idx + 1
                else:
                    msg = f"警告：段落 '{para[:30]}...' 无法与词序列匹配，该段落将不按空行断句"
                    print(msg)
                    merge_warnings.append(msg)

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
            force_break_punc = [False] * len(words)
            for pos in punc_positions:
                tidx = char_to_token[pos]
                if 0 <= tidx < len(words) - 1:
                    force_break_punc[tidx] = True

    final_force_break = [False] * len(words)
    if force_break:
        for i, v in enumerate(force_break):
            if v: final_force_break[i] = True
    if force_break_punc:
        for i, v in enumerate(force_break_punc):
            if v: final_force_break[i] = True

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

        timestamp_str = time.strftime("%Y%m%d_%H%M%S")
        prefix = generate_output_filename(audio, timestamp_str, default_name="align")
        merged_path = ALIGN_OUTPUT_DIR / f"{prefix}_merged_custom.srt"
        with open(merged_path, "w", encoding="utf-8") as f:
            f.write(merged_srt)

    if merge_warnings:
        try:
            gr.Warning("\n".join(merge_warnings))
        except:
            pass

    progress(0.9, desc="清理...")
    manager.cleanup_temp()
    progress(1.0, desc="完成")
    return word_srt, sent_srt, merged_srt

# ==================== 修复 Bug 5：跨平台打开文件夹函数 ====================
def open_file_or_dir(path: str):
    """跨平台打开文件/文件夹"""
    path = str(path)
    if sys.platform == "win32":
        os.startfile(path)
    elif sys.platform == "darwin":
        subprocess.Popen(["open", path])
    else:
        subprocess.Popen(["xdg-open", path])

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

    with gr.Blocks(title="FireRedASR2S WebUI 增强融合版", theme=gr.themes.Default()) as demo:
        gr.Markdown(f"""
        # FireRedASR2S 语音识别系统 增强融合版
        **支持 VAD、LID、标点恢复、时间戳、SRT字幕生成**
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

        # 修复 Bug 2：增加操作提示组件
        load_msg = gr.Textbox(label="操作提示", interactive=False, visible=True)

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

        # 基础按钮绑定（修复 Bug 2：输出到 load_msg 和 status_display）
        load_btn.click(
            load_model_click,
            inputs=[asr_model_type, use_gpu, use_half, enable_vad, enable_lid, enable_punc,
                    beam_size, nbest, decode_max_len, softmax_smoothing, aed_length_penalty,
                    eos_penalty, elm_weight,
                    vad_min_speech_frame, vad_max_speech_frame, vad_min_silence_frame,
                    vad_speech_threshold, vad_smooth_window_size,
                    punc_threshold],
            outputs=[load_msg, status_display]
        )
        unload_btn.click(unload_model_click, outputs=[load_msg, status_display])
        refresh_btn.click(refresh_status, outputs=[status_display])

        gr.Markdown("---")

        # ========== 主标签页 ==========
        with gr.Tabs():
            # ---------- 音频识别 ----------
            with gr.Tab("音频识别"):
                # 修复 Bug 1：声明共享的音频路径状态
                audio_path_state = gr.State()

                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### 上传音频")

                        input_mode = gr.Radio(
                            choices=["文件上传（推荐大文件）", "麦克风/音频组件（小文件）"],
                            value="文件上传（推荐大文件）",
                            label="音频输入方式"
                        )
                        audio_file = gr.File(
                            label="选择音频文件",
                            file_types=[".wav", ".mp3", ".m4a", ".flac", ".ogg"],
                            type="filepath",
                            visible=True
                        )
                        audio_mic = gr.Audio(
                            label="录制或选择音频",
                            type="filepath",
                            sources=["upload", "microphone"],
                            visible=False
                        )

                        force_preprocess_audio = gr.Checkbox(label="⚡ 强制预处理为 16kHz 单声道 (推荐)", value=True)

                        # 字幕合并参数
                        with gr.Accordion("字幕合并参数", open=True):
                            merge_max_duration = gr.Slider(1.0, 20.0, 10.0, step=0.5, label="单条最大时长 (秒)")
                            merge_max_chars = gr.Slider(5, 100, 30, step=5, label="单条最大字符数")
                            merge_punctuations = gr.Textbox(value="。！？.!?", label="句末标点")
                            merge_silence_threshold = gr.Slider(0.1, 1.0, 0.3, step=0.05, label="静音阈值 (秒)")

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

                def toggle_input(mode):
                    return (gr.update(visible=(mode == "文件上传（推荐大文件）")),
                            gr.update(visible=(mode != "文件上传（推荐大文件）")))
                input_mode.change(toggle_input, inputs=input_mode, outputs=[audio_file, audio_mic])

                def get_audio_path(mode, file_path, mic_path):
                    return file_path if mode == "文件上传（推荐大文件）" else mic_path

                transcribe_btn.click(
                    get_audio_path,
                    inputs=[input_mode, audio_file, audio_mic],
                    outputs=[audio_path_state]   # 保存到共享状态
                ).then(
                    transcribe_audio,
                    inputs=[audio_path_state, asr_model_type, use_gpu, use_half, enable_vad, enable_lid, enable_punc,
                            beam_size, nbest, decode_max_len, softmax_smoothing, aed_length_penalty,
                            eos_penalty, elm_weight,
                            vad_min_speech_frame, vad_max_speech_frame, vad_min_silence_frame,
                            vad_speech_threshold, vad_smooth_window_size,
                            punc_threshold,
                            merge_max_duration, merge_max_chars, merge_punctuations, merge_silence_threshold,
                            force_preprocess_audio],
                    outputs=[text_output, json_output, srt_output]
                ).then(refresh_status, outputs=[status_display])

                clear_btn.click(
                    lambda: [None, "", "", ""],
                    outputs=[audio_file, text_output, json_output, srt_output]
                )

            # ---------- 视频字幕（移除 subtitle_mode） ----------
            with gr.Tab("视频字幕"):
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### 上传视频")
                        video_input = gr.Video(
                            label="选择视频文件",
                            sources=["upload"],
                            interactive=True
                        )
                        force_preprocess_video = gr.Checkbox(label="⚡ 强制预处理音频 (推荐)", value=True)

                        # 字幕合并参数
                        with gr.Accordion("字幕合并参数", open=True):
                            video_merge_max_duration = gr.Slider(1.0, 20.0, 10.0, step=0.5, label="单条最大时长 (秒)")
                            video_merge_max_chars = gr.Slider(5, 100, 30, step=5, label="单条最大字符数")
                            video_merge_punctuations = gr.Textbox(value="。！？.!?", label="句末标点")
                            video_merge_silence_threshold = gr.Slider(0.1, 1.0, 0.3, step=0.05, label="静音阈值 (秒)")

                        with gr.Row():
                            video_transcribe_btn = gr.Button("提取字幕", variant="primary")
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
                            punc_threshold,
                            video_merge_max_duration, video_merge_max_chars, video_merge_punctuations, video_merge_silence_threshold,
                            force_preprocess_video],
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
                        force_preprocess_batch = gr.Checkbox(label="⚡ 强制预处理为 16kHz 单声道", value=True)
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
                            punc_threshold,
                            force_preprocess_batch],
                    outputs=[batch_output]
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
                        force_preprocess_align = gr.Checkbox(label="⚡ 强制预处理音频 (推荐)", value=True)

                        with gr.Accordion("字幕合并参数", open=True):
                            merge_punctuations_align = gr.Textbox(
                                label="句末标点符号", value="。！？.!?",
                                info="遇到这些符号时强制断句"
                            )
                            with gr.Row():
                                align_max_words = gr.Slider(
                                    label="最大词数", minimum=5, maximum=50, value=20, step=1,
                                    info="单条字幕最多包含多少个词"
                                )
                                align_max_chars = gr.Slider(
                                    label="最大字符数", minimum=5, maximum=100, value=30, step=5,
                                    info="单条字幕最多包含多少个字符（中文按字数）"
                                )
                            with gr.Row():
                                align_max_duration = gr.Slider(
                                    label="最大时长 (秒)", minimum=1.0, maximum=20.0, value=10.0, step=0.5,
                                    info="单条字幕最大时长"
                                )
                                align_silence_threshold = gr.Slider(
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
                            merge_punctuations_align, align_max_words, align_max_chars, align_max_duration, align_silence_threshold,
                            merge_by_punc, merge_by_silence, merge_by_wordcount, merge_by_charcount, merge_by_duration,
                            merge_by_newline,
                            force_preprocess_align],
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
                        preview_max_size = gr.Slider(
                            label="字幕预览最大文件大小 (MB)", minimum=1, maximum=100, value=manager.settings.get("preview_max_size_mb", 5), step=1,
                            info="超过此大小的音频将不会在预览中加载，避免浏览器卡顿"
                        )
                    with gr.Row():
                        open_output_btn = gr.Button("打开输出目录")
                        open_log_btn = gr.Button("打开日志文件夹")
                        clear_cache_btn = gr.Button("清理临时文件")
                    with gr.Row():
                        save_config_btn = gr.Button("保存当前配置", variant="primary")
                        preset_files = sorted([f.name for f in PRESET_DIR.glob("preset_*.json")], reverse=True)
                        preset_selector = gr.Dropdown(label="选择预设文件", choices=preset_files, value=None, interactive=True)
                        load_config_btn = gr.Button("加载所选配置", variant="secondary")
                        refresh_preset_btn = gr.Button("刷新列表", variant="secondary", size="sm")
                    config_status = gr.Textbox(label="配置状态", interactive=False)

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

                # 修复 Bug 5：使用跨平台函数
                def open_output():
                    open_file_or_dir(str(OUTPUT_DIR))
                    return "已打开输出目录"
                open_output_btn.click(open_output, outputs=[config_status])

                def open_log():
                    open_file_or_dir(str(LOG_DIR))
                    return "已打开日志文件夹"
                open_log_btn.click(open_log, outputs=[config_status])

                def clear_cache():
                    cleaned = manager.cleanup_temp()
                    return f"清理了 {cleaned} 个临时文件"
                clear_cache_btn.click(clear_cache, outputs=[config_status])

                # 保存当前配置（增加 merge_punctuations_align 的保存）
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
                        "merge_punctuations": merge_punctuations.value,        # 音频识别页
                        "merge_punctuations_align": merge_punctuations_align.value,  # 强制对齐页
                        "merge_max_words": align_max_words.value,
                        "merge_max_chars": align_max_chars.value,
                        "merge_max_duration": align_max_duration.value,
                        "merge_silence_threshold": align_silence_threshold.value,
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
                    new_choices = sorted([f.name for f in PRESET_DIR.glob("preset_*.json")], reverse=True)
                    return f"配置已保存到 {preset_path}", gr.update(choices=new_choices)
                save_config_btn.click(
                    save_current_config,
                    outputs=[config_status, preset_selector]
                )

                def refresh_preset_list():
                    new_choices = sorted([f.name for f in PRESET_DIR.glob("preset_*.json")], reverse=True)
                    return gr.update(choices=new_choices)
                refresh_preset_btn.click(refresh_preset_list, outputs=[preset_selector])

                # 加载所选配置（动态更新，匹配 components 列表长度）
                # 注意：输出列表顺序与下方 keys 严格对应
                _PARAM_KEYS = [
                    "asr_model_type", "use_gpu", "use_half", "enable_vad", "enable_lid", "enable_punc",
                    "beam_size", "nbest", "decode_max_len", "softmax_smoothing", "aed_length_penalty", "eos_penalty", "elm_weight",
                    "vad_min_speech_frame", "vad_max_speech_frame", "vad_min_silence_frame", "vad_speech_threshold", "vad_smooth_window_size", "punc_threshold",
                    "merge_punctuations", "merge_max_words", "merge_max_chars", "merge_max_duration", "merge_silence_threshold",
                    "merge_by_punc", "merge_by_silence", "merge_by_wordcount", "merge_by_charcount", "merge_by_duration", "merge_by_newline",
                    "preview_max_size_mb", "merge_punctuations_align"
                ]
                _PARAM_DEFAULTS = {
                    "asr_model_type": "aed", "use_gpu": True, "use_half": False, "enable_vad": True, "enable_lid": True, "enable_punc": True,
                    "beam_size": 3, "nbest": 1, "decode_max_len": 0, "softmax_smoothing": 1.25, "aed_length_penalty": 0.6, "eos_penalty": 1.0, "elm_weight": 0.0,
                    "vad_min_speech_frame": 20, "vad_max_speech_frame": 2000, "vad_min_silence_frame": 20, "vad_speech_threshold": 0.4, "vad_smooth_window_size": 5, "punc_threshold": 0.45,
                    "merge_punctuations": "。！？.!?", "merge_max_words": 20, "merge_max_chars": 30, "merge_max_duration": 10.0, "merge_silence_threshold": 0.3,
                    "merge_by_punc": True, "merge_by_silence": True, "merge_by_wordcount": True, "merge_by_charcount": True, "merge_by_duration": True, "merge_by_newline": False,
                    "preview_max_size_mb": 5, "merge_punctuations_align": "。！？.!?"
                }

                def load_selected_config(filename):
                    if not filename:
                        return ["请先选择一个预设文件"] + [gr.update() for _ in _PARAM_KEYS]

                    file_path = PRESET_DIR / filename
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            cfg = json.load(f)
                    except Exception as e:
                        return [f"加载失败: {e}"] + [gr.update() for _ in _PARAM_KEYS]

                    updates = []
                    for key in _PARAM_KEYS:
                        val = cfg.get(key, _PARAM_DEFAULTS.get(key))
                        updates.append(gr.update(value=val))
                    return [f"配置已加载: {filename}"] + updates

                # outputs 列表与 _PARAM_KEYS 一一对应，前面加 config_status
                load_config_btn.click(
                    load_selected_config,
                    inputs=[preset_selector],
                    outputs=[config_status] + [
                        asr_model_type, use_gpu, use_half, enable_vad, enable_lid, enable_punc,
                        beam_size, nbest, decode_max_len, softmax_smoothing, aed_length_penalty, eos_penalty, elm_weight,
                        vad_min_speech_frame, vad_max_speech_frame, vad_min_silence_frame, vad_speech_threshold, vad_smooth_window_size, punc_threshold,
                        merge_punctuations, align_max_words, align_max_chars, align_max_duration, align_silence_threshold,
                        merge_by_punc, merge_by_silence, merge_by_wordcount, merge_by_charcount, merge_by_duration, merge_by_newline,
                        preview_max_size, merge_punctuations_align
                    ]
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

    demo = create_interface()
    demo.queue(default_concurrency_limit=1)
    demo.launch(
        server_name="127.0.0.1",
        server_port=18006,
        inbrowser=True,
        show_error=True,
        max_file_size=500 * 1024 * 1024   
    )

if __name__ == "__main__":
    main()