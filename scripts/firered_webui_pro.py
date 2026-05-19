#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
FireRedASR2S WebUI pro 专业版

Copyright 2026 光影的故事2018
Licensed under the Apache License, Version 2.0
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
import uuid
import re
import subprocess
import shutil
from pathlib import Path
from datetime import timedelta

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
ROOT_DIR = BASE_DIR.parent
DEFAULT_OUTPUT_DIR = ROOT_DIR / "output"
OUTPUT_DIR = DEFAULT_OUTPUT_DIR

CACHE_DIR = ROOT_DIR / "cache"
CACHE_DIR.mkdir(exist_ok=True)

PRESET_DIR = ROOT_DIR / "preset"
PRESET_DIR.mkdir(exist_ok=True)
CONFIG_FILE = PRESET_DIR / "settings.json"

config_lock = threading.RLock()

# ==================== FFmpeg 配置 ====================
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

# ==================== 配置保存/加载 ====================
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

# ==================== 其他依赖 ====================
try:
    import gradio as gr
    import torch
    import numpy as np
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
        """自动查找 ASR 模型目录（统一大写）"""
        model_type_upper = model_type.upper()
        candidates = [
            ROOT_DIR / "pretrained_models" / f"FireRedASR2-{model_type_upper}",
            ROOT_DIR / "pretrained_models" / f"FireRedASR2-{model_type_upper}-2025",
            # 回退到已知目录
            ROOT_DIR / "pretrained_models" / "FireRedASR2-AED",
            ROOT_DIR / "pretrained_models" / "FireRedASR2-AED-2025",
        ]
        for p in candidates:
            if p.exists():
                print(f"✅ 自动检测到模型目录: {p}")
                return str(p)
        return None

    def _find_vad_model_dir(self):
        base = ROOT_DIR / "pretrained_models" / "FireRedVAD"
        if not base.exists():
            return None
        patterns = ["*.pt", "*.pth", "*.pth.tar", "*.onnx", "*.bin", "*.tar"]
        def has_model(p):
            if not p.is_dir():
                return False
            for pat in patterns:
                if list(p.glob(pat)):
                    return True
            return False
        sub_dirs = ["vad", "VAD", "Stream-VAD", "AED", "stream_vad", "aed"]
        for sub in sub_dirs:
            path = base / sub
            if path.exists() and has_model(path):
                print(f"✅ 自动检测到 VAD 模型目录: {path}")
                return str(path)
        if has_model(base):
            return str(base)
        return None

    def _find_lid_model_dir(self):
        path = ROOT_DIR / "pretrained_models" / "FireRedLID"
        if path.exists() and any(path.iterdir()):
            return str(path)
        return None

    def _find_punc_model_dir(self):
        path = ROOT_DIR / "pretrained_models" / "FireRedPunc"
        if path.exists() and any(path.iterdir()):
            return str(path)
        return None

    def load_system(self, config_dict=None, advanced_params=None):
        with self.lock:
            if self.asr_system is not None:
                self.unload_system()
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
                if not default_config["use_gpu"]:
                    default_config["use_half"] = False

                model_dir_override = config_dict.get("model_dir", None) if config_dict else None
                if model_dir_override and Path(model_dir_override).exists():
                    model_dir = model_dir_override
                else:
                    model_dir = self._find_model_dir(default_config['asr_model_type'].upper())
                if not model_dir or not Path(model_dir).exists():
                    return False, "ASR模型目录不存在"

                adv_defaults = {
                    "beam_size": 3,
                    "nbest": 1,
                    "decode_max_len": 0,
                    "softmax_smoothing": 1.25,
                    "aed_length_penalty": 0.6,
                    "eos_penalty": 1.0,
                    "elm_weight": 0.0,
                    "vad_min_speech_frame": 20,
                    "vad_max_speech_frame": 2000,
                    "vad_min_silence_frame": 20,
                    "vad_speech_threshold": 0.4,
                    "vad_smooth_window_size": 5,
                    "punc_threshold": 0.45,
                }
                if advanced_params is None:
                    advanced_params = {}
                for k, v in adv_defaults.items():
                    if k not in advanced_params:
                        advanced_params[k] = v

                vad_config = FireRedVadConfig(use_gpu=default_config["use_gpu"])
                lid_config = FireRedLidConfig(use_gpu=default_config["use_gpu"])
                asr_config = FireRedAsr2Config(
                    use_gpu=default_config["use_gpu"],
                    use_half=default_config["use_half"],
                    return_timestamp=True
                )
                punc_config = FireRedPuncConfig(use_gpu=default_config["use_gpu"])

                asr_config.beam_size = advanced_params["beam_size"]
                asr_config.nbest = advanced_params["nbest"]
                asr_config.decode_max_len = advanced_params["decode_max_len"]
                asr_config.softmax_smoothing = advanced_params["softmax_smoothing"]
                asr_config.aed_length_penalty = advanced_params["aed_length_penalty"]
                asr_config.eos_penalty = advanced_params["eos_penalty"]
                asr_config.elm_weight = advanced_params["elm_weight"]

                vad_config.min_speech_frame = advanced_params["vad_min_speech_frame"]
                vad_config.max_speech_frame = advanced_params["vad_max_speech_frame"]
                vad_config.min_silence_frame = advanced_params["vad_min_silence_frame"]
                vad_config.speech_threshold = advanced_params["vad_speech_threshold"]
                vad_config.smooth_window_size = advanced_params["vad_smooth_window_size"]

                try:
                    punc_config.threshold = advanced_params["punc_threshold"]
                except:
                    pass

                vad_model_dir = self._find_vad_model_dir()
                if not vad_model_dir:
                    return False, "VAD模型目录未找到"

                lid_model_dir = self._find_lid_model_dir()
                if not lid_model_dir and default_config["enable_lid"]:
                    return False, "LID模型目录未找到"

                punc_model_dir = self._find_punc_model_dir()
                if not punc_model_dir and default_config["enable_punc"]:
                    return False, "Punc模型目录未找到"

                system_config = FireRedAsr2SystemConfig(
                    vad_model_dir=vad_model_dir,
                    lid_model_dir=lid_model_dir if lid_model_dir else "",
                    asr_model_dir=str(model_dir),
                    punc_model_dir=punc_model_dir if punc_model_dir else "",
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
                self.config['advanced'] = advanced_params
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
        if self.asr_system is None:
            return None, None, "系统未加载"
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

    def _prepare_audio(self, audio_input, return_waveform=False, force_preprocess=True):
        """预处理音频为 16k 单声道 wav，归一化浮点数据"""
        try:
            # 如果不强制预处理且输入已是理想格式，直接返回
            if not force_preprocess and isinstance(audio_input, str) and os.path.exists(audio_input):
                try:
                    info = sf.info(audio_input)
                    if info.samplerate == 16000 and info.channels == 1 and info.subtype == 'PCM_16':
                        if return_waveform:
                            data, sr = sf.read(audio_input, dtype='float32')
                            return audio_input, data, sr
                        return audio_input
                except:
                    pass

            # 处理元组输入（麦克风/录音），归一化到 [-1, 1]
            if isinstance(audio_input, tuple):
                sr, data = audio_input
                if data.ndim > 1:
                    data = np.mean(data, axis=1)
                # 检测是否为整数格式，若是则归一化
                if data.dtype.kind == 'i' or np.max(np.abs(data)) > 1.0:
                    data = data.astype(np.float32)
                    max_val = np.iinfo(np.int16).max if data.dtype == np.int16 else np.max(np.abs(data))
                    data /= max_val
                else:
                    data = data.astype(np.float32)
                input_path = CACHE_DIR / f"input_{uuid.uuid4().hex}_{int(time.time())}.wav"
                sf.write(str(input_path), data, sr)
                self.temp_files.append(str(input_path))
                need_cleanup_input = True
            elif isinstance(audio_input, str) and os.path.exists(audio_input):
                input_path = audio_input
                need_cleanup_input = False
            else:
                return None

            # 转码输出
            out_path = CACHE_DIR / f"temp_audio_{uuid.uuid4().hex}_{int(time.time())}.wav"
            cmd = [
                FFMPEG_PATH, "-y", "-i", str(input_path),
                "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", str(out_path)
            ]
            try:
                subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            except subprocess.CalledProcessError as e:
                error_detail = e.stderr.decode(errors='replace') if e.stderr else str(e)
                logging.error(f"FFmpeg转换失败: {error_detail}")
                return None
            self.temp_files.append(str(out_path))

            if need_cleanup_input and str(input_path) in self.temp_files:
                self.temp_files.remove(str(input_path))
                try:
                    os.unlink(str(input_path))
                except:
                    pass

            if return_waveform:
                data, sr = sf.read(str(out_path), dtype='float32')
                return str(out_path), data, sr
            else:
                return str(out_path)
        except Exception as e:
            logging.error(f"音频预处理失败: {e}")
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
        return "无结果", "{}", "", [], "{}"

    text = result.get("text", "")
    sentences = result.get("sentences", [])
    words = result.get("words", [])
    vad_segments = result.get("vad_segments_ms", [])

    # 词级 segments 优先用 words，否则用 sentences
    word_segments = []
    if words:
        for w in words:
            word_segments.append({
                "start": w.get("start_ms", 0) / 1000.0,
                "end": w.get("end_ms", 0) / 1000.0,
                "text": w.get("text", "")
            })
    elif sentences:
        for s in sentences:
            word_segments.append({
                "start": s.get("start_ms", 0) / 1000.0,
                "end": s.get("end_ms", 0) / 1000.0,
                "text": s.get("text", "")
            })

    # 句子级 segments（供直接输出）
    sent_segments = []
    if sentences:
        for s in sentences:
            sent_segments.append({
                "start": s.get("start_ms", 0) / 1000.0,
                "end": s.get("end_ms", 0) / 1000.0,
                "text": s.get("text", "")
            })
    elif word_segments:
        sent_segments = word_segments[:]

    word_json = json.dumps(word_segments, ensure_ascii=False, indent=2)
    sent_json = json.dumps(sent_segments, ensure_ascii=False, indent=2)

    # SRT 基于句子级
    srt_lines = []
    for i, seg in enumerate(sent_segments, 1):
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

    return full_text, sent_json, srt_text, word_segments, word_json

def merge_timestamps_to_sentences(timestamps, words,
                                   sentence_endings="。！？.!?",
                                   max_words=20, max_chars=50, max_duration=10.0,
                                   silence_threshold=0.3,
                                   merge_by_punc=True, merge_by_silence=True,
                                   merge_by_wordcount=True, merge_by_charcount=True,
                                   merge_by_duration=True, force_break_indices=None):
    if len(timestamps) == 0:
        return []
    has_cjk = any(re.search(r'[\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af]', w) for w in words)
    join_str = "" if has_cjk else " "
    sentences = []
    current_start = timestamps[0][0]
    current_words = []
    last_end = timestamps[0][1]

    for i, ((start, end), word) in enumerate(zip(timestamps, words)):
        should_break = False
        # 强制断句索引
        if force_break_indices and i < len(force_break_indices) and force_break_indices[i]:
            should_break = True
        else:
            # 句末标点断句
            if merge_by_punc and any(word.endswith(p) for p in sentence_endings):
                should_break = True
            # 静音阈值断句（仅当已有前词）
            if not should_break and merge_by_silence and i > 0:
                if start - last_end > silence_threshold:
                    should_break = True
            # 词数限制
            if not should_break and merge_by_wordcount and len(current_words) + 1 >= max_words:
                should_break = True
            # 字符数限制（注意：如果当前只有一个词且已超长，此处无法触发，因此后续单独处理）
            if not should_break and merge_by_charcount and current_words:
                new_text = join_str.join(current_words + [word])
                if len(new_text) >= max_chars:
                    should_break = True
            # 时长限制
            if not should_break and merge_by_duration and current_words:
                if (last_end - current_start) + (end - start) >= max_duration:
                    should_break = True

        # 处理第一个词超长的情况
        if not should_break and not current_words:
            # 单独检查单个词是否已经超过字符数或时长限制
            if merge_by_charcount and len(word) >= max_chars:
                should_break = True
            if merge_by_duration and (end - start) >= max_duration:
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
        start = seconds_to_srt_time(sent["start"])
        end = seconds_to_srt_time(sent["end"])
        srt_lines.append(str(i))
        srt_lines.append(f"{start} --> {end}")
        srt_lines.append(sent["text"])
        srt_lines.append("")
    return "\n".join(srt_lines)

def save_outputs(base_name, full_text, sent_json, srt_text, language, model_info):
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
    if sent_json and sent_json != "{}":
        json_path = OUTPUT_DIR / f"{prefix}.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            f.write(sent_json)
        saved['sent_json'] = str(json_path)
    if srt_text.strip():
        srt_path = OUTPUT_DIR / f"{prefix}.srt"
        with open(srt_path, 'w', encoding='utf-8') as f:
            f.write(srt_text)
        saved['srt'] = str(srt_path)
    return saved, prefix

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
    info.append(f"日志文件: {log_file}")
    info.append(f"缓存目录: {CACHE_DIR}")
    return "\n".join(info)

# ==================== 显示截断 ====================
def truncate_all_for_display(full_text, word_json, sent_json, srt_text,
                              text_limit=1000, word_item_limit=300, sent_item_limit=300, srt_line_limit=200):
    disp_text = full_text
    if len(disp_text) > text_limit:
        disp_text = disp_text[:text_limit] + "\n... [内容过长已截断，完整内容已保存]"

    disp_word_json = word_json
    try:
        word_data = json.loads(word_json)
        if isinstance(word_data, list) and len(word_data) > word_item_limit:
            word_data = word_data[:word_item_limit]
            disp_word_json = json.dumps(word_data, ensure_ascii=False, indent=2) + f"\n... [截断至前{word_item_limit}条]"
    except:
        if len(disp_word_json) > 2000:
            disp_word_json = disp_word_json[:2000] + "\n... [截断]"

    disp_sent_json = sent_json
    try:
        sent_data = json.loads(sent_json)
        if isinstance(sent_data, list) and len(sent_data) > sent_item_limit:
            sent_data = sent_data[:sent_item_limit]
            disp_sent_json = json.dumps(sent_data, ensure_ascii=False, indent=2) + f"\n... [截断至前{sent_item_limit}条]"
    except:
        if len(disp_sent_json) > 2000:
            disp_sent_json = disp_sent_json[:2000] + "\n... [截断]"

    lines = srt_text.splitlines()
    disp_srt = srt_text
    if len(lines) > srt_line_limit:
        disp_srt = "\n".join(lines[:srt_line_limit]) + f"\n... [截断至前{srt_line_limit}行]"

    return disp_text, disp_word_json, disp_sent_json, disp_srt

def ensure_model_loaded(config_params, advanced_params):
    with manager.lock:
        need_reload = manager.asr_system is None
        if not need_reload:
            for k, v in config_params.items():
                if manager.config.get(k) != v:
                    need_reload = True
                    break
        if not need_reload and manager.config and 'advanced' in manager.config:
            saved_adv = manager.config['advanced']
            for k, v in advanced_params.items():
                if saved_adv.get(k) != v:
                    need_reload = True
                    break
        if need_reload:
            success, msg = manager.load_system(config_params, advanced_params)
            if not success:
                raise RuntimeError(f"加载失败: {msg}")
    return manager

# ==================== 标点重新注入 ====================
def inject_punctuation_to_words(word_segments, full_text_with_punc, punctuation_chars="。！？.!?"):
    """
    将 full_text_with_punc 中的标点附加到对应的 word_segments 文本末尾。
    假设 word_segments 的 text 顺序与去标点后的汉字序列一致。
    """
    if not word_segments:
        return word_segments
    # 提取所有汉字字符
    han_chars = re.findall(r'[\u4e00-\u9fff]', full_text_with_punc)
    if len(han_chars) != len(word_segments):
        return word_segments  # 长度不匹配则跳过

    # 构建一个带索引的标点位置映射
    punct_positions = []
    idx = 0
    for ch in full_text_with_punc:
        if ch in punctuation_chars:
            punct_positions.append((idx, ch))
        elif re.match(r'[\u4e00-\u9fff]', ch):
            idx += 1
        # 其他字符（如空格）忽略，可能影响索引，这里简单处理：只统计汉字
    # 将标点分配给前一个汉字对应的 word_segment
    seg_idx = 0
    for (han_idx, punct_char) in punct_positions:
        while seg_idx < len(word_segments) and seg_idx < han_idx:
            seg_idx += 1
        if seg_idx > 0:
            # 标点附着到前一个词
            word_segments[seg_idx - 1]["text"] += punct_char
    return word_segments

# ==================== 识别函数（音频、视频、批量均使用全局合并参数） ====================
def transcribe_audio(audio, asr_model_type, use_gpu, use_half, enable_vad, enable_lid, enable_punc,
                     beam_size, nbest, decode_max_len, softmax_smoothing, aed_length_penalty,
                     eos_penalty, elm_weight,
                     vad_min_speech_frame, vad_max_speech_frame, vad_min_silence_frame,
                     vad_speech_threshold, vad_smooth_window_size,
                     punc_threshold,
                     enable_duration, merge_max_duration,
                     enable_charcount, merge_max_chars,
                     enable_punc_merge, merge_punctuations,
                     enable_silence, merge_silence_threshold,
                     force_preprocess,
                     progress=gr.Progress()):
    if not FIRERED_AVAILABLE:
        return "错误: FireRedASR2S 模块不可用", "", "", ""

    if audio is None:
        return "请上传或录制音频", "", "", ""

    progress(0, desc="初始化...")
    config_params = {
        "use_gpu": use_gpu, "use_half": use_half,
        "enable_vad": enable_vad, "enable_lid": enable_lid, "enable_punc": enable_punc,
        "asr_model_type": asr_model_type
    }
    advanced_params = {
        "beam_size": beam_size, "nbest": nbest, "decode_max_len": decode_max_len,
        "softmax_smoothing": softmax_smoothing, "aed_length_penalty": aed_length_penalty,
        "eos_penalty": eos_penalty, "elm_weight": elm_weight,
        "vad_min_speech_frame": vad_min_speech_frame, "vad_max_speech_frame": vad_max_speech_frame,
        "vad_min_silence_frame": vad_min_silence_frame, "vad_speech_threshold": vad_speech_threshold,
        "vad_smooth_window_size": vad_smooth_window_size, "punc_threshold": punc_threshold,
    }
    try:
        ensure_model_loaded(config_params, advanced_params)
    except RuntimeError as e:
        return str(e), "", "", ""

    progress(0.3, desc="识别中...")
    result, audio_path, error = manager.transcribe(audio, force_preprocess=force_preprocess)
    if error:
        return f"错误: {error}", "", "", ""

    progress(0.7, desc="生成输出...")
    full_text_with_punc, sent_json_initial, srt_text_initial, word_segments, word_json = format_result_to_outputs(result)

    # --- 标点注入 ---
    if word_segments and result.get("text"):
        word_segments = inject_punctuation_to_words(word_segments, result["text"], punctuation_chars=merge_punctuations)

    # 句子合并（使用注入标点后的 word_segments）
    merged_sentences = []
    if word_segments:
        ts_data = [(s["start"], s["end"]) for s in word_segments]
        texts = [s["text"] for s in word_segments]
        merged_sentences = merge_timestamps_to_sentences(
            ts_data, texts,
            sentence_endings=merge_punctuations,
            max_words=50, max_chars=merge_max_chars, max_duration=merge_max_duration,
            silence_threshold=merge_silence_threshold,
            merge_by_punc=enable_punc_merge, merge_by_silence=enable_silence,
            merge_by_wordcount=False, merge_by_charcount=enable_charcount,
            merge_by_duration=enable_duration
        )
        srt_text = sentences_to_srt(merged_sentences)
        sent_json = json.dumps(merged_sentences, ensure_ascii=False, indent=2)
    else:
        srt_text = srt_text_initial
        sent_json = sent_json_initial

    base_name = audio if isinstance(audio, str) and os.path.exists(audio) else None
    saved, prefix = save_outputs(base_name, full_text_with_punc, sent_json, srt_text, "自动检测", asr_model_type)
    if word_segments:
        word_json_path = OUTPUT_DIR / f"{prefix}_words.json"
        with open(word_json_path, 'w', encoding='utf-8') as f:
            f.write(word_json)
        saved['word_json'] = str(word_json_path)

    save_info = "文件已保存:\n"
    for k, v in saved.items():
        save_info += f"  {Path(v).name}\n"
    full_text_disp = save_info + "\n" + full_text_with_punc

    disp_text, disp_word_json, disp_sent_json, disp_srt = truncate_all_for_display(
        full_text_disp, word_json, sent_json, srt_text
    )

    progress(0.9, desc="清理...")
    manager.cleanup_temp()
    progress(1.0, desc="完成")
    return disp_text, disp_word_json, disp_sent_json, disp_srt

def transcribe_video(video, asr_model_type, use_gpu, use_half, enable_vad, enable_lid, enable_punc,
                     beam_size, nbest, decode_max_len, softmax_smoothing, aed_length_penalty,
                     eos_penalty, elm_weight,
                     vad_min_speech_frame, vad_max_speech_frame, vad_min_silence_frame,
                     vad_speech_threshold, vad_smooth_window_size,
                     punc_threshold,
                     enable_duration, merge_max_duration,
                     enable_charcount, merge_max_chars,
                     enable_punc_merge, merge_punctuations,
                     enable_silence, merge_silence_threshold,
                     force_preprocess,
                     progress=gr.Progress()):
    temp_audio_path = None
    try:
        if not FIRERED_AVAILABLE:
            return "错误: FireRedASR2S 模块不可用", "", "", ""

        if video is None:
            return "请上传视频文件", "", "", ""

        progress(0, desc="初始化...")
        config_params = {
            "use_gpu": use_gpu, "use_half": use_half,
            "enable_vad": enable_vad, "enable_lid": enable_lid, "enable_punc": enable_punc,
            "asr_model_type": asr_model_type
        }
        advanced_params = {
            "beam_size": beam_size, "nbest": nbest, "decode_max_len": decode_max_len,
            "softmax_smoothing": softmax_smoothing, "aed_length_penalty": aed_length_penalty,
            "eos_penalty": eos_penalty, "elm_weight": elm_weight,
            "vad_min_speech_frame": vad_min_speech_frame, "vad_max_speech_frame": vad_max_speech_frame,
            "vad_min_silence_frame": vad_min_silence_frame, "vad_speech_threshold": vad_speech_threshold,
            "vad_smooth_window_size": vad_smooth_window_size, "punc_threshold": punc_threshold,
        }
        try:
            ensure_model_loaded(config_params, advanced_params)
        except RuntimeError as e:
            return str(e), "", "", ""

        progress(0.2, desc="提取视频音频...")
        audio_path = CACHE_DIR / f"video_audio_{uuid.uuid4().hex}_{int(time.time())}.wav"
        temp_audio_path = str(audio_path)
        cmd = [
            str(FFMPEG_PATH), "-i", video,
            "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
            "-y", temp_audio_path
        ]
        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as e:
            error_detail = e.stderr if e.stderr else str(e)
            return f"音频提取失败: {error_detail}", "", "", ""

        progress(0.4, desc="识别音频...")
        result, _, error = manager.transcribe(temp_audio_path, force_preprocess=force_preprocess)
        if error:
            return f"识别失败: {error}", "", "", ""

        progress(0.6, desc="生成字幕...")
        full_text_with_punc, sent_json_initial, srt_text_initial, word_segments, word_json = format_result_to_outputs(result)

        if word_segments and result.get("text"):
            word_segments = inject_punctuation_to_words(word_segments, result["text"], punctuation_chars=merge_punctuations)

        merged_sentences = []
        if word_segments:
            ts_data = [(s["start"], s["end"]) for s in word_segments]
            texts = [s["text"] for s in word_segments]
            merged_sentences = merge_timestamps_to_sentences(
                ts_data, texts,
                sentence_endings=merge_punctuations,
                max_words=50, max_chars=merge_max_chars, max_duration=merge_max_duration,
                silence_threshold=merge_silence_threshold,
                merge_by_punc=enable_punc_merge, merge_by_silence=enable_silence,
                merge_by_wordcount=False, merge_by_charcount=enable_charcount,
                merge_by_duration=enable_duration
            )
            srt_text = sentences_to_srt(merged_sentences)
            sent_json = json.dumps(merged_sentences, ensure_ascii=False, indent=2)
        else:
            srt_text = srt_text_initial
            sent_json = sent_json_initial

        base_name = video if isinstance(video, str) and os.path.exists(video) else None
        saved, prefix = save_outputs(base_name, full_text_with_punc, sent_json, srt_text, "自动检测", asr_model_type)
        if word_segments:
            word_json_path = OUTPUT_DIR / f"{prefix}_words.json"
            with open(word_json_path, 'w', encoding='utf-8') as f:
                f.write(word_json)
            saved['word_json'] = str(word_json_path)

        save_info = "文件已保存:\n"
        for k, v in saved.items():
            save_info += f"  {Path(v).name}\n"
        combined_text = f"{save_info}\n\n【识别文本】\n{full_text_with_punc}"

        disp_text, disp_word_json, disp_sent_json, disp_srt = truncate_all_for_display(
            combined_text, word_json, sent_json, srt_text
        )

        progress(0.9, desc="清理...")
        manager.cleanup_temp()
        progress(1.0, desc="完成")
        return disp_text, disp_word_json, disp_sent_json, disp_srt
    except Exception as e:
        logging.error(traceback.format_exc())
        return f"处理视频时发生未知错误: {str(e)}", "", "", ""
    finally:
        if temp_audio_path and os.path.exists(temp_audio_path):
            try:
                os.unlink(temp_audio_path)
            except:
                pass

def transcribe_batch(files, asr_model_type, use_gpu, use_half, enable_vad, enable_lid, enable_punc,
                     beam_size, nbest, decode_max_len, softmax_smoothing, aed_length_penalty,
                     eos_penalty, elm_weight,
                     vad_min_speech_frame, vad_max_speech_frame, vad_min_silence_frame,
                     vad_speech_threshold, vad_smooth_window_size,
                     punc_threshold, force_preprocess,
                     enable_duration, merge_max_duration,
                     enable_charcount, merge_max_chars,
                     enable_punc_merge, merge_punctuations,
                     enable_silence, merge_silence_threshold,
                     progress=gr.Progress()):
    if not files:
        return "请选择音频文件"
    config_params = {
        "use_gpu": use_gpu, "use_half": use_half,
        "enable_vad": enable_vad, "enable_lid": enable_lid, "enable_punc": enable_punc,
        "asr_model_type": asr_model_type
    }
    advanced_params = {
        "beam_size": beam_size, "nbest": nbest, "decode_max_len": decode_max_len,
        "softmax_smoothing": softmax_smoothing, "aed_length_penalty": aed_length_penalty,
        "eos_penalty": eos_penalty, "elm_weight": elm_weight,
        "vad_min_speech_frame": vad_min_speech_frame, "vad_max_speech_frame": vad_max_speech_frame,
        "vad_min_silence_frame": vad_min_silence_frame, "vad_speech_threshold": vad_speech_threshold,
        "vad_smooth_window_size": vad_smooth_window_size, "punc_threshold": punc_threshold,
    }
    try:
        ensure_model_loaded(config_params, advanced_params)
    except RuntimeError as e:
        return str(e)

    results_summary = []
    total = len(files)
    for i, file_obj in enumerate(files, 1):
        file_path = file_obj.name if hasattr(file_obj, 'name') else str(file_obj)
        progress(i/total, desc=f"处理 {i}/{total}: {os.path.basename(file_path)}")
        result, _, error = manager.transcribe(file_path, force_preprocess=force_preprocess)
        if error:
            results_summary.append(f"【{os.path.basename(file_path)}】错误: {error}")
        else:
            full_text_with_punc, sent_json_initial, srt_text_initial, word_segments, word_json = format_result_to_outputs(result)
            if word_segments and result.get("text"):
                word_segments = inject_punctuation_to_words(word_segments, result["text"], punctuation_chars=merge_punctuations)

            if word_segments:
                ts_data = [(s["start"], s["end"]) for s in word_segments]
                texts = [s["text"] for s in word_segments]
                merged = merge_timestamps_to_sentences(
                    ts_data, texts,
                    sentence_endings=merge_punctuations,
                    max_chars=merge_max_chars, max_duration=merge_max_duration,
                    silence_threshold=merge_silence_threshold,
                    merge_by_punc=enable_punc_merge, merge_by_silence=enable_silence,
                    merge_by_wordcount=False, merge_by_charcount=enable_charcount,
                    merge_by_duration=enable_duration
                )
                srt_text = sentences_to_srt(merged)
                sent_json = json.dumps(merged, ensure_ascii=False, indent=2)
            else:
                srt_text = srt_text_initial
                sent_json = sent_json_initial

            saved, prefix = save_outputs(file_path, full_text_with_punc, sent_json, srt_text, "自动检测", asr_model_type)
            if word_segments:
                word_json_path = OUTPUT_DIR / f"{prefix}_words.json"
                with open(word_json_path, 'w', encoding='utf-8') as f:
                    f.write(word_json)
                saved['word_json'] = str(word_json_path)
            saved_files = [Path(v).name for v in saved.values()]
            results_summary.append(f"【{os.path.basename(file_path)}】已保存: {', '.join(saved_files)}")
    progress(1.0, desc="完成")
    manager.cleanup_temp()
    summary = f"批量处理完成，共 {total} 个文件。\n" + "\n".join(results_summary)
    summary += f"\n\n所有结果文件已保存至输出目录: {OUTPUT_DIR}"
    return summary

def load_model_click(asr_model_type, use_gpu, use_half, enable_vad, enable_lid, enable_punc,
                     beam_size, nbest, decode_max_len, softmax_smoothing, aed_length_penalty,
                     eos_penalty, elm_weight,
                     vad_min_speech_frame, vad_max_speech_frame, vad_min_silence_frame,
                     vad_speech_threshold, vad_smooth_window_size,
                     punc_threshold):
    config = {
        "use_gpu": use_gpu, "use_half": use_half,
        "enable_vad": enable_vad, "enable_lid": enable_lid, "enable_punc": enable_punc,
        "asr_model_type": asr_model_type
    }
    advanced = {
        "beam_size": beam_size, "nbest": nbest, "decode_max_len": decode_max_len,
        "softmax_smoothing": softmax_smoothing, "aed_length_penalty": aed_length_penalty,
        "eos_penalty": eos_penalty, "elm_weight": elm_weight,
        "vad_min_speech_frame": vad_min_speech_frame, "vad_max_speech_frame": vad_max_speech_frame,
        "vad_min_silence_frame": vad_min_silence_frame, "vad_speech_threshold": vad_speech_threshold,
        "vad_smooth_window_size": vad_smooth_window_size, "punc_threshold": punc_threshold,
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

def open_file_or_dir(path: str):
    path = str(path)
    if sys.platform == "win32":
        os.startfile(path)
    elif sys.platform == "darwin":
        subprocess.Popen(["open", path])
    else:
        subprocess.Popen(["xdg-open", path])

def clear_cache_fully():
    count = 0
    for f in CACHE_DIR.glob("*.wav"):
        try:
            os.unlink(f)
            count += 1
        except Exception:
            pass
    return f"已删除 {count} 个缓存文件"

# ==================== 创建界面 ====================
def create_interface():
    settings = manager.settings
    default_output_dir = settings.get("output_dir", str(DEFAULT_OUTPUT_DIR))
    global OUTPUT_DIR
    with config_lock:
        OUTPUT_DIR = Path(default_output_dir)
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    llm_dir = ROOT_DIR / "pretrained_models" / "FireRedASR2-LLM"
    model_choices = ["aed"]
    if llm_dir.exists():
        model_choices.append("llm")

    with gr.Blocks(title="FireRedASR2S WebUI 专业版", theme=gr.themes.Default()) as demo:
        gr.Markdown(f"""
        # FireRedASR2S 语音识别系统 专业版（修复版）
        **支持 VAD、LID、标点恢复、四种时间戳预览及词级 JSON 输出**
        输出目录: `{OUTPUT_DIR}` | 缓存目录: `{CACHE_DIR}`
        """)

        with gr.Accordion("系统状态信息", open=False):
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

        with gr.Row():
            asr_model_type = gr.Dropdown(
                label="ASR 模型类型",
                choices=model_choices,
                value="aed",               
                scale=2
            )
            load_msg = gr.Textbox(
                label="操作提示",
                interactive=False,
                visible=True,
                scale=3
            )

        with gr.Row():
            load_btn = gr.Button("加载模型", variant="primary", scale=1)
            unload_btn = gr.Button("卸载模型", variant="stop", scale=1)
            use_gpu = gr.Checkbox(label="使用 GPU", value=torch.cuda.is_available(), scale=1)
            default_half = torch.cuda.is_available() and torch.cuda.get_device_properties(0).total_memory < 10e9
            use_half = gr.Checkbox(
                label="使用半精度 (FP16)", value=default_half, scale=1   
            )
            enable_vad = gr.Checkbox(label="启用 VAD", value=True, scale=1)
            enable_lid = gr.Checkbox(label="启用 LID", value=True, scale=1)
            enable_punc = gr.Checkbox(label="启用 标点恢复", value=True, scale=1)

        with gr.Accordion("高级参数", open=False):
            gr.Markdown("### 解码参数")
            with gr.Row():
                with gr.Column():
                    beam_size = gr.Slider(1, 10, 3, step=1, label="Beam 大小")
                    nbest = gr.Slider(1, 5, 1, step=1, label="候选结果数")
                    decode_max_len = gr.Slider(0, 500, 0, step=10, label="最大解码长度")
                with gr.Column():
                    softmax_smoothing = gr.Slider(0.5, 2.0, 1.25, step=0.05, label="Softmax 平滑")
                    aed_length_penalty = gr.Slider(-2.0, 2.0, 0.6, step=0.1, label="长度惩罚")
                    eos_penalty = gr.Slider(0.5, 2.0, 1.0, step=0.1, label="结束符惩罚")
            with gr.Row():
                elm_weight = gr.Slider(0.0, 1.0, 0.0, step=0.05, label="外部语言模型权重")
            gr.Markdown("### VAD 参数")
            with gr.Row():
                with gr.Column():
                    vad_speech_threshold = gr.Slider(0.1, 0.9, 0.4, step=0.05, label="语音阈值")
                    vad_min_speech_frame = gr.Slider(1, 50, 20, step=1, label="最小语音帧数")
                with gr.Column():
                    vad_max_speech_frame = gr.Slider(100, 3000, 2000, step=50, label="最大语音帧数")
                    vad_min_silence_frame = gr.Slider(5, 50, 20, step=1, label="最小静音帧数")
            vad_smooth_window_size = gr.Slider(1, 20, 5, step=1, label="平滑窗口")
            punc_threshold = gr.Slider(0.1, 0.9, 0.45, step=0.05, label="标点阈值")

        load_btn.click(
            load_model_click,
            inputs=[asr_model_type, use_gpu, use_half, enable_vad, enable_lid, enable_punc,
                    beam_size, nbest, decode_max_len, softmax_smoothing, aed_length_penalty,
                    eos_penalty, elm_weight,
                    vad_min_speech_frame, vad_max_speech_frame, vad_min_silence_frame,
                    vad_speech_threshold, vad_smooth_window_size, punc_threshold],
            outputs=[load_msg, status_display]
        )
        unload_btn.click(unload_model_click, outputs=[load_msg, status_display])
        refresh_btn.click(refresh_status, outputs=[status_display])

        gr.Markdown("---")
        with gr.Tabs():
            # ===== 音频识别 =====
            with gr.Tab("音频识别"):
                with gr.Row():
                    with gr.Column(scale=1):
                        audio_input = gr.File(
                            label="选择音频文件",
                            file_types=[".wav", ".mp3", ".m4a", ".flac", ".ogg"],
                            type="filepath"
                        )
                        audio_preview = gr.Audio(
                            label="🎧 音频预览", type="filepath", interactive=False, visible=False
                        )
                        force_preprocess_check = gr.Checkbox(
                            label="⚡ 强制预处理为 16kHz 单声道 (推荐大文件)", value=True, interactive=True
                        )
                        with gr.Accordion("全局字幕合并参数", open=True):
                            enable_duration = gr.Checkbox(label="启用单条最大时长限制", value=False)
                            merge_max_duration = gr.Slider(1.0, 20.0, 10.0, step=0.5, label="单条最大时长 (秒)")
                            enable_charcount = gr.Checkbox(label="启用单条最大字符数限制", value=False)
                            merge_max_chars = gr.Slider(5, 100, 30, step=5, label="单条最大字符数")
                            enable_silence = gr.Checkbox(label="启用静音阈值分句", value=False)
                            merge_silence_threshold = gr.Slider(0.1, 1.0, 0.3, step=0.05, label="静音阈值 (秒)")
                            enable_punc_merge = gr.Checkbox(label="启用句末标点分句", value=True)
                            merge_punctuations = gr.Textbox(value="。！？.!?", label="句末标点")
                        with gr.Row():
                            transcribe_btn = gr.Button("开始识别", variant="primary")
                            c_btn = gr.Button("清空", variant="secondary")
                    with gr.Column(scale=2):
                        with gr.Tabs():
                            with gr.Tab("识别文本"):
                                text_output = gr.Textbox(label="结果", lines=15, show_copy_button=True)
                            with gr.Tab("字级时间戳 (JSON)"):
                                word_json_output = gr.Textbox(label="逐词时间戳", lines=15, show_copy_button=True)
                            with gr.Tab("句子级时间戳 (JSON)"):
                                sent_json_output = gr.Textbox(label="句子级时间戳", lines=15, show_copy_button=True)
                            with gr.Tab("SRT字幕"):
                                srt_output = gr.Textbox(label="SRT字幕", lines=15, show_copy_button=True)

                def update_audio_preview(file_path):
                    if file_path and os.path.exists(file_path):
                        return gr.update(value=file_path, visible=True)
                    return gr.update(value=None, visible=False)

                audio_input.change(update_audio_preview, inputs=[audio_input], outputs=[audio_preview])

                transcribe_btn.click(
                    transcribe_audio,
                    inputs=[audio_input, asr_model_type, use_gpu, use_half, enable_vad, enable_lid, enable_punc,
                            beam_size, nbest, decode_max_len, softmax_smoothing, aed_length_penalty,
                            eos_penalty, elm_weight,
                            vad_min_speech_frame, vad_max_speech_frame, vad_min_silence_frame,
                            vad_speech_threshold, vad_smooth_window_size,
                            punc_threshold,
                            enable_duration, merge_max_duration,
                            enable_charcount, merge_max_chars,
                            enable_punc_merge, merge_punctuations,
                            enable_silence, merge_silence_threshold,
                            force_preprocess_check],
                    outputs=[text_output, word_json_output, sent_json_output, srt_output]
                ).then(refresh_status, outputs=[status_display])

                c_btn.click(
                    lambda: [None, gr.update(value=None, visible=False), True, "", "", "", ""],
                    outputs=[audio_input, audio_preview, force_preprocess_check,
                             text_output, word_json_output, sent_json_output, srt_output]
                )

            # ===== 视频字幕（移除独立合并参数，使用全局） =====
            with gr.Tab("视频字幕"):
                with gr.Row():
                    with gr.Column(scale=1):
                        video_input = gr.Video(label="选择视频文件", sources=["upload"])
                        force_preprocess_video = gr.Checkbox(
                            label="⚡ 强制预处理音频 (推荐大文件)", value=False, interactive=True
                        )
                        gr.Markdown("> 字幕合并参数使用上方「音频识别」Tab 中的全局设置")
                        with gr.Row():
                            video_transcribe_btn = gr.Button("提取字幕", variant="primary")
                            video_clear_btn = gr.Button("清空", variant="secondary")
                    with gr.Column(scale=2):
                        with gr.Tabs():
                            with gr.Tab("识别文本"):
                                video_text_output = gr.Textbox(label="结果", lines=15, show_copy_button=True)
                            with gr.Tab("字级时间戳 (JSON)"):
                                video_word_json_output = gr.Textbox(label="逐词时间戳", lines=15, show_copy_button=True)
                            with gr.Tab("句子级时间戳 (JSON)"):
                                video_sent_json_output = gr.Textbox(label="句子级时间戳", lines=15, show_copy_button=True)
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
                            enable_duration, merge_max_duration,
                            enable_charcount, merge_max_chars,
                            enable_punc_merge, merge_punctuations,
                            enable_silence, merge_silence_threshold,
                            force_preprocess_video],
                    outputs=[video_text_output, video_word_json_output, video_sent_json_output, video_srt_output]
                ).then(refresh_status, outputs=[status_display])

                video_clear_btn.click(
                    lambda: [None, "", "", "", ""],
                    outputs=[video_input, video_text_output, video_word_json_output, video_sent_json_output, video_srt_output]
                )

            # ===== 批量处理（使用全局合并参数） =====
            with gr.Tab("批量处理"):
                with gr.Row():
                    with gr.Column(scale=1):
                        file_input = gr.File(
                            label="上传多个音频文件",
                            file_types=[".wav", ".mp3", ".m4a", ".flac", ".ogg"],
                            file_count="multiple",
                            type="filepath"
                        )
                        force_preprocess_batch = gr.Checkbox(label="⚡ 强制预处理", value=True)
                        gr.Markdown("> 字幕合并参数使用上方「音频识别」Tab 中的全局设置")
                        with gr.Row():
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
                            vad_speech_threshold, vad_smooth_window_size, punc_threshold,
                            force_preprocess_batch,
                            enable_duration, merge_max_duration,
                            enable_charcount, merge_max_chars,
                            enable_punc_merge, merge_punctuations,
                            enable_silence, merge_silence_threshold],
                    outputs=[batch_output]
                ).then(refresh_status, outputs=[status_display])

                batch_clear.click(
                    lambda: [None, ""],
                    outputs=[file_input, batch_output]
                )

            # ===== 系统信息 =====
            with gr.Tab("系统信息"):
                system_info_text = gr.Textbox(value=get_system_info(), lines=20, label="详细信息", show_copy_button=True)
                with gr.Row():
                    output_dir_input = gr.Textbox(value=str(OUTPUT_DIR), label="输出目录", interactive=True, scale=3)
                    update_output_btn = gr.Button("更新输出目录", variant="secondary", scale=1)
                with gr.Row():
                    open_output_btn = gr.Button("打开输出目录")
                    open_log_btn = gr.Button("打开日志文件夹")
                with gr.Row():
                    clear_temp_btn = gr.Button("清理已跟踪的临时文件", variant="secondary")
                    clear_all_cache_btn = gr.Button("清空整个缓存目录", variant="secondary")
                with gr.Row():
                    save_config_btn = gr.Button("保存当前配置", variant="primary")
                    preset_files = sorted([f.name for f in PRESET_DIR.glob("preset_*.json")], reverse=True)
                    preset_selector = gr.Dropdown(label="选择预设文件", choices=preset_files, value=None, interactive=True)
                    load_config_btn = gr.Button("加载所选配置", variant="secondary")
                    refresh_preset_btn = gr.Button("刷新列表", variant="secondary", size="sm")
                config_status = gr.Textbox(label="配置状态", interactive=False)

                def update_output_dir(new_dir):
                    global OUTPUT_DIR
                    try:
                        p = Path(new_dir)
                        p.mkdir(parents=True, exist_ok=True)
                        with config_lock:
                            OUTPUT_DIR = p
                        manager.settings["output_dir"] = str(p)
                        save_settings(manager.settings)
                        return f"输出目录已更新为 {p}", get_system_info()
                    except Exception as e:
                        return f"更新失败: {e}", get_system_info()

                update_output_btn.click(update_output_dir, inputs=[output_dir_input], outputs=[config_status, system_info_text])

                open_output_btn.click(
                    lambda: (open_file_or_dir(str(OUTPUT_DIR)), "已打开输出目录")[1],
                    outputs=[config_status]
                )
                open_log_btn.click(
                    lambda: (open_file_or_dir(str(LOG_DIR)), "已打开日志文件夹")[1],
                    outputs=[config_status]
                )

                clear_temp_btn.click(lambda: f"清理了 {manager.cleanup_temp()} 个临时文件", outputs=[config_status])
                clear_all_cache_btn.click(clear_cache_fully, outputs=[config_status])

                def save_current_config():
                    config = {
                        "asr_model_type": asr_model_type.value,
                        "use_gpu": use_gpu.value, "use_half": use_half.value,
                        "enable_vad": enable_vad.value, "enable_lid": enable_lid.value, "enable_punc": enable_punc.value,
                        "beam_size": beam_size.value, "nbest": nbest.value, "decode_max_len": decode_max_len.value,
                        "softmax_smoothing": softmax_smoothing.value, "aed_length_penalty": aed_length_penalty.value,
                        "eos_penalty": eos_penalty.value, "elm_weight": elm_weight.value,
                        "vad_min_speech_frame": vad_min_speech_frame.value, "vad_max_speech_frame": vad_max_speech_frame.value,
                        "vad_min_silence_frame": vad_min_silence_frame.value, "vad_speech_threshold": vad_speech_threshold.value,
                        "vad_smooth_window_size": vad_smooth_window_size.value, "punc_threshold": punc_threshold.value,
                        "enable_duration": enable_duration.value, "merge_max_duration": merge_max_duration.value,
                        "enable_charcount": enable_charcount.value, "merge_max_chars": merge_max_chars.value,
                        "enable_punc_merge": enable_punc_merge.value, "merge_punctuations": merge_punctuations.value,
                        "enable_silence": enable_silence.value, "merge_silence_threshold": merge_silence_threshold.value,
                        "force_preprocess": force_preprocess_check.value,
                    }
                    timestamp = time.strftime("%Y%m%d_%H%M%S")
                    preset_path = PRESET_DIR / f"preset_{timestamp}.json"
                    with open(preset_path, "w", encoding="utf-8") as f:
                        json.dump(config, f, ensure_ascii=False, indent=2)
                    new_choices = sorted([f.name for f in PRESET_DIR.glob("preset_*.json")], reverse=True)
                    return f"配置已保存到 {preset_path}", gr.update(choices=new_choices)
                save_config_btn.click(save_current_config, outputs=[config_status, preset_selector])

                def refresh_preset_list():
                    return gr.update(choices=sorted([f.name for f in PRESET_DIR.glob("preset_*.json")], reverse=True))
                refresh_preset_btn.click(refresh_preset_list, outputs=[preset_selector])

                _PARAM_KEYS = [
                    "asr_model_type", "use_gpu", "use_half", "enable_vad", "enable_lid", "enable_punc",
                    "beam_size", "nbest", "decode_max_len", "softmax_smoothing", "aed_length_penalty", "eos_penalty", "elm_weight",
                    "vad_min_speech_frame", "vad_max_speech_frame", "vad_min_silence_frame", "vad_speech_threshold", "vad_smooth_window_size", "punc_threshold",
                    "enable_duration", "merge_max_duration", "enable_charcount", "merge_max_chars",
                    "enable_punc_merge", "merge_punctuations", "enable_silence", "merge_silence_threshold",
                    "force_preprocess"
                ]
                _PARAM_DEFAULTS = {
                    "asr_model_type": "aed", "use_gpu": True, "use_half": False, "enable_vad": True, "enable_lid": True, "enable_punc": True,
                    "beam_size": 3, "nbest": 1, "decode_max_len": 0, "softmax_smoothing": 1.25, "aed_length_penalty": 0.6, "eos_penalty": 1.0, "elm_weight": 0.0,
                    "vad_min_speech_frame": 20, "vad_max_speech_frame": 2000, "vad_min_silence_frame": 20, "vad_speech_threshold": 0.4, "vad_smooth_window_size": 5, "punc_threshold": 0.45,
                    "enable_duration": True, "merge_max_duration": 10.0, "enable_charcount": True, "merge_max_chars": 30,
                    "enable_punc_merge": True, "merge_punctuations": "。！？.!?", "enable_silence": True, "merge_silence_threshold": 0.3,
                    "force_preprocess": True
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
                    # 验证模型类型
                    if cfg.get("asr_model_type") not in model_choices:
                        cfg["asr_model_type"] = model_choices[0]
                        msg = f"⚠️ 预设中的模型类型无效，已重置为 {model_choices[0]}"
                    else:
                        msg = f"配置已加载: {filename}"
                    updates = [gr.update(value=cfg.get(key, _PARAM_DEFAULTS.get(key))) for key in _PARAM_KEYS]
                    return [msg] + updates

                load_config_btn.click(
                    load_selected_config,
                    inputs=[preset_selector],
                    outputs=[config_status,
                             asr_model_type, use_gpu, use_half, enable_vad, enable_lid, enable_punc,
                             beam_size, nbest, decode_max_len, softmax_smoothing, aed_length_penalty, eos_penalty, elm_weight,
                             vad_min_speech_frame, vad_max_speech_frame, vad_min_silence_frame, vad_speech_threshold, vad_smooth_window_size, punc_threshold,
                             enable_duration, merge_max_duration, enable_charcount, merge_max_chars,
                             enable_punc_merge, merge_punctuations, enable_silence, merge_silence_threshold,
                             force_preprocess_check]
                )

        gr.Markdown("---")
        gr.Markdown(f"""
        <div style="text-align: center; color: #666; font-size: 0.9em;">
        <p>本软件包不提供任何模型文件，模型由用户自行从官方渠道获取。用户需自行遵守模型的原许可证。</p>
        <p>本软件包按“原样”提供，不提供任何明示或暗示的担保。使用本软件所产生的一切风险由用户自行承担。</p>
        <p><strong>更新请关注B站up主：光影的故事2018</strong></p>
        </div>
        """)

        demo.load(refresh_status, outputs=[status_display])

    return demo

@atexit.register
def cleanup():
    print("正在退出，清理资源...")
    manager.unload_system()
    manager.cleanup_temp()
    clean_old_logs()
    print("清理完成")

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
        max_file_size=200 * 1024 * 1024   # 200 MB
    )

if __name__ == "__main__":
    main()