#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
FireRedASR2S WebUI pro 专业版 (v2.1)

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
CURRENT_DIR = Path(__file__).resolve().parent
if (CURRENT_DIR / "FireRedASR2S").exists():
    ROOT_DIR = CURRENT_DIR
else:
    ROOT_DIR = CURRENT_DIR.parent

LOG_DIR = ROOT_DIR / "logs"
LOG_DIR.mkdir(exist_ok=True)

def clean_old_logs(days=7):
    cutoff = time.time() - days * 24 * 3600
    # 修复：清理所有日志文件，而不只是 error_*.log
    for f in LOG_DIR.glob("*.log"):
        if f.stat().st_mtime < cutoff:
            try:
                f.unlink()
            except:
                pass

clean_old_logs()
log_file = LOG_DIR / f"error_{time.strftime('%Y%m%d')}.log"

# 文件日志记录 INFO 级别，便于排查问题
file_handler = logging.FileHandler(log_file, encoding='utf-8')
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

console_handler = logging.StreamHandler(sys.stderr)
console_handler.setLevel(logging.WARNING)
console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

logging.basicConfig(level=logging.INFO, handlers=[file_handler, console_handler])

# ==================== 路径设置 ====================
PROJECT_ROOT = ROOT_DIR
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
ROOT_DIR = BASE_DIR.parent if not (BASE_DIR / "FireRedASR2S").exists() else BASE_DIR
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
    with config_lock:
        if CONFIG_FILE.exists():
            try:
                with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except:
                return {}
        return {}

def save_settings(settings):
    with config_lock:
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
        """自动查找 ASR 模型目录（仅匹配指定类型，禁止跨类型回退）"""
        model_type_upper = model_type.upper()
        # 标准候选路径
        candidates = [
            ROOT_DIR / "pretrained_models" / f"FireRedASR2-{model_type_upper}",
            ROOT_DIR / "pretrained_models" / f"FireRedASR2-{model_type_upper}-2025",
        ]
        for p in candidates:
            if p.exists():
                print(f"✅ 自动检测到模型目录: {p}")
                return str(p)

        # 增强：扫描 pretrained_models 下的所有一级子目录，匹配包含 FireRedASR2- 且包含目标类型的目录
        pretrained_root = ROOT_DIR / "pretrained_models"
        if pretrained_root.exists():
            for subdir in pretrained_root.iterdir():
                if subdir.is_dir():
                    name = subdir.name.lower()
                    if "fireredasr2" in name and model_type_upper.lower() in name:
                        print(f"✅ 扫描检测到模型目录: {subdir}")
                        return str(subdir)
        return None

    def _find_vad_model_dir(self):
        """查找非流式 VAD 模型目录（FireRedVad 需要 cmvn.ark + 模型文件）"""
        base = ROOT_DIR / "pretrained_models" / "FireRedVAD"
        if not base.exists():
            # 尝试扫描 pretrained_models 下所有目录
            pretrained_root = ROOT_DIR / "pretrained_models"
            if pretrained_root.exists():
                for subdir in pretrained_root.iterdir():
                    if subdir.is_dir() and "vad" in subdir.name.lower():
                        if self._is_valid_vad_dir(subdir):
                            print(f"✅ 扫描检测到 VAD 模型目录: {subdir}")
                            return str(subdir)
            return None

        def has_model(p):
            if not p.is_dir():
                return False
            if not (p / "cmvn.ark").exists():
                return False
            for pat in ["*.pt", "*.pth", "*.pth.tar", "*.onnx", "*.bin", "*.tar"]:
                if list(p.glob(pat)):
                    return True
            return False

        # 仅匹配非流式 VAD，避免误选 Stream-VAD / AED
        sub_dirs = ["VAD", "vad"]
        for sub in sub_dirs:
            path = base / sub
            if has_model(path):
                print(f"✅ 自动检测到 VAD 模型目录: {path}")
                return str(path)
        if has_model(base):
            return str(base)
        return None

    def _is_valid_vad_dir(self, p):
        if not p.is_dir():
            return False
        if not (p / "cmvn.ark").exists():
            return False
        for pat in ["*.pt", "*.pth", "*.pth.tar", "*.onnx", "*.bin", "*.tar"]:
            if list(p.glob(pat)):
                return True
        return False

    def _find_lid_model_dir(self):
        path = ROOT_DIR / "pretrained_models" / "FireRedLID"
        if path.exists() and any(path.iterdir()):
            return str(path)
        # 扫描
        pretrained_root = ROOT_DIR / "pretrained_models"
        if pretrained_root.exists():
            for subdir in pretrained_root.iterdir():
                if subdir.is_dir() and "lid" in subdir.name.lower():
                    return str(subdir)
        return None

    def _find_punc_model_dir(self):
        path = ROOT_DIR / "pretrained_models" / "FireRedPunc"
        if path.exists() and any(path.iterdir()):
            return str(path)
        # 扫描
        pretrained_root = ROOT_DIR / "pretrained_models"
        if pretrained_root.exists():
            for subdir in pretrained_root.iterdir():
                if subdir.is_dir() and "punc" in subdir.name.lower():
                    return str(subdir)
        return None

    def load_system(self, config_dict=None, advanced_params=None):
        with self.lock:
            # 先完成所有校验和新系统构造，成功后再替换旧系统
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

                # GPU 可用性校验：GPU 不可用回退 CPU；CPU 模式不支持 FP16
                if default_config["use_gpu"] and not torch.cuda.is_available():
                    logging.warning("CUDA不可用，自动回退 CPU")
                    default_config["use_gpu"] = False
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
                    return_timestamp=(default_config["asr_model_type"] == "aed")
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
                    asr_type=default_config["asr_model_type"],
                    asr_model_dir=str(model_dir),
                    punc_model_dir=punc_model_dir if punc_model_dir else "",
                    vad_config=vad_config,
                    lid_config=lid_config,
                    asr_config=asr_config,
                    punc_config=punc_config,
                    enable_vad=bool(default_config["enable_vad"]),
                    enable_lid=bool(default_config["enable_lid"]),
                    enable_punc=bool(default_config["enable_punc"])
                )

                # 构造新系统
                new_system = FireRedAsr2System(system_config)
                # 成功后再卸载旧系统并替换
                if self.asr_system is not None:
                    self.unload_system()
                self.asr_system = new_system
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

            if isinstance(audio_input, tuple):
                sr, data = audio_input
                if data.ndim > 1:
                    data = np.mean(data, axis=1)
                orig_dtype = data.dtype
                data = data.astype(np.float32)
                if np.issubdtype(orig_dtype, np.integer):
                    info = np.iinfo(orig_dtype)
                    if info.min < 0:
                        # 有符号整数(int16/int32)：除以满幅归一化到 [-1, 1]
                        data /= float(info.max)
                    else:
                        # 无符号整数(uint8 等)：先去除直流中点再归一化到 [-1, 1]
                        data = (data - float(info.max) / 2.0) / (float(info.max) / 2.0)
                else:
                    # 浮点：仅当峰值超出 [-1,1] 时压缩，避免无谓失真
                    peak = float(np.max(np.abs(data)))
                    if peak > 1.0:
                        data /= peak
                input_path = CACHE_DIR / f"input_{uuid.uuid4().hex}_{int(time.time())}.wav"
                sf.write(str(input_path), data, sr)
                self.temp_files.append(str(input_path))
                need_cleanup_input = True
            elif isinstance(audio_input, str) and os.path.exists(audio_input):
                input_path = audio_input
                need_cleanup_input = False
            else:
                return None

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
                # 修复：统一转换为字符串进行比较和删除
                if need_cleanup_input and str(input_path) in self.temp_files:
                    self.temp_files.remove(str(input_path))
                    try:
                        os.unlink(str(input_path))
                    except Exception:
                        pass
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
    ms = int(round((td.total_seconds() - total_seconds) * 1000))
    if ms >= 1000:
        ms = 999
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{ms:03d}"

def format_result_to_outputs(result):
    if not result or not isinstance(result, dict):
        return "无结果", "{}", "", [], "{}"

    text = result.get("text", "")
    sentences = result.get("sentences", [])
    words = result.get("words", [])
    vad_segments = result.get("vad_segments_ms", [])

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

    srt_lines = []
    for i, seg in enumerate(sent_segments, 1):
        start = seconds_to_srt_time(seg["start"])
        end = seconds_to_srt_time(seg["end"])
        srt_lines.append(str(i))
        srt_lines.append(f"{start} --> {end}")
        srt_lines.append(seg["text"])
        srt_lines.append("")
    srt_text = "\n".join(srt_lines)

    # 元数据信息（增加语种聚合，防止除零）
    extra = f"VAD段: {len(vad_segments)}"
    if sentences and sentences[0].get("asr_confidence") is not None:
        extra += f" | 置信度: {sentences[0]['asr_confidence']:.3f}"
    # 语种统计
    langs = [s.get("lang") for s in sentences if s.get("lang")]
    if langs:
        from collections import Counter
        top_lang = Counter(langs).most_common(1)[0][0]
        confs = [s.get("lang_confidence", 0) or 0 for s in sentences if s.get("lang") == top_lang]
        if confs:
            avg_conf = sum(confs) / len(confs)
            extra += f" | 语种: {top_lang} ({avg_conf:.2f})"
        else:
            extra += f" | 语种: {top_lang}"
    if result.get("dur_s"):
        extra += f" | 时长: {result['dur_s']:.1f}s"
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
        if force_break_indices and i < len(force_break_indices) and force_break_indices[i]:
            should_break = True
        else:
            if merge_by_silence and i > 0:
                if start - last_end > silence_threshold:
                    if current_words:
                        sentences.append({
                            "start": current_start,
                            "end": last_end,
                            "text": join_str.join(current_words).strip()
                        })
                        current_words = []
                        current_start = None
            if not should_break and merge_by_punc and any(word.endswith(p) for p in sentence_endings):
                should_break = True
            if not should_break and merge_by_wordcount and len(current_words) + 1 >= max_words:
                should_break = True
            if not should_break and merge_by_charcount and current_words:
                new_text = join_str.join(current_words + [word])
                if len(new_text) >= max_chars:
                    should_break = True
            if not should_break and merge_by_duration and current_words:
                # 使用 end - current_start 直接计算句子时长，避免包含词间静音
                if (end - current_start) >= max_duration:
                    should_break = True
            if not should_break and not current_words:
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
    # 加入毫秒时间戳避免同名冲突
    timestamp = time.strftime("%Y%m%d_%H%M%S") + f"_{int(time.time()*1000)%1000:03d}"
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
            # 使用近似比较，避免浮点精度导致误判
            def is_close(a, b):
                if isinstance(a, (int, float)) and isinstance(b, (int, float)):
                    return abs(a - b) < 1e-6
                return a == b

            for k, v in config_params.items():
                if not is_close(manager.config.get(k), v):
                    need_reload = True
                    break
        if not need_reload and manager.config and 'advanced' in manager.config:
            saved_adv = manager.config['advanced']
            for k, v in advanced_params.items():
                if not is_close(saved_adv.get(k), v):
                    need_reload = True
                    break
        if need_reload:
            success, msg = manager.load_system(config_params, advanced_params)
            if not success:
                raise RuntimeError(f"加载失败: {msg}")
    return manager

# ==================== 标点注入增强 ====================
_ALIGN_SKIP = set("，。！？；：、,.:;!?…‥—–·~～\"'“”‘’「」『』（）()【】[]《》〈〉")

def inject_punctuation_to_words(word_segments, full_text_with_punc,
                                 punctuation_chars="。！？.!?"):
    if not word_segments or not full_text_with_punc:
        return None                      # 返回 None 表示"放弃注入"
    punct_set = set(punctuation_chars)

    # 定义可忽略字符判断：空白、_ALIGN_SKIP、句末标点，以及所有非字母数字非中日韩字符
    def is_ignorable(ch):
        if ch.isspace() or ch in _ALIGN_SKIP or ch in punct_set:
            return True
        # 若不是字母、数字、中日韩统一表意文字，则视为可忽略符号
        if not (ch.isalnum() or '\u4e00' <= ch <= '\u9fff' or '\u3040' <= ch <= '\u30ff' or '\uac00' <= ch <= '\ud7af'):
            return True
        return False

    total = sum(len(seg['text']) for seg in word_segments)
    spans, acc = [], 0
    for seg in word_segments:
        spans.append((acc, acc + len(seg['text'])))
        acc += len(seg['text'])

    # 一致性校验：除去所有可忽略字符后的有效字符数必须相等
    stripped = sum(1 for ch in full_text_with_punc if not is_ignorable(ch))
    if stripped != total:
        logging.warning(f"标点注入长度不匹配(text={stripped}, words={total})，回退原生分句")
        return None

    char_idx, word_ptr, last_word_idx = 0, 0, -1
    for ch in full_text_with_punc:
        if ch in punct_set:
            if last_word_idx >= 0:
                word_segments[last_word_idx]['text'] += ch
        elif is_ignorable(ch):
            continue                     # 忽略所有标点、空白、未知符号
        else:
            if char_idx >= total:
                break
            while word_ptr < len(spans) and char_idx >= spans[word_ptr][1]:
                word_ptr += 1
            last_word_idx = word_ptr
            char_idx += 1
    return word_segments

# 续词集合：几乎不可能出现在句首的字（不要加 啊/哦/嗯/吧/吗/呢/了 等可开句的语气词）
STRONG_CONT = set("们的地得着过么之于与和或及而并且之")

def sanitize_word_punctuation(word_segments, punctuation_chars="。！？.!?",
                              min_gap=0.05):
    if not word_segments:
        return word_segments
    enders = set(punctuation_chars)
    for i, w in enumerate(word_segments[:-1]):
        nxt = word_segments[i + 1]
        # 循环剥离末尾句末标点（保留至少 1 个字符，防止词被剥空）
        while len(w['text']) > 1 and w['text'][-1] in enders:
            gap = nxt['start'] - w['end']
            if nxt['text'][:1] in STRONG_CONT or gap < min_gap:
                w['text'] = w['text'][:-1]      # 剥离误插的句末标点
            else:
                break
    return word_segments

def merge_short_sentences(sentences, min_chars=4, max_gap=0.6,
                          enders="。！？.!?"):
    """最小句长合并兜底：将过短的句子合并到前一句。
    合并时若前句以句末标点结尾且当前句以续词开头，剥离该误断标点。"""
    if not sentences:
        return sentences
    ender_set = set(enders)
    out = []
    for s in sentences:
        if out and len(s['text']) < min_chars and s['start'] - out[-1]['end'] < max_gap:
            prev_text = out[-1]['text']
            # 前句以句末标点结尾、当前句以续词开头 → 剥掉这个误断的标点
            if prev_text and prev_text[-1] in ender_set and s['text'][:1] in STRONG_CONT:
                out[-1]['text'] = prev_text[:-1]
            out[-1]['end'] = s['end']
            out[-1]['text'] += s['text']
        else:
            out.append(s)
    return out

# ==================== 识别函数 ====================
def transcribe_audio(audio, asr_model_type, use_gpu, use_half, enable_vad, enable_lid, enable_punc,
                     beam_size, nbest, decode_max_len, softmax_smoothing, aed_length_penalty,
                     eos_penalty, elm_weight,
                     vad_min_speech_frame, vad_max_speech_frame, vad_min_silence_frame,
                     vad_speech_threshold, vad_smooth_window_size,
                     enable_duration, merge_max_duration,
                     enable_charcount, merge_max_chars,
                     enable_punc_merge, merge_punctuations,
                     enable_silence, merge_silence_threshold,
                     sanitize_gap,
                     enable_min_sentence_merge, min_sentence_chars, min_sentence_gap,
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
        "vad_smooth_window_size": vad_smooth_window_size,
    }
    try:
        ensure_model_loaded(config_params, advanced_params)
    except RuntimeError as e:
        return str(e), "", "", ""

    progress(0.3, desc="识别中...")
    try:
        result, audio_path, error = manager.transcribe(audio, force_preprocess=force_preprocess)
        if error:
            return f"错误: {error}", "", "", ""

        progress(0.7, desc="生成输出...")
        full_text_with_punc, sent_json_initial, srt_text_initial, word_segments, word_json = format_result_to_outputs(result)

        # --- 标点注入（Patch 1）---
        injection_ok = False
        if word_segments and result.get("words") and result.get("text"):
            injected = inject_punctuation_to_words(word_segments, result["text"], punctuation_chars=merge_punctuations)
            if injected is not None:
                word_segments = injected
                # 句号合理性检查（Patch 2）
                word_segments = sanitize_word_punctuation(
                    word_segments, punctuation_chars=merge_punctuations, min_gap=sanitize_gap)
                injection_ok = True
            else:
                # 注入失败，回退到原生句子级结果
                word_segments = []
            # 注入成功后重新生成词级 JSON（带标点）；失败则保留原始版本
            if injection_ok:
                word_json = json.dumps(word_segments, ensure_ascii=False, indent=2)

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
            # 可选：最小句长合并（Patch 3）
            if enable_min_sentence_merge:
                merged_sentences = merge_short_sentences(
                    merged_sentences,
                    min_chars=min_sentence_chars,
                    max_gap=min_sentence_gap,
                    enders=merge_punctuations
                )
            srt_text = sentences_to_srt(merged_sentences)
            sent_json = json.dumps(merged_sentences, ensure_ascii=False, indent=2)
        else:
            srt_text = srt_text_initial
            sent_json = sent_json_initial

        base_name = audio if isinstance(audio, str) and os.path.exists(audio) else None
        saved, prefix = save_outputs(base_name, full_text_with_punc, sent_json, srt_text, "自动检测", asr_model_type)
        if word_json and word_json != "[]" and word_json != "{}":
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
    except Exception as e:
        logging.error(traceback.format_exc())
        return f"发生未知错误: {str(e)}", "", "", ""
    finally:
        manager.cleanup_temp()  # 确保临时文件清理

def transcribe_video(video, asr_model_type, use_gpu, use_half, enable_vad, enable_lid, enable_punc,
                     beam_size, nbest, decode_max_len, softmax_smoothing, aed_length_penalty,
                     eos_penalty, elm_weight,
                     vad_min_speech_frame, vad_max_speech_frame, vad_min_silence_frame,
                     vad_speech_threshold, vad_smooth_window_size,
                     enable_duration, merge_max_duration,
                     enable_charcount, merge_max_chars,
                     enable_punc_merge, merge_punctuations,
                     enable_silence, merge_silence_threshold,
                     sanitize_gap,
                     enable_min_sentence_merge, min_sentence_chars, min_sentence_gap,
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
            "vad_smooth_window_size": vad_smooth_window_size,
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

        # --- 标点注入（Patch 1）---
        injection_ok = False
        if word_segments and result.get("words") and result.get("text"):
            injected = inject_punctuation_to_words(word_segments, result["text"], punctuation_chars=merge_punctuations)
            if injected is not None:
                word_segments = injected
                word_segments = sanitize_word_punctuation(
                    word_segments, punctuation_chars=merge_punctuations, min_gap=sanitize_gap)
                injection_ok = True
            else:
                word_segments = []
            if injection_ok:
                word_json = json.dumps(word_segments, ensure_ascii=False, indent=2)

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
            if enable_min_sentence_merge:
                merged_sentences = merge_short_sentences(
                    merged_sentences,
                    min_chars=min_sentence_chars,
                    max_gap=min_sentence_gap,
                    enders=merge_punctuations
                )
            srt_text = sentences_to_srt(merged_sentences)
            sent_json = json.dumps(merged_sentences, ensure_ascii=False, indent=2)
        else:
            srt_text = srt_text_initial
            sent_json = sent_json_initial

        base_name = video if isinstance(video, str) and os.path.exists(video) else None
        saved, prefix = save_outputs(base_name, full_text_with_punc, sent_json, srt_text, "自动检测", asr_model_type)
        if word_json and word_json != "[]" and word_json != "{}":
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
        manager.cleanup_temp()

def transcribe_batch(files, asr_model_type, use_gpu, use_half, enable_vad, enable_lid, enable_punc,
                     beam_size, nbest, decode_max_len, softmax_smoothing, aed_length_penalty,
                     eos_penalty, elm_weight,
                     vad_min_speech_frame, vad_max_speech_frame, vad_min_silence_frame,
                     vad_speech_threshold, vad_smooth_window_size,
                     force_preprocess,
                     enable_duration, merge_max_duration,
                     enable_charcount, merge_max_chars,
                     enable_punc_merge, merge_punctuations,
                     enable_silence, merge_silence_threshold,
                     sanitize_gap,
                     enable_min_sentence_merge, min_sentence_chars, min_sentence_gap,
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
        "vad_smooth_window_size": vad_smooth_window_size,
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
        try:
            result, _, error = manager.transcribe(file_path, force_preprocess=force_preprocess)
            if error:
                results_summary.append(f"【{os.path.basename(file_path)}】错误: {error}")
            else:
                full_text_with_punc, sent_json_initial, srt_text_initial, word_segments, word_json = format_result_to_outputs(result)

                # --- 标点注入（Patch 1）---
                injection_ok = False
                if word_segments and result.get("words") and result.get("text"):
                    injected = inject_punctuation_to_words(word_segments, result["text"], punctuation_chars=merge_punctuations)
                    if injected is not None:
                        word_segments = injected
                        word_segments = sanitize_word_punctuation(
                            word_segments, punctuation_chars=merge_punctuations, min_gap=sanitize_gap)
                        injection_ok = True
                    else:
                        word_segments = []
                    if injection_ok:
                        word_json = json.dumps(word_segments, ensure_ascii=False, indent=2)

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
                    if enable_min_sentence_merge:
                        merged = merge_short_sentences(
                            merged,
                            min_chars=min_sentence_chars,
                            max_gap=min_sentence_gap,
                            enders=merge_punctuations
                        )
                    srt_text = sentences_to_srt(merged)
                    sent_json = json.dumps(merged, ensure_ascii=False, indent=2)
                else:
                    srt_text = srt_text_initial
                    sent_json = sent_json_initial

                saved, prefix = save_outputs(file_path, full_text_with_punc, sent_json, srt_text, "自动检测", asr_model_type)
                if word_json and word_json != "[]" and word_json != "{}":
                    word_json_path = OUTPUT_DIR / f"{prefix}_words.json"
                    with open(word_json_path, 'w', encoding='utf-8') as f:
                        f.write(word_json)
                    saved['word_json'] = str(word_json_path)
                saved_files = [Path(v).name for v in saved.values()]
                results_summary.append(f"【{os.path.basename(file_path)}】已保存: {', '.join(saved_files)}")
        except Exception as e:
            logging.error(traceback.format_exc())
            results_summary.append(f"【{os.path.basename(file_path)}】处理异常: {e}")
        finally:
            # 每个文件处理后进行显存清理
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    progress(1.0, desc="完成")
    manager.cleanup_temp()
    summary = f"批量处理完成，共 {total} 个文件。\n" + "\n".join(results_summary)
    summary += f"\n\n所有结果文件已保存至输出目录: {OUTPUT_DIR}"
    return summary

def load_model_click(asr_model_type, use_gpu, use_half, enable_vad, enable_lid, enable_punc,
                     beam_size, nbest, decode_max_len, softmax_smoothing, aed_length_penalty,
                     eos_penalty, elm_weight,
                     vad_min_speech_frame, vad_max_speech_frame, vad_min_silence_frame,
                     vad_speech_threshold, vad_smooth_window_size):
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
        "vad_smooth_window_size": vad_smooth_window_size,
    }
    # 注意：load_system 内部已经处理了安全替换逻辑，此处无需先卸载
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
    """清空整个缓存目录（删除所有文件，保留目录本身）"""
    count = 0
    for f in CACHE_DIR.iterdir():
        try:
            if f.is_file():
                f.unlink()
                count += 1
            elif f.is_dir():
                shutil.rmtree(f)
                count += 1
        except Exception as e:
            logging.warning(f"删除缓存失败 {f}: {e}")
    return f"已清理缓存目录，共删除 {count} 个项目"

# ==================== 创建界面 ====================
def create_interface():
    settings = manager.settings
    default_output_dir = settings.get("output_dir", str(DEFAULT_OUTPUT_DIR))
    global OUTPUT_DIR
    with config_lock:
        try:
            OUTPUT_DIR = Path(default_output_dir)
            OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            logging.error(f"输出目录不可用({default_output_dir}): {e}，回退默认目录")
            OUTPUT_DIR = DEFAULT_OUTPUT_DIR
            OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    llm_dir = ROOT_DIR / "pretrained_models" / "FireRedASR2-LLM"
    model_choices = ["aed"]
    if llm_dir.exists():
        model_choices.append("llm")

    with gr.Blocks(title="FireRedASR2S WebUI 专业版", theme=gr.themes.Default()) as demo:
        gr.Markdown(f"""
        # FireRedASR2S 语音识别系统 专业版
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
            # 小显存(<10GB)默认开启 FP16 省显存：整套模型 float32 约 8GB+，8GB 卡不开 FP16 会 OOM。
            # FP16 在 Pascal 等无张量核心的卡上虽无加速，但能把显存占用减半，是跑通本模型的必要条件。
            default_half = torch.cuda.is_available() and torch.cuda.get_device_properties(0).total_memory < 10e9
            use_half = gr.Checkbox(
                label="使用半精度 (FP16)", value=default_half, scale=1
            )
            enable_vad = gr.Checkbox(label="启用 VAD", value=True, scale=1)
            enable_lid = gr.Checkbox(label="启用 LID", value=True, scale=1)
            enable_punc = gr.Checkbox(label="启用 标点恢复", value=True, scale=1)

        # GPU 与半精度联动
        def on_gpu_change(gpu_val):
            if not gpu_val:
                return gr.update(value=False)
            return gr.update()
        use_gpu.change(on_gpu_change, inputs=[use_gpu], outputs=[use_half])

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

        load_btn.click(
            load_model_click,
            inputs=[asr_model_type, use_gpu, use_half, enable_vad, enable_lid, enable_punc,
                    beam_size, nbest, decode_max_len, softmax_smoothing, aed_length_penalty,
                    eos_penalty, elm_weight,
                    vad_min_speech_frame, vad_max_speech_frame, vad_min_silence_frame,
                    vad_speech_threshold, vad_smooth_window_size],
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
                            enable_silence = gr.Checkbox(label="启用静音阈值分句", value=True)          # 默认开启
                            merge_silence_threshold = gr.Slider(0.1, 1.0, 0.4, step=0.05, label="静音阈值 (秒)")
                            enable_punc_merge = gr.Checkbox(label="启用句末标点分句", value=False)       # 默认关闭
                            merge_punctuations = gr.Textbox(value="。！？.!?", label="句末标点")
                            sanitize_gap = gr.Slider(
                                0.0, 0.3, 0.05, step=0.01,
                                label="标点剥离阈值 (秒，0=仅按续词剥离)",
                                info="误插句号与下一词间隔小于此值时剥离。语速快的素材建议调小；口播/播客可用 0.05~0.1"
                            )
                            # 最小句长合并控件
                            with gr.Row():
                                enable_min_sentence_merge = gr.Checkbox(
                                    label="启用最小句长合并（兜底）",
                                    value=False,
                                    info="将过短的句子合并到前一句，避免碎片字幕"
                                )
                                min_sentence_chars = gr.Slider(
                                    2, 10, 4, step=1,
                                    label="最小句长（字符）"
                                )
                                min_sentence_gap = gr.Slider(
                                    0.1, 2.0, 0.6, step=0.1,
                                    label="最大合并间隔（秒）"
                                )
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
                            enable_duration, merge_max_duration,
                            enable_charcount, merge_max_chars,
                            enable_punc_merge, merge_punctuations,
                            enable_silence, merge_silence_threshold,
                            sanitize_gap,
                            enable_min_sentence_merge, min_sentence_chars, min_sentence_gap,
                            force_preprocess_check],
                    outputs=[text_output, word_json_output, sent_json_output, srt_output]
                ).then(refresh_status, outputs=[status_display])

                c_btn.click(
                    lambda: [None, gr.update(value=None, visible=False), True, "", "", "", ""],
                    outputs=[audio_input, audio_preview, force_preprocess_check,
                             text_output, word_json_output, sent_json_output, srt_output]
                )

            # ===== 视频字幕 =====
            with gr.Tab("视频字幕"):
                with gr.Row():
                    with gr.Column(scale=1):
                        video_input = gr.Video(label="选择视频文件", sources=["upload"])
                        force_preprocess_video = gr.Checkbox(
                            label="⚡ 强制预处理音频 (推荐大文件)", value=True, interactive=True  # 修复：默认 True 与音频一致
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
                            enable_duration, merge_max_duration,
                            enable_charcount, merge_max_chars,
                            enable_punc_merge, merge_punctuations,
                            enable_silence, merge_silence_threshold,
                            sanitize_gap,
                            enable_min_sentence_merge, min_sentence_chars, min_sentence_gap,
                            force_preprocess_video],
                    outputs=[video_text_output, video_word_json_output, video_sent_json_output, video_srt_output]
                ).then(refresh_status, outputs=[status_display])

                video_clear_btn.click(
                    lambda: [None, True, "", "", "", ""],
                    outputs=[video_input, force_preprocess_video,
                            video_text_output, video_word_json_output,
                            video_sent_json_output, video_srt_output]
                )

            # ===== 批量处理 =====
            with gr.Tab("批量处理"):
                with gr.Row():
                    with gr.Column(scale=1):
                        file_input = gr.File(
                            label="上传多个音频文件",
                            file_types=[".wav", ".mp3", ".m4a", ".flac", ".ogg"],
                            file_count="multiple",
                            type="filepath"
                        )
                        force_preprocess_batch = gr.Checkbox(label="⚡ 强制预处理", value=True)  # 默认 True，与音频一致
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
                            vad_speech_threshold, vad_smooth_window_size,
                            force_preprocess_batch,
                            enable_duration, merge_max_duration,
                            enable_charcount, merge_max_chars,
                            enable_punc_merge, merge_punctuations,
                            enable_silence, merge_silence_threshold,
                            sanitize_gap,
                            enable_min_sentence_merge, min_sentence_chars, min_sentence_gap],
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

                # 修复：保存配置使用回调参数，不再直接读取 .value
                def save_current_config(
                    asr_model_type, use_gpu, use_half, enable_vad, enable_lid, enable_punc,
                    beam_size, nbest, decode_max_len, softmax_smoothing, aed_length_penalty,
                    eos_penalty, elm_weight,
                    vad_min_speech_frame, vad_max_speech_frame, vad_min_silence_frame,
                    vad_speech_threshold, vad_smooth_window_size,
                    enable_duration, merge_max_duration,
                    enable_charcount, merge_max_chars,
                    enable_punc_merge, merge_punctuations,
                    enable_silence, merge_silence_threshold,
                    sanitize_gap,
                    enable_min_sentence_merge, min_sentence_chars, min_sentence_gap,
                    force_preprocess_check, force_preprocess_video, force_preprocess_batch
                ):
                    config = {
                        "asr_model_type": asr_model_type,
                        "use_gpu": use_gpu,
                        "use_half": use_half,
                        "enable_vad": enable_vad,
                        "enable_lid": enable_lid,
                        "enable_punc": enable_punc,
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
                        "enable_duration": enable_duration,
                        "merge_max_duration": merge_max_duration,
                        "enable_charcount": enable_charcount,
                        "merge_max_chars": merge_max_chars,
                        "enable_punc_merge": enable_punc_merge,
                        "merge_punctuations": merge_punctuations,
                        "enable_silence": enable_silence,
                        "merge_silence_threshold": merge_silence_threshold,
                        "sanitize_gap": sanitize_gap,
                        "enable_min_sentence_merge": enable_min_sentence_merge,
                        "min_sentence_chars": min_sentence_chars,
                        "min_sentence_gap": min_sentence_gap,
                        "force_preprocess": force_preprocess_check,
                        "force_preprocess_video": force_preprocess_video,
                        "force_preprocess_batch": force_preprocess_batch,
                    }
                    timestamp = time.strftime("%Y%m%d_%H%M%S")
                    preset_path = PRESET_DIR / f"preset_{timestamp}.json"
                    with open(preset_path, "w", encoding='utf-8') as f:
                        json.dump(config, f, ensure_ascii=False, indent=2)
                    new_choices = sorted([f.name for f in PRESET_DIR.glob("preset_*.json")], reverse=True)
                    return f"配置已保存到 {preset_path}", gr.update(choices=new_choices)

                save_config_btn.click(
                    save_current_config,
                    inputs=[
                        asr_model_type, use_gpu, use_half, enable_vad, enable_lid, enable_punc,
                        beam_size, nbest, decode_max_len, softmax_smoothing, aed_length_penalty,
                        eos_penalty, elm_weight,
                        vad_min_speech_frame, vad_max_speech_frame, vad_min_silence_frame,
                        vad_speech_threshold, vad_smooth_window_size,
                        enable_duration, merge_max_duration,
                        enable_charcount, merge_max_chars,
                        enable_punc_merge, merge_punctuations,
                        enable_silence, merge_silence_threshold,
                        sanitize_gap,
                        enable_min_sentence_merge, min_sentence_chars, min_sentence_gap,
                        force_preprocess_check, force_preprocess_video, force_preprocess_batch
                    ],
                    outputs=[config_status, preset_selector]
                )

                def refresh_preset_list():
                    return gr.update(choices=sorted([f.name for f in PRESET_DIR.glob("preset_*.json")], reverse=True))
                refresh_preset_btn.click(refresh_preset_list, outputs=[preset_selector])

                _PARAM_KEYS = [
                    "asr_model_type", "use_gpu", "use_half", "enable_vad", "enable_lid", "enable_punc",
                    "beam_size", "nbest", "decode_max_len", "softmax_smoothing", "aed_length_penalty", "eos_penalty", "elm_weight",
                    "vad_min_speech_frame", "vad_max_speech_frame", "vad_min_silence_frame", "vad_speech_threshold", "vad_smooth_window_size",
                    "enable_duration", "merge_max_duration", "enable_charcount", "merge_max_chars",
                    "enable_punc_merge", "merge_punctuations", "enable_silence", "merge_silence_threshold",
                    "sanitize_gap",
                    "enable_min_sentence_merge", "min_sentence_chars", "min_sentence_gap",
                    "force_preprocess", "force_preprocess_video", "force_preprocess_batch"
                ]
                _PARAM_DEFAULTS = {
                    "asr_model_type": "aed", "use_gpu": True, "use_half": False, "enable_vad": True, "enable_lid": True, "enable_punc": True,
                    "beam_size": 3, "nbest": 1, "decode_max_len": 0, "softmax_smoothing": 1.25, "aed_length_penalty": 0.6, "eos_penalty": 1.0, "elm_weight": 0.0,
                    "vad_min_speech_frame": 20, "vad_max_speech_frame": 2000, "vad_min_silence_frame": 20, "vad_speech_threshold": 0.4, "vad_smooth_window_size": 5,
                    "enable_duration": False, "merge_max_duration": 10.0, "enable_charcount": False, "merge_max_chars": 30,
                    "enable_punc_merge": False, "merge_punctuations": "。！？.!?", "enable_silence": True, "merge_silence_threshold": 0.4,
                    "sanitize_gap": 0.05,
                    "enable_min_sentence_merge": False, "min_sentence_chars": 4, "min_sentence_gap": 0.6,
                    "force_preprocess": True, "force_preprocess_video": True, "force_preprocess_batch": True
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
                             vad_min_speech_frame, vad_max_speech_frame, vad_min_silence_frame, vad_speech_threshold, vad_smooth_window_size,
                             enable_duration, merge_max_duration, enable_charcount, merge_max_chars,
                             enable_punc_merge, merge_punctuations, enable_silence, merge_silence_threshold,
                             sanitize_gap,
                             enable_min_sentence_merge, min_sentence_chars, min_sentence_gap,
                             force_preprocess_check, force_preprocess_video, force_preprocess_batch]
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