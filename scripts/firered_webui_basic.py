#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
FireRedASR2S WebUI 融合版 

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
import uuid
import re
import subprocess
import shutil
import zipfile
from pathlib import Path
from datetime import timedelta
from collections import Counter

# ==================== 日志设置 ====================
LOG_DIR = Path(__file__).parent.parent / "logs"
LOG_DIR.mkdir(exist_ok=True)

def clean_old_logs(days=7):
    cutoff = time.time() - days * 24 * 3600
    for f in LOG_DIR.glob("error_*.log"):
        if f.stat().st_mtime < cutoff:
            try:
                f.unlink()
            except Exception:
                pass

clean_old_logs()
log_file = LOG_DIR / f"error_{time.strftime('%Y%m%d')}.log"

console_handler = logging.StreamHandler(sys.stderr)
console_handler.setLevel(logging.WARNING)
file_handler = logging.FileHandler(log_file, encoding='utf-8')
file_handler.setLevel(logging.INFO)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[console_handler, file_handler]
)

# ==================== 路径设置 ====================
CURRENT_DIR = Path(__file__).parent.absolute()
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
ROOT_DIR = PROJECT_ROOT
DEFAULT_OUTPUT_DIR = ROOT_DIR / "output"
OUTPUT_DIR = DEFAULT_OUTPUT_DIR
ALIGN_OUTPUT_DIR = OUTPUT_DIR / "字幕自动打轴"
ALIGN_OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

CACHE_DIR = ROOT_DIR / "cache"
CACHE_DIR.mkdir(exist_ok=True)

PRESET_DIR = ROOT_DIR / "preset"
PRESET_DIR.mkdir(exist_ok=True)
CONFIG_FILE = PRESET_DIR / "settings.json"

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
    with config_lock:
        if CONFIG_FILE.exists():
            try:
                with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception:
                return {}
        return {}

def save_settings(settings):
    with config_lock:
        try:
            with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
                json.dump(settings, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logging.error(f"保存配置失败: {e}")

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

    def _find_vad_model_dir(self):
        vad_base = ROOT_DIR / "pretrained_models" / "FireRedVAD"
        if not vad_base.exists():
            logging.error(f"VAD 根目录不存在: {vad_base}")
            return None
        patterns = ["*.pt", "*.pth", "*.pth.tar", "*.onnx", "*.bin", "*.tar"]
        for sub in ["VAD", "Stream-VAD", "AED", "vad", "stream-vad", "aed"]:
            sub_path = vad_base / sub
            if sub_path.is_dir():
                for pat in patterns:
                    if list(sub_path.glob(pat)):
                        logging.info(f"✅ 找到 VAD 模型目录: {sub_path}")
                        return str(sub_path)
        for root, dirs, files in os.walk(vad_base):
            for pat in patterns:
                if list(Path(root).glob(pat)):
                    logging.info(f"✅ 递归找到 VAD 模型目录: {root}")
                    return root
        logging.error("❌ 未找到任何 VAD 模型文件")
        return None

    def _find_model_dir(self, model_type="AED", user_dir=None):
        if user_dir and Path(user_dir).exists():
            return str(user_dir)
        model_type = model_type.upper()
        candidates = [
            ROOT_DIR / "pretrained_models" / f"FireRedASR2-{model_type}",
            ROOT_DIR / "pretrained_models" / f"FireRedASR2-{model_type}-2025",
        ]
        for p in candidates:
            if p.exists():
                print(f"✅ 自动检测到模型目录: {p}")
                return str(p)
        return None

    def load_system(self, config_dict=None, advanced_params=None):
        with self.lock:
            try:
                default_config = {
                    "use_gpu": torch.cuda.is_available(),
                    "use_half": False,
                    "enable_vad": True,
                    "enable_lid": True,
                    "enable_punc": True,
                    "asr_model_type": "aed",
                    "model_dir": None,
                }
                if config_dict:
                    default_config.update(config_dict)

                # CPU 下强制关闭半精度
                if default_config["use_half"] and not default_config["use_gpu"]:
                    logging.warning("use_half 仅在 GPU 模式下有效，已自动关闭")
                    default_config["use_half"] = False
                if default_config["use_gpu"] and not torch.cuda.is_available():
                    return False, "已勾选 GPU 但当前 CUDA 不可用，请取消勾选或检查驱动"

                model_dir = self._find_model_dir(default_config['asr_model_type'].upper(),
                                                 user_dir=default_config.get("model_dir"))
                if not model_dir or not Path(model_dir).exists():
                    return False, f"模型目录不存在（{default_config['asr_model_type']}），请放入 pretrained_models/FireRedASR2-{default_config['asr_model_type'].upper()} 或指定 model_dir"

                vad_model_dir = self._find_vad_model_dir()
                if not vad_model_dir:
                    return False, "VAD 模型目录未找到，请检查 pretrained_models/FireRedVAD 中是否有模型文件"

                lid_model_dir = str(ROOT_DIR / "pretrained_models" / "FireRedLID")
                if not os.path.exists(lid_model_dir):
                    return False, "LID 模型目录不存在"

                punc_model_dir = str(ROOT_DIR / "pretrained_models" / "FireRedPunc")
                if not os.path.exists(punc_model_dir):
                    return False, "标点模型目录不存在"

                vad_config = FireRedVadConfig(use_gpu=default_config["use_gpu"])
                lid_config = FireRedLidConfig(use_gpu=default_config["use_gpu"])
                asr_config = FireRedAsr2Config(
                    use_gpu=default_config["use_gpu"],
                    use_half=default_config["use_half"],
                    return_timestamp=True
                )
                punc_config = FireRedPuncConfig(use_gpu=default_config["use_gpu"])

                if advanced_params:
                    for k, v in advanced_params.items():
                        if k == "beam_size": asr_config.beam_size = v
                        elif k == "nbest": asr_config.nbest = v
                        elif k == "decode_max_len": asr_config.decode_max_len = v
                        elif k == "softmax_smoothing": asr_config.softmax_smoothing = v
                        elif k == "aed_length_penalty": asr_config.aed_length_penalty = v
                        elif k == "eos_penalty": asr_config.eos_penalty = v
                        elif k == "elm_weight": asr_config.elm_weight = v
                        elif k == "vad_min_speech_frame": vad_config.min_speech_frame = v
                        elif k == "vad_max_speech_frame": vad_config.max_speech_frame = v
                        elif k == "vad_min_silence_frame": vad_config.min_silence_frame = v
                        elif k == "vad_speech_threshold": vad_config.speech_threshold = v
                        elif k == "vad_smooth_window_size": vad_config.smooth_window_size = v
                        elif k == "punc_threshold":
                            try: punc_config.threshold = v
                            except: pass

                system_config = FireRedAsr2SystemConfig(
                    vad_model_dir=vad_model_dir,
                    lid_model_dir=lid_model_dir,
                    asr_model_dir=str(model_dir),
                    punc_model_dir=punc_model_dir,
                    vad_config=vad_config,
                    lid_config=lid_config,
                    asr_config=asr_config,
                    punc_config=punc_config,
                    enable_vad=int(default_config["enable_vad"]),
                    enable_lid=int(default_config["enable_lid"]),
                    enable_punc=int(default_config["enable_punc"])
                )

                # 先构造新系统，成功后再替换旧系统
                new_system = FireRedAsr2System(system_config)
                if self.asr_system is not None:
                    self.unload_system()
                self.asr_system = new_system
                self.config = {**default_config, "advanced": (advanced_params or {})}
                return True, f"系统加载成功 (ASR: {default_config['asr_model_type']})"
            except Exception as e:
                logging.error(traceback.format_exc())
                return False, f"加载失败: {str(e)}"

    def unload_system(self):
        with self.lock:
            if self.asr_system:
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
                try: os.unlink(audio_path)
                except: pass
            return None, None, f"识别失败: {str(e)}"

    def _prepare_audio(self, audio_input, force_preprocess=True, return_waveform=False):
        try:
            input_path = None
            need_transcode = True

            if isinstance(audio_input, tuple):
                sr, data = audio_input
                if data.ndim > 1:
                    data = np.mean(data, axis=1)
                # 归一化：确保音频数据在[-1,1]范围内
                if np.issubdtype(data.dtype, np.integer) or np.max(np.abs(data)) > 1.0:
                    orig_dtype = data.dtype
                    data = data.astype(np.float32)
                    if np.issubdtype(orig_dtype, np.integer):
                        max_val = float(np.iinfo(orig_dtype).max)
                    else:
                        max_val = np.max(np.abs(data))
                    data /= max_val
                else:
                    data = data.astype(np.float32)
                input_path = CACHE_DIR / f"input_{uuid.uuid4().hex}_{int(time.time())}.wav"
                sf.write(str(input_path), data, sr)
                self.temp_files.append(str(input_path))
                need_transcode = True
            elif isinstance(audio_input, str) and os.path.exists(audio_input):
                input_path = audio_input
                if not force_preprocess:
                    try:
                        info = sf.info(input_path)
                        if info.samplerate == 16000 and info.channels == 1 and info.subtype == 'PCM_16':
                            need_transcode = False
                    except:
                        pass
                else:
                    need_transcode = True
            else:
                return None

            if not need_transcode:
                if return_waveform:
                    data, sr = sf.read(input_path, dtype='float32')
                    return input_path, data, sr
                return input_path

            out_path = CACHE_DIR / f"temp_audio_{uuid.uuid4().hex}_{int(time.time())}.wav"
            cmd = [
                FFMPEG_PATH, "-y", "-i", str(input_path),
                "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", str(out_path)
            ]
            try:
                subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            except Exception as e:
                logging.warning(f"FFmpeg 预处理失败，回退到 librosa: {e}")
                data, sr = librosa.load(input_path, sr=16000, mono=True)
                sf.write(str(out_path), data.astype(np.float32), 16000)
            self.temp_files.append(str(out_path))

            # 清理元组输入产生的临时文件
            if isinstance(audio_input, tuple) and str(input_path) in self.temp_files:
                self.temp_files.remove(str(input_path))
                try:
                    os.unlink(str(input_path))
                except:
                    pass

            if return_waveform:
                data, sr = sf.read(str(out_path), dtype='float32')
                return str(out_path), data, sr
            return str(out_path)
        except Exception as e:
            logging.error(f"音频预处理失败: {e}")
            if isinstance(audio_input, tuple) and input_path and str(input_path) in self.temp_files:
                self.temp_files.remove(str(input_path))
                try: os.unlink(str(input_path))
                except: pass
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
            except Exception: pass
        self.temp_files = []
        return cleaned

    def _seconds_to_srt_time(self, seconds):
        seconds = max(0.0, float(seconds))
        td = timedelta(seconds=seconds)
        total_seconds = int(td.total_seconds())
        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60
        secs = total_seconds % 60
        ms = int((td.total_seconds() - total_seconds) * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{ms:03d}"

    def force_align(self, audio_input, reference_text, force_preprocess=True):
        if self.asr_system is None:
            return None, None, None, None, None, "模型未加载"
        if self.config['asr_model_type'] != 'aed':
            return None, None, None, None, None, "强制对齐仅支持 AED 模型"
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
                    return None, None, None, None, None, "音频过短"
                frame_shift = duration / T

            tokens, token_ids = asr.tokenizer.tokenize(reference_text)
            if len(token_ids) == 0:
                return None, None, None, None, None, "参考文本为空"
            yseq = torch.tensor(token_ids, device=enc_outputs.device)
            hyps = [[{"yseq": yseq}]]
            nbest_hyps = asr.model.get_token_timestamp_torchaudio(enc_outputs, enc_lengths, hyps)
            timestamp = nbest_hyps[0][0].get("timestamp")
            if timestamp is None:
                return None, None, None, None, None, "时间戳为空"
            starts, ends = timestamp
            if len(starts) == 0:
                return None, None, None, None, None, "时间戳为空"

            # ===== 时间戳单位判断（增强） =====
            max_start = max(starts)
            # 尝试帧索引假设
            frame_scale = max_start * frame_shift
            # 尝试秒假设
            sec_scale = max_start
            # 尝试毫秒假设
            ms_scale = max_start / 1000.0

            candidates = []
            if frame_shift > 0 and frame_scale <= duration * 1.5:
                candidates.append(('frame', [(s * frame_shift, e * frame_shift) for s, e in zip(starts, ends)]))
            if sec_scale <= duration * 1.5:
                candidates.append(('second', list(zip(starts, ends))))
            if ms_scale <= duration * 1.5:
                candidates.append(('millisecond', [(s / 1000.0, e / 1000.0) for s, e in zip(starts, ends)]))

            # 选择最佳候选：结束时间戳单调递增且长度不超过音频，覆盖率最高
            best_ts = None
            best_score = -1
            for unit, ts in candidates:
                # 过滤异常
                if any(e < s for s, e in ts):
                    continue
                if ts[-1][1] > duration * 1.5 + 1.0:
                    continue
                # 计算覆盖率
                covered = sum(e - s for s, e in ts)
                coverage = min(covered / duration, 1.0)
                # 检查单调性
                monotonic = all(ts[i][1] <= ts[i+1][0] + 0.1 for i in range(len(ts)-1))
                if not monotonic:
                    continue
                score = coverage
                if score > best_score:
                    best_score = score
                    best_ts = ts
                    best_unit = unit

            if best_ts is None:
                return None, None, None, None, None, "时间戳单位无法确定或结果不合理"
            timestamps_sec = best_ts
            logging.info(f"时间戳单位判断为: {best_unit}")

            # 最终 sanity check
            if any(e < s for s, e in timestamps_sec):
                return None, None, None, None, None, "时间戳异常：存在 end < start"
            if timestamps_sec[-1][1] > duration * 1.2 + 1.0:
                return None, None, None, None, None, f"时间戳异常：末尾超出音频时长"

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
                try: os.unlink(audio_path)
                except: pass

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

    if sentences:
        sent_segments = []
        for s in sentences:
            sent_segments.append({
                "start": s.get("start_ms", 0) / 1000.0,
                "end": s.get("end_ms", 0) / 1000.0,
                "text": s.get("text", "")
            })
    else:
        sent_segments = word_segments[:] if word_segments else []

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

    extra = f"VAD段: {len(vad_segments)}"
    if sentences and sentences[0].get("asr_confidence") is not None:
        extra += f" | 置信度: {sentences[0]['asr_confidence']:.3f}"
    # 语种聚合：取句子中出现次数最多的语种及其平均置信度，并防止除零
    langs = [s.get("lang") for s in sentences if s.get("lang")]
    if langs:
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

def inject_punctuation_to_words(word_segments, full_text_with_punc, punctuation_chars="。！？.!?"):
    """
    将带标点的全文中的标点符号，精确附加到对应词的末尾。
    支持中英文混合及数字等非标点字符。
    """
    if not word_segments or not full_text_with_punc:
        return word_segments
    punct_set = set(punctuation_chars)
    # 预先计算每个词的原始字符区间（不含后续添加的标点）
    spans = []
    total = 0
    for seg in word_segments:
        start = total
        total += len(seg['text'])
        spans.append((start, total))
    # 遍历全文，仅对非标点字符推进索引；标点则附加到最近一个词
    last_word_idx = -1
    char_idx = 0
    word_ptr = 0
    for ch in full_text_with_punc:
        if ch in punct_set:
            if last_word_idx >= 0:
                word_segments[last_word_idx]['text'] += ch
        elif not ch.isspace():  # 任何非空白、非标点字符都视为有效字符（汉字、英文、数字等）
            if char_idx >= total:
                break
            # 找到该字符所属的词
            while word_ptr < len(spans) and char_idx >= spans[word_ptr][1]:
                word_ptr += 1
            if word_ptr < len(spans) and spans[word_ptr][0] <= char_idx < spans[word_ptr][1]:
                last_word_idx = word_ptr
            char_idx += 1
        # 空白字符忽略
    return word_segments

def truncate_long_text(text, max_chars=8000):
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n\n... [内容过长已截断，完整内容已保存至文件]"

def truncate_long_srt(srt_text, max_lines=400):
    lines = srt_text.splitlines()
    if len(lines) <= max_lines:
        return srt_text
    return "\n".join(lines[:max_lines]) + "\n\n... [字幕过长已截断，完整文件已保存]"

def truncate_long_json(json_str, max_items=300, max_chars=10000):
    try:
        data = json.loads(json_str)
        if isinstance(data, list) and len(data) > max_items:
            truncated = json.dumps(data[:max_items], ensure_ascii=False, indent=2)
            return truncated + f"\n\n... [截断至前{max_items}条]"
        if isinstance(data, dict) and len(json_str) > max_chars:
            return json_str[:max_chars] + "\n... [JSON过长已截断]"
    except:
        if len(json_str) > max_chars:
            return json_str[:max_chars] + "\n... [JSON过长已截断]"
    return json_str

def truncate_all_for_display(full_text, word_json, sent_json, srt_text):
    return (
        truncate_long_text(full_text),
        truncate_long_json(word_json, max_items=300),
        truncate_long_json(sent_json, max_items=300),
        truncate_long_srt(srt_text, max_lines=400)
    )

def merge_timestamps_to_sentences(timestamps, words,
                                   sentence_endings="。！？.!?",
                                   max_words=20, max_chars=50, max_duration=10.0,
                                   silence_threshold=0.3,
                                   merge_by_punc=True, merge_by_silence=True,
                                   merge_by_wordcount=True, merge_by_charcount=True,
                                   merge_by_duration=True, force_break_indices=None):
    """
    将 token 级时间戳合并为句子，支持多种断句规则。
    修复点：
    - 静音断句后新句子开始，后续条件不应立即再次断句。
    - 时长计算使用 end - current_start，避免包含词间静音。
    """
    if not timestamps or not words:
        return []
    has_cjk = any(re.search(r'[\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af]', w) for w in words)
    join_str = "" if has_cjk else " "
    sentences = []
    current_start = timestamps[0][0]
    current_words = []
    last_end = timestamps[0][1]
    force_break_indices = force_break_indices or [False] * len(words)

    for i, ((start, end), word) in enumerate(zip(timestamps, words)):
        # 检查强制断句（最高优先级）
        if force_break_indices[i]:
            if current_words:
                sentences.append({
                    "start": current_start,
                    "end": last_end,
                    "text": join_str.join(current_words).strip()
                })
            current_start = start
            current_words = []
            last_end = end
            current_words.append(word)
            last_end = end
            # 强制断句后该词已经作为新句子开头，不再检查其他条件
            continue

        # 静音断句：当前词与上一词之间的间隔大于阈值
        if merge_by_silence and i > 0 and start - last_end > silence_threshold:
            if current_words:
                sentences.append({
                    "start": current_start,
                    "end": last_end,
                    "text": join_str.join(current_words).strip()
                })
                current_words = []
                current_start = None

        # 如果 current_words 为空，说明新句子刚开始，设置起始时间
        if not current_words:
            current_start = start

        # 添加当前词
        current_words.append(word)
        last_end = end

        # 检查其他断句条件（仅在当前句子至少有一个词后检查）
        should_break = False
        # 标点断句
        if merge_by_punc and any(word.endswith(p) for p in sentence_endings):
            should_break = True
        # 词数限制（当前词加入后总词数 >= max_words）
        if not should_break and merge_by_wordcount and len(current_words) >= max_words:
            should_break = True
        # 字符数限制
        if not should_break and merge_by_charcount:
            new_text = join_str.join(current_words)
            if len(new_text) >= max_chars:
                should_break = True
        # 时长限制（使用句子首尾时间差）
        if not should_break and merge_by_duration and (end - current_start) >= max_duration:
            should_break = True

        if should_break:
            sentences.append({
                "start": current_start,
                "end": last_end,
                "text": join_str.join(current_words).strip()
            })
            current_words = []
            current_start = None

    # 处理剩余词
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

def save_outputs(base_name, full_text, sent_json, srt_text, language, model_info):
    # 增加毫秒级时间戳和随机短码，避免同一秒内冲突
    timestamp = time.strftime("%Y%m%d_%H%M%S") + f"_{int(time.time()*1000)%1000:03d}"
    safe = re.sub(r'[^\w\u4e00-\u9fff\-\.]', '', Path(base_name).stem) if base_name else "firered"
    prefix = f"{safe}_{timestamp}"
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
        if manager.asr_system and manager.config:
            info.append(f"ASR系统: 已加载 ({manager.config['asr_model_type']})")
            info.append(f"  VAD: {'启用' if manager.config['enable_vad'] else '禁用'}")
            info.append(f"  LID: {'启用' if manager.config['enable_lid'] else '禁用'}")
            info.append(f"  Punc: {'启用' if manager.config['enable_punc'] else '禁用'}")
        else:
            info.append("ASR系统: 未加载")
    info.append(f"输出目录: {OUTPUT_DIR}")
    info.append(f"字幕打轴输出: {ALIGN_OUTPUT_DIR}")
    info.append(f"缓存目录: {CACHE_DIR}")
    info.append(f"日志文件: {log_file}")
    return "\n".join(info)

def ensure_model_loaded(config_params, advanced_params):
    with manager.lock:
        need_reload = manager.asr_system is None
        if not need_reload and manager.config:
            for k, v in config_params.items():
                if manager.config.get(k) != v:
                    need_reload = True
                    break
            if not need_reload:
                saved_adv = manager.config.get('advanced', {})
                for k, v in advanced_params.items():
                    if saved_adv.get(k) != v:
                        need_reload = True
                        break
        if need_reload:
            success, msg = manager.load_system(config_params, advanced_params)
            if not success:
                raise RuntimeError(f"加载失败: {msg}")
    return manager

# ==================== 音频识别（四输出）====================
def transcribe_audio(audio_file, asr_model_type, use_gpu, use_half, enable_vad, enable_lid, enable_punc,
                     beam_size, nbest, decode_max_len, softmax_smoothing, aed_length_penalty,
                     eos_penalty, elm_weight,
                     vad_min_speech_frame, vad_max_speech_frame, vad_min_silence_frame,
                     vad_speech_threshold, vad_smooth_window_size,
                     punc_threshold,
                     audio_enable_duration, audio_merge_max_duration,
                     audio_enable_charcount, audio_merge_max_chars,
                     audio_enable_punc, audio_merge_punctuations,
                     audio_enable_silence, audio_merge_silence_threshold,
                     force_preprocess, model_dir,
                     progress=gr.Progress()):
    if not FIRERED_AVAILABLE:
        return "错误: FireRedASR2S 模块不可用", "", "", "", None
    if not audio_file or (isinstance(audio_file, str) and not os.path.exists(audio_file)):
        return "请上传有效的音频文件", "", "", "", None

    progress(0, desc="初始化...")
    config = {
        "use_gpu": use_gpu, "use_half": use_half,
        "enable_vad": enable_vad, "enable_lid": enable_lid, "enable_punc": enable_punc,
        "asr_model_type": asr_model_type,
        "model_dir": model_dir if model_dir and model_dir.strip() else None
    }
    advanced = {
        "beam_size": beam_size, "nbest": nbest, "decode_max_len": decode_max_len,
        "softmax_smoothing": softmax_smoothing, "aed_length_penalty": aed_length_penalty,
        "eos_penalty": eos_penalty, "elm_weight": elm_weight,
        "vad_min_speech_frame": vad_min_speech_frame, "vad_max_speech_frame": vad_max_speech_frame,
        "vad_min_silence_frame": vad_min_silence_frame, "vad_speech_threshold": vad_speech_threshold,
        "vad_smooth_window_size": vad_smooth_window_size, "punc_threshold": punc_threshold,
    }
    try:
        ensure_model_loaded(config, advanced)
    except RuntimeError as e:
        return str(e), "", "", "", None

    progress(0.3, desc="识别中...")
    result, _, error = manager.transcribe(audio_file, force_preprocess=force_preprocess)
    if error:
        return f"错误: {error}", "", "", "", None

    progress(0.7, desc="生成输出...")
    full_text, sent_json, srt_text, word_segments, word_json = format_result_to_outputs(result)

    # 标点注入
    if word_segments and result.get("text"):
        word_segments = inject_punctuation_to_words(word_segments, result["text"], audio_merge_punctuations)
        word_json = json.dumps(word_segments, ensure_ascii=False, indent=2)

    if word_segments:
        ts_data = [(s["start"], s["end"]) for s in word_segments]
        texts = [s["text"] for s in word_segments]
        merged = merge_timestamps_to_sentences(
            ts_data, texts,
            sentence_endings=audio_merge_punctuations,
            max_chars=audio_merge_max_chars, max_duration=audio_merge_max_duration,
            silence_threshold=audio_merge_silence_threshold,
            merge_by_punc=audio_enable_punc, merge_by_silence=audio_enable_silence,
            merge_by_wordcount=False, merge_by_charcount=audio_enable_charcount,
            merge_by_duration=audio_enable_duration
        )
        srt_text = sentences_to_srt(merged)
        sent_json = json.dumps(merged, ensure_ascii=False, indent=2)

    base_name = audio_file if isinstance(audio_file, str) and os.path.exists(audio_file) else None
    saved, prefix = save_outputs(base_name, full_text, sent_json, srt_text, "自动检测", asr_model_type)
    if word_segments:
        word_json_path = OUTPUT_DIR / f"{prefix}_words.json"
        with open(word_json_path, 'w', encoding='utf-8') as f:
            f.write(word_json)
        saved['word_json'] = str(word_json_path)

    save_info = "文件已保存:\n"
    for k, v in saved.items():
        save_info += f"  {Path(v).name}\n"
    full_text_disp = save_info + "\n" + full_text

    disp_text, disp_word, disp_sent, disp_srt = truncate_all_for_display(
        full_text_disp, word_json, sent_json, srt_text
    )

    progress(0.9, desc="清理...")
    manager.cleanup_temp()
    progress(1.0, desc="完成")
    return disp_text, disp_word, disp_sent, disp_srt, list(saved.values())

# ==================== 视频字幕（四输出）====================
def transcribe_video(video, asr_model_type, use_gpu, use_half, enable_vad, enable_lid, enable_punc,
                     beam_size, nbest, decode_max_len, softmax_smoothing, aed_length_penalty,
                     eos_penalty, elm_weight,
                     vad_min_speech_frame, vad_max_speech_frame, vad_min_silence_frame,
                     vad_speech_threshold, vad_smooth_window_size,
                     punc_threshold,
                     v_enable_duration, v_merge_max_duration,
                     v_enable_charcount, v_merge_max_chars,
                     v_enable_punc, v_merge_punctuations,
                     v_enable_silence, v_merge_silence_threshold,
                     force_preprocess, model_dir,
                     progress=gr.Progress()):
    temp_audio_path = None
    try:
        if not FIRERED_AVAILABLE:
            return "错误: FireRedASR2S 模块不可用", "", "", "", None
        if video is None:
            return "请上传视频文件", "", "", "", None

        progress(0, desc="初始化...")
        config = {
            "use_gpu": use_gpu, "use_half": use_half,
            "enable_vad": enable_vad, "enable_lid": enable_lid, "enable_punc": enable_punc,
            "asr_model_type": asr_model_type,
            "model_dir": model_dir if model_dir and model_dir.strip() else None
        }
        advanced = {
            "beam_size": beam_size, "nbest": nbest, "decode_max_len": decode_max_len,
            "softmax_smoothing": softmax_smoothing, "aed_length_penalty": aed_length_penalty,
            "eos_penalty": eos_penalty, "elm_weight": elm_weight,
            "vad_min_speech_frame": vad_min_speech_frame, "vad_max_speech_frame": vad_max_speech_frame,
            "vad_min_silence_frame": vad_min_silence_frame, "vad_speech_threshold": vad_speech_threshold,
            "vad_smooth_window_size": vad_smooth_window_size, "punc_threshold": punc_threshold,
        }
        try:
            ensure_model_loaded(config, advanced)
        except RuntimeError as e:
            return str(e), "", "", "", None

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
            return f"音频提取失败: {e.stderr}", "", "", "", None

        progress(0.4, desc="识别音频...")
        result, _, error = manager.transcribe(temp_audio_path, force_preprocess=force_preprocess)
        if error:
            return f"识别失败: {error}", "", "", "", None

        progress(0.6, desc="生成字幕...")
        full_text, sent_json, srt_text, word_segments, word_json = format_result_to_outputs(result)

        if word_segments and result.get("text"):
            word_segments = inject_punctuation_to_words(word_segments, result["text"], v_merge_punctuations)
            word_json = json.dumps(word_segments, ensure_ascii=False, indent=2)

        if word_segments:
            ts_data = [(s["start"], s["end"]) for s in word_segments]
            texts = [s["text"] for s in word_segments]
            merged = merge_timestamps_to_sentences(
                ts_data, texts,
                sentence_endings=v_merge_punctuations,
                max_chars=v_merge_max_chars, max_duration=v_merge_max_duration,
                silence_threshold=v_merge_silence_threshold,
                merge_by_punc=v_enable_punc, merge_by_silence=v_enable_silence,
                merge_by_wordcount=False, merge_by_charcount=v_enable_charcount,
                merge_by_duration=v_enable_duration
            )
            srt_text = sentences_to_srt(merged)
            sent_json = json.dumps(merged, ensure_ascii=False, indent=2)

        base_name = video if isinstance(video, str) and os.path.exists(video) else None
        saved, prefix = save_outputs(base_name, full_text, sent_json, srt_text, "自动检测", asr_model_type)
        if word_segments:
            word_json_path = OUTPUT_DIR / f"{prefix}_words.json"
            with open(word_json_path, 'w', encoding='utf-8') as f:
                f.write(word_json)
            saved['word_json'] = str(word_json_path)

        save_info = "文件已保存:\n"
        for k, v in saved.items():
            save_info += f"  {Path(v).name}\n"
        combined_text = f"音频识别完成！字幕文件已生成。\n{save_info}\n\n【识别文本】\n{full_text}"

        disp_text, disp_word, disp_sent, disp_srt = truncate_all_for_display(
            combined_text, word_json, sent_json, srt_text
        )

        progress(0.9, desc="清理...")
        manager.cleanup_temp()
        progress(1.0, desc="完成")
        return disp_text, disp_word, disp_sent, disp_srt, list(saved.values())
    except Exception as e:
        logging.error(traceback.format_exc())
        return f"处理视频时发生未知错误: {str(e)}", "", "", "", None
    finally:
        if temp_audio_path and os.path.exists(temp_audio_path):
            try: os.unlink(temp_audio_path)
            except: pass

# ==================== 批量处理（四输出）====================
def transcribe_batch(files, asr_model_type, use_gpu, use_half, enable_vad, enable_lid, enable_punc,
                     beam_size, nbest, decode_max_len, softmax_smoothing, aed_length_penalty,
                     eos_penalty, elm_weight,
                     vad_min_speech_frame, vad_max_speech_frame, vad_min_silence_frame,
                     vad_speech_threshold, vad_smooth_window_size,
                     punc_threshold, force_preprocess, model_dir,
                     batch_enable_duration, batch_merge_max_duration,
                     batch_enable_charcount, batch_merge_max_chars,
                     batch_enable_punc, batch_merge_punctuations,
                     batch_enable_silence, batch_merge_silence_threshold,
                     progress=gr.Progress()):
    if not files:
        return "请选择音频文件", None
    config = {
        "use_gpu": use_gpu, "use_half": use_half,
        "enable_vad": enable_vad, "enable_lid": enable_lid, "enable_punc": enable_punc,
        "asr_model_type": asr_model_type,
        "model_dir": model_dir if model_dir and model_dir.strip() else None
    }
    advanced = {
        "beam_size": beam_size, "nbest": nbest, "decode_max_len": decode_max_len,
        "softmax_smoothing": softmax_smoothing, "aed_length_penalty": aed_length_penalty,
        "eos_penalty": eos_penalty, "elm_weight": elm_weight,
        "vad_min_speech_frame": vad_min_speech_frame, "vad_max_speech_frame": vad_max_speech_frame,
        "vad_min_silence_frame": vad_min_silence_frame, "vad_speech_threshold": vad_speech_threshold,
        "vad_smooth_window_size": vad_smooth_window_size, "punc_threshold": punc_threshold,
    }
    try:
        ensure_model_loaded(config, advanced)
    except RuntimeError as e:
        return str(e), None

    results_summary = []
    all_saved = []
    total = len(files)
    for i, file_obj in enumerate(files, 1):
        file_path = file_obj.name if hasattr(file_obj, 'name') else str(file_obj)
        progress(i/total, desc=f"处理 {i}/{total}: {os.path.basename(file_path)}")
        try:
            result, _, error = manager.transcribe(file_path, force_preprocess=force_preprocess)
            if error:
                results_summary.append(f"【{os.path.basename(file_path)}】错误: {error}")
            else:
                full_text, sent_json, srt_text, word_segments, word_json = format_result_to_outputs(result)
                if word_segments and result.get("text"):
                    word_segments = inject_punctuation_to_words(word_segments, result["text"], batch_merge_punctuations)
                    word_json = json.dumps(word_segments, ensure_ascii=False, indent=2)

                if word_segments:
                    ts_data = [(s["start"], s["end"]) for s in word_segments]
                    texts = [s["text"] for s in word_segments]
                    merged = merge_timestamps_to_sentences(
                        ts_data, texts,
                        sentence_endings=batch_merge_punctuations,
                        max_chars=batch_merge_max_chars, max_duration=batch_merge_max_duration,
                        silence_threshold=batch_merge_silence_threshold,
                        merge_by_punc=batch_enable_punc, merge_by_silence=batch_enable_silence,
                        merge_by_wordcount=False, merge_by_charcount=batch_enable_charcount,
                        merge_by_duration=batch_enable_duration
                    )
                    srt_text = sentences_to_srt(merged)
                    sent_json = json.dumps(merged, ensure_ascii=False, indent=2)

                saved, prefix = save_outputs(file_path, full_text, sent_json, srt_text, "自动检测", asr_model_type)
                if word_segments:
                    word_json_path = OUTPUT_DIR / f"{prefix}_words.json"
                    with open(word_json_path, 'w', encoding='utf-8') as f:
                        f.write(word_json)
                    saved['word_json'] = str(word_json_path)
                saved_files = [Path(v).name for v in saved.values() if v]
                all_saved.extend(saved.values())
                results_summary.append(f"【{os.path.basename(file_path)}】已保存: {', '.join(saved_files)}")
        except Exception as e:
            logging.error(traceback.format_exc())
            results_summary.append(f"【{os.path.basename(file_path)}】处理异常: {e}")
        finally:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    progress(1.0, desc="完成")
    manager.cleanup_temp()
    summary = f"批量处理完成，共 {total} 个文件。\n" + "\n".join(results_summary)
    summary += f"\n\n所有结果文件已保存至: {OUTPUT_DIR}"

    zip_path = None
    if all_saved:
        try:
            zip_path = OUTPUT_DIR / f"batch_{time.strftime('%Y%m%d_%H%M%S')}.zip"
            with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
                for fp in all_saved:
                    zf.write(fp, arcname=Path(fp).name)
            summary += f"\n打包文件: {zip_path.name}"
        except Exception as e:
            logging.error(f"批量打包失败: {e}")
            summary += f"\n[打包失败: {e}]"
            zip_path = None

    if len(summary) > 10000:
        summary = summary[:10000] + "\n\n... [批量结果过长，已截断]"
    return summary, str(zip_path) if zip_path else None

def load_model_click(asr_model_type, use_gpu, use_half, enable_vad, enable_lid, enable_punc,
                     beam_size, nbest, decode_max_len, softmax_smoothing, aed_length_penalty,
                     eos_penalty, elm_weight,
                     vad_min_speech_frame, vad_max_speech_frame, vad_min_silence_frame,
                     vad_speech_threshold, vad_smooth_window_size,
                     punc_threshold, model_dir):
    config = {
        "use_gpu": use_gpu, "use_half": use_half,
        "enable_vad": enable_vad, "enable_lid": enable_lid, "enable_punc": enable_punc,
        "asr_model_type": asr_model_type,
        "model_dir": model_dir if model_dir and model_dir.strip() else None
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

# ==================== 强制对齐包装（含字级JSON保存与合并）====================
def force_align_wrapper(audio, text, asr_model_type, use_gpu, use_half,
                        enable_vad, enable_lid, enable_punc,
                        beam_size, nbest, decode_max_len,
                        softmax_smoothing, aed_length_penalty,
                        eos_penalty, elm_weight,
                        vad_min_speech_frame, vad_max_speech_frame,
                        vad_min_silence_frame, vad_speech_threshold,
                        vad_smooth_window_size, punc_threshold, model_dir,
                        merge_punctuations, align_max_words, align_max_chars, align_max_duration,
                        align_silence_threshold,
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
    config = {
        "use_gpu": use_gpu, "use_half": use_half,
        "enable_vad": enable_vad, "enable_lid": enable_lid, "enable_punc": enable_punc,
        "asr_model_type": asr_model_type,
        "model_dir": model_dir if model_dir and model_dir.strip() else None
    }
    advanced = {
        "beam_size": beam_size, "nbest": nbest, "decode_max_len": decode_max_len,
        "softmax_smoothing": softmax_smoothing, "aed_length_penalty": aed_length_penalty,
        "eos_penalty": eos_penalty, "elm_weight": elm_weight,
        "vad_min_speech_frame": vad_min_speech_frame, "vad_max_speech_frame": vad_max_speech_frame,
        "vad_min_silence_frame": vad_min_silence_frame, "vad_speech_threshold": vad_speech_threshold,
        "vad_smooth_window_size": vad_smooth_window_size, "punc_threshold": punc_threshold,
    }
    try:
        ensure_model_loaded(config, advanced)
    except RuntimeError as e:
        return str(e), "", ""

    progress(0.3, desc="强制对齐中...")
    word_srt, sent_srt, timestamps, words, token_ids_out, error = manager.force_align(
        audio, text, force_preprocess=force_preprocess)
    if error:
        return f"错误: {error}", "", ""

    word_segments = []
    if timestamps and words:
        for (start, end), w in zip(timestamps, words):
            word_segments.append({"start": start, "end": end, "text": w})
        word_json_str = json.dumps(word_segments, ensure_ascii=False, indent=2)
        timestamp_str = time.strftime("%Y%m%d_%H%M%S")
        prefix = generate_output_filename(audio, timestamp_str, default_name="align")
        word_json_path = ALIGN_OUTPUT_DIR / f"{prefix}_words.json"
        with open(word_json_path, "w", encoding="utf-8") as f:
            f.write(word_json_str)

    force_break = [False] * len(words)
    merge_warnings = []

    # 空行断句
    if merge_by_newline and words and timestamps and token_ids_out:
        asr = manager.asr_system.asr
        paragraphs = [p.strip() for p in text.split('\n') if p.strip()]
        if len(paragraphs) > 1:
            current_pos = 0
            match_success = True
            for para in paragraphs:
                para_tokens, para_ids = asr.tokenizer.tokenize(para)
                if len(para_ids) == 0:
                    continue
                if not match_success:
                    merge_warnings.append(f"警告：由于前一段落未匹配，后续段落均跳过")
                    break
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
                    msg = f"警告：段落 '{para[:30]}...' 无法完全匹配，后续段落可能错位"
                    merge_warnings.append(msg)
                    match_success = False

    # 标点断句
    if merge_by_punc and words and timestamps:
        asr = manager.asr_system.asr
        tokens, token_ids = asr.tokenizer.tokenize(text)
        char_to_token = [-1] * len(text)
        cur = 0
        SPECIAL_PREFIXES = ("<", "[")
        for token_idx, token in enumerate(tokens):
            if token.startswith(SPECIAL_PREFIXES) and token.endswith((">", "]")):
                continue
            token_len = len(token)
            for i in range(token_len):
                if cur + i < len(text):
                    char_to_token[cur + i] = token_idx
            cur += token_len

        punc_positions = [idx for idx, ch in enumerate(text) if ch in merge_punctuations]
        for pos in punc_positions:
            tidx = char_to_token[pos]
            if 0 <= tidx < len(words) - 1:
                force_break[tidx] = True

    merged_srt = ""
    if timestamps and words:
        sentences = merge_timestamps_to_sentences(
            timestamps, words,
            sentence_endings=merge_punctuations,
            max_words=align_max_words,
            max_chars=align_max_chars,
            max_duration=align_max_duration,
            silence_threshold=align_silence_threshold,
            merge_by_punc=False,
            merge_by_silence=merge_by_silence,
            merge_by_wordcount=merge_by_wordcount,
            merge_by_charcount=merge_by_charcount,
            merge_by_duration=merge_by_duration,
            force_break_indices=force_break
        )
        merged_srt = sentences_to_srt(sentences)

        timestamp_str = time.strftime("%Y%m%d_%H%M%S")
        prefix = generate_output_filename(audio, timestamp_str, default_name="align")
        merged_path = ALIGN_OUTPUT_DIR / f"{prefix}_merged_custom.srt"
        with open(merged_path, "w", encoding="utf-8") as f:
            f.write(merged_srt)

    if merge_warnings:
        warning_text = "\n".join(merge_warnings)
        logging.warning(warning_text)
        try:
            gr.Warning(warning_text)
        except:
            pass
        merged_srt += f"\n\n[警告] {warning_text}"

    word_srt = truncate_long_srt(word_srt, max_lines=400) if word_srt else ""
    sent_srt = truncate_long_srt(sent_srt, max_lines=400) if sent_srt else ""
    merged_srt = truncate_long_srt(merged_srt, max_lines=400)

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
        try:
            OUTPUT_DIR = Path(default_output_dir)
            ALIGN_OUTPUT_DIR = OUTPUT_DIR / "字幕自动打轴"
            ALIGN_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            logging.error(f"输出目录不可用({default_output_dir}): {e}，回退默认目录")
            OUTPUT_DIR = DEFAULT_OUTPUT_DIR
            ALIGN_OUTPUT_DIR = OUTPUT_DIR / "字幕自动打轴"
            ALIGN_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    llm_dir = ROOT_DIR / "pretrained_models" / "FireRedASR2-LLM"
    model_choices = ["aed"]
    if llm_dir.exists():
        model_choices.append("llm")

    with gr.Blocks(title="FireRedASR2S WebUI 融合版", theme=gr.themes.Default()) as demo:
        gr.Markdown(f"""
        # FireRedASR2S 语音识别系统 融合版
        输出目录: `{OUTPUT_DIR}` | 打轴目录: `{ALIGN_OUTPUT_DIR}` | 缓存: `{CACHE_DIR}`
        """)

        with gr.Accordion("系统状态信息", open=False):
            with gr.Row():
                status_display = gr.Textbox(label="系统状态", value=get_system_info(), lines=6, interactive=False, scale=4)
                with gr.Column(scale=1):
                    refresh_btn = gr.Button("刷新状态", variant="secondary")
                    health_btn = gr.Button("健康检查", variant="secondary")

        def health_check():
            info = get_system_info()
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
            use_gpu = gr.Checkbox(label="启用 GPU", value=torch.cuda.is_available(), scale=1)
            default_half = torch.cuda.is_available() and torch.cuda.get_device_properties(0).total_memory < 10 * 1024**3
            use_half = gr.Checkbox(label="开启半精度 (FP16)", value=default_half, scale=1)
            enable_vad = gr.Checkbox(label="启用 VAD", value=True, scale=1)
            enable_lid = gr.Checkbox(label="启用 LID", value=True, scale=1)
            enable_punc = gr.Checkbox(label="启用 标点恢复", value=True, scale=1)

        # GPU 与半精度联动
        def on_gpu_change(gpu_val):
            if not gpu_val:
                return gr.update(value=False)  # 取消 GPU 时自动关闭半精度
            return gr.update()
        use_gpu.change(on_gpu_change, inputs=[use_gpu], outputs=[use_half])

        with gr.Accordion("高级参数", open=False):
            model_dir = gr.Textbox(label="ASR 模型目录（可选，留空自动探测）", value="")
            gr.Markdown("### 解码参数")
            with gr.Row():
                beam_size = gr.Slider(1, 10, 3, step=1, label="Beam 大小")
                nbest = gr.Slider(1, 5, 1, step=1, label="候选结果数")
                decode_max_len = gr.Slider(0, 500, 0, step=10, label="最大解码长度")
            with gr.Row():
                softmax_smoothing = gr.Slider(0.5, 2.0, 1.25, label="Softmax 平滑")
                aed_length_penalty = gr.Slider(-2.0, 2.0, 0.6, label="长度惩罚")
                eos_penalty = gr.Slider(0.5, 2.0, 1.0, label="结束符惩罚")
            elm_weight = gr.Slider(0.0, 1.0, 0.0, label="外部语言模型权重")
            gr.Markdown("### VAD 参数")
            with gr.Row():
                vad_speech_threshold = gr.Slider(0.1, 0.9, 0.4, label="语音阈值")
                vad_min_speech_frame = gr.Slider(1, 50, 20, label="最小语音帧数")
            with gr.Row():
                vad_max_speech_frame = gr.Slider(100, 3000, 2000, label="最大语音帧数")
                vad_min_silence_frame = gr.Slider(5, 50, 20, label="最小静音帧数")
            vad_smooth_window_size = gr.Slider(1, 20, 5, label="平滑窗口大小")
            punc_threshold = gr.Slider(0.1, 0.9, 0.45, label="标点阈值")

        load_btn.click(load_model_click,
                       inputs=[asr_model_type, use_gpu, use_half, enable_vad, enable_lid, enable_punc,
                               beam_size, nbest, decode_max_len, softmax_smoothing, aed_length_penalty,
                               eos_penalty, elm_weight,
                               vad_min_speech_frame, vad_max_speech_frame, vad_min_silence_frame,
                               vad_speech_threshold, vad_smooth_window_size, punc_threshold, model_dir],
                       outputs=[load_msg, status_display])
        unload_btn.click(unload_model_click, outputs=[load_msg, status_display])
        refresh_btn.click(refresh_status, outputs=[status_display])

        gr.Markdown("---")

        with gr.Tabs():
            # --- 音频识别（四窗口）---
            with gr.Tab("音频识别"):
                with gr.Row():
                    with gr.Column(scale=1):
                        input_mode = gr.Radio(["文件上传（推荐大文件）", "麦克风/音频组件（小文件）"],
                                             value="文件上传（推荐大文件）", label="输入方式")
                        audio_file = gr.File(file_types=[".wav", ".mp3", ".m4a", ".flac", ".ogg"],
                                             type="filepath", visible=True, label="选择音频文件")
                        audio_mic = gr.Audio(type="filepath", sources=["upload", "microphone"],
                                             visible=False, label="录制或选择音频")
                        audio_preview = gr.Audio(type="filepath", interactive=False, visible=False, label="🎧 音频预览")
                        force_preprocess_audio = gr.Checkbox(label="⚡ 强制预处理为 16kHz 单声道", value=True)

                        def toggle_input(mode):
                            file_vis = (mode == "文件上传（推荐大文件）")
                            mic_vis = not file_vis
                            return gr.update(visible=file_vis), gr.update(visible=mic_vis)
                        input_mode.change(toggle_input, [input_mode], [audio_file, audio_mic])

                        def get_audio_path(mode, file_path, mic_path):
                            return file_path if mode == "文件上传（推荐大文件）" else mic_path
                        audio_path_state = gr.State()

                        with gr.Accordion("字幕合并参数", open=True):
                            with gr.Row():
                                audio_enable_duration = gr.Checkbox(label="启用时长限制", value=False)
                                audio_merge_max_duration = gr.Slider(1.0, 20.0, 10.0, step=0.5, label="单条最大时长 (秒)")
                            with gr.Row():
                                audio_enable_charcount = gr.Checkbox(label="启用字符数限制", value=False)
                                audio_merge_max_chars = gr.Slider(5, 100, 30, step=5, label="单条最大字符数")
                            with gr.Row():
                                audio_enable_silence = gr.Checkbox(label="启用静音阈值分句", value=False)
                                audio_merge_silence_threshold = gr.Slider(0.1, 1.0, 0.3, step=0.05, label="静音阈值 (秒)")
                            with gr.Row():
                                audio_enable_punc = gr.Checkbox(label="启用句末标点分句", value=True)
                                audio_merge_punctuations = gr.Textbox(value="。！？.!?", label="句末标点")

                        with gr.Row():
                            transcribe_btn = gr.Button("开始识别", variant="primary")
                            clear_btn = gr.Button("清空", variant="secondary")

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
                        audio_download = gr.File(label="📦 下载结果文件", file_count="multiple", interactive=False)

                def update_audio_preview(file_path):
                    if file_path and os.path.exists(file_path):
                        return gr.update(value=file_path, visible=True)
                    return gr.update(value=None, visible=False)
                audio_file.change(update_audio_preview, [audio_file], [audio_preview])
                audio_mic.change(update_audio_preview, [audio_mic], [audio_preview])

                transcribe_btn.click(
                    get_audio_path, [input_mode, audio_file, audio_mic], [audio_path_state]
                ).then(
                    transcribe_audio,
                    [audio_path_state, asr_model_type, use_gpu, use_half,
                     enable_vad, enable_lid, enable_punc,
                     beam_size, nbest, decode_max_len, softmax_smoothing, aed_length_penalty,
                     eos_penalty, elm_weight,
                     vad_min_speech_frame, vad_max_speech_frame, vad_min_silence_frame,
                     vad_speech_threshold, vad_smooth_window_size, punc_threshold,
                     audio_enable_duration, audio_merge_max_duration,
                     audio_enable_charcount, audio_merge_max_chars,
                     audio_enable_punc, audio_merge_punctuations,
                     audio_enable_silence, audio_merge_silence_threshold,
                     force_preprocess_audio, model_dir],
                    [text_output, word_json_output, sent_json_output, srt_output, audio_download]
                ).then(refresh_status, outputs=[status_display])

                clear_btn.click(
                    lambda: [None, None, gr.update(value=None, visible=False), True, "", "", "", "", None],
                    [audio_file, audio_mic, audio_preview, force_preprocess_audio,
                     text_output, word_json_output, sent_json_output, srt_output, audio_download]
                )

            # --- 视频字幕（四窗口）---
            with gr.Tab("视频字幕"):
                with gr.Row():
                    with gr.Column(scale=1):
                        video_input = gr.Video(sources=["upload"], label="上传视频文件")
                        force_preprocess_video = gr.Checkbox(
                            label="⚡ 强制二次预处理（提取时已转16k单声道，通常无需开启）", value=False)
                        with gr.Accordion("字幕合并参数", open=True):
                            with gr.Row():
                                v_enable_duration = gr.Checkbox(label="启用时长限制", value=False)
                                v_merge_max_duration = gr.Slider(1.0, 20.0, 10.0, step=0.5, label="单条最大时长 (秒)")
                            with gr.Row():
                                v_enable_charcount = gr.Checkbox(label="启用字符数限制", value=False)
                                v_merge_max_chars = gr.Slider(5, 100, 30, step=5, label="单条最大字符数")
                            with gr.Row():
                                v_enable_silence = gr.Checkbox(label="启用静音阈值分句", value=False)
                                v_merge_silence_threshold = gr.Slider(0.1, 1.0, 0.3, step=0.05, label="静音阈值 (秒)")
                            with gr.Row():
                                v_enable_punc = gr.Checkbox(label="启用句末标点分句", value=True)
                                v_merge_punctuations = gr.Textbox(value="。！？.!?", label="句末标点")
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
                        video_download = gr.File(label="📦 下载结果文件", file_count="multiple", interactive=False)

                video_transcribe_btn.click(
                    transcribe_video,
                    [video_input, asr_model_type, use_gpu, use_half, enable_vad, enable_lid, enable_punc,
                     beam_size, nbest, decode_max_len, softmax_smoothing, aed_length_penalty,
                     eos_penalty, elm_weight,
                     vad_min_speech_frame, vad_max_speech_frame, vad_min_silence_frame,
                     vad_speech_threshold, vad_smooth_window_size, punc_threshold,
                     v_enable_duration, v_merge_max_duration,
                     v_enable_charcount, v_merge_max_chars,
                     v_enable_punc, v_merge_punctuations,
                     v_enable_silence, v_merge_silence_threshold,
                     force_preprocess_video, model_dir],
                    [video_text_output, video_word_json_output, video_sent_json_output, video_srt_output, video_download]
                ).then(refresh_status, outputs=[status_display])

                video_clear_btn.click(
                    lambda: [None, False, "", "", "", "", None],
                    [video_input, force_preprocess_video,
                     video_text_output, video_word_json_output, video_sent_json_output, video_srt_output, video_download]
                )

            # --- 批量处理（四输出）---
            with gr.Tab("批量处理"):
                with gr.Row():
                    with gr.Column(scale=1):
                        file_input = gr.File(file_types=[".wav", ".mp3", ".m4a", ".flac", ".ogg"],
                                             file_count="multiple", type="filepath", label="上传多个音频文件")
                        force_preprocess_batch = gr.Checkbox(label="⚡ 强制预处理", value=True)
                        with gr.Accordion("字幕合并参数", open=True):
                            with gr.Row():
                                batch_enable_duration = gr.Checkbox(label="启用时长限制", value=False)
                                batch_merge_max_duration = gr.Slider(1.0, 20.0, 10.0, step=0.5, label="单条最大时长 (秒)")
                            with gr.Row():
                                batch_enable_charcount = gr.Checkbox(label="启用字符数限制", value=False)
                                batch_merge_max_chars = gr.Slider(5, 100, 30, step=5, label="单条最大字符数")
                            with gr.Row():
                                batch_enable_silence = gr.Checkbox(label="启用静音阈值分句", value=False)
                                batch_merge_silence_threshold = gr.Slider(0.1, 1.0, 0.3, step=0.05, label="静音阈值 (秒)")
                            with gr.Row():
                                batch_enable_punc = gr.Checkbox(label="启用句末标点分句", value=True)
                                batch_merge_punctuations = gr.Textbox(value="。！？.!?", label="句末标点")
                        with gr.Row():
                            batch_transcribe_btn = gr.Button("批量识别", variant="primary")
                            batch_clear = gr.Button("清空", variant="secondary")
                    with gr.Column(scale=2):
                        batch_output = gr.Textbox(label="批量结果", lines=20, show_copy_button=True)
                        batch_download = gr.File(label="📦 下载打包结果 (zip)", interactive=False)

                batch_transcribe_btn.click(
                    transcribe_batch,
                    [file_input, asr_model_type, use_gpu, use_half, enable_vad, enable_lid, enable_punc,
                     beam_size, nbest, decode_max_len, softmax_smoothing, aed_length_penalty,
                     eos_penalty, elm_weight,
                     vad_min_speech_frame, vad_max_speech_frame, vad_min_silence_frame,
                     vad_speech_threshold, vad_smooth_window_size, punc_threshold,
                     force_preprocess_batch, model_dir,
                     batch_enable_duration, batch_merge_max_duration,
                     batch_enable_charcount, batch_merge_max_chars,
                     batch_enable_punc, batch_merge_punctuations,
                     batch_enable_silence, batch_merge_silence_threshold],
                    [batch_output, batch_download]
                ).then(refresh_status, outputs=[status_display])

                batch_clear.click(
                    lambda: [None, "", None],
                    outputs=[file_input, batch_output, batch_download]
                )

            # --- 字幕自动打轴 ---
            with gr.Tab("字幕自动打轴（文稿生字幕）"):
                with gr.Row():
                    with gr.Column(scale=1):
                        align_audio = gr.Audio(sources=["upload", "microphone"], type="filepath", label="上传或录制音频")
                        align_text = gr.Textbox(label="粘贴稿子文本", lines=8, placeholder="与音频内容一致，空行分隔段落")
                        force_preprocess_align = gr.Checkbox(label="⚡ 强制预处理音频", value=True)
                        with gr.Accordion("字幕合并参数", open=True):
                            merge_punctuations_align = gr.Textbox(value="。！？.!?", label="句末标点")
                            with gr.Row():
                                align_max_words = gr.Slider(5, 50, 20, step=1, label="最大词数")
                                align_max_chars = gr.Slider(5, 100, 30, step=5, label="最大字符数")
                            with gr.Row():
                                align_max_duration = gr.Slider(1.0, 20.0, 10.0, step=0.5, label="最大时长 (秒)")
                                align_silence_threshold = gr.Slider(0.1, 1.0, 0.3, step=0.05, label="静音阈值 (秒)")
                            with gr.Row():
                                merge_by_punc = gr.Checkbox(label="根据标点断句", value=False)
                                merge_by_silence = gr.Checkbox(label="根据静音断句", value=False)
                                merge_by_wordcount = gr.Checkbox(label="根据词数断句", value=False)
                            with gr.Row():
                                merge_by_charcount = gr.Checkbox(label="根据字符数断句", value=False)
                                merge_by_duration = gr.Checkbox(label="根据时长断句", value=False)
                                merge_by_newline = gr.Checkbox(label="根据空行断句", value=True)
                        with gr.Row():
                            align_btn = gr.Button("生成精准字幕", variant="primary")
                            align_clear = gr.Button("清空", variant="secondary")
                    with gr.Column(scale=2):
                        with gr.Tabs():
                            with gr.Tab("逐词 SRT"):
                                align_word_output = gr.Textbox(label="逐词 SRT", lines=42, show_copy_button=True)
                            with gr.Tab("整句 SRT"):
                                align_sent_output = gr.Textbox(label="整句 SRT", lines=42, show_copy_button=True)
                            with gr.Tab("合并字幕（自定义）"):
                                align_merged_output = gr.Textbox(label="合并字幕（自定义）", lines=42, show_copy_button=True)

                align_btn.click(
                    force_align_wrapper,
                    [align_audio, align_text, asr_model_type, use_gpu, use_half,
                     enable_vad, enable_lid, enable_punc,
                     beam_size, nbest, decode_max_len, softmax_smoothing, aed_length_penalty,
                     eos_penalty, elm_weight,
                     vad_min_speech_frame, vad_max_speech_frame, vad_min_silence_frame,
                     vad_speech_threshold, vad_smooth_window_size, punc_threshold, model_dir,
                     merge_punctuations_align, align_max_words, align_max_chars, align_max_duration,
                     align_silence_threshold,
                     merge_by_punc, merge_by_silence, merge_by_wordcount, merge_by_charcount, merge_by_duration,
                     merge_by_newline, force_preprocess_align],
                    [align_word_output, align_sent_output, align_merged_output]
                ).then(refresh_status, outputs=[status_display])

                align_clear.click(
                    lambda: [None, "", "", "", ""],
                    [align_audio, align_text, align_word_output, align_sent_output, align_merged_output]
                )

            # --- 系统信息 ---
            with gr.Tab("系统信息"):
                system_info_text = gr.Textbox(value=get_system_info(), lines=20, label="详细信息", show_copy_button=True)
                with gr.Row():
                    output_dir_input = gr.Textbox(value=str(OUTPUT_DIR), label="输出目录", interactive=True, scale=3)
                    update_output_btn = gr.Button("更新输出目录", variant="secondary", scale=1)
                with gr.Row():
                    open_output_btn = gr.Button("打开输出目录")
                    open_log_btn = gr.Button("打开日志文件夹")
                    clear_cache_btn = gr.Button("清理缓存文件", variant="secondary")
                with gr.Row():
                    save_config_btn = gr.Button("保存当前配置", variant="primary")
                    preset_files = sorted([f.name for f in PRESET_DIR.glob("preset_*.json")], reverse=True)
                    preset_selector = gr.Dropdown(label="选择预设文件", choices=preset_files, value=None, interactive=True)
                    load_config_btn = gr.Button("加载所选配置", variant="secondary")
                    refresh_preset_btn = gr.Button("刷新列表", variant="secondary")
                config_status = gr.Textbox(label="配置状态", interactive=False)

                def update_output_dir(new_dir):
                    global OUTPUT_DIR, ALIGN_OUTPUT_DIR
                    try:
                        p = Path(new_dir)
                        p.mkdir(parents=True, exist_ok=True)
                        with config_lock:
                            OUTPUT_DIR = p
                            ALIGN_OUTPUT_DIR = p / "字幕自动打轴"
                            ALIGN_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
                            manager.settings["output_dir"] = str(p)
                            save_settings(manager.settings)
                        return f"输出目录已更新为 {p}", get_system_info()
                    except Exception as e:
                        return f"更新失败: {e}", get_system_info()
                update_output_btn.click(update_output_dir, [output_dir_input], [config_status, system_info_text])

                open_output_btn.click(lambda: (open_file_or_dir(str(OUTPUT_DIR)), "已打开输出目录")[1], outputs=[config_status])
                open_log_btn.click(lambda: (open_file_or_dir(str(LOG_DIR)), "已打开日志文件夹")[1], outputs=[config_status])

                def clear_cache_fully():
                    count = 0
                    for f in CACHE_DIR.glob("*.wav"):
                        try:
                            os.unlink(f)
                            count += 1
                        except Exception:
                            pass
                    manager.cleanup_temp()
                    return f"已彻底删除 {count} 个缓存文件（并清理临时记录）"
                clear_cache_btn.click(clear_cache_fully, outputs=[config_status])

                # 保存配置时包含所有参数（包括 force_preprocess 相关）
                def save_current_config():
                    config = {
                        "asr_model_type": asr_model_type.value,
                        "use_gpu": use_gpu.value, "use_half": use_half.value,
                        "enable_vad": enable_vad.value, "enable_lid": enable_lid.value, "enable_punc": enable_punc.value,
                        "model_dir": model_dir.value,
                        "beam_size": beam_size.value, "nbest": nbest.value, "decode_max_len": decode_max_len.value,
                        "softmax_smoothing": softmax_smoothing.value, "aed_length_penalty": aed_length_penalty.value,
                        "eos_penalty": eos_penalty.value, "elm_weight": elm_weight.value,
                        "vad_min_speech_frame": vad_min_speech_frame.value, "vad_max_speech_frame": vad_max_speech_frame.value,
                        "vad_min_silence_frame": vad_min_silence_frame.value, "vad_speech_threshold": vad_speech_threshold.value,
                        "vad_smooth_window_size": vad_smooth_window_size.value, "punc_threshold": punc_threshold.value,
                        "audio_enable_duration": audio_enable_duration.value, "audio_merge_max_duration": audio_merge_max_duration.value,
                        "audio_enable_charcount": audio_enable_charcount.value, "audio_merge_max_chars": audio_merge_max_chars.value,
                        "audio_enable_punc": audio_enable_punc.value, "audio_merge_punctuations": audio_merge_punctuations.value,
                        "audio_enable_silence": audio_enable_silence.value, "audio_merge_silence_threshold": audio_merge_silence_threshold.value,
                        "force_preprocess_audio": force_preprocess_audio.value,
                        "v_enable_duration": v_enable_duration.value, "v_merge_max_duration": v_merge_max_duration.value,
                        "v_enable_charcount": v_enable_charcount.value, "v_merge_max_chars": v_merge_max_chars.value,
                        "v_enable_punc": v_enable_punc.value, "v_merge_punctuations": v_merge_punctuations.value,
                        "v_enable_silence": v_enable_silence.value, "v_merge_silence_threshold": v_merge_silence_threshold.value,
                        "force_preprocess_video": force_preprocess_video.value,
                        "force_preprocess_batch": force_preprocess_batch.value,
                        "merge_punctuations_align": merge_punctuations_align.value, "align_max_words": align_max_words.value,
                        "align_max_chars": align_max_chars.value, "align_max_duration": align_max_duration.value,
                        "align_silence_threshold": align_silence_threshold.value,
                        "merge_by_punc": merge_by_punc.value, "merge_by_silence": merge_by_silence.value,
                        "merge_by_wordcount": merge_by_wordcount.value, "merge_by_charcount": merge_by_charcount.value,
                        "merge_by_duration": merge_by_duration.value, "merge_by_newline": merge_by_newline.value,
                        "force_preprocess_align": force_preprocess_align.value,
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
                    "model_dir",
                    "beam_size", "nbest", "decode_max_len", "softmax_smoothing", "aed_length_penalty", "eos_penalty", "elm_weight",
                    "vad_min_speech_frame", "vad_max_speech_frame", "vad_min_silence_frame", "vad_speech_threshold", "vad_smooth_window_size", "punc_threshold",
                    "audio_enable_duration", "audio_merge_max_duration", "audio_enable_charcount", "audio_merge_max_chars",
                    "audio_enable_punc", "audio_merge_punctuations", "audio_enable_silence", "audio_merge_silence_threshold",
                    "force_preprocess_audio",
                    "v_enable_duration", "v_merge_max_duration", "v_enable_charcount", "v_merge_max_chars",
                    "v_enable_punc", "v_merge_punctuations", "v_enable_silence", "v_merge_silence_threshold",
                    "force_preprocess_video",
                    "force_preprocess_batch",
                    "merge_punctuations_align", "align_max_words", "align_max_chars", "align_max_duration", "align_silence_threshold",
                    "merge_by_punc", "merge_by_silence", "merge_by_wordcount", "merge_by_charcount", "merge_by_duration", "merge_by_newline",
                    "force_preprocess_align",
                ]
                _PARAM_DEFAULTS = {
                    "asr_model_type": "aed", "use_gpu": True, "use_half": False, "enable_vad": True, "enable_lid": True, "enable_punc": True,
                    "model_dir": "",
                    "beam_size": 3, "nbest": 1, "decode_max_len": 0, "softmax_smoothing": 1.25, "aed_length_penalty": 0.6, "eos_penalty": 1.0, "elm_weight": 0.0,
                    "vad_min_speech_frame": 20, "vad_max_speech_frame": 2000, "vad_min_silence_frame": 20, "vad_speech_threshold": 0.4, "vad_smooth_window_size": 5, "punc_threshold": 0.45,
                    "audio_enable_duration": False, "audio_merge_max_duration": 10.0,
                    "audio_enable_charcount": False, "audio_merge_max_chars": 30,
                    "audio_enable_punc": True, "audio_merge_punctuations": "。！？.!?",
                    "audio_enable_silence": False, "audio_merge_silence_threshold": 0.3,
                    "force_preprocess_audio": True,
                    "v_enable_duration": False, "v_merge_max_duration": 10.0,
                    "v_enable_charcount": False, "v_merge_max_chars": 30,
                    "v_enable_punc": True, "v_merge_punctuations": "。！？.!?",
                    "v_enable_silence": False, "v_merge_silence_threshold": 0.3,
                    "force_preprocess_video": False,
                    "force_preprocess_batch": True,
                    "merge_punctuations_align": "。！？.!?", "align_max_words": 20, "align_max_chars": 30, "align_max_duration": 10.0, "align_silence_threshold": 0.3,
                    "merge_by_punc": False, "merge_by_silence": False, "merge_by_wordcount": False, "merge_by_charcount": False, "merge_by_duration": False, "merge_by_newline": True,
                    "force_preprocess_align": True,
                }

                def load_selected_config(filename):
                    if not filename:
                        return ["请先选择一个预设文件"] + [gr.update() for _ in _PARAM_KEYS]
                    try:
                        with open(PRESET_DIR / filename, 'r', encoding='utf-8') as f:
                            cfg = json.load(f)
                    except Exception as e:
                        return [f"加载失败: {e}"] + [gr.update() for _ in _PARAM_KEYS]
                    if cfg.get("asr_model_type") not in model_choices:
                        cfg["asr_model_type"] = model_choices[0]
                        msg = f"⚠️ 预设中的模型类型无效，已重置为 {model_choices[0]}"
                    else:
                        msg = f"配置已加载: {filename}"
                    updates = [gr.update(value=cfg.get(k, _PARAM_DEFAULTS.get(k))) for k in _PARAM_KEYS]
                    return [msg] + updates

                load_config_btn.click(
                    load_selected_config, [preset_selector],
                    [config_status,
                     asr_model_type, use_gpu, use_half, enable_vad, enable_lid, enable_punc,
                     model_dir,
                     beam_size, nbest, decode_max_len, softmax_smoothing, aed_length_penalty, eos_penalty, elm_weight,
                     vad_min_speech_frame, vad_max_speech_frame, vad_min_silence_frame, vad_speech_threshold, vad_smooth_window_size, punc_threshold,
                     audio_enable_duration, audio_merge_max_duration, audio_enable_charcount, audio_merge_max_chars,
                     audio_enable_punc, audio_merge_punctuations, audio_enable_silence, audio_merge_silence_threshold,
                     force_preprocess_audio,
                     v_enable_duration, v_merge_max_duration, v_enable_charcount, v_merge_max_chars,
                     v_enable_punc, v_merge_punctuations, v_enable_silence, v_merge_silence_threshold,
                     force_preprocess_video,
                     force_preprocess_batch,
                     merge_punctuations_align, align_max_words, align_max_chars, align_max_duration, align_silence_threshold,
                     merge_by_punc, merge_by_silence, merge_by_wordcount, merge_by_charcount, merge_by_duration, merge_by_newline,
                     force_preprocess_align]
                )

        gr.Markdown("---")
        gr.Markdown("""<div style="text-align: center; color: #666; font-size: 0.9em;">
        <p>本软件包不提供任何模型文件，模型由用户自行从官方渠道获取。用户需自行遵守模型的原许可证。</p>
        <p>本软件包按“原样”提供，不提供任何明示或暗示的担保。使用本软件所产生的一切风险由用户自行承担。</p>
        <p><strong>更新请关注B站up主：光影的故事2018</strong></p>
        </div>""")
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