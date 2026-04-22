#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
FireRedASR2S 文稿对齐 + 双语字幕生成（独立版）- 最终修复版
修复内容：
- 批量处理文件路径错误
- 批量处理输出绑定冲突
- 进度条边界保护
- 标点断句边界优化
- 日志记录完善
Copyright 2026 光影的故事2018
"""

import sys
import os
import re
import time
import json
import gc
import logging
import threading
import atexit
import tempfile
import hashlib
import shutil
import subprocess
from pathlib import Path
from datetime import timedelta
from typing import List, Dict, Optional, Tuple, Union

# ==================== 日志配置 ====================
LOG_DIR = Path(__file__).parent.parent / "logs"
LOG_DIR.mkdir(exist_ok=True)
LOG_FILE = LOG_DIR / f"align_{time.strftime('%Y%m%d')}.log"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE, encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ==================== 路径设置 ====================
CURRENT_DIR = Path(__file__).parent.absolute()
PROJECT_ROOT = CURRENT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT / "FireRedASR2S"))

# 导入 FireRedASR2S 模块
try:
    from fireredasr2s import FireRedAsr2System, FireRedAsr2SystemConfig
    from fireredasr2s.fireredasr2 import FireRedAsr2Config
    from fireredasr2s.fireredvad import FireRedVadConfig
    from fireredasr2s.fireredlid import FireRedLidConfig
    from fireredasr2s.fireredpunc import FireRedPuncConfig
    FIRERED_AVAILABLE = True
except ImportError as e:
    logger.error(f"导入 FireRedASR2S 失败: {e}")
    print("请确保 FireRedASR2S 模块已正确放置在 FireRedASR2S 目录下")
    sys.exit(1)

# 导入 Gradio 和音频处理库
try:
    import gradio as gr
    import torch
    import numpy as np
    import librosa
    import soundfile as sf
except ImportError as e:
    logger.error(f"缺少基础依赖库: {e}")
    sys.exit(1)

# ==================== 路径设置 ====================
BASE_DIR = Path(__file__).parent.absolute()
ROOT_DIR = BASE_DIR.parent
OUTPUT_DIR = ROOT_DIR / "output" / "字幕自动打轴"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ==================== FFmpeg 配置 ====================
PORTABLE_FFMPEG_DIR = ROOT_DIR / "ffmpeg" / "bin"
if sys.platform == "win32":
    PORTABLE_FFMPEG_EXE = PORTABLE_FFMPEG_DIR / "ffmpeg.exe"
else:
    PORTABLE_FFMPEG_EXE = PORTABLE_FFMPEG_DIR / "ffmpeg"

if PORTABLE_FFMPEG_EXE.exists():
    os.environ["PATH"] = str(PORTABLE_FFMPEG_DIR) + os.pathsep + os.environ.get("PATH", "")
    FFMPEG_PATH = str(PORTABLE_FFMPEG_EXE)
    logger.info(f"已加载内置 FFmpeg: {FFMPEG_PATH}")
else:
    system_ffmpeg = shutil.which("ffmpeg")
    if system_ffmpeg:
        FFMPEG_PATH = system_ffmpeg
        logger.info(f"使用系统 FFmpeg: {FFMPEG_PATH}")
    else:
        FFMPEG_PATH = "ffmpeg"
        logger.warning("未找到 FFmpeg，视频处理功能不可用")

# ==================== 模型管理器（线程安全） ====================
class FireRedAlignManager:
    def __init__(self):
        self.asr_system = None
        self.config = None
        self.lock = threading.RLock()
        self.temp_files = []
        self.model_dir = None  # 实际使用的模型目录

    def find_model_dir(self, preferred_type="AED"):
        """自动查找模型目录，支持 AED 或 AED-2025 等变体"""
        candidates = [
            ROOT_DIR / "pretrained_models" / f"FireRedASR2-{preferred_type}",
            ROOT_DIR / "pretrained_models" / f"FireRedASR2-{preferred_type}-2025",
            ROOT_DIR / "pretrained_models" / "FireRedASR2-AED",
            ROOT_DIR / "pretrained_models" / "FireRedASR2-AED-2025",
        ]
        for p in candidates:
            if p.exists():
                logger.info(f"使用模型目录: {p}")
                return str(p)
        return None

    def load_system(self, use_gpu=True, use_half=False, model_dir_override=None):
        with self.lock:
            if self.asr_system is not None:
                return True, "系统已加载"
            try:
                config = {
                    "use_gpu": use_gpu and torch.cuda.is_available(),
                    "use_half": use_half,
                    "enable_vad": True,
                    "enable_lid": True,
                    "enable_punc": True,
                    "asr_model_type": "aed",
                }

                if model_dir_override and Path(model_dir_override).exists():
                    model_dir = str(model_dir_override)
                else:
                    model_dir = self.find_model_dir()
                    if not model_dir:
                        return False, "未找到 AED 模型目录，请将模型放在 pretrained_models/FireRedASR2-AED 或指定路径"

                vad_config = FireRedVadConfig(use_gpu=config["use_gpu"])
                lid_config = FireRedLidConfig(use_gpu=config["use_gpu"])
                asr_config = FireRedAsr2Config(
                    use_gpu=config["use_gpu"],
                    use_half=config["use_half"],
                    return_timestamp=True
                )
                punc_config = FireRedPuncConfig(use_gpu=config["use_gpu"])

                vad_model_dir = str(ROOT_DIR / "pretrained_models" / "FireRedVAD" / "vad")
                if not os.path.exists(vad_model_dir):
                    alt_vad_dir = str(ROOT_DIR / "pretrained_models" / "FireRedVAD" / "VAD")
                    if os.path.exists(alt_vad_dir):
                        vad_model_dir = alt_vad_dir

                system_config = FireRedAsr2SystemConfig(
                    vad_model_dir=vad_model_dir,
                    lid_model_dir=str(ROOT_DIR / "pretrained_models" / "FireRedLID"),
                    asr_model_dir=model_dir,
                    punc_model_dir=str(ROOT_DIR / "pretrained_models" / "FireRedPunc"),
                    vad_config=vad_config,
                    lid_config=lid_config,
                    asr_config=asr_config,
                    punc_config=punc_config,
                    enable_vad=int(config["enable_vad"]),
                    enable_lid=int(config["enable_lid"]),
                    enable_punc=int(config["enable_punc"])
                )

                self.asr_system = FireRedAsr2System(system_config)
                self.config = config
                self.model_dir = model_dir
                logger.info("FireRedASR2S 系统加载成功")
                return True, "模型加载成功"
            except Exception as e:
                logger.error(f"加载模型失败: {e}", exc_info=True)
                return False, f"加载失败: {str(e)}"

    def unload_system(self):
        with self.lock:
            if self.asr_system is not None:
                del self.asr_system
            self.asr_system = None
            self.config = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
            logger.info("模型已卸载，GPU 显存已清理")
            return True, "系统已卸载"

    def _prepare_audio(self, audio_input):
        # 安全处理音频路径
        if isinstance(audio_input, str):
            audio_path = audio_input
        elif isinstance(audio_input, tuple) and len(audio_input) > 0:
            audio_path = audio_input[0]
        elif isinstance(audio_input, dict):
            audio_path = audio_input.get("name")
        else:
            return None
        if not os.path.exists(audio_path):
            logger.error(f"音频文件不存在: {audio_path}")
            return None

        try:
            data, sr = librosa.load(audio_path, sr=None)
            if sr != 16000:
                data = librosa.resample(data.astype(np.float32), orig_sr=sr, target_sr=16000)
                sr = 16000
            temp_hash = hashlib.md5(data.tobytes() + str(time.time()).encode()).hexdigest()[:8]
            temp_path = os.path.join(tempfile.gettempdir(), f"firered_align_{temp_hash}.wav")
            sf.write(temp_path, data, sr)
            self.temp_files.append(temp_path)
            return temp_path
        except Exception as e:
            logger.error(f"音频处理失败: {e}", exc_info=True)
            return None

    def cleanup_temp(self):
        cleaned = 0
        for f in self.temp_files[:]:
            try:
                if os.path.exists(f):
                    os.unlink(f)
                    cleaned += 1
            except Exception as e:
                logger.warning(f"临时文件删除失败 {f}: {e}")
        self.temp_files = []
        return cleaned

    def force_align(self, audio_input, reference_text, progress_callback=None):
        if self.asr_system is None:
            return None, None, None, None, "模型未加载"

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

            if self.config['use_gpu'] and torch.cuda.is_available():
                feats = feats.cuda()
                lengths = lengths.cuda()
                if asr.config.use_half:
                    feats = feats.half()

            asr.model.eval()
            with torch.no_grad():
                enc_outputs, enc_lengths, _ = asr.model.encoder(feats, lengths)
                T = enc_outputs.size(1)
                if T == 0:
                    return None, None, None, None, "音频过短，无法对齐"
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
            max_start = max(starts)
            if max_start <= duration * 1.5 and max_start > 0.01 * duration:
                timestamps_sec = list(zip(starts, ends))
            else:
                hypothetical_max_sec = max_start * frame_shift
                if abs(hypothetical_max_sec - duration) < 0.2 * duration:
                    timestamps_sec = [(s * frame_shift, e * frame_shift) for s, e in zip(starts, ends)]
                else:
                    if max_start > duration * 1000:
                        scale = duration / max_start * 0.95
                        timestamps_sec = [(s * scale, e * scale) for s, e in zip(starts, ends)]
                    else:
                        timestamps_sec = [(s * frame_shift, e * frame_shift) for s, e in zip(starts, ends)]

            min_len = min(len(timestamps_sec), len(token_ids))
            timestamps_sec = timestamps_sec[:min_len]
            token_ids = token_ids[:min_len]
            tokens = tokens[:min_len]

            token_texts = [asr.tokenizer.detokenize([tid]) for tid in token_ids]

            word_srt_lines = []
            for i, ((start, end), txt) in enumerate(zip(timestamps_sec, token_texts), 1):
                word_srt_lines.append(str(i))
                word_srt_lines.append(f"{self._seconds_to_srt_time(start)} --> {self._seconds_to_srt_time(end)}")
                word_srt_lines.append(txt)
                word_srt_lines.append("")
            word_srt = "\n".join(word_srt_lines)

            if timestamps_sec:
                start_all = timestamps_sec[0][0]
                end_all = timestamps_sec[-1][1]
                full_text = asr.tokenizer.detokenize(token_ids)
                sentence_srt = f"1\n{self._seconds_to_srt_time(start_all)} --> {self._seconds_to_srt_time(end_all)}\n{full_text}\n"
            else:
                sentence_srt = ""

            return word_srt, sentence_srt, timestamps_sec, token_texts, None

        except Exception as e:
            logger.error(f"强制对齐失败: {e}", exc_info=True)
            return None, None, None, None, f"强制对齐失败: {str(e)}"
        finally:
            if audio_path in self.temp_files:
                self.temp_files.remove(audio_path)
                try:
                    os.unlink(audio_path)
                except:
                    pass

    def _seconds_to_srt_time(self, seconds):
        td = timedelta(seconds=seconds)
        total_seconds = int(td.total_seconds())
        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60
        secs = total_seconds % 60
        ms = int((td.total_seconds() - total_seconds) * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{ms:03d}"

manager = FireRedAlignManager()

# ==================== 工具函数 ====================
def seconds_to_srt_time(seconds: float) -> str:
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds - int(seconds)) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"

def sentences_to_srt(sentences: List[Dict]) -> str:
    lines = []
    for i, sent in enumerate(sentences, 1):
        lines.append(str(i))
        lines.append(f"{seconds_to_srt_time(sent['start'])} --> {seconds_to_srt_time(sent['end'])}")
        lines.append(sent["text"])
        lines.append("")
    return "\n".join(lines)

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

def get_system_status():
    lines = []
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        total_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        lines.append(f"显卡: {gpu_name} ({total_mem:.1f} GB)")
    else:
        lines.append("设备: CPU 模式")
    with manager.lock:
        if manager.asr_system is not None:
            lines.append(f"ASR系统: 已加载 (模型: {Path(manager.model_dir).name if manager.model_dir else '未知'})")
        else:
            lines.append("ASR系统: 未加载")
    lines.append(f"输出目录: {OUTPUT_DIR}")
    lines.append(f"日志文件: {LOG_FILE}")
    return "\n".join(lines)

def safe_audio_path(audio_input) -> Optional[str]:
    if audio_input is None:
        return None
    if isinstance(audio_input, str):
        return os.path.abspath(audio_input)
    if isinstance(audio_input, tuple) and len(audio_input) > 0:
        return os.path.abspath(audio_input[0])
    if isinstance(audio_input, dict):
        return os.path.abspath(audio_input.get("name")) if audio_input.get("name") else None
    return None

# ==================== 单次对齐处理 ====================
def run_alignment(
    audio_file, primary_text, secondary_text, secondary_lang, enable_dual,
    use_gpu, use_half, model_dir_override,
    merge_punctuations, merge_max_words, merge_max_chars, merge_max_duration,
    merge_silence_threshold, merge_by_punc, merge_by_silence, merge_by_wordcount,
    merge_by_charcount, merge_by_duration, merge_by_newline,
    progress=gr.Progress()
):
    if audio_file is None:
        return "错误: 请上传音频文件", "", "", "", "", "", get_system_status()
    if not primary_text or not primary_text.strip():
        return "错误: 请粘贴主文稿", "", "", "", "", "", get_system_status()

    audio_path = safe_audio_path(audio_file)
    if not audio_path or not os.path.exists(audio_path):
        return "错误: 无法获取有效的音频文件路径", "", "", "", "", "", get_system_status()

    progress(0.05, desc="加载模型...")
    success, msg = manager.load_system(use_gpu, use_half, model_dir_override)
    if not success:
        return f"错误: {msg}", "", "", "", "", "", get_system_status()

    progress(0.2, desc="强制对齐中...")
    word_srt, sent_srt, timestamps, words, error = manager.force_align(audio_path, primary_text)
    if error:
        return f"错误: {error}", "", "", "", "", "", get_system_status()

    if not timestamps or not words:
        return "错误: 未获取到有效时间戳", "", "", "", "", "", get_system_status()

    asr = manager.asr_system.asr
    force_break = None
    merge_warnings = []

    progress(0.4, desc="处理空行断句...")
    # === 精确空行断句 ===
    if merge_by_newline and words and timestamps:
        paragraphs = [p.strip() for p in primary_text.split('\n') if p.strip()]
        if len(paragraphs) > 1:
            force_break = [False] * len(words)
            current_pos = 0
            total_words = len(words)
            for para in paragraphs:
                para_tokens, _ = asr.tokenizer.tokenize(para)
                if len(para_tokens) == 0:
                    continue
                found = -1
                for start in range(current_pos, total_words - len(para_tokens) + 1):
                    if words[start:start+len(para_tokens)] == para_tokens:
                        found = start
                        break
                if found >= 0:
                    end_idx = found + len(para_tokens) - 1
                    if end_idx < total_words - 1:
                        force_break[end_idx] = True
                    current_pos = end_idx + 1
                else:
                    para_clean = re.sub(r'[^\w\u4e00-\u9fff]', '', para)
                    for start in range(current_pos, total_words - len(para_tokens) + 1):
                        segment = ''.join(words[start:start+len(para_tokens)])
                        segment_clean = re.sub(r'[^\w\u4e00-\u9fff]', '', segment)
                        if segment_clean == para_clean:
                            found = start
                            break
                    if found >= 0:
                        end_idx = found + len(para_tokens) - 1
                        if end_idx < total_words - 1:
                            force_break[end_idx] = True
                        current_pos = end_idx + 1
                    else:
                        msg = f"警告：段落 '{para[:30]}...' 无法与词序列匹配"
                        merge_warnings.append(msg)
                # 进度保护
                progress(0.4 + 0.1 * min(current_pos / total_words, 1.0), desc="处理空行断句...")

    progress(0.55, desc="处理标点断句...")
    # === 精确标点断句 ===
    force_break_punc = None
    if merge_by_punc and words and timestamps:
        punc_positions = [idx for idx, ch in enumerate(primary_text) if ch in merge_punctuations]
        if punc_positions:
            tokens_all, _ = asr.tokenizer.tokenize(primary_text)
            char_to_token = [-1] * len(primary_text)
            cur = 0
            for token_idx, token in enumerate(tokens_all):
                token_len = len(token)
                for i in range(token_len):
                    if cur + i < len(primary_text):
                        char_to_token[cur + i] = token_idx
                cur += token_len
            force_break_punc = [False] * len(words)
            for pos in punc_positions:
                if pos < len(char_to_token):
                    tidx = char_to_token[pos]
                    if 0 <= tidx < len(words):
                        force_break_punc[tidx] = True

    # 合并断句索引
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

    progress(0.7, desc="生成合并字幕...")
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

    # 双语挂载
    dual_srt = ""
    secondary_srt_str = ""
    warning_msg = ""

    progress(0.8, desc="处理双语挂载...")
    if enable_dual and secondary_text and secondary_text.strip():
        sec_paragraphs = [p.strip() for p in secondary_text.split('\n') if p.strip()]
        len_diff = abs(len(sec_paragraphs) - len(sentences))
        if len_diff <= 1:
            if len(sec_paragraphs) > len(sentences):
                sec_paragraphs = sec_paragraphs[:len(sentences)]
                warning_msg = f"⚠️ 副文稿段落数多 {len_diff} 段，已自动截断"
            elif len(sentences) > len(sec_paragraphs):
                sec_paragraphs += [""] * (len(sentences) - len(sec_paragraphs))
                warning_msg = f"⚠️ 副文稿段落数少 {len_diff} 段，已补充空行"

            sec_lines = []
            for i, (seg, sec_text) in enumerate(zip(sentences, sec_paragraphs), 1):
                sec_lines.append(str(i))
                sec_lines.append(f"{seconds_to_srt_time(seg['start'])} --> {seconds_to_srt_time(seg['end'])}")
                sec_lines.append(sec_text)
                sec_lines.append("")
            secondary_srt_str = "\n".join(sec_lines)

            dual_lines = []
            for i, (seg, sec_text) in enumerate(zip(sentences, sec_paragraphs), 1):
                dual_lines.append(str(i))
                dual_lines.append(f"{seconds_to_srt_time(seg['start'])} --> {seconds_to_srt_time(seg['end'])}")
                dual_lines.append(seg['text'])
                dual_lines.append(sec_text)
                dual_lines.append("")
            dual_srt = "\n".join(dual_lines)
        else:
            warning_msg = f"⚠️ 段落数相差 {len_diff} 段（超过1），跳过双语生成"

    # 合并警告信息
    if merge_warnings:
        warning_msg = warning_msg + "\n" + "\n".join(merge_warnings) if warning_msg else "\n".join(merge_warnings)

    # 保存文件
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    base_name = Path(audio_path).stem if audio_path else "align"
    prefix = f"{base_name}_align_{timestamp}"

    word_path = OUTPUT_DIR / f"{prefix}_words.srt"
    sent_path = OUTPUT_DIR / f"{prefix}_sentence.srt"
    merged_path = OUTPUT_DIR / f"{prefix}_merged.srt"

    with open(word_path, "w", encoding="utf-8") as f:
        f.write(word_srt)
    with open(sent_path, "w", encoding="utf-8") as f:
        f.write(sent_srt)
    with open(merged_path, "w", encoding="utf-8") as f:
        f.write(merged_srt)

    status = f"✅ 对齐完成！\n逐词字幕: {word_path.name}\n整句字幕: {sent_path.name}\n合并字幕: {merged_path.name}"

    # 过滤语言标记（只允许字母数字下划线横线）
    safe_lang_tag = re.sub(r'[^\w\-]', '', secondary_lang.strip()) if secondary_lang else ""
    safe_lang_tag = f"_{safe_lang_tag}" if safe_lang_tag else ""

    if secondary_srt_str:
        sec_path = OUTPUT_DIR / f"{prefix}{safe_lang_tag}_secondary.srt"
        with open(sec_path, "w", encoding="utf-8") as f:
            f.write(secondary_srt_str)
        status += f"\n副文稿单语: {sec_path.name}"

    if dual_srt:
        dual_path = OUTPUT_DIR / f"{prefix}{safe_lang_tag}_dual.srt"
        with open(dual_path, "w", encoding="utf-8") as f:
            f.write(dual_srt)
        status += f"\n双语字幕: {dual_path.name}"

    if warning_msg:
        status += f"\n{warning_msg}"
        gr.Warning(warning_msg)

    manager.cleanup_temp()
    progress(1.0, desc="完成")
    return status, word_srt, sent_srt, merged_srt, secondary_srt_str, dual_srt, get_system_status()

def clear_outputs():
    return "等待开始", "", "", "", "", "", get_system_status()

# ==================== 批量处理 ====================
def batch_process(
    audio_files, text_files, enable_dual_batch,
    use_gpu, use_half, model_dir_override,
    merge_punctuations, merge_max_words, merge_max_chars, merge_max_duration,
    merge_silence_threshold, merge_by_punc, merge_by_silence, merge_by_wordcount,
    merge_by_charcount, merge_by_duration, merge_by_newline,
    progress=gr.Progress()
):
    if not audio_files or not text_files:
        return "请上传音频文件和对应的文稿文件（数量相同，顺序对应）", "", get_system_status()
    if len(audio_files) != len(text_files):
        return f"音频文件数量 ({len(audio_files)}) 与文稿文件数量 ({len(text_files)}) 不一致，请检查", "", get_system_status()

    # 加载模型（复用，只需一次）
    progress(0.02, desc="加载模型...")
    success, msg = manager.load_system(use_gpu, use_half, model_dir_override)
    if not success:
        return f"模型加载失败: {msg}", "", get_system_status()

    results = []
    total = len(audio_files)
    for idx, (audio_obj, text_obj) in enumerate(zip(audio_files, text_files)):
        progress(idx / total, desc=f"处理 {idx+1}/{total}...")
        # 修复：audio_obj 和 text_obj 直接是路径字符串
        audio_path = safe_audio_path(audio_obj)
        if not audio_path or not os.path.exists(audio_path):
            results.append(f"❌ {os.path.basename(audio_obj) if isinstance(audio_obj, str) else '未知'}: 音频文件无效")
            continue

        # 读取文稿内容
        try:
            text_path = text_obj if isinstance(text_obj, str) else (text_obj.name if hasattr(text_obj, 'name') else str(text_obj))
            primary_text = Path(text_path).read_text(encoding='utf-8')
            if not primary_text.strip():
                results.append(f"❌ {os.path.basename(audio_path)}: 文稿内容为空")
                continue
        except Exception as e:
            results.append(f"❌ {os.path.basename(audio_path)}: 读取文稿失败 - {e}")
            continue

        # 执行对齐
        word_srt, sent_srt, timestamps, words, error = manager.force_align(audio_path, primary_text)
        if error:
            results.append(f"❌ {os.path.basename(audio_path)}: 对齐失败 - {error}")
            continue

        if not timestamps or not words:
            results.append(f"❌ {os.path.basename(audio_path)}: 未获取到有效时间戳")
            continue

        # 断句处理
        asr = manager.asr_system.asr
        force_break = None
        if merge_by_newline and words:
            paragraphs = [p.strip() for p in primary_text.split('\n') if p.strip()]
            if len(paragraphs) > 1:
                force_break = [False] * len(words)
                current_pos = 0
                for para in paragraphs:
                    para_tokens, _ = asr.tokenizer.tokenize(para)
                    if len(para_tokens) == 0:
                        continue
                    for start in range(current_pos, len(words) - len(para_tokens) + 1):
                        if words[start:start+len(para_tokens)] == para_tokens:
                            end_idx = start + len(para_tokens) - 1
                            if end_idx < len(words) - 1:
                                force_break[end_idx] = True
                            current_pos = end_idx + 1
                            break

        force_break_punc = None
        if merge_by_punc and words:
            punc_positions = [idx for idx, ch in enumerate(primary_text) if ch in merge_punctuations]
            if punc_positions:
                tokens_all, _ = asr.tokenizer.tokenize(primary_text)
                char_to_token = [-1] * len(primary_text)
                cur = 0
                for token_idx, token in enumerate(tokens_all):
                    token_len = len(token)
                    for i in range(token_len):
                        if cur + i < len(primary_text):
                            char_to_token[cur + i] = token_idx
                    cur += token_len
                force_break_punc = [False] * len(words)
                for pos in punc_positions:
                    if pos < len(char_to_token):
                        tidx = char_to_token[pos]
                        if 0 <= tidx < len(words):
                            force_break_punc[tidx] = True

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

        # 保存文件
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        base_name = Path(audio_path).stem
        prefix = f"{base_name}_align_{timestamp}"

        word_path = OUTPUT_DIR / f"{prefix}_words.srt"
        sent_path = OUTPUT_DIR / f"{prefix}_sentence.srt"
        merged_path = OUTPUT_DIR / f"{prefix}_merged.srt"

        with open(word_path, "w", encoding="utf-8") as f:
            f.write(words_to_srt_from_tokens(words, timestamps))
        with open(sent_path, "w", encoding="utf-8") as f:
            f.write(sentences_to_srt(sentences))
        with open(merged_path, "w", encoding="utf-8") as f:
            f.write(merged_srt)

        results.append(f"✅ {os.path.basename(audio_path)}: 已生成")

    manager.cleanup_temp()
    progress(1.0, desc="完成")
    # 返回结果字符串、空占位、系统状态
    return "\n".join(results), "", get_system_status()

def words_to_srt_from_tokens(words, timestamps):
    """辅助函数：从 words 和 timestamps 生成逐词 SRT"""
    lines = []
    for i, (word, (start, end)) in enumerate(zip(words, timestamps), 1):
        lines.append(str(i))
        lines.append(f"{seconds_to_srt_time(start)} --> {seconds_to_srt_time(end)}")
        lines.append(word)
        lines.append("")
    return "\n".join(lines)

# ==================== 创建界面 ====================
def create_ui():
    with gr.Blocks(title="FireRedASR2S 文稿对齐 + 双语字幕生成", theme=gr.themes.Default()) as demo:
        gr.Markdown("# 🎬 FireRedASR2S 文稿对齐 + 双语字幕生成（正式版）")

        with gr.Tabs():
            # ---------- 单次处理标签页 ----------
            with gr.Tab("单次处理"):
                with gr.Row():
                    with gr.Column(scale=1):
                        audio_input = gr.Audio(label="选择音频文件", type="filepath", sources=["upload"])
                        primary_text = gr.Textbox(
                            label="主文稿（对齐用）",
                            lines=18,
                            placeholder="粘贴与音频内容一致的稿子...\n段落之间用空行分隔"
                        )
                        secondary_text = gr.Textbox(
                            label="副文稿（挂载用，可选）",
                            lines=18,
                            placeholder="粘贴翻译稿...\n段落结构尽量与主文稿一致"
                        )
                        with gr.Row():
                            secondary_lang = gr.Textbox(label="副文稿语言标记", placeholder="如：en / ja / fr", value="", scale=1)
                            enable_dual = gr.Checkbox(label="生成双语字幕", value=False, scale=1)

                    with gr.Column(scale=2):
                        with gr.Row():
                            system_status = gr.Textbox(label="系统状态", value=get_system_status(), lines=4, interactive=False, scale=1)
                            task_status = gr.Textbox(label="任务状态", value="等待开始", lines=4, interactive=False, scale=1)

                        with gr.Accordion("⚙️ 模型设置", open=True):
                            with gr.Row():
                                use_gpu = gr.Checkbox(label="使用 GPU", value=torch.cuda.is_available())
                                use_half = gr.Checkbox(label="使用半精度 (FP16)", value=False)
                            model_dir_override = gr.Textbox(label="模型目录（可选）", placeholder="留空自动检测，或指定完整路径", value="")

                        with gr.Accordion("📝 字幕合并规则", open=True):
                            with gr.Row():
                                merge_newline = gr.Checkbox(label="按空行分段（推荐）", value=True)
                                merge_punc = gr.Checkbox(label="按标点断句", value=True)
                                merge_silence = gr.Checkbox(label="按静音断句", value=True)
                            with gr.Row():
                                merge_wordcount = gr.Checkbox(label="按词数断句", value=True)
                                merge_charcount = gr.Checkbox(label="按字符数断句", value=True)
                                merge_duration = gr.Checkbox(label="按时长断句", value=True)
                            with gr.Row():
                                punc_box = gr.Textbox(label="句末标点", value="。！？.!?", scale=2)
                                silence_slider = gr.Slider(label="静音阈值 (秒)", minimum=0.1, maximum=1.0, value=0.3, step=0.05, scale=1)
                            with gr.Row():
                                max_words_slider = gr.Slider(label="最大词数", minimum=5, maximum=50, value=20, step=1)
                                max_chars_slider = gr.Slider(label="最大字符数", minimum=5, maximum=100, value=30, step=5)
                                max_duration_slider = gr.Slider(label="最大时长 (秒)", minimum=1.0, maximum=20.0, value=10.0, step=0.5)

                        with gr.Row():
                            run_btn = gr.Button("开始对齐", variant="primary", size="lg")
                            clear_btn = gr.Button("清空", variant="secondary")

                        with gr.Tabs():
                            with gr.Tab("逐词 SRT"):
                                word_output = gr.Textbox(label="逐词字幕", lines=20, show_copy_button=True)
                            with gr.Tab("整句 SRT"):
                                sent_output = gr.Textbox(label="整句字幕", lines=20, show_copy_button=True)
                            with gr.Tab("合并字幕"):
                                merged_output = gr.Textbox(label="合并后的字幕", lines=20, show_copy_button=True)
                            with gr.Tab("副文稿单语 SRT"):
                                secondary_output = gr.Textbox(label="副文稿字幕", lines=20, show_copy_button=True)
                            with gr.Tab("双语 SRT"):
                                dual_output = gr.Textbox(label="双语字幕", lines=20, show_copy_button=True)

            # ---------- 批量处理标签页 ----------
            with gr.Tab("批量处理"):
                with gr.Row():
                    with gr.Column(scale=1):
                        audio_files = gr.File(label="上传音频文件（可多选）", file_count="multiple", file_types=[".wav", ".mp3", ".m4a", ".flac", ".ogg"])
                        text_files = gr.File(label="上传对应的文稿文件（顺序对应）", file_count="multiple", file_types=[".txt"])
                        enable_dual_batch = gr.Checkbox(label="生成双语字幕（批量时副文稿为空）", value=False)
                    with gr.Column(scale=2):
                        batch_status = gr.Textbox(label="批量处理状态", lines=10, interactive=False)
                        batch_system = gr.Textbox(label="系统状态", value=get_system_status(), lines=4, interactive=False)
                        batch_run_btn = gr.Button("开始批量对齐", variant="primary", size="lg")

            # ---------- 帮助标签页 ----------
            with gr.Tab("帮助"):
                gr.Markdown("""
                ## 使用说明

                ### 单次处理
                1. 上传音频文件（支持 wav/mp3/m4a/flac/ogg）
                2. 粘贴主文稿（与音频内容一致，段落间用空行分隔）
                3. （可选）粘贴副文稿（翻译稿）并勾选“生成双语字幕”
                4. 调整模型设置和合并规则
                5. 点击“开始对齐”

                ### 批量处理
                1. 上传多个音频文件（顺序任意）
                2. 上传对应的文稿文件（.txt，与音频顺序一一对应）
                3. 点击“开始批量对齐”
                4. 系统会为每对音频+文稿生成字幕文件

                ### 合并规则说明
                - **按空行分段**：主文稿中的空行会强制切分字幕（推荐）
                - **按标点断句**：遇到句末标点（。！？）时切分
                - **按静音断句**：词间静音超过阈值时切分
                - **按词数/字符数/时长**：达到上限时强制切分

                ### 模型路径
                - 默认自动检测 `pretrained_models/FireRedASR2-AED` 或 `FireRedASR2-AED-2025`
                - 如需指定，在“模型目录”中输入完整路径

                ### 输出文件
                所有字幕文件保存在 `output/字幕自动打轴/` 目录下。
                """)

        # 绑定事件（单次）
        run_btn.click(
            run_alignment,
            inputs=[
                audio_input, primary_text, secondary_text, secondary_lang, enable_dual,
                use_gpu, use_half, model_dir_override,
                punc_box, max_words_slider, max_chars_slider, max_duration_slider,
                silence_slider, merge_punc, merge_silence, merge_wordcount,
                merge_charcount, merge_duration, merge_newline
            ],
            outputs=[task_status, word_output, sent_output, merged_output, secondary_output, dual_output, system_status]
        )
        clear_btn.click(
            clear_outputs,
            outputs=[task_status, word_output, sent_output, merged_output, secondary_output, dual_output, system_status]
        ).then(
            lambda: [None, "", "", "", False],
            outputs=[audio_input, primary_text, secondary_text, secondary_lang, enable_dual]
        )

        # 批量处理事件（修复输出绑定）
        batch_run_btn.click(
            batch_process,
            inputs=[
                audio_files, text_files, enable_dual_batch,
                use_gpu, use_half, model_dir_override,
                punc_box, max_words_slider, max_chars_slider, max_duration_slider,
                silence_slider, merge_punc, merge_silence, merge_wordcount,
                merge_charcount, merge_duration, merge_newline
            ],
            outputs=[batch_status, gr.State(), batch_system]  # 第二个返回值用 State 占位，避免覆盖
        )

        # 页脚
        gr.HTML("""
        <div style="text-align: center; color: #666; font-size: 0.85em; margin-top: 20px;">
            <p>© 2026 光影紐扣 | 基于 FireRedASR2S (Apache 2.0)</p>
            <p>更新请关注B站：光影的故事2018 | 日志文件: logs/align_*.log</p>
        </div>
        """)

    return demo

# ==================== 退出清理 ====================
@atexit.register
def cleanup():
    logger.info("正在退出，清理资源...")
    manager.unload_system()
    manager.cleanup_temp()
    logger.info("清理完成")

def main():
    if not FIRERED_AVAILABLE:
        logger.error("FireRedASR2S 模块不可用，请检查环境。")
        return

    demo = create_ui()
    demo.launch(
        server_name="127.0.0.1",
        server_port=18001,
        inbrowser=True,
        show_error=True
    )

if __name__ == "__main__":
    main()