#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
FireRedASR2S 文稿对齐 + 双语字幕生成（锚点增强·稳定版）
- 智能时间戳检测，精准对齐
- 可选 FFmpeg 预处理（安全回退）
- 临时文件存放 PROJECT_ROOT/cache（永久缓存）
- 新增“强制断行锚点”：完全按稿子换行断句，专为规整文稿设计
- 帮助页完全外挂：scripts/help_content.json
Copyright 2026 光影的故事2018
"""

import sys, os, re, time, json, gc, logging, threading, atexit, tempfile, hashlib, shutil, subprocess
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

try:
    from fireredasr2s import FireRedAsr2System, FireRedAsr2SystemConfig
    from fireredasr2s.fireredasr2 import FireRedAsr2Config
    from fireredasr2s.fireredvad import FireRedVadConfig
    from fireredasr2s.fireredlid import FireRedLidConfig
    from fireredasr2s.fireredpunc import FireRedPuncConfig
    FIRERED_AVAILABLE = True
except ImportError as e:
    logger.error(f"导入 FireRedASR2S 失败: {e}")
    sys.exit(1)

try:
    import gradio as gr
    import torch
    import numpy as np
    import librosa
    import soundfile as sf
except ImportError as e:
    logger.error(f"缺少基础依赖库: {e}")
    sys.exit(1)

BASE_DIR = CURRENT_DIR
ROOT_DIR = PROJECT_ROOT
OUTPUT_DIR = ROOT_DIR / "output" / "字幕自动打轴"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------- 缓存目录 ----------
CACHE_DIR = ROOT_DIR / "cache"
CACHE_DIR.mkdir(exist_ok=True)

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
        logger.warning("未找到 FFmpeg，预处理功能可能受限")

# ==================== 安全截断 ====================
def safe_preview(text: str, max_len: int = 20000) -> str:
    if not text:
        return ""
    if len(text) <= max_len:
        return text
    return text[:max_len] + "\n...（文本过长已截断，完整内容请查看输出目录）"

# ==================== 模型管理器 ====================
class FireRedAlignManager:
    def __init__(self):
        self.asr_system = None
        self.config = None
        self.lock = threading.RLock()
        self.temp_files = []
        self.model_dir = None

    def find_model_dir(self, preferred_type="AED"):
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
                    logger.info(f"使用指定模型目录: {model_dir}")
                else:
                    model_dir = self.find_model_dir()
                    if not model_dir:
                        return False, "未找到 AED 模型目录，请将模型放在 pretrained_models/FireRedASR2-AED 或通过“模型目录”指定正确路径"

                # ---------- 检查子模型目录 ----------
                vad_model_dir = str(ROOT_DIR / "pretrained_models" / "FireRedVAD" / "vad")
                if not os.path.exists(vad_model_dir):
                    alt_vad_dir = str(ROOT_DIR / "pretrained_models" / "FireRedVAD" / "VAD")
                    if os.path.exists(alt_vad_dir):
                        vad_model_dir = alt_vad_dir
                    else:
                        return False, "未找到 VAD 模型，请确保 pretrained_models/FireRedVAD/vad 或 pretrained_models/FireRedVAD/VAD 存在"
                lid_model_dir = str(ROOT_DIR / "pretrained_models" / "FireRedLID")
                punc_model_dir = str(ROOT_DIR / "pretrained_models" / "FireRedPunc")
                if not os.path.exists(lid_model_dir):
                    return False, f"LID 模型目录不存在: {lid_model_dir}"
                if not os.path.exists(punc_model_dir):
                    return False, f"标点模型目录不存在: {punc_model_dir}"

                vad_config = FireRedVadConfig(use_gpu=config["use_gpu"])
                lid_config = FireRedLidConfig(use_gpu=config["use_gpu"])
                asr_config = FireRedAsr2Config(
                    use_gpu=config["use_gpu"],
                    use_half=config["use_half"],
                    return_timestamp=True
                )
                punc_config = FireRedPuncConfig(use_gpu=config["use_gpu"])

                system_config = FireRedAsr2SystemConfig(
                    vad_model_dir=vad_model_dir,
                    lid_model_dir=lid_model_dir,
                    asr_model_dir=model_dir,
                    punc_model_dir=punc_model_dir,
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

    def _prepare_audio(self, audio_input, force_preprocess=False):
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

        if force_preprocess and FFMPEG_PATH != "ffmpeg":
            try:
                with open(audio_path, 'rb') as f:
                    file_hash = hashlib.md5(f.read(4096)).hexdigest()
            except Exception:
                file_hash = hashlib.md5(audio_path.encode()).hexdigest()
            cache_name = f"prep_{file_hash}_{os.path.basename(audio_path)}_16k.wav"
            cache_path = CACHE_DIR / cache_name

            if not cache_path.exists():
                logger.info(f"FFmpeg 预处理: {audio_path} -> {cache_path}")
                cmd = [
                    FFMPEG_PATH, "-y", "-i", audio_path,
                    "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le",
                    str(cache_path)
                ]
                try:
                    subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                except subprocess.CalledProcessError as e:
                    err = e.stderr.decode(errors="ignore")[:200] if e.stderr else ""
                    logger.error(f"FFmpeg 预处理失败: {err}")
                    if cache_path.exists():
                        cache_path.unlink()
                    force_preprocess = False
            if force_preprocess and cache_path.exists():
                try:
                    data, sr = sf.read(str(cache_path), dtype='float32')
                    if sr != 16000:
                        data = librosa.resample(data, orig_sr=sr, target_sr=16000)
                        sr = 16000
                    return str(cache_path), data, sr
                except Exception as e:
                    logger.error(f"读取缓存文件失败: {e}，回退 librosa")
                    force_preprocess = False

        try:
            data, sr = librosa.load(audio_path, sr=None)
            if sr != 16000:
                data = librosa.resample(data.astype(np.float32), orig_sr=sr, target_sr=16000)
                sr = 16000
            return audio_path, data, sr
        except Exception as e:
            logger.error(f"librosa 加载音频失败: {e}")
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

    def force_align(self, audio_input, reference_text, progress_callback=None, force_preprocess=False):
        if self.asr_system is None:
            return None, None, None, None, None, "模型未加载"

        audio_prepare_result = self._prepare_audio(audio_input, force_preprocess)
        if audio_prepare_result is None:
            return None, None, None, None, None, "音频处理失败"

        audio_path, waveform, sr = audio_prepare_result
        duration = len(waveform) / sr

        try:
            asr = self.asr_system.asr

            feats, lengths, _, _, _ = asr.feat_extractor([(16000, waveform)], ["tmp"])
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
                    return None, None, None, None, None, "音频过短，无法对齐"
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

            # ========== 修复：稳健的时间戳单位检测 ==========
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

            del feats, enc_outputs, enc_lengths, yseq
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            return word_srt, sentence_srt, timestamps_sec, token_texts, token_ids, None

        except Exception as e:
            logger.error(f"强制对齐失败: {e}", exc_info=True)
            return None, None, None, None, None, f"强制对齐失败: {str(e)}"

    def _seconds_to_srt_time(self, seconds):
        if seconds < 0:
            seconds = 0
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
    if seconds < 0:
        seconds = 0
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
    """修复版合并函数：支持 CJK 检测，英文自动空格连接"""
    if len(timestamps) == 0:
        return []

    # 检测是否含有中日韩文字
    has_cjk = any(re.search(r'[\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af]', w) for w in words)
    join_str = "" if has_cjk else " "

    sentences = []
    current_start = timestamps[0][0]
    current_words = []
    last_end = timestamps[0][1]

    for i, ((start, end), word) in enumerate(zip(timestamps, words)):
        break_before = False
        if current_words:
            if merge_by_silence and i > 0:
                gap = start - last_end
                if gap > silence_threshold:
                    break_before = True
            if not break_before and merge_by_wordcount and len(current_words) >= max_words:
                break_before = True
            if not break_before and merge_by_charcount:
                new_text = join_str.join(current_words + [word])
                if len(new_text) > max_chars:
                    break_before = True
            if not break_before and merge_by_duration:
                current_dur = last_end - current_start
                if current_dur + (end - start) >= max_duration:
                    break_before = True

        if break_before:
            sentences.append({
                "start": current_start,
                "end": last_end,
                "text": join_str.join(current_words).strip()
            })
            current_start = start
            current_words = []

        if not current_words:
            current_start = start
        current_words.append(word)
        last_end = end

        should_break_after = False
        if force_break_indices and i < len(force_break_indices) and force_break_indices[i]:
            should_break_after = True
        elif merge_by_punc and any(word.endswith(p) for p in sentence_endings):
            should_break_after = True

        if should_break_after:
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

# ==================== 锚点增强 ====================
def clean_text_for_anchor(text: str) -> str:
    text = re.sub(r'[^\u4e00-\u9fff\u3000-\u303f\uff00-\uffef a-zA-Z0-9，。！？；：“”‘’（）【】\n]', '', text)
    text = re.sub(r' +', ' ', text)
    return text

def anchor_align_segments(words, word_timestamps, force_break_indices, audio_duration,
                          anchor_char_count=3,
                          use_anchor_start=False, use_anchor_end=False, use_anchor_mean=False):
    """
    灵活锚点增强：
    - use_anchor_start: 段落开始时间取前 anchor_char_count 个汉字的开始时间均值
    - use_anchor_end:   段落结束时间取后 anchor_char_count 个汉字的结束时间均值
    - use_anchor_mean:  开始/结束时间取前后锚点的均值（优先级高于单独启用的前/后锚点）
    未开启任何锚点或段落无汉字时回退到原始词边界。
    """
    if not words or not word_timestamps:
        return []

    segments = []
    cur_words = []
    cur_ts = []
    for i, (word, (s, e)) in enumerate(zip(words, word_timestamps)):
        cur_words.append(word)
        cur_ts.append((s, e))
        if force_break_indices[i]:
            segments.append((cur_words, cur_ts))
            cur_words = []
            cur_ts = []
    if cur_words:
        segments.append((cur_words, cur_ts))

    result_sentences = []
    for seg_words, seg_ts in segments:
        seg_text = "".join(seg_words).strip()
        if not seg_text:
            continue

        chinese_indices = [i for i, w in enumerate(seg_words) if re.search(r'[\u4e00-\u9fff]', w)]
        start_default = seg_ts[0][0]
        end_default = seg_ts[-1][1]

        if chinese_indices:
            n_front = min(anchor_char_count, len(chinese_indices))
            front_idx = chinese_indices[:n_front]
            start_front = sum(seg_ts[i][0] for i in front_idx) / n_front
            end_front = sum(seg_ts[i][1] for i in front_idx) / n_front
        else:
            start_front = start_default
            end_front = end_default

        if chinese_indices:
            n_back = min(anchor_char_count, len(chinese_indices))
            back_idx = chinese_indices[-n_back:]
            start_back = sum(seg_ts[i][0] for i in back_idx) / n_back
            end_back = sum(seg_ts[i][1] for i in back_idx) / n_back
        else:
            start_back = start_default
            end_back = end_default

        if use_anchor_mean:
            seg_start = (start_front + start_back) / 2
        elif use_anchor_start:
            seg_start = start_front
        else:
            seg_start = start_default

        if use_anchor_mean:
            seg_end = (end_front + end_back) / 2
        elif use_anchor_end:
            seg_end = end_back
        else:
            seg_end = end_default

        if seg_end <= seg_start:
            seg_end = seg_start + 0.5

        result_sentences.append({
            "start": seg_start,
            "end": seg_end,
            "text": seg_text
        })
    return result_sentences

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
    lines.append(f"缓存目录: {CACHE_DIR}")
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
    use_anchor_start, use_anchor_end, use_anchor_mean, anchor_char_count,
    force_preprocess,
    force_linebreak_mode=False,   # 新增：强制断行锚点模式
    progress=gr.Progress()
):
    if audio_file is None:
        return "错误: 请上传音频文件", "", "", "", "", "", "", get_system_status()
    if not primary_text or not primary_text.strip():
        return "错误: 请粘贴主文稿", "", "", "", "", "", "", get_system_status()

    # 在强制断行模式下直接使用原始文稿，不清洗
    if force_linebreak_mode:
        primary_text_cleaned = primary_text
    else:
        anchor_enabled = use_anchor_start or use_anchor_end or use_anchor_mean
        primary_text_cleaned = clean_text_for_anchor(primary_text) if anchor_enabled else primary_text

    audio_path = safe_audio_path(audio_file)
    if not audio_path or not os.path.exists(audio_path):
        return "错误: 无法获取有效的音频文件路径", "", "", "", "", "", "", get_system_status()

    if manager.asr_system is None:
        progress(0.05, desc="自动加载模型...")
        success, msg = manager.load_system(use_gpu, use_half, model_dir_override)
        if not success:
            return f"错误: {msg}", "", "", "", "", "", "", get_system_status()

    progress(0.2, desc="强制对齐中...")
    word_srt, sent_srt, timestamps, words, token_ids_out, error = manager.force_align(
        audio_path, primary_text_cleaned, force_preprocess=force_preprocess
    )
    if error:
        return f"错误: {error}", "", "", "", "", "", "", get_system_status()

    if not timestamps or not words:
        return "错误: 未获取到有效时间戳", "", "", "", "", "", "", get_system_status()

    asr = manager.asr_system.asr
    final_force_break = None
    merge_warnings = []

    # ---------- 强制断行锚点模式 ----------
    if force_linebreak_mode:
        lines = [line.strip() for line in primary_text.split('\n') if line.strip()]
        if lines:
            force_break = [False] * len(words)
            current_pos = 0
            total_words = len(words)
            for line in lines:
                line_tokens, line_ids = asr.tokenizer.tokenize(line)
                if not line_tokens:
                    continue
                # 使用 token ID 匹配，精确度更高
                found = -1
                for start in range(current_pos, total_words - len(line_ids) + 1):
                    if token_ids_out[start:start+len(line_ids)] == line_ids:
                        found = start
                        break
                if found >= 0:
                    end_idx = found + len(line_ids) - 1
                    if end_idx < total_words - 1:
                        force_break[end_idx] = True
                    current_pos = end_idx + 1
                else:
                    # 尝试降级字符串匹配
                    for start in range(current_pos, total_words - len(line_tokens) + 1):
                        if words[start:start+len(line_tokens)] == line_tokens:
                            found = start
                            break
                    if found >= 0:
                        end_idx = found + len(line_tokens) - 1
                        if end_idx < total_words - 1:
                            force_break[end_idx] = True
                        current_pos = end_idx + 1
                    else:
                        merge_warnings.append(f"强制断行失败：无法匹配行 '{line[:30]}...'，将忽略该行")
                        current_pos += max(len(line_tokens), 1)
            final_force_break = force_break
        # 在强制断行模式下，所有自动断句规则都设为无效
        merge_by_punc = False
        merge_by_silence = False
        merge_by_wordcount = False
        merge_by_charcount = False
        merge_by_duration = False

    # ---------- 常规断句逻辑（非强制模式下） ----------
    if not force_linebreak_mode:
        force_break = None
        # 空行断句
        progress(0.4, desc="处理空行断句...")
        if merge_by_newline and words and timestamps:
            paragraphs = [p.strip() for p in primary_text_cleaned.split('\n') if p.strip()]
            if len(paragraphs) > 1:
                force_break = [False] * len(words)
                current_pos = 0
                total_words = len(words)
                for para in paragraphs:
                    para_tokens, para_ids = asr.tokenizer.tokenize(para)
                    if len(para_ids) == 0:
                        continue
                    found = -1
                    for start in range(current_pos, total_words - len(para_ids) + 1):
                        if token_ids_out[start:start+len(para_ids)] == para_ids:
                            found = start
                            break
                    if found >= 0:
                        end_idx = found + len(para_ids) - 1
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
                            current_pos += len(para_tokens)
                            merge_warnings.append(f"警告：段落 '{para[:30]}...' 无法匹配")
                    progress(0.4 + 0.1 * min(current_pos / total_words, 1.0), desc="处理空行断句...")

        # 标点断句
        progress(0.55, desc="处理标点断句...")
        force_break_punc = None
        if merge_by_punc and words and timestamps:
            punc_positions = [idx for idx, ch in enumerate(primary_text_cleaned) if ch in merge_punctuations]
            if punc_positions:
                tokens_all, token_ids_all = asr.tokenizer.tokenize(primary_text_cleaned)
                char_to_token = [-1] * len(primary_text_cleaned)
                cur = 0
                for token_idx, token in enumerate(tokens_all):
                    token_len = len(token)
                    for i in range(token_len):
                        if cur + i < len(primary_text_cleaned):
                            char_to_token[cur + i] = token_idx
                    cur += token_len
                force_break_punc = [False] * len(words)
                for pos in punc_positions:
                    if pos < len(char_to_token):
                        tidx = char_to_token[pos]
                        if 0 <= tidx < len(words) - 1:   # 修复：末尾标点不断句
                            force_break_punc[tidx] = True

        if force_break is not None or force_break_punc is not None:
            final_force_break = [False] * len(words)
            if force_break:
                for i, v in enumerate(force_break):
                    if v: final_force_break[i] = True
            if force_break_punc:
                for i, v in enumerate(force_break_punc):
                    if v: final_force_break[i] = True

    # ---------- 生成合并字幕 ----------
    progress(0.7, desc="生成合并字幕...")
    sentences = merge_timestamps_to_sentences(
        timestamps, words,
        sentence_endings=merge_punctuations,
        max_words=merge_max_words,
        max_chars=merge_max_chars,
        max_duration=merge_max_duration,
        silence_threshold=merge_silence_threshold,
        merge_by_punc=merge_by_punc,
        merge_by_silence=merge_by_silence,
        merge_by_wordcount=merge_by_wordcount,
        merge_by_charcount=merge_by_charcount,
        merge_by_duration=merge_by_duration,
        force_break_indices=final_force_break
    )
    merged_srt = sentences_to_srt(sentences)

    # 锚点增强（在 final_force_break 存在且启用锚点时）
    anchor_srt = ""
    anchor_enabled = use_anchor_start or use_anchor_end or use_anchor_mean
    if anchor_enabled and final_force_break and words and timestamps:
        audio_duration = timestamps[-1][1] + 0.5 if timestamps else 0.0
        anchor_sentences = anchor_align_segments(
            words, timestamps, final_force_break, audio_duration,
            anchor_char_count=anchor_char_count,
            use_anchor_start=use_anchor_start,
            use_anchor_end=use_anchor_end,
            use_anchor_mean=use_anchor_mean
        )
        anchor_srt = sentences_to_srt(anchor_sentences)

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

    if merge_warnings:
        warning_msg = warning_msg + "\n" + "\n".join(merge_warnings) if warning_msg else "\n".join(merge_warnings)

    # 保存文件
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    base_name = Path(audio_path).stem if audio_path else "align"
    prefix = f"{base_name}_align_{timestamp}"

    word_path = OUTPUT_DIR / f"{prefix}_words.srt"
    sent_path = OUTPUT_DIR / f"{prefix}_sentence.srt"
    merged_path = OUTPUT_DIR / f"{prefix}_merged.srt"
    with open(word_path, "w", encoding="utf-8") as f: f.write(word_srt)
    with open(sent_path, "w", encoding="utf-8") as f: f.write(sent_srt)
    with open(merged_path, "w", encoding="utf-8") as f: f.write(merged_srt)

    if anchor_srt:
        anchor_path = OUTPUT_DIR / f"{prefix}_anchor.srt"
        with open(anchor_path, "w", encoding="utf-8") as f: f.write(anchor_srt)

    safe_lang_tag = re.sub(r'[^\w\-]', '', secondary_lang.strip()) if secondary_lang else ""
    safe_lang_tag = f"_{safe_lang_tag}" if safe_lang_tag else ""

    if secondary_srt_str:
        sec_path = OUTPUT_DIR / f"{prefix}{safe_lang_tag}_secondary.srt"
        with open(sec_path, "w", encoding="utf-8") as f: f.write(secondary_srt_str)

    if dual_srt:
        dual_path = OUTPUT_DIR / f"{prefix}{safe_lang_tag}_dual.srt"
        with open(dual_path, "w", encoding="utf-8") as f: f.write(dual_srt)

    status = f"✅ 对齐完成！\n逐词字幕: {word_path.name}\n整句字幕: {sent_path.name}\n合并字幕: {merged_path.name}"
    if anchor_srt:
        status += f"\n锚点字幕: {prefix}_anchor.srt"
    if secondary_srt_str:
        status += f"\n副文稿单语: {prefix}{safe_lang_tag}_secondary.srt"
    if dual_srt:
        status += f"\n双语字幕: {prefix}{safe_lang_tag}_dual.srt"
    if warning_msg:
        status += f"\n{warning_msg}"
        try:
            gr.Warning(warning_msg)
        except:
            logger.warning(warning_msg)

    manager.cleanup_temp()
    progress(1.0, desc="完成")

    return (
        safe_preview(status, 5000),
        safe_preview(word_srt),
        safe_preview(sent_srt),
        safe_preview(merged_srt),
        safe_preview(secondary_srt_str),
        safe_preview(dual_srt),
        safe_preview(anchor_srt),
        get_system_status()
    )

def clear_outputs():
    return "等待开始", "", "", "", "", "", "", get_system_status()

# ==================== 批量处理 ====================
def batch_process(
    audio_files, text_files, enable_dual_batch,
    use_gpu, use_half, model_dir_override,
    merge_punctuations, merge_max_words, merge_max_chars, merge_max_duration,
    merge_silence_threshold, merge_by_punc, merge_by_silence, merge_by_wordcount,
    merge_by_charcount, merge_by_duration, merge_by_newline,
    force_preprocess,
    progress=gr.Progress()
):
    if not audio_files or not text_files:
        return "请上传音频文件和对应的文稿文件", get_system_status()
    if len(audio_files) != len(text_files):
        return f"音频文件数量 ({len(audio_files)}) 与文稿文件数量 ({len(text_files)}) 不一致", get_system_status()

    if manager.asr_system is None:
        progress(0.02, desc="自动加载模型...")
        success, msg = manager.load_system(use_gpu, use_half, model_dir_override)
        if not success:
            return f"模型加载失败: {msg}", get_system_status()

    results = []
    total = len(audio_files)
    for idx, (audio_obj, text_obj) in enumerate(zip(audio_files, text_files)):
        progress(idx / total, desc=f"处理 {idx+1}/{total}...")
        audio_path = safe_audio_path(audio_obj)
        if not audio_path or not os.path.exists(audio_path):
            results.append(f"❌ {os.path.basename(audio_obj) if isinstance(audio_obj, str) else '未知'}: 音频文件无效")
            continue

        try:
            text_path = text_obj if isinstance(text_obj, str) else (text_obj.name if hasattr(text_obj, 'name') else str(text_obj))
            primary_text = Path(text_path).read_text(encoding='utf-8')
            if not primary_text.strip():
                results.append(f"❌ {os.path.basename(audio_path)}: 文稿内容为空")
                continue
        except Exception as e:
            results.append(f"❌ {os.path.basename(audio_path)}: 读取文稿失败 - {e}")
            continue

        word_srt, sent_srt, timestamps, words, token_ids_out, error = manager.force_align(audio_path, primary_text, force_preprocess=force_preprocess)
        if error:
            results.append(f"❌ {os.path.basename(audio_path)}: 对齐失败 - {error}")
            continue
        if not timestamps or not words:
            results.append(f"❌ {os.path.basename(audio_path)}: 未获取到有效时间戳")
            continue

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
                    found = False
                    for start in range(current_pos, len(words) - len(para_tokens) + 1):
                        if words[start:start+len(para_tokens)] == para_tokens:
                            end_idx = start + len(para_tokens) - 1
                            if end_idx < len(words) - 1:
                                force_break[end_idx] = True
                            current_pos = end_idx + 1
                            found = True
                            break
                    if not found:
                        current_pos += len(para_tokens)

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

        final_force_break = [False] * len(words)
        if force_break:
            for i, v in enumerate(force_break):
                if v: final_force_break[i] = True
        if force_break_punc:
            for i, v in enumerate(force_break_punc):
                if v: final_force_break[i] = True

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

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        base_name = Path(audio_path).stem
        prefix = f"{base_name}_align_{timestamp}"
        word_path = OUTPUT_DIR / f"{prefix}_words.srt"
        sent_path = OUTPUT_DIR / f"{prefix}_sentence.srt"
        merged_path = OUTPUT_DIR / f"{prefix}_merged.srt"
        with open(word_path, "w", encoding="utf-8") as f: f.write(word_srt)
        with open(sent_path, "w", encoding="utf-8") as f: f.write(sent_srt)
        with open(merged_path, "w", encoding="utf-8") as f: f.write(merged_srt)
        results.append(f"✅ {os.path.basename(audio_path)}: 已生成")

    manager.cleanup_temp()
    progress(1.0, desc="完成")
    return "\n".join(results), get_system_status()

# ==================== 创建 UI ====================
def create_ui():
    help_data = {}
    help_file = Path(__file__).parent / "help_content.json"
    if help_file.exists():
        try:
            with open(help_file, 'r', encoding='utf-8') as f:
                help_data = json.load(f)
        except Exception as e:
            logger.warning(f"加载帮助文件失败: {e}")

    with gr.Blocks(title="FireRedASR2S 文稿对齐 + 双语字幕生成（锚点增强）", theme=gr.themes.Default()) as demo:
        gr.Markdown("# 🎬 FireRedASR2S 文稿对齐 + 双语字幕生成（锚点增强版）")

        with gr.Tabs():
            # ---------- 单次处理 ----------
            with gr.Tab("单次处理"):
                with gr.Row():
                    with gr.Column(scale=1):
                        audio_input = gr.File(label="选择音频文件", file_types=[".wav", ".mp3", ".m4a", ".flac", ".ogg"])
                        force_preprocess_check = gr.Checkbox(label="⚡ 强制预处理为 16kHz 单声道 (推荐)", value=True)
                        primary_text = gr.Textbox(label="主文稿（对齐用）", lines=18, placeholder="粘贴与音频内容一致的稿子...\n段落之间用空行分隔")
                        secondary_text = gr.Textbox(label="副文稿（挂载用，可选）", lines=18, placeholder="粘贴翻译稿...\n段落结构尽量与主文稿一致")
                        with gr.Row():
                            secondary_lang = gr.Textbox(label="副文稿语言标记", placeholder="en", value="", scale=1)
                            enable_dual = gr.Checkbox(label="生成双语字幕", value=False, scale=1)

                    with gr.Column(scale=2):
                        with gr.Row():
                            system_status = gr.Textbox(label="系统状态", value=get_system_status(), lines=4, interactive=False, scale=1)
                            task_status = gr.Textbox(label="任务状态", value="等待开始", lines=4, interactive=False, scale=1)

                        with gr.Accordion("⚙️ 模型控制", open=True):
                            with gr.Row():
                                use_gpu = gr.Checkbox(label="使用 GPU", value=torch.cuda.is_available())
                                use_half = gr.Checkbox(label="使用半精度 (FP16)", value=False)
                            model_dir_override = gr.Textbox(label="模型目录（可选）", placeholder="留空自动检测，或指定完整路径", value="")
                            with gr.Row():
                                load_model_btn = gr.Button("加载模型", variant="primary")
                                unload_model_btn = gr.Button("卸载模型", variant="secondary")
                                refresh_status_btn = gr.Button("刷新状态", variant="secondary")

                        with gr.Accordion("📝 字幕合并规则", open=True):
                            # 新增：强制断行锚点（最上方优先显示）
                            force_linebreak_cb = gr.Checkbox(
                                label="📌 强制断行锚点（按稿子换行强制分段，忽略所有自动规则，专为规整文稿设计）",
                                value=False,
                                info="开启后，将完全依据文稿的原始换行进行分段，首字/尾字定位起止时间，忽略标点、静音、时长等断句规则"
                            )
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
                                gr.Markdown("**锚点增强选项**（需先启用空行分段，独立于强制断行模式）")
                            with gr.Row():
                                use_anchor_start = gr.Checkbox(label="前锚点", value=False, info="段落开头取前几个汉字校准开始时间")
                                use_anchor_end = gr.Checkbox(label="后锚点", value=False, info="段落结尾取后几个汉字校准结束时间")
                                use_anchor_mean = gr.Checkbox(label="前后均值", value=False, info="开始/结束取前后锚点的平均值，覆盖单独选项")
                            with gr.Row():
                                anchor_char_count = gr.Slider(1, 5, value=3, step=1, label="锚点参考汉字数")

                        with gr.Row():
                            run_btn = gr.Button("开始对齐", variant="primary", size="lg")
                            clear_btn = gr.Button("清空", variant="secondary")
                            open_output_btn = gr.Button("打开输出目录", variant="secondary")

                        with gr.Tabs():
                            with gr.Tab("逐词 SRT"):
                                word_output = gr.Textbox(label="逐词字幕", lines=20, show_copy_button=True)
                            with gr.Tab("整句 SRT"):
                                sent_output = gr.Textbox(label="整句字幕", lines=20, show_copy_button=True)
                            with gr.Tab("合并字幕"):
                                merged_output = gr.Textbox(label="合并后的字幕", lines=20, show_copy_button=True)
                            with gr.Tab("锚点增强字幕"):
                                anchor_output = gr.Textbox(label="锚点增强字幕", lines=20, show_copy_button=True)
                            with gr.Tab("副文稿单语 SRT"):
                                secondary_output = gr.Textbox(label="副文稿字幕", lines=20, show_copy_button=True)
                            with gr.Tab("双语 SRT"):
                                dual_output = gr.Textbox(label="双语字幕", lines=20, show_copy_button=True)

            # ---------- 批量处理 ----------
            with gr.Tab("批量处理"):
                with gr.Row():
                    with gr.Column(scale=1):
                        audio_files = gr.File(label="上传音频文件（可多选）", file_count="multiple", file_types=[".wav",".mp3",".m4a",".flac",".ogg"])
                        text_files = gr.File(label="上传对应的文稿文件（顺序对应）", file_count="multiple", file_types=[".txt"])
                        force_preprocess_batch = gr.Checkbox(label="⚡ 强制预处理为 16kHz 单声道", value=True)
                        enable_dual_batch = gr.Checkbox(label="生成双语字幕（批量暂不支持）", value=False, interactive=False)
                    with gr.Column(scale=2):
                        batch_status = gr.Textbox(label="批量处理状态", lines=10, interactive=False)
                        batch_system = gr.Textbox(label="系统状态", value=get_system_status(), lines=4, interactive=False)
                        batch_run_btn = gr.Button("开始批量对齐", variant="primary", size="lg")

            # ---------- 帮助 ----------
            with gr.Tab("帮助"):
                if help_file.exists() and help_data:
                    for section, content in help_data.items():
                        if content and isinstance(content, str):
                            gr.Markdown(f"## {section}\n\n{content}")
                else:
                    gr.Markdown("""
                    ## 使用说明
                    1. 上传音频文件（支持 wav/mp3/m4a/flac/ogg）
                    2. 粘贴主文稿（与音频内容一致，段落间用空行分隔）
                    3. （可选）粘贴副文稿（翻译稿）并勾选“生成双语字幕”
                    4. 调整模型设置和合并规则
                    5. 可选：启用强制断行锚点，完全按稿子原始换行分段，首尾字定位
                    6. 可选：启用锚点增强，利用段落前几个汉字精准校准段边界
                    7. 点击“开始对齐”，完成后字幕文件保存在输出目录，可点击“打开输出目录”获取。
                    """)

        # ---------- 事件绑定 ----------
        def load_model_action(gpu, half, model_dir):
            success, msg = manager.load_system(gpu, half, model_dir)
            return msg, get_system_status()

        def unload_model_action():
            success, msg = manager.unload_system()
            return msg, get_system_status()

        def refresh_status_action():
            return get_system_status()

        load_model_btn.click(load_model_action, inputs=[use_gpu, use_half, model_dir_override], outputs=[task_status, system_status])
        unload_model_btn.click(unload_model_action, outputs=[task_status, system_status])
        refresh_status_btn.click(refresh_status_action, outputs=[system_status])

        run_btn.click(
            run_alignment,
            inputs=[
                audio_input, primary_text, secondary_text, secondary_lang, enable_dual,
                use_gpu, use_half, model_dir_override,
                punc_box, max_words_slider, max_chars_slider, max_duration_slider,
                silence_slider, merge_punc, merge_silence, merge_wordcount,
                merge_charcount, merge_duration, merge_newline,
                use_anchor_start, use_anchor_end, use_anchor_mean, anchor_char_count,
                force_preprocess_check,
                force_linebreak_cb   # 新增输入
            ],
            outputs=[task_status, word_output, sent_output, merged_output,
                     secondary_output, dual_output, anchor_output, system_status]
        )

        clear_btn.click(
            clear_outputs,
            outputs=[task_status, word_output, sent_output, merged_output,
                     secondary_output, dual_output, anchor_output, system_status]
        ).then(
            lambda: [None, "", "", "", False, False, False, False, 3, True, False],
            outputs=[audio_input, primary_text, secondary_text, secondary_lang,
                     enable_dual, use_anchor_start, use_anchor_end, use_anchor_mean,
                     anchor_char_count, force_preprocess_check, force_linebreak_cb]
        )

        def open_output_dir():
            if sys.platform == "win32":
                os.startfile(str(OUTPUT_DIR))
            elif sys.platform == "darwin":
                subprocess.run(["open", str(OUTPUT_DIR)])
            else:
                subprocess.run(["xdg-open", str(OUTPUT_DIR)])
            return "已打开输出目录"

        open_output_btn.click(open_output_dir, outputs=[])

        batch_run_btn.click(
            batch_process,
            inputs=[
                audio_files, text_files, enable_dual_batch,
                use_gpu, use_half, model_dir_override,
                punc_box, max_words_slider, max_chars_slider, max_duration_slider,
                silence_slider, merge_punc, merge_silence, merge_wordcount,
                merge_charcount, merge_duration, merge_newline,
                force_preprocess_batch
            ],
            outputs=[batch_status, batch_system]
        )

        gr.HTML("""
        <div style="text-align: center; color: #666; font-size: 0.85em; margin-top: 20px;">
            <p>© 2026 光影紐扣 | 基于 FireRedASR2S (Apache 2.0)</p>
            <p>更新请关注B站：光影的故事2018 | 日志文件: logs/align_*.log</p>
        </div>
        """)

    return demo

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

    model_root = ROOT_DIR / "pretrained_models"
    if not model_root.exists():
        print(f"警告: 模型目录 {model_root} 不存在，请确保模型已下载。")

    demo = create_ui()
    demo.queue(default_concurrency_limit=1, max_size=2)

    ports = [18001, 18002, 18003, 18004, 18005]
    for p in ports:
        try:
            demo.launch(
                server_name="127.0.0.1",
                server_port=p,
                inbrowser=True,
                show_error=True,
                max_file_size=100 * 1024 * 1024
            )
            break
        except OSError:
            print(f"端口 {p} 被占用，尝试下一个...")
            continue
    else:
        print("所有端口均被占用，请手动指定空闲端口。")

if __name__ == "__main__":
    main()