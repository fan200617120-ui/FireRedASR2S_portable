#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
FireRedASR2S 文稿对齐 + 双语字幕生成（锚点增强版 v3）

"""

import sys, os, re, time, json, gc, logging, threading, atexit, hashlib, shutil, subprocess
from pathlib import Path
from datetime import timedelta
from typing import List, Dict, Optional, Union, Tuple, Any

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

FIRERED_AVAILABLE = False
IMPORT_ERROR = None
try:
    from fireredasr2s import FireRedAsr2System, FireRedAsr2SystemConfig
    from fireredasr2s.fireredasr2 import FireRedAsr2Config
    from fireredasr2s.fireredvad import FireRedVadConfig
    from fireredasr2s.fireredlid import FireRedLidConfig
    from fireredasr2s.fireredpunc import FireRedPuncConfig
    FIRERED_AVAILABLE = True
except ImportError as e:
    IMPORT_ERROR = str(e)
    logger.error(f"导入 FireRedASR2S 失败: {e}")

try:
    import gradio as gr
    import torch
    import numpy as np
    import librosa
    import soundfile as sf
except ImportError as e:
    logger.error(f"缺少基础依赖库: {e}")
    raise ImportError(f"缺少基础依赖库: {e}")

BASE_DIR = CURRENT_DIR
ROOT_DIR = PROJECT_ROOT
OUTPUT_DIR = ROOT_DIR / "output" / "字幕自动打轴"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CACHE_DIR = ROOT_DIR / "cache"
CACHE_DIR.mkdir(exist_ok=True)

LONG_AUDIO_CHUNK_THRESHOLD = 300.0

AUDIO_EXTS = [".wav", ".mp3", ".m4a", ".flac", ".ogg", ".aac", ".wma", ".opus"]

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

# ==================== 通用工具 ====================
def safe_preview(text: str, max_len: int = 20000) -> str:
    if not text:
        return ""
    if len(text) <= max_len:
        return text
    return text[:max_len] + "\n...（文本过长已截断，完整内容请查看输出目录）"

def seconds_to_srt_time(seconds: float) -> str:
    if seconds is None or seconds < 0:
        seconds = 0
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int(round((seconds - int(seconds)) * 1000))
    if millis >= 1000:
        millis = 999
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"

def srt_to_vtt(srt_content: str) -> str:
    out = ["WEBVTT", ""]
    for block in srt_content.split("\n\n"):
        lines = [l for l in block.split("\n") if l.strip()]
        if not lines:
            continue
        if len(lines) >= 2 and "-->" in lines[1]:
            out.append(lines[0])
            out.append(lines[1].replace(",", "."))
            out.extend(lines[2:])
            out.append("")
        else:
            out.extend(lines)
            out.append("")
    return "\n".join(out)

def cap_warnings(warnings: List[str], n: int = 8) -> str:
    if not warnings:
        return ""
    shown = warnings[:n]
    txt = "\n".join("⚠️ " + w for w in shown)
    extra = len(warnings) - n
    if extra > 0:
        txt += f"\n⚠️ …另有 {extra} 条警告，详见日志"
    return txt

def read_text_robust(path) -> str:
    last_err = None
    for enc in ("utf-8-sig", "utf-8", "gb18030", "big5"):
        try:
            return Path(path).read_text(encoding=enc)
        except (UnicodeDecodeError, LookupError) as e:
            last_err = e
            continue
    raise ValueError(f"无法识别文稿编码（已尝试 utf-8/utf-8-sig/gb18030/big5）: {last_err}")

def gradio_file_path(obj) -> Optional[str]:
    if obj is None:
        return None
    if isinstance(obj, str):
        return obj
    if isinstance(obj, dict):
        return obj.get("name")
    if isinstance(obj, (tuple, list)) and len(obj) > 0:
        return obj[0]
    if hasattr(obj, "name"):
        return getattr(obj, "name")
    return None

def safe_audio_path(audio_input) -> Optional[str]:
    p = gradio_file_path(audio_input)
    if not p:
        return None
    return os.path.abspath(str(p))

def _file_md5(path, chunk: int = 1 << 20) -> str:
    h = hashlib.md5()
    with open(path, 'rb') as f:
        for blk in iter(lambda: f.read(chunk), b""):
            h.update(blk)
    return h.hexdigest()

# ==================== 字幕构建工具 ====================
def sentences_to_srt(sentences: List[Dict]) -> str:
    lines = []
    for i, sent in enumerate(sentences, 1):
        lines.append(str(i))
        lines.append(f"{seconds_to_srt_time(sent['start'])} --> {seconds_to_srt_time(sent['end'])}")
        lines.append(sent["text"])
        lines.append("")
    return "\n".join(lines)

def refine_sentences(sentences: List[Dict], min_duration: float = 0.0) -> List[Dict]:
    if not sentences:
        return []
    out = [dict(s) for s in sentences if str(s.get("text", "")).strip()]
    for s in out:
        if s["end"] < s["start"]:
            s["end"] = s["start"]
        if s["end"] - s["start"] < 0.05:
            s["end"] = s["start"] + 0.05
    if min_duration and min_duration > 0:
        n = len(out)
        for i, s in enumerate(out):
            if s["end"] - s["start"] >= min_duration:
                continue
            limit = out[i + 1]["start"] - 0.01 if i + 1 < n else s["start"] + min_duration
            s["end"] = max(s["end"], min(s["start"] + min_duration, limit))
    for a, b in zip(out, out[1:]):
        if b["start"] < a["end"]:
            a["end"] = max(a["start"] + 0.05, b["start"] - 0.01)
            if b["start"] < a["end"]:
                b["start"] = a["end"]
    for s in out:
        if s["end"] <= s["start"]:
            s["end"] = s["start"] + 0.05
    return out

def build_sentence_srt_from_tokens(timestamps, token_texts, endings="。！？.!?") -> str:
    if not timestamps or not token_texts:
        return ""
    has_cjk = any(re.search(r'[\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af]', t) for t in token_texts)
    join_str = "" if has_cjk else " "
    sents, cur_start, cur = [], None, []
    for (s, e), t in zip(timestamps, token_texts):
        if cur_start is None:
            cur_start = s
        cur.append(t)
        if any(t.rstrip().endswith(p) for p in endings):
            sents.append({"start": cur_start, "end": e, "text": join_str.join(cur).strip()})
            cur_start, cur = None, []
    if cur:
        sents.append({"start": cur_start, "end": timestamps[-1][1], "text": join_str.join(cur).strip()})
    sents = refine_sentences(sents, min_duration=0.0)
    return sentences_to_srt(sents) if sents else ""

def or_breaks(a: Optional[List[bool]], b: Optional[List[bool]]) -> Optional[List[bool]]:
    if a is None:
        return list(b) if b is not None else None
    if b is None:
        return a
    n = max(len(a), len(b))
    out = [False] * n
    for i in range(n):
        out[i] = (i < len(a) and a[i]) or (i < len(b) and b[i])
    return out

def tokenize_with_fallback(tokenizer, text):
    """安全调用 tokenizer.tokenize，兼容返回一个或两个值的情况"""
    try:
        result = tokenizer.tokenize(text)
        if isinstance(result, tuple) and len(result) >= 2:
            return result[0], result[1]
        else:
            return result, None
    except Exception as e:
        logger.warning(f"tokenize 失败: {e}")
        return [], None

def compute_paragraph_breaks(paragraphs, words, token_ids, tokenizer, warnings) -> Optional[List[bool]]:
    if not paragraphs or not words or token_ids is None:
        return None
    total = len(words)
    force_break = [False] * total
    current_pos, matched = 0, 0
    for para in paragraphs:
        para_tokens, para_ids = tokenize_with_fallback(tokenizer, para)
        para_ids = [int(t) for t in (para_ids or [])]
        if not para_ids:
            continue
        found, n = -1, len(para_ids)
        # ID 匹配
        for start in range(current_pos, total - n + 1):
            if [int(t) for t in token_ids[start:start + n]] == para_ids:
                found = start
                break
        # 字符串兜底
        if found < 0 and para_tokens:
            para_clean = re.sub(r'[^\w\u4e00-\u9fff]', '', para)
            for start in range(current_pos, total - len(para_tokens) + 1):
                seg = ''.join(words[start:start + len(para_tokens)])
                if re.sub(r'[^\w\u4e00-\u9fff]', '', seg) == para_clean:
                    found = start
                    n = len(para_tokens)
                    break
        if found >= 0:
            end_idx = found + n - 1
            if end_idx < total - 1:
                force_break[end_idx] = True
            current_pos = end_idx + 1
            matched += 1
        else:
            current_pos += max(n, 1)
            warnings.append(f"段落匹配失败（已跳过）: '{para[:30]}...'")
    return force_break if matched else None

def compute_punc_breaks(text, words, tokenizer, punc_chars, warnings) -> Optional[List[bool]]:
    if not text or not words:
        return None
    positions = [i for i, ch in enumerate(text) if ch in punc_chars]
    if not positions:
        return None
    try:
        tokens_all, _ = tokenize_with_fallback(tokenizer, text)
    except Exception as e:
        warnings.append(f"标点断句分词失败: {e}")
        return None
    tokens_all = list(tokens_all or [])[:len(words)]
    char_to_token = [-1] * len(text)
    cur = 0
    for tidx, tok in enumerate(tokens_all):
        tl = len(tok)
        for i in range(tl):
            if cur + i < len(text):
                char_to_token[cur + i] = tidx
        cur += tl
        if cur >= len(text):
            break
    fb = [False] * len(words)
    used = 0
    for pos in positions:
        if pos < len(char_to_token):
            tidx = char_to_token[pos]
            if 0 <= tidx < len(words) - 1:
                fb[tidx] = True
                used += 1
    return fb if used else None

def compute_line_breaks(lines, words, token_ids, tokenizer, warnings):
    if not lines or not words or token_ids is None:
        return None, 0, len(lines or [])
    total = len(words)
    force_break = [False] * total
    current_pos, matched = 0, 0
    for line in lines:
        line_tokens, line_ids = tokenize_with_fallback(tokenizer, line)
        line_ids = [int(t) for t in (line_ids or [])]
        if not line_ids:
            continue
        found, n = -1, len(line_ids)
        for start in range(current_pos, total - n + 1):
            if [int(t) for t in token_ids[start:start + n]] == line_ids:
                found = start
                break
        if found < 0 and line_tokens:
            for start in range(current_pos, total - len(line_tokens) + 1):
                if list(words[start:start + len(line_tokens)]) == list(line_tokens):
                    found = start
                    n = len(line_tokens)
                    break
        if found >= 0:
            end_idx = found + n - 1
            if end_idx < total - 1:
                force_break[end_idx] = True
            current_pos = end_idx + 1
            matched += 1
        else:
            current_pos += max(n, 1)
            warnings.append(f"强制断行：行匹配失败 '{line[:30]}...'")
    return (force_break if matched else None), matched, len(lines)

def build_dual_outputs(sentences, sec_paragraphs):
    len_diff = abs(len(sec_paragraphs) - len(sentences))
    if len_diff > 1:
        return "", "", f"⚠️ 段落数相差 {len_diff} 段（超过1），跳过双语生成"
    warning = ""
    if len(sec_paragraphs) > len(sentences):
        sec_paragraphs = sec_paragraphs[:len(sentences)]
        warning = f"⚠️ 副文稿段落数多 {len_diff} 段，已自动截断"
    elif len(sentences) > len(sec_paragraphs):
        sec_paragraphs = sec_paragraphs + [""] * (len(sentences) - len(sec_paragraphs))
        warning = f"⚠️ 副文稿段落数少 {len_diff} 段，已补充空行"
    bad = pairs = 0
    for seg, sec in zip(sentences, sec_paragraphs):
        a, b = len(str(seg.get("text", ""))), len(sec)
        if a > 0 and b > 0:
            pairs += 1
            r = b / a
            if r < 0.2 or r > 5.0:
                bad += 1
    if pairs and bad / pairs > 0.5:
        warning += "\n⚠️ 多数段落主/副文稿长度差异过大，请核对副文稿是否与主稿逐段对应"
    sec_lines, dual_lines = [], []
    for i, (seg, sec_text) in enumerate(zip(sentences, sec_paragraphs), 1):
        st, et = seconds_to_srt_time(seg['start']), seconds_to_srt_time(seg['end'])
        sec_lines += [str(i), f"{st} --> {et}", sec_text, ""]
        dual_lines += [str(i), f"{st} --> {et}", seg['text'], sec_text, ""]
    return "\n".join(sec_lines), "\n".join(dual_lines), warning

def clean_text_for_anchor(text: str) -> str:
    text = re.sub(r'[^\u4e00-\u9fff\u3000-\u303f\uff00-\uffef a-zA-Z0-9，。！？；：“”‘’（）【】\n]', '', text)
    text = re.sub(r' +', ' ', text)
    return text

def ensure_break_length(breaks: Optional[List[bool]], target_len: int) -> Optional[List[bool]]:
    """确保断点列表长度与 target_len 一致，不足补 False，超出截断"""
    if breaks is None:
        return None
    if len(breaks) < target_len:
        return breaks + [False] * (target_len - len(breaks))
    return breaks[:target_len]

# ==================== 模型管理器 ====================
class FireRedAlignManager:
    def __init__(self):
        self.asr_system = None
        self.config = None
        self.lock = threading.RLock()
        self.temp_files = []
        self.model_dir = None

    def find_model_dir(self, preferred_type="AED"):
        base = ROOT_DIR / "pretrained_models"
        candidates = []
        for name in (f"FireRedASR2-{preferred_type}",
                     f"FireRedASR2-{preferred_type}-2025",
                     "FireRedASR2-AED", "FireRedASR2-AED-2025"):
            p = base / name
            if p.exists():
                candidates.append(p)
        if base.exists():
            for p in sorted(base.iterdir()):
                if not p.is_dir() or p in candidates:
                    continue
                if preferred_type.lower() in p.name.lower() or "aed" in p.name.lower():
                    has_model = any(p.glob("*.pt")) or any(p.glob("*.bin")) or \
                                any(p.glob("*.safetensors")) or any(p.glob("*.yaml")) or any(p.glob("*.json"))
                    if has_model:
                        candidates.append(p)
        for p in candidates:
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
                    use_gpu=config["use_gpu"], use_half=config["use_half"], return_timestamp=True
                )
                punc_config = FireRedPuncConfig(use_gpu=config["use_gpu"])
                system_config = FireRedAsr2SystemConfig(
                    vad_model_dir=vad_model_dir, lid_model_dir=lid_model_dir,
                    asr_model_dir=model_dir, punc_model_dir=punc_model_dir,
                    vad_config=vad_config, lid_config=lid_config,
                    asr_config=asr_config, punc_config=punc_config,
                    enable_vad=int(config["enable_vad"]), enable_lid=int(config["enable_lid"]),
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

    def get_asr(self):
        with self.lock:
            return self.asr_system.asr if self.asr_system is not None else None

    def _prepare_audio(self, audio_input, force_preprocess=False):
        audio_path = gradio_file_path(audio_input)
        if not audio_path or not os.path.exists(audio_path):
            logger.error(f"音频文件不存在: {audio_path}")
            return None

        cache_path = None
        if force_preprocess and FFMPEG_PATH != "ffmpeg":
            try:
                file_hash = _file_md5(audio_path)
            except Exception:
                file_hash = hashlib.md5(audio_path.encode()).hexdigest()
            cache_name = f"prep_{file_hash}_{os.path.basename(audio_path)}_16k.wav"
            cache_path = CACHE_DIR / cache_name
            if not cache_path.exists():
                logger.info(f"FFmpeg 预处理: {audio_path} -> {cache_path}")
                cmd = [FFMPEG_PATH, "-y", "-i", audio_path,
                       "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", str(cache_path)]
                try:
                    subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                    # 将生成的缓存文件加入临时列表
                    self.temp_files.append(str(cache_path))
                except subprocess.CalledProcessError as e:
                    err = e.stderr.decode(errors="ignore")[:200] if e.stderr else ""
                    logger.error(f"FFmpeg 预处理失败: {err}")
                    if cache_path.exists():
                        cache_path.unlink()
                    force_preprocess = False

        if force_preprocess and cache_path is not None and cache_path.exists():
            try:
                data, sr = sf.read(str(cache_path), dtype='float32')
                if data.ndim > 1:
                    data = data.mean(axis=1)
                if sr != 16000:
                    data = librosa.resample(data, orig_sr=sr, target_sr=16000)
                    sr = 16000
                return str(cache_path), data.astype(np.float32), sr
            except Exception as e:
                logger.error(f"读取缓存文件失败: {e}，回退 librosa")
                force_preprocess = False

        try:
            data, sr = librosa.load(audio_path, sr=None, mono=True)
            if sr != 16000:
                data = librosa.resample(data.astype(np.float32), orig_sr=sr, target_sr=16000)
                sr = 16000
            return audio_path, data.astype(np.float32), sr
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

    def _find_silences(self, waveform, sr, frame_len=0.05, top_db=35, min_silence=0.25) -> List[float]:
        try:
            w = np.asarray(waveform, dtype=np.float32)
            fl = max(int(sr * frame_len), 1)
            n = len(w) // fl
            if n < 4:
                return []
            rms = np.sqrt(np.mean(w[:n * fl].reshape(n, fl) ** 2, axis=1) + 1e-12)
            thresh = rms.max() * (10 ** (-top_db / 20))
            quiet = rms < thresh
            mids, start = [], None
            for i, q in enumerate(quiet):
                if q and start is None:
                    start = i
                elif not q and start is not None:
                    if (i - start) * frame_len >= min_silence:
                        mids.append(((start + i) / 2) * frame_len)
                    start = None
            if start is not None and (len(quiet) - start) * frame_len >= min_silence:
                mids.append(((start + len(quiet)) / 2) * frame_len)
            return sorted(mids)
        except Exception as e:
            logger.warning(f"静音检测失败: {e}")
            return []

    def _resolve_timestamp_units(self, starts, ends, duration, frame_shift):
        if not starts:
            return None
        max_start = max(starts)
        hyp = max_start * frame_shift
        if duration > 0 and frame_shift > 0 and abs(hyp - duration) <= 0.25 * duration:
            logger.info(f"时间戳单位检测：帧索引 (frame_shift={frame_shift:.4f}s, max={max_start:.0f})")
            return [(s * frame_shift, e * frame_shift) for s, e in zip(starts, ends)]
        if max_start <= duration * 1.2:
            logger.info(f"时间戳单位检测：秒 (max={max_start:.2f}, duration={duration:.2f})")
            return list(zip(starts, ends))
        if max_start > 0:
            scale = duration / max_start
            logger.info(f"时间戳单位检测：未知单位，线性缩放 scale={scale:.6f}")
            return [(s * scale, e * scale) for s, e in zip(starts, ends)]
        return None

    def _align_single(self, waveform, sr, text, asr, warnings):
        duration = float(len(waveform)) / sr
        if duration < 0.05:
            return None, None, None, "音频段过短"
        feats = enc_outputs = enc_lengths = yseq = None
        try:
            feats, lengths, _, _, _ = asr.feat_extractor([(16000, waveform)], ["tmp"])
            if not isinstance(lengths, torch.Tensor):
                lengths = torch.tensor(lengths, dtype=torch.long)
            else:
                lengths = lengths.long()
            use_gpu = bool(self.config and self.config.get("use_gpu"))
            if use_gpu and torch.cuda.is_available():
                feats = feats.cuda()
                lengths = lengths.cuda()
            if getattr(asr.config, "use_half", False):
                feats = feats.half()
            asr.model.eval()
            with torch.no_grad():
                enc_outputs, enc_lengths, _ = asr.model.encoder(feats, lengths)
            T = int(enc_outputs.size(1))
            if T <= 0:
                return None, None, None, "音频过短，无法对齐"
            frame_shift = duration / T

            try:
                tokens, token_ids = tokenize_with_fallback(asr.tokenizer, text)
            except Exception as te:
                cleaned = clean_text_for_anchor(text)
                try:
                    tokens, token_ids = tokenize_with_fallback(asr.tokenizer, cleaned)
                    warnings.append("原稿分词失败，已用清洗文本回退（对齐锚点可能轻微偏移）")
                except Exception as te2:
                    return None, None, None, f"文本分词失败: {te2}"
            token_ids = [int(t) for t in (token_ids or [])]
            if len(token_ids) == 0:
                return None, None, None, "参考文本分词后为空"

            yseq = torch.tensor(token_ids, device=enc_outputs.device)
            hyps = [[{"yseq": yseq}]]
            with torch.no_grad():
                nbest_hyps = asr.model.get_token_timestamp_torchaudio(enc_outputs, enc_lengths, hyps)
            timestamp = nbest_hyps[0][0].get("timestamp") if nbest_hyps and nbest_hyps[0] else None
            if timestamp is None:
                return None, None, None, "模型未返回时间戳"
            starts, ends = timestamp
            if starts is None or len(starts) == 0:
                return None, None, None, "时间戳为空"

            starts = [float(s) for s in starts]
            ends = [float(e) for e in ends]
            ts = self._resolve_timestamp_units(starts, ends, duration, frame_shift)
            if ts is None:
                return None, None, None, "无法解析时间戳单位"

            cleaned_ts = []
            for s, e in ts:
                s = max(0.0, s)
                e = max(s, min(e, duration))
                cleaned_ts.append((s, e))
            return cleaned_ts, list(tokens), token_ids, None
        except Exception as e:
            logger.error(f"对齐子任务失败: {e}", exc_info=True)
            return None, None, None, f"对齐失败: {e}"
        finally:
            del feats, enc_outputs, enc_lengths, yseq
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def _force_align_chunked(self, waveform, sr, paragraphs, asr, warnings):
        duration = float(len(waveform)) / sr
        char_lens = [max(len(p), 1) for p in paragraphs]
        total_chars = sum(char_lens)
        silences = self._find_silences(waveform, sr)

        boundaries = [0.0]
        acc = 0
        for cl in char_lens[:-1]:
            acc += cl
            target = duration * acc / total_chars
            b = None
            if silences:
                best = min(silences, key=lambda t: abs(t - target))
                if abs(best - target) <= 3.0:
                    b = best
            if b is None:
                b = target
            if b - boundaries[-1] < 0.5:
                b = min(boundaries[-1] + 0.5, duration)
            boundaries.append(b)
        boundaries.append(duration)
        for i in range(1, len(boundaries)):
            if boundaries[i] <= boundaries[i - 1]:
                boundaries[i] = boundaries[i - 1] + 0.01
        if boundaries[-1] > duration:
            boundaries[-1] = duration

        all_ts, all_tokens, all_ids, seg_last = [], [], [], []
        for i, para in enumerate(paragraphs):
            s, e = boundaries[i], boundaries[i + 1]
            if e - s < 0.1:
                warnings.append(f"分块 {i+1} 时长过短 ({e-s:.2f}s)，但仍尝试对齐...")
            seg = waveform[int(s * sr):int(e * sr)]
            ts, toks, tids, err = self._align_single(seg, sr, para, asr, warnings)
            if err:
                warnings.append(f"段落 {i + 1} 对齐失败: {err}")
                continue
            all_ts.extend([(a + s, b2 + s) for a, b2 in ts])
            all_tokens.extend(toks)
            all_ids.extend(tids)
            seg_last.append(len(all_ids) - 1)

        if not all_ts:
            return None
        breaks = [False] * len(all_ids)
        for idx in seg_last[:-1]:
            if 0 <= idx < len(all_ids) - 1:
                breaks[idx] = True
        logger.info(f"分块对齐完成：{len(all_ids)} tokens / {len(paragraphs)} 段")
        return all_ts, all_tokens, all_ids, breaks

    def _build_word_srt(self, timestamps_sec, token_texts):
        lines = []
        for i, ((s, e), txt) in enumerate(zip(timestamps_sec, token_texts), 1):
            lines.append(str(i))
            lines.append(f"{seconds_to_srt_time(s)} --> {seconds_to_srt_time(e)}")
            lines.append(txt)
            lines.append("")
        return "\n".join(lines)

    def force_align(self, audio_input, reference_text,
                    progress_callback=None, force_preprocess=False, enable_chunk=True):
        warnings_acc: List[str] = []

        def _err(msg):
            return None, None, None, None, None, None, msg, warnings_acc

        with self.lock:
            if self.asr_system is None:
                return _err("模型未加载")
            prep = self._prepare_audio(audio_input, force_preprocess)
            if prep is None:
                return _err("音频处理失败")
            audio_path, waveform, sr = prep
            if waveform is None or len(waveform) == 0:
                return _err("音频数据为空")

            duration = float(len(waveform)) / sr
            asr = self.asr_system.asr
            paragraphs = [p.strip() for p in reference_text.split('\n') if p.strip()]

            use_chunk = bool(enable_chunk and duration > LONG_AUDIO_CHUNK_THRESHOLD and len(paragraphs) >= 2)
            timestamps_sec = tokens = token_ids = para_breaks = None
            if use_chunk:
                logger.info(f"音频 {duration:.1f}s > {LONG_AUDIO_CHUNK_THRESHOLD:.0f}s，启用分块对齐（{len(paragraphs)} 段）")
                res = self._force_align_chunked(waveform, sr, paragraphs, asr, warnings_acc)
                if res is not None:
                    timestamps_sec, tokens, token_ids, para_breaks = res
                else:
                    warnings_acc.append("分块对齐失败，已回退整体对齐（大音频可能占用较多显存）")
                    use_chunk = False
            if not use_chunk:
                timestamps_sec, tokens, token_ids, err = self._align_single(
                    waveform, sr, reference_text, asr, warnings_acc)
                if err:
                    return _err(err)
                para_breaks = None

            if not timestamps_sec:
                return _err("未获取到有效时间戳")

            if len(timestamps_sec) < len(token_ids):
                dropped = len(token_ids) - len(timestamps_sec)
                warnings_acc.append(f"{dropped} 个 token 未获得时间戳（尾部内容可能缺失）")
            min_len = min(len(timestamps_sec), len(token_ids))
            timestamps_sec = timestamps_sec[:min_len]
            token_ids = [int(t) for t in token_ids[:min_len]]
            tokens = list(tokens[:min_len])
            token_texts = [asr.tokenizer.detokenize([tid]) for tid in token_ids]

            joined = "".join(token_texts)
            orig_cmp = re.sub(r'\s+', '', reference_text)
            if orig_cmp and any(re.search(r'[\u4e00-\u9fff]', t) for t in token_texts):
                if abs(len(joined) - len(orig_cmp)) > max(3, 0.15 * len(orig_cmp)):
                    warnings_acc.append("分词结果与原文字符数差异较大（可能为 BPE 分词器），段落/标点匹配可能不准")

            word_srt = self._build_word_srt(timestamps_sec, token_texts)
            sentence_srt = build_sentence_srt_from_tokens(timestamps_sec, token_texts)
            return word_srt, sentence_srt, timestamps_sec, token_texts, token_ids, para_breaks, None, warnings_acc


manager = FireRedAlignManager()

# ==================== 断句合并函数（原缺失，现提供实现） ====================
def merge_timestamps_to_sentences(
    timestamps, words,
    sentence_endings="。！？.!?",
    max_words=20, max_chars=30, max_duration=10.0,
    silence_threshold=0.3,
    merge_by_punc=False, merge_by_silence=False,
    merge_by_wordcount=False, merge_by_charcount=False,
    merge_by_duration=False,
    force_break_indices=None
):
    """
    将 token 级时间戳合并为句子级字幕。
    支持多种断句规则：
    - force_break_indices: 强制断点索引列表（True表示在此token后断句）
    - merge_by_punc: 根据标点断句（句末标点）
    - merge_by_silence: 根据 token 间静音时长断句
    - merge_by_wordcount: 达到最大词数断句
    - merge_by_charcount: 达到最大字符数断句
    - merge_by_duration: 达到最大时长断句
    返回 List[Dict]，包含 start, end, text
    """
    if not timestamps or not words:
        return []
    n = len(words)
    # 确保 force_break_indices 长度匹配
    if force_break_indices is None:
        force_break_indices = [False] * n
    else:
        force_break_indices = ensure_break_length(force_break_indices, n)

    sentences = []
    cur_start = timestamps[0][0]
    cur_end = timestamps[0][1]
    cur_text_parts = [words[0]]
    word_count = 1
    char_count = len(words[0])

    def should_break(i):
        """判断是否在索引 i 处断句（即当前句子的最后一个 token 是 i）"""
        if force_break_indices[i]:
            return True
        # 标点断句：当前词以句末标点结尾
        if merge_by_punc and any(words[i].rstrip().endswith(p) for p in sentence_endings):
            return True
        # 词数限制
        if merge_by_wordcount and word_count >= max_words:
            return True
        # 字符数限制
        if merge_by_charcount and char_count >= max_chars:
            return True
        # 时长限制（当前句子总时长达到阈值）
        if merge_by_duration and (timestamps[i][1] - cur_start) >= max_duration:
            return True
        # 静音断句：与下一个 token 之间的静音时长超过阈值
        if merge_by_silence and i < n - 1:
            gap = timestamps[i+1][0] - timestamps[i][1]
            if gap >= silence_threshold:
                return True
        return False

    for i in range(1, n):
        if should_break(i-1):
            sentences.append({
                "start": cur_start,
                "end": cur_end,
                "text": "".join(cur_text_parts).strip()
            })
            cur_start = timestamps[i][0]
            cur_end = timestamps[i][1]
            cur_text_parts = [words[i]]
            word_count = 1
            char_count = len(words[i])
        else:
            cur_end = timestamps[i][1]
            cur_text_parts.append(words[i])
            word_count += 1
            char_count += len(words[i])

    # 添加最后一句
    if cur_text_parts:
        sentences.append({
            "start": cur_start,
            "end": cur_end,
            "text": "".join(cur_text_parts).strip()
        })
    return sentences

# ==================== 锚点增强 ====================
def anchor_align_segments(words, word_timestamps, force_break_indices,
                          anchor_char_count=3,
                          use_anchor_start=False, use_anchor_end=False, use_anchor_mean=False):
    if not words or not word_timestamps:
        return []
    # 确保断点长度与 words 一致
    force_break_indices = ensure_break_length(force_break_indices, len(words))
    segments, cur_words, cur_ts = [], [], []
    for i, (word, (s, e)) in enumerate(zip(words, word_timestamps)):
        cur_words.append(word)
        cur_ts.append((s, e))
        if i < len(force_break_indices) and force_break_indices[i]:
            segments.append((cur_words, cur_ts))
            cur_words, cur_ts = [], []
    if cur_words:
        segments.append((cur_words, cur_ts))

    result_sentences = []
    for seg_words, seg_ts in segments:
        seg_text = "".join(seg_words).strip()
        if not seg_text:
            continue
        chinese_indices = [i for i, w in enumerate(seg_words) if re.search(r'[\u4e00-\u9fff]', w)]
        start_default, end_default = seg_ts[0][0], seg_ts[-1][1]
        if chinese_indices:
            n_front = min(anchor_char_count, len(chinese_indices))
            front_idx = chinese_indices[:n_front]
            start_front = sum(seg_ts[i][0] for i in front_idx) / n_front
        else:
            start_front = start_default
        if chinese_indices:
            n_back = min(anchor_char_count, len(chinese_indices))
            back_idx = chinese_indices[-n_back:]
            end_back = sum(seg_ts[i][1] for i in back_idx) / n_back
        else:
            end_back = end_default

        if use_anchor_mean:
            seg_start = (start_front + start_default) / 2
            seg_end = (end_back + end_default) / 2
        elif use_anchor_start:
            seg_start, seg_end = start_front, end_default
        elif use_anchor_end:
            seg_start, seg_end = start_default, end_back
        else:
            seg_start, seg_end = start_default, end_default

        if seg_end <= seg_start:
            seg_end = seg_start + 0.5
        result_sentences.append({"start": seg_start, "end": seg_end, "text": seg_text})

    result_sentences.sort(key=lambda s: s["start"])
    return refine_sentences(result_sentences, min_duration=0.0)

def get_system_status():
    lines = []
    if not FIRERED_AVAILABLE:
        lines.append(f"⚠️ FireRedASR2S 导入失败: {IMPORT_ERROR}")
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

# ==================== 单次对齐处理 ====================
def run_alignment(
    audio_file, primary_text, secondary_text, secondary_lang, enable_dual,
    use_gpu, use_half, model_dir_override,
    merge_punctuations, merge_max_words, merge_max_chars, merge_max_duration,
    merge_silence_threshold,
    merge_by_punc, merge_by_silence, merge_by_wordcount, merge_by_charcount,
    merge_by_duration, merge_by_newline,
    use_anchor_start, use_anchor_end, use_anchor_mean, anchor_char_count,
    force_preprocess, force_linebreak_mode=False,
    min_sent_duration=0.0, enable_chunk=True,
    progress=gr.Progress()
):
    if audio_file is None:
        return "错误: 请上传音频文件", "", "", "", "", "", "", get_system_status()
    if not primary_text or not primary_text.strip():
        return "错误: 请粘贴主文稿", "", "", "", "", "", "", get_system_status()

    audio_path = safe_audio_path(audio_file)
    if not audio_path or not os.path.exists(audio_path):
        return "错误: 无法获取有效的音频文件路径", "", "", "", "", "", "", get_system_status()

    if manager.asr_system is None:
        progress(0.05, desc="自动加载模型...")
        success, msg = manager.load_system(use_gpu, use_half, model_dir_override)
        if not success:
            return f"错误: {msg}", "", "", "", "", "", "", get_system_status()

    progress(0.2, desc="强制对齐中...")
    word_srt, sent_srt, timestamps, words, token_ids_out, para_breaks, error, align_warnings = \
        manager.force_align(audio_path, primary_text, force_preprocess=force_preprocess,
                            enable_chunk=enable_chunk)
    merge_warnings = list(align_warnings or [])
    if error:
        status = f"错误: {error}"
        w = cap_warnings(merge_warnings)
        if w:
            status += "\n" + w
        return status, "", "", "", "", "", "", get_system_status()
    if not timestamps or not words:
        return "错误: 未获取到有效时间戳", "", "", "", "", "", "", get_system_status()

    asr = manager.get_asr()
    if asr is None:
        return "错误: 模型在处理过程中被卸载，请重新加载后再试", "", "", "", "", "", "", get_system_status()

    # 保存自动断句规则快照（供强制断行失败回退）
    saved_auto = (merge_by_punc, merge_by_silence, merge_by_wordcount,
                  merge_by_charcount, merge_by_duration, merge_by_newline)
    anchor_enabled = use_anchor_start or use_anchor_end or use_anchor_mean
    final_force_break = None
    linebreak_active = False

    # ---------- 强制断行锚点模式 ----------
    if force_linebreak_mode:
        if para_breaks is not None:
            final_force_break = list(para_breaks)
            linebreak_active = True
            merge_warnings.append("长音频分块模式：强制断行直接采用分块段落边界")
        else:
            lines = [line.strip() for line in primary_text.split('\n') if line.strip()]
            if lines:
                fb, matched, total_lines = compute_line_breaks(
                    lines, words, token_ids_out, asr.tokenizer, merge_warnings)
                if fb and matched > 0:
                    final_force_break = fb
                    linebreak_active = True
                    if matched < total_lines:
                        merge_warnings.append(f"强制断行：{total_lines - matched}/{total_lines} 行未匹配，按可匹配行分段")
                else:
                    merge_warnings.append("强制断行：所有行均未匹配，已回退自动断句规则")
            else:
                merge_warnings.append("强制断行：文稿无有效行，已回退自动断句规则")
        if linebreak_active:
            merge_by_punc = merge_by_silence = merge_by_wordcount = False
            merge_by_charcount = merge_by_duration = merge_by_newline = False
        else:
            (merge_by_punc, merge_by_silence, merge_by_wordcount,
             merge_by_charcount, merge_by_duration, merge_by_newline) = saved_auto
            # 如果所有自动规则均为 False，则默认启用标点断句
            if not any([merge_by_punc, merge_by_silence, merge_by_wordcount,
                        merge_by_charcount, merge_by_duration, merge_by_newline]):
                merge_by_punc = True
                merge_warnings.append("强制断行失败且无其他断句规则，已自动启用按标点断句")
            final_force_break = list(para_breaks) if para_breaks is not None else None

    # ---------- 常规断句逻辑 ----------
    if not linebreak_active:
        progress(0.4, desc="处理空行断句...")
        if para_breaks is not None:
            final_force_break = list(para_breaks)
        elif merge_by_newline:
            paragraphs = [p.strip() for p in primary_text.split('\n') if p.strip()]
            if len(paragraphs) > 1:
                final_force_break = compute_paragraph_breaks(
                    paragraphs, words, token_ids_out, asr.tokenizer, merge_warnings)

        progress(0.55, desc="处理标点断句...")
        if merge_by_punc:
            punc_b = compute_punc_breaks(primary_text, words, asr.tokenizer,
                                         merge_punctuations, merge_warnings)
            final_force_break = or_breaks(final_force_break, punc_b)

    # ---------- 生成合并字幕 ----------
    progress(0.7, desc="生成合并字幕...")
    sentences = merge_timestamps_to_sentences(
        timestamps, words,
        sentence_endings=merge_punctuations,
        max_words=merge_max_words, max_chars=merge_max_chars,
        max_duration=merge_max_duration, silence_threshold=merge_silence_threshold,
        merge_by_punc=merge_by_punc, merge_by_silence=merge_by_silence,
        merge_by_wordcount=merge_by_wordcount, merge_by_charcount=merge_by_charcount,
        merge_by_duration=merge_by_duration,
        force_break_indices=final_force_break
    )
    sentences = refine_sentences(sentences, min_duration=min_sent_duration)
    merged_srt = sentences_to_srt(sentences)

    # 锚点增强
    anchor_srt = ""
    if anchor_enabled:
        if final_force_break and any(final_force_break):
            anchor_sentences = anchor_align_segments(
                words, timestamps, final_force_break,
                anchor_char_count=anchor_char_count,
                use_anchor_start=use_anchor_start,
                use_anchor_end=use_anchor_end,
                use_anchor_mean=use_anchor_mean
            )
            anchor_sentences = refine_sentences(anchor_sentences, min_duration=min_sent_duration)
            anchor_srt = sentences_to_srt(anchor_sentences)
        else:
            merge_warnings.append("锚点增强需要分段锚点（空行分段 / 强制断行 / 长音频分块），当前无分段，已跳过")

    # 双语挂载
    dual_srt = ""
    secondary_srt_str = ""
    warning_msg = ""
    progress(0.8, desc="处理双语挂载...")
    if enable_dual and secondary_text and secondary_text.strip():
        sec_paragraphs = [p.strip() for p in secondary_text.split('\n') if p.strip()]
        secondary_srt_str, dual_srt, dual_warn = build_dual_outputs(sentences, sec_paragraphs)
        warning_msg = dual_warn

    if merge_warnings:
        wtxt = "\n".join(merge_warnings)
        warning_msg = (warning_msg + "\n" + wtxt) if warning_msg else wtxt

    # 保存文件
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    base_name = Path(audio_path).stem
    prefix = f"{base_name}_align_{timestamp}"
    word_path = OUTPUT_DIR / f"{prefix}_words.srt"
    sent_path = OUTPUT_DIR / f"{prefix}_sentence.srt"
    merged_path = OUTPUT_DIR / f"{prefix}_merged.srt"
    merged_vtt_path = OUTPUT_DIR / f"{prefix}_merged.vtt"
    with open(word_path, "w", encoding="utf-8") as f:
        f.write(word_srt)
    with open(sent_path, "w", encoding="utf-8") as f:
        f.write(sent_srt)
    with open(merged_path, "w", encoding="utf-8") as f:
        f.write(merged_srt)
    with open(merged_vtt_path, "w", encoding="utf-8") as f:
        f.write(srt_to_vtt(merged_srt))

    anchor_path = None
    if anchor_srt:
        anchor_path = OUTPUT_DIR / f"{prefix}_anchor.srt"
        with open(anchor_path, "w", encoding="utf-8") as f:
            f.write(anchor_srt)

    safe_lang_tag = re.sub(r'[^\w\-]', '', secondary_lang.strip()) if secondary_lang else ""
    safe_lang_tag = f"_{safe_lang_tag}" if safe_lang_tag else ""
    sec_path = dual_path = dual_vtt_path = None
    if secondary_srt_str:
        sec_path = OUTPUT_DIR / f"{prefix}{safe_lang_tag}_secondary.srt"
        with open(sec_path, "w", encoding="utf-8") as f:
            f.write(secondary_srt_str)
    if dual_srt:
        dual_path = OUTPUT_DIR / f"{prefix}{safe_lang_tag}_dual.srt"
        with open(dual_path, "w", encoding="utf-8") as f:
            f.write(dual_srt)
        dual_vtt_path = OUTPUT_DIR / f"{prefix}{safe_lang_tag}_dual.vtt"
        with open(dual_vtt_path, "w", encoding="utf-8") as f:
            f.write(srt_to_vtt(dual_srt))

    status = (f"✅ 对齐完成！\n逐词字幕: {word_path.name}\n整句字幕: {sent_path.name}\n"
              f"合并字幕: {merged_path.name} (+ .vtt)")
    if anchor_path:
        status += f"\n锚点字幕: {anchor_path.name}"
    if sec_path:
        status += f"\n副文稿单语: {sec_path.name}"
    if dual_path:
        status += f"\n双语字幕: {dual_path.name} (+ .vtt)"
    w = cap_warnings(merge_warnings)
    if w:
        status += "\n" + w

    try:
        if merge_warnings:
            gr.Warning(f"处理完成，但有 {len(merge_warnings)} 条警告，详见任务状态/日志")
    except Exception:
        logger.warning(warning_msg or "")

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

def reset_inputs_defaults():
    """返回所有可重置输入组件的默认值"""
    return [
        None,               # audio_input
        "",                 # primary_text
        "",                 # secondary_text
        "",                 # secondary_lang
        False,              # enable_dual
        torch.cuda.is_available(),  # use_gpu
        False,              # use_half
        "",                 # model_dir_override
        "。！？.!?",        # punc_box
        20,                 # max_words_slider
        30,                 # max_chars_slider
        10.0,               # max_duration_slider
        0.3,                # silence_slider
        False,              # merge_punc
        False,              # merge_silence
        False,              # merge_wordcount
        False,              # merge_charcount
        False,              # merge_duration
        True,               # merge_newline
        False,              # use_anchor_start
        False,              # use_anchor_end
        False,              # use_anchor_mean
        3,                  # anchor_char_count
        True,               # force_preprocess_check
        False,              # force_linebreak_cb
        0.0,                # min_dur_slider
        True                # chunk_cb
    ]

# ==================== 批量处理 ====================
def batch_process(
    audio_files, text_files, text_files_secondary, enable_dual_batch, secondary_lang_batch,
    use_gpu, use_half, model_dir_override,
    merge_punctuations, merge_max_words, merge_max_chars, merge_max_duration,
    merge_silence_threshold,
    merge_by_punc, merge_by_silence, merge_by_wordcount, merge_by_charcount,
    merge_by_duration, merge_by_newline,
    force_preprocess, enable_chunk=True,
    progress=gr.Progress()
):
    if not audio_files or not text_files:
        return "请上传音频文件和对应的文稿文件", get_system_status()
    if len(audio_files) != len(text_files):
        return f"音频文件数量 ({len(audio_files)}) 与文稿文件数量 ({len(text_files)}) 不一致", get_system_status()

    sec_file_list = list(text_files_secondary) if text_files_secondary else []
    if enable_dual_batch:
        if len(sec_file_list) != len(audio_files):
            return (f"已勾选双语字幕，但副文稿数量 ({len(sec_file_list)}) 与音频数量 ({len(audio_files)}) 不一致；"
                    f"如不需双语请取消勾选", get_system_status())

    if manager.asr_system is None:
        progress(0.02, desc="自动加载模型...")
        success, msg = manager.load_system(use_gpu, use_half, model_dir_override)
        if not success:
            return f"模型加载失败: {msg}", get_system_status()

    results = []
    total = len(audio_files)
    safe_lang_tag = re.sub(r'[^\w\-]', '', (secondary_lang_batch or "").strip())
    safe_lang_tag = f"_{safe_lang_tag}" if safe_lang_tag else ""

    for idx, (audio_obj, text_obj) in enumerate(zip(audio_files, text_files)):
        progress(idx / total, desc=f"处理 {idx + 1}/{total}...")
        audio_path = safe_audio_path(audio_obj)
        name = os.path.basename(audio_path) if audio_path else "未知"
        if not audio_path or not os.path.exists(audio_path):
            results.append(f"❌ {name}: 音频文件无效")
            continue
        try:
            text_path = gradio_file_path(text_obj)
            if not text_path or not os.path.exists(text_path):
                results.append(f"❌ {name}: 文稿文件无效")
                continue
            primary_text = read_text_robust(text_path)
            if not primary_text.strip():
                results.append(f"❌ {name}: 文稿内容为空")
                continue
        except Exception as e:
            results.append(f"❌ {name}: 读取文稿失败 - {e}")
            continue

        try:
            word_srt, sent_srt, timestamps, words, token_ids_out, para_breaks, error, align_warnings = \
                manager.force_align(audio_path, primary_text,
                                    force_preprocess=force_preprocess, enable_chunk=enable_chunk)
            merge_warnings = list(align_warnings or [])
            if error:
                results.append(f"❌ {name}: 对齐失败 - {error}")
                continue
            if not timestamps or not words:
                results.append(f"❌ {name}: 未获取到有效时间戳")
                continue

            asr = manager.get_asr()
            if asr is None:
                results.append(f"❌ {name}: 模型在处理过程中被卸载")
                continue

            # 段落 / 标点断句
            final_force_break = None
            if para_breaks is not None:
                final_force_break = list(para_breaks)
            elif merge_by_newline:
                paragraphs = [p.strip() for p in primary_text.split('\n') if p.strip()]
                if len(paragraphs) > 1:
                    final_force_break = compute_paragraph_breaks(
                        paragraphs, words, token_ids_out, asr.tokenizer, merge_warnings)
            if merge_by_punc:
                punc_b = compute_punc_breaks(primary_text, words, asr.tokenizer,
                                             merge_punctuations, merge_warnings)
                final_force_break = or_breaks(final_force_break, punc_b)

            sentences = merge_timestamps_to_sentences(
                timestamps, words,
                sentence_endings=merge_punctuations,
                max_words=merge_max_words, max_chars=merge_max_chars,
                max_duration=merge_max_duration, silence_threshold=merge_silence_threshold,
                merge_by_punc=merge_by_punc, merge_by_silence=merge_by_silence,
                merge_by_wordcount=merge_by_wordcount, merge_by_charcount=merge_by_charcount,
                merge_by_duration=merge_by_duration,
                force_break_indices=final_force_break
            )
            sentences = refine_sentences(sentences, min_duration=0.0)
            merged_srt = sentences_to_srt(sentences)

            # 双语
            dual_note = ""
            if enable_dual_batch and idx < len(sec_file_list):
                try:
                    sec_path_txt = gradio_file_path(sec_file_list[idx])
                    sec_text = read_text_robust(sec_path_txt)
                    sec_paragraphs = [p.strip() for p in sec_text.split('\n') if p.strip()]
                    secondary_srt, dual_srt, dual_warn = build_dual_outputs(sentences, sec_paragraphs)
                    if dual_warn:
                        merge_warnings.append(dual_warn.strip())
                    if dual_srt:
                        with open(OUTPUT_DIR / f"{Path(audio_path).stem}_batch_dual{safe_lang_tag}.srt",
                                  "w", encoding="utf-8") as f:
                            f.write(dual_srt)
                        with open(OUTPUT_DIR / f"{Path(audio_path).stem}_batch_dual{safe_lang_tag}.vtt",
                                  "w", encoding="utf-8") as f:
                            f.write(srt_to_vtt(dual_srt))
                        dual_note = " + 双语"
                    if secondary_srt:
                        with open(OUTPUT_DIR / f"{Path(audio_path).stem}_batch_secondary{safe_lang_tag}.srt",
                                  "w", encoding="utf-8") as f:
                            f.write(secondary_srt)
                except Exception as e:
                    merge_warnings.append(f"双语生成失败: {e}")

            timestamp = time.strftime("%Y%m%d_%H%M%S")
            prefix = f"{Path(audio_path).stem}_align_{timestamp}"
            with open(OUTPUT_DIR / f"{prefix}_words.srt", "w", encoding="utf-8") as f:
                f.write(word_srt)
            with open(OUTPUT_DIR / f"{prefix}_sentence.srt", "w", encoding="utf-8") as f:
                f.write(sent_srt)
            with open(OUTPUT_DIR / f"{prefix}_merged.srt", "w", encoding="utf-8") as f:
                f.write(merged_srt)
            with open(OUTPUT_DIR / f"{prefix}_merged.vtt", "w", encoding="utf-8") as f:
                f.write(srt_to_vtt(merged_srt))

            msg = f"✅ {name}: 已生成{dual_note}"
            if merge_warnings:
                msg += f"（{len(merge_warnings)} 条警告: " + "；".join(merge_warnings[:3]) + ("…" if len(merge_warnings) > 3 else "") + "）"
            results.append(msg)
        except Exception as e:
            logger.error(f"批量处理 {name} 异常: {e}", exc_info=True)
            results.append(f"❌ {name}: 处理异常 - {e}")
        finally:
            # 显存清理
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

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

    with gr.Blocks(title="FireRedASR2S 文稿对齐 + 双语字幕生成（锚点增强版）",
                   theme=gr.themes.Default()) as demo:
        gr.Markdown("# 🎬 FireRedASR2S 文稿对齐 + 双语字幕生成（锚点增强版）")

        with gr.Tabs():
            # ---------- 单次处理 ----------
            with gr.Tab("单次处理"):
                with gr.Row():
                    with gr.Column(scale=1):
                        audio_input = gr.File(label="选择音频文件", file_types=AUDIO_EXTS)
                        force_preprocess_check = gr.Checkbox(label="⚡ 强制预处理为 16kHz 单声道 (推荐)", value=True)
                        chunk_cb = gr.Checkbox(
                            label="🧩 长音频自动分块对齐（>5分钟按段落切分）", value=True,
                            info="音频超过 5 分钟且文稿有多个段落时，按段落字符比例 + 静音点切分音频逐段对齐，防止显存溢出")
                        primary_text = gr.Textbox(label="主文稿（对齐用）", lines=18,
                                                  placeholder="粘贴与音频内容一致的稿子...\n段落之间用空行分隔")
                        secondary_text = gr.Textbox(label="副文稿（挂载用，可选）", lines=18,
                                                    placeholder="粘贴翻译稿...\n段落结构尽量与主文稿一致")
                        with gr.Row():
                            secondary_lang = gr.Textbox(label="副文稿语言标记", placeholder="en", value="", scale=1)
                            enable_dual = gr.Checkbox(label="生成双语字幕", value=False, scale=1)

                    with gr.Column(scale=2):
                        with gr.Row():
                            system_status = gr.Textbox(label="系统状态", value=get_system_status(),
                                                       lines=4, interactive=False, scale=1)
                            task_status = gr.Textbox(label="任务状态", value="等待开始",
                                                     lines=4, interactive=False, scale=1)

                with gr.Accordion("⚙️ 模型控制", open=True):
                    with gr.Row():
                        use_gpu = gr.Checkbox(label="使用 GPU", value=torch.cuda.is_available())
                        default_half = torch.cuda.is_available() and torch.cuda.get_device_properties(0).total_memory < 10 * 1024**3
                        use_half = gr.Checkbox(label="使用半精度 (FP16)", value=default_half)
                        model_dir_override = gr.Textbox(label="模型目录（可选）",
                                                        placeholder="留空自动检测，或指定完整路径", value="")
                    with gr.Row():
                        load_model_btn = gr.Button("加载模型", variant="primary")
                        unload_model_btn = gr.Button("卸载模型", variant="secondary")
                        refresh_status_btn = gr.Button("刷新状态", variant="secondary")

                with gr.Accordion("📝 断句规则（推荐只勾选“按空行分段”）", open=True):
                    merge_newline = gr.Checkbox(label="按空行分段（推荐）", value=True)
                    with gr.Accordion("更多规则（可选）", open=False):
                        with gr.Row():
                            merge_punc = gr.Checkbox(label="按标点断句", value=False)
                            merge_silence = gr.Checkbox(label="按静音断句", value=False)
                            merge_wordcount = gr.Checkbox(label="按词数断句", value=False)
                        with gr.Row():
                            merge_charcount = gr.Checkbox(label="按字符数断句", value=False)
                            merge_duration = gr.Checkbox(label="按时长断句", value=False)
                        with gr.Row():
                            punc_box = gr.Textbox(label="句末标点", value="。！？.!?", scale=2)
                            silence_slider = gr.Slider(0.1, 1.0, value=0.3, step=0.05, label="静音阈值 (秒)")
                        with gr.Row():
                            max_words_slider = gr.Slider(5, 50, value=20, step=1, label="最大词数")
                            max_chars_slider = gr.Slider(5, 100, value=30, step=5, label="最大字符数")
                            max_duration_slider = gr.Slider(1.0, 20.0, value=10.0, step=0.5, label="最大时长 (秒)")
                        min_dur_slider = gr.Slider(0.0, 3.0, value=0.0, step=0.1,
                                                   label="最短字幕时长 (秒，0=不限制)",
                                                   info="过短的字幕会向后延长，不会与下一条重叠")

                with gr.Accordion("🎯 锚点增强与分段 (实验性，默认关闭)", open=False):
                    with gr.Row():
                        gr.Markdown("**字符锚点微调**（需配合空行分段）")
                    with gr.Row():
                        use_anchor_start = gr.Checkbox(label="前锚点", value=False, info="段落开头取前几个汉字校准开始时间")
                        use_anchor_end = gr.Checkbox(label="后锚点", value=False, info="段落结尾取后几个汉字校准结束时间")
                        use_anchor_mean = gr.Checkbox(label="前后均值", value=False, info="开始/结束取前后锚点的平均值，覆盖单独选项")
                    with gr.Row():
                        anchor_char_count = gr.Slider(1, 5, value=3, step=1, label="锚点参考汉字数")
                    gr.Markdown("以下选项为特殊场景设计，**通常无需开启**。空行分段已能覆盖绝大多数文稿。")
                    force_linebreak_cb = gr.Checkbox(
                        label="📌 强制断行锚点（按稿子原始换行强制分段）", value=False,
                        info="开启后完全依照换行分段，忽略标点、静音等自动规则。仅推荐极规整的纯文本使用。"
                             "若所有行均匹配失败，将自动回退到常规断句规则。"
                    )

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
                        audio_files = gr.File(label="上传音频文件（可多选）", file_count="multiple", file_types=AUDIO_EXTS)
                        text_files = gr.File(label="上传对应的文稿文件（顺序对应）", file_count="multiple", file_types=[".txt"])
                        text_files_secondary = gr.File(label="上传对应的副文稿/翻译稿（可选，需与音频数量一致）",
                                                       file_count="multiple", file_types=[".txt"])
                        with gr.Row():
                            enable_dual_batch = gr.Checkbox(label="生成双语字幕（需上传副文稿）", value=False)
                            secondary_lang_batch = gr.Textbox(label="语言标记", placeholder="en", value="", scale=1)
                        force_preprocess_batch = gr.Checkbox(label="⚡ 强制预处理为 16kHz 单声道", value=True)
                        chunk_cb_batch = gr.Checkbox(label="🧩 长音频自动分块对齐（>5分钟）", value=True)
                    with gr.Column(scale=2):
                        batch_status = gr.Textbox(label="批量处理状态", lines=12, interactive=False)
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
1. 上传音频文件（支持 wav/mp3/m4a/flac/ogg/aac/wma/opus）
2. 粘贴主文稿（与音频内容一致，段落间用空行分隔）
3. （可选）粘贴副文稿（翻译稿）并勾选“生成双语字幕”
4. 调整模型设置和合并规则
5. 可选：启用强制断行锚点，完全按稿子原始换行分段（全部匹配失败会自动回退常规规则）
6. 可选：启用锚点增强，利用段落前几个汉字精准校准段边界
7. 点击“开始对齐”，完成后字幕文件保存在输出目录，同时导出 .vtt 版本

## 注意事项
- **长音频**：超过 5 分钟且文稿有多个段落时，默认启用分块对齐（按段落字符比例 + 静音点切分），防止显存溢出
- **分词器**：段落/标点匹配依赖字符级 tokenizer；若为 BPE 类分词器，界面上会出现一致性警告，此时建议使用“按空行分段”并以警告为准
- **任务取消**：暂不支持中途取消，长任务请耐心等待；处理期间卸载模型会在任务结束后生效
- **双语对齐**：副文稿按空行分段后与主文稿段落数需一致（±1）；长度差异过大会给出错配提示
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

        load_model_btn.click(load_model_action,
                             inputs=[use_gpu, use_half, model_dir_override],
                             outputs=[task_status, system_status])
        unload_model_btn.click(unload_model_action, outputs=[task_status, system_status])
        refresh_status_btn.click(refresh_status_action, outputs=[system_status])

        run_btn.click(
            run_alignment,
            inputs=[
                audio_input, primary_text, secondary_text, secondary_lang, enable_dual,
                use_gpu, use_half, model_dir_override,
                punc_box, max_words_slider, max_chars_slider, max_duration_slider, silence_slider,
                merge_punc, merge_silence, merge_wordcount, merge_charcount, merge_duration, merge_newline,
                use_anchor_start, use_anchor_end, use_anchor_mean, anchor_char_count,
                force_preprocess_check, force_linebreak_cb,
                min_dur_slider, chunk_cb
            ],
            outputs=[task_status, word_output, sent_output, merged_output,
                     secondary_output, dual_output, anchor_output, system_status]
        )

        clear_btn.click(
            clear_outputs,
            outputs=[task_status, word_output, sent_output, merged_output,
                     secondary_output, dual_output, anchor_output, system_status]
        ).then(
            reset_inputs_defaults,
            outputs=[audio_input, primary_text, secondary_text, secondary_lang, enable_dual,
                     use_gpu, use_half, model_dir_override,
                     punc_box, max_words_slider, max_chars_slider, max_duration_slider, silence_slider,
                     merge_punc, merge_silence, merge_wordcount, merge_charcount, merge_duration, merge_newline,
                     use_anchor_start, use_anchor_end, use_anchor_mean, anchor_char_count,
                     force_preprocess_check, force_linebreak_cb,
                     min_dur_slider, chunk_cb]
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
                audio_files, text_files, text_files_secondary, enable_dual_batch, secondary_lang_batch,
                use_gpu, use_half, model_dir_override,
                punc_box, max_words_slider, max_chars_slider, max_duration_slider, silence_slider,
                merge_punc, merge_silence, merge_wordcount, merge_charcount, merge_duration, merge_newline,
                force_preprocess_batch, chunk_cb_batch
            ],
            outputs=[batch_status, batch_system]
        )

        gr.HTML("""
        <div style="text-align: center; color: #666; font-size: 0.85em; margin-top: 20px;">
            <p>© 2026 光影紐扣 | 基于 FireRedASR2S (Apache 2.0) 修复版 v3</p>
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
        msg = f"FireRedASR2S 模块不可用，请检查环境。错误: {IMPORT_ERROR}"
        logger.error(msg)
        print(msg)
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