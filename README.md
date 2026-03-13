# 🎬 FireRedASR2S 便携版 - 语音字幕工作站

<div align="center">
  <h3>本地离线 · 高效便捷 · 开源免费</h3>
  <p>一个集语音识别、强制对齐、字幕清洗、格式转换、翻译助手于一体的全能字幕工具箱</p>
</div>

---

## ✨ 功能亮点

- 🎤 **高精度语音转字幕**：基于 FireRedASR2S 工业级 ASR 系统，中文识别准确率顶尖，支持长音频、视频文件。
- ⏱️ **强制对齐（自动拍唱词）**：上传稿子 + 配音，一键生成精准时间轴，支持空行断句、字数/词数/时长限制。
- 🧹 **字幕清洗**：一键去除语气词、自定义替换规则、简繁转换。
- 🔄 **格式转换**：SRT / TXT / LRC 互转，双语合并（中文在上，英文在下）。
- 🌐 **在线翻译助手**：内置专业提示词模板，一键打开 DeepL、豆包、DeepSeek 等网站，方便人工二次优化。
- 🤖 **双语字幕 API 版**：支持 DeepSeek / 通义千问 / OpenAI 等 API，批量上下文翻译，输出双语字幕。
- 🧩 **模块化设计**：各功能独立脚本，互不干扰，便于维护和扩展。
- 📂 **统一输出目录**：所有生成文件自动保存至 `output/` 文件夹，井井有条。
- ⚙️ **可配置高级参数**：解码参数、VAD 阈值、标点阈值等，满足专业用户需求。
- 🔒 **完全离线**：所有处理均在本地，不上传任何数据，保护隐私。

---

## 📦 下载与安装

### 系统要求
- Windows 10 / 11 64位
- NVIDIA 显卡（建议显存 ≥ 8GB，GTX 1080 实测流畅）
- 已安装 Visual C++ Redistributable 2015-2022

### 获取便携包
从 - B站 主页：[光影的故事2018](https://space.bilibili.com/381518712)`，解压到**纯英文路径**（如 `D:\ASR_Tools`），切勿放在中文目录下。

---

## 🚀 快速开始

### 第一步：下载模型
本工具**不包含模型文件**，你需要自行下载以下四个模型，并放置到 `pretrained_models` 文件夹中。

**国内用户（推荐使用 ModelScope）**：
```bash
# 安装 modelscope
python_embeded\python.exe -m pip install modelscope

# 下载模型（在软件根目录打开命令行执行）
python_embeded\python.exe -m modelscope download --model FireRedTeam/FireRedASR2-AED --local_dir ./pretrained_models/FireRedASR2-AED
python_embeded\python.exe -m modelscope download --model FireRedTeam/FireRedVAD --local_dir ./pretrained_models/FireRedVAD
python_embeded\python.exe -m modelscope download --model FireRedTeam/FireRedLID --local_dir ./pretrained_models/FireRedLID
python_embeded\python.exe -m modelscope download --model FireRedTeam/FireRedPunc --local_dir ./pretrained_models/FireRedPunc
```

**国际用户（Hugging Face）**：
```bash
# 安装 huggingface_hub
python_embeded\python.exe -m pip install huggingface_hub[cli]

# 下载模型
python_embeded\python.exe -m huggingface_hub.cli download FireRedTeam/FireRedASR2-AED --local-dir ./pretrained_models/FireRedASR2-AED
python_embeded\python.exe -m huggingface_hub.cli download FireRedTeam/FireRedVAD --local-dir ./pretrained_models/FireRedVAD
python_embeded\python.exe -m huggingface_hub.cli download FireRedTeam/FireRedLID --local-dir ./pretrained_models/FireRedLID
python_embeded\python.exe -m huggingface_hub.cli download FireRedTeam/FireRedPunc --local-dir ./pretrained_models/FireRedPunc
```

下载完成后，目录结构应如下：
```
pretrained_models/
├── FireRedASR2-AED/
├── FireRedLID/
├── FireRedPunc/
└── FireRedVAD/
    └── VAD/
```

### 第二步：启动主界面
双击根目录下的 `启动器.bat`，稍等片刻浏览器将自动打开主界面（地址 `http://127.0.0.1:7868`）。

如果你需要手动启动，也可以执行：
```cmd
python_embeded\python.exe Index.py
```

---

## 🧩 各模块使用说明

### 1️⃣ 语音转字幕 (`firered_webui_pro.py`)
- **功能**：音频/视频识别，生成带时间戳的 SRT 字幕。
- **操作**：
  - 上传音频（或录制），点击「开始识别」。
  - 可调节高级参数（解码、VAD、标点等）。
  - 结果可在“识别文本”、“时间戳”、“SRT字幕”标签页查看，并可复制或保存。

### 2️⃣ 字幕清洗 (`clean_subtitle.py`)
- **功能**：去除语气词、自定义替换、简繁转换。
- **操作**：
  - 上传字幕文件或粘贴文本，选择清洗模式（温和/激进）。
  - 支持自定义词库（每行一个词），可加载/保存词库文件。
  - 清洗后文件自动保存至 `output/字幕清洗`。

### 3️⃣ 格式转换与双语合并 (`subtitle_utils.py`)
- **功能**：双语合并、SRT转TXT、LRC转SRT、中文字幕添加拼音。
- **操作**：
  - 在对应标签页上传文件，设置参数，点击转换即可。

### 4️⃣ 在线AI助手 (`AI_translator.py`)
- **功能**：内置专业提示词模板，一键打开常用翻译/大模型网站。
- **操作**：
  - 选择模板，复制提示词，点击按钮打开网站，粘贴使用。

### 5️⃣ 双语字幕API版 (`subtitle_translator_pro.py`)
- **功能**：通过 API 批量翻译字幕（需自备 API Key）。
- **操作**：
  - 上传字幕文件，填写 API 信息，点击开始翻译。
  - 支持上下文窗口，翻译更连贯。

---

## ⚙️ 高级参数与自定义

- **强制对齐参数**：在“字幕自动打轴”标签页，可设置句末标点、最大词数、最大字符数、最大时长、静音阈值等，并选择断句规则（标点/静音/词数/字数/空行）。
- **系统信息**：可自定义输出目录、预览文件大小阈值、清理临时文件、保存/加载配置。
- **配置文件**：所有设置自动保存在 `preset/settings.json`，也可导出为预设文件。

---

## ⚠️ 重要注意事项

1. **路径限制**：请勿将软件包放在**包含中文或空格**的路径下，否则可能导致部分库加载失败。
2. **不要随意改名**：根目录文件夹名建议保持 `FireRedASR2S_portable_win`，否则可能影响路径引用。
3. **核心文件**：以下文件/文件夹请勿移动或删除：
   - `python_embeded/`（独立 Python 环境）
   - `FireRedASR2S/`（核心代码）
   - `ffmpeg/`（视频处理工具）
   - `scripts/`（各功能脚本）
   - `Index.py`、`启动器.bat`、`download_models.py`
4. **可清理目录**：`output/`、`preset/`、`logs/` 中的文件可以安全删除。
5. **模型下载**：务必从官方渠道下载模型，并保留其中的许可证文件。

---

## 📜 版权与免责声明

- 本项目基于 **FireRedASR2S**（[Apache 2.0 许可证](https://github.com/FireRedTeam/FireRedASR2S)）二次开发。
- 原创代码（WebUI、启动脚本、辅助工具）© 2026 [光影的故事2018](https://space.bilibili.com/381518712)，保留所有权利。
- 本软件包**不包含任何模型文件**，模型版权归原开发者所有，用户需自行下载并遵守其许可证。
- 软件按“原样”提供，不提供任何担保。使用本软件所产生的一切风险由用户自行承担。
- 本软件包仅限个人学习与研究使用，禁止用于商业用途或侵权行为。

---

## 🤝 贡献指南

欢迎提交 Issue 和 Pull Request！如果你想贡献代码，请遵循以下原则：
- 保持代码风格一致，添加必要注释。
- 确保修改不影响原有功能。
- 涉及模型调用时，注意兼容性。

---

## 📬 联系与反馈

- B站 主页：[光影的故事2018](https://space.bilibili.com/381518712)
- 问题反馈：在 GitHub 提交 Issue，或通过 B站 私信。

---

## 🌟 支持项目

如果这个工具对你有所帮助，欢迎给项目点一个 ⭐，让更多人看到！  
你的支持是我持续更新的动力。

---

**© 2026 光影紐扣 版权所有 | 基于 FireRedASR2S (Apache 2.0) 二次开发**
