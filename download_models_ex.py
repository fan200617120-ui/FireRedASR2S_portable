#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
FireRedASR2S WebUI Professional Edition
Copyright 2026 光影的故事2018

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

模型下载脚本 - 手动运行
下载 FireRedASR2-AED、FireRedVAD、FireRedLID、FireRedPunc 以及可选的 FireRedASR2-LLM
"""

import os
import subprocess
import sys
from pathlib import Path

BASE_DIR = Path(__file__).parent.absolute()
PYTHON_EXE = BASE_DIR / "python_embeded" / "python.exe"
SCRIPTS_DIR = BASE_DIR / "python_embeded" / "Scripts"

if not PYTHON_EXE.exists():
    print("❌ 未找到嵌入版 Python，请确保 python_embeded 目录存在")
    input("按 Enter 键退出...")
    sys.exit(1)

MODEL_DIR = BASE_DIR / "pretrained_models"
MODEL_DIR.mkdir(exist_ok=True)

# 模型列表（核心四模型 + 可选的 LLM）
CORE_MODELS = ["FireRedASR2-AED", "FireRedVAD", "FireRedLID", "FireRedPunc"]
LLM_MODEL = "FireRedASR2-LLM"

def get_model_id(model_name, source):
    """根据下载源返回完整的模型 ID"""
    if source == '1':  # ModelScope
        return f"xukaituo/{model_name}"
    else:              # Hugging Face
        # Hugging Face 上官方路径为 FireRedTeam/，如需改用 xukaituo 请自行修改
        return f"FireRedTeam/{model_name}"

def run_cmd(cmd):
    """执行命令并打印输出"""
    print(f"执行: {' '.join(cmd)}")
    try:
        subprocess.run(cmd, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"命令执行失败: {e}")
        return False

def download_models(source_choice, download_llm):
    """根据用户选择下载模型"""
    steps = []

    if source_choice == '1':
        # 国内用户 (ModelScope)
        print("\n使用 ModelScope 下载，适合国内用户")
        if not run_cmd([str(PYTHON_EXE), "-m", "pip", "install", "-U", "modelscope", "-i", "https://pypi.tuna.tsinghua.edu.cn/simple"]):
            return False
        modelscope_exe = SCRIPTS_DIR / "modelscope.exe"
        if not modelscope_exe.exists():
            modelscope_exe = SCRIPTS_DIR / "modelscope"
        if not modelscope_exe.exists():
            print("❌ 找不到 modelscope 可执行文件，请检查安装")
            return False

        # 核心模型
        for model in CORE_MODELS:
            model_id = get_model_id(model, source_choice)
            steps.append([
                str(modelscope_exe), "download", "--model", model_id,
                "--local_dir", str(MODEL_DIR / model)
            ])
        # 可选 LLM
        if download_llm:
            model_id = get_model_id(LLM_MODEL, source_choice)
            steps.append([
                str(modelscope_exe), "download", "--model", model_id,
                "--local_dir", str(MODEL_DIR / LLM_MODEL)
            ])
    else:
        # 国际用户 (Hugging Face)
        print("\n使用 Hugging Face 下载，适合国际用户")
        if not run_cmd([str(PYTHON_EXE), "-m", "pip", "install", "-U", "huggingface_hub[cli]", "-i", "https://pypi.tuna.tsinghua.edu.cn/simple"]):
            return False
        cli_exe = SCRIPTS_DIR / "huggingface-cli.exe"
        if not cli_exe.exists():
            cli_exe = SCRIPTS_DIR / "huggingface-cli"
        if not cli_exe.exists():
            print("❌ 找不到 huggingface-cli 可执行文件，请检查安装")
            return False

        # 核心模型
        for model in CORE_MODELS:
            model_id = get_model_id(model, source_choice)
            steps.append([
                str(cli_exe), "download", model_id,
                "--local-dir", str(MODEL_DIR / model)
            ])
        # 可选 LLM
        if download_llm:
            model_id = get_model_id(LLM_MODEL, source_choice)
            steps.append([
                str(cli_exe), "download", model_id,
                "--local-dir", str(MODEL_DIR / LLM_MODEL)
            ])

    for cmd in steps:
        if not run_cmd(cmd):
            print("\n下载过程中出现错误，请检查网络或手动下载。")
            return False
    return True

def main():
    print("=" * 50)
    print("       FireRed 模型专用下载工具")
    print("=" * 50)
    print("本工具可下载以下模型：")
    print("  [核心] FireRedASR2-AED (语音识别)")
    print("  [核心] FireRedVAD      (语音活动检测)")
    print("  [核心] FireRedLID      (语种识别)")
    print("  [核心] FireRedPunc     (标点恢复)")
    print("  [可选] FireRedASR2-LLM (大语言模型，需额外确认)")
    print(f"\n模型将保存至: {MODEL_DIR}")
    print("\n请选择下载源：")
    print("1. 国内用户 (ModelScope，推荐，使用 xukaituo 仓库)")
    print("2. 国际用户 (Hugging Face，使用 FireRedTeam 仓库)")
    print("0. 退出")

    choice = input("请输入数字 (0/1/2): ").strip()
    if choice == '0':
        print("退出下载")
        input("按 Enter 键退出...")
        return
    elif choice not in ('1', '2'):
        print("无效输入，退出")
        input("按 Enter 键退出...")
        return

    # 询问是否下载 LLM
    llm_choice = input("\n是否下载可选模型 FireRedASR2-LLM？(y/n，默认 n): ").strip().lower()
    download_llm = llm_choice == 'y'

    print("\n开始下载...")
    if download_models(choice, download_llm):
        print("\n✅ 所选模型下载完成！")
    else:
        print("\n⚠️ 部分模型下载失败，请重试或手动下载。")

    input("\n按 Enter 键退出...")

if __name__ == "__main__":
    main()
