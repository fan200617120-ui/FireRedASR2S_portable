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
下载 FireRedASR2-AED、FireRedVAD、FireRedLID、FireRedPunc 模型
"""

import os
import subprocess
import sys
from pathlib import Path

# 项目根目录
BASE_DIR = Path(__file__).parent.absolute()
PYTHON_EXE = BASE_DIR / "python_embeded" / "python.exe"
SCRIPTS_DIR = BASE_DIR / "python_embeded" / "Scripts"

if not PYTHON_EXE.exists():
    print("❌ 未找到嵌入版 Python，请确保 python_embeded 目录存在")
    input("按 Enter 键退出...")
    sys.exit(1)

MODEL_DIR = BASE_DIR / "pretrained_models"
MODEL_DIR.mkdir(exist_ok=True)

def run_cmd(cmd):
    """执行命令并打印输出"""
    print(f"执行: {' '.join(cmd)}")
    try:
        subprocess.run(cmd, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"命令执行失败: {e}")
        return False

def download_models(choice):
    """根据用户选择下载模型"""
    if choice == '1':
        # 国内用户 (ModelScope)
        print("\n使用 ModelScope 下载，适合国内用户")
        # 先安装 modelscope
        if not run_cmd([str(PYTHON_EXE), "-m", "pip", "install", "-U", "modelscope"]):
            return False
        # 找到 modelscope 可执行文件
        modelscope_exe = SCRIPTS_DIR / "modelscope.exe"
        if not modelscope_exe.exists():
            modelscope_exe = SCRIPTS_DIR / "modelscope"  # 无扩展名
        if not modelscope_exe.exists():
            print("❌ 找不到 modelscope 可执行文件，请检查安装")
            return False
        # 执行下载命令
        steps = [
            [str(modelscope_exe), "download", "--model", "FireRedTeam/FireRedASR2-AED", "--local_dir", str(MODEL_DIR / "FireRedASR2-AED")],
            [str(modelscope_exe), "download", "--model", "FireRedTeam/FireRedVAD", "--local_dir", str(MODEL_DIR / "FireRedVAD")],
            [str(modelscope_exe), "download", "--model", "FireRedTeam/FireRedLID", "--local_dir", str(MODEL_DIR / "FireRedLID")],
            [str(modelscope_exe), "download", "--model", "FireRedTeam/FireRedPunc", "--local_dir", str(MODEL_DIR / "FireRedPunc")],
        ]
    else:
        # 国际用户 (Hugging Face)
        print("\n使用 Hugging Face 下载，适合国际用户")
        # 先安装 huggingface_hub
        if not run_cmd([str(PYTHON_EXE), "-m", "pip", "install", "-U", "huggingface_hub[cli]"]):
            return False
        # huggingface_hub 安装后会生成 huggingface-cli 命令
        cli_exe = SCRIPTS_DIR / "huggingface-cli.exe"
        if not cli_exe.exists():
            cli_exe = SCRIPTS_DIR / "huggingface-cli"
        if not cli_exe.exists():
            print("❌ 找不到 huggingface-cli 可执行文件，请检查安装")
            return False
        steps = [
            [str(cli_exe), "download", "FireRedTeam/FireRedASR2-AED", "--local-dir", str(MODEL_DIR / "FireRedASR2-AED")],
            [str(cli_exe), "download", "FireRedTeam/FireRedVAD", "--local-dir", str(MODEL_DIR / "FireRedVAD")],
            [str(cli_exe), "download", "FireRedTeam/FireRedLID", "--local-dir", str(MODEL_DIR / "FireRedLID")],
            [str(cli_exe), "download", "FireRedTeam/FireRedPunc", "--local-dir", str(MODEL_DIR / "FireRedPunc")],
        ]

    for cmd in steps:
        if not run_cmd(cmd):
            print("\n下载过程中出现错误，请检查网络或手动下载。")
            return False
    return True

def main():
    print("=" * 50)
    print("       FireRedASR2S 模型下载工具")
    print("=" * 50)
    print("本脚本将下载以下四个模型：")
    print("  - FireRedASR2-AED")
    print("  - FireRedVAD")
    print("  - FireRedLID")
    print("  - FireRedPunc")
    print(f"\n模型将保存至: {MODEL_DIR}")
    print("\n请选择下载源：")
    print("1. 国内用户 (ModelScope，推荐)")
    print("2. 国际用户 (Hugging Face)")
    print("0. 退出")

    choice = input("请输入数字 (0/1/2): ").strip()
    if choice == '0':
        print("退出下载")
        return
    elif choice not in ('1', '2'):
        print("无效输入，退出")
        return

    if download_models(choice):
        print("\n✅ 所有模型下载完成！")
    else:
        print("\n⚠️ 部分模型下载失败，请重试或手动下载。")

    input("\n按 Enter 键退出...")

if __name__ == "__main__":
    main()