@echo off
chcp 65001 >nul
title 语音字幕工作站 凡哥制作
REM:: FireRedASR2S 项目版权所有 (c) 2026 FireRedTeam (原作者)
REM:: 本启动脚本为二次开发内容，基于 Apache 2.0 协议发布

color 0B
setlocal enabledelayedexpansion

set "PROJECT_ROOT=%~dp0"
set "PROJECT_ROOT=%PROJECT_ROOT:~0,-1%"

set "PYTHON_DIR=%PROJECT_ROOT%\python_embeded"
set "PYTHON_EXE=%PYTHON_DIR%\python.exe"
set "MODEL_DIR=%PROJECT_ROOT%\pretrained_models"
set "INDEX_SCRIPT=%PROJECT_ROOT%\Index_Public_release.py"
set "DOWNLOAD_SCRIPT=%PROJECT_ROOT%\download_models.py"

if not exist "%PYTHON_EXE%" (
    echo [错误] 未找到嵌入版 Python，请确保 python_embeded 目录完整。
    pause
    exit /b 1
)

if not exist "%INDEX_SCRIPT%" (
    echo [错误] 未找到主界面脚本 Index_Public_release.py，请重新下载完整包。
    pause
    exit /b 1
)

if not exist "%MODEL_DIR%" (
    echo [提示] 未找到模型文件夹，将为你打开模型下载工具。
    echo 请根据提示选择下载源，模型下载完成后会自动启动主界面。
    echo 按任意键继续...
    pause >nul
    "%PYTHON_EXE%" "%DOWNLOAD_SCRIPT%"
) else (
    dir "%MODEL_DIR%\FireRedASR2-AED" >nul 2>&1
    if errorlevel 1 (
        echo [提示] 模型文件夹为空或缺少必要模型，将为你打开模型下载工具。
        echo 按任意键继续...
        pause >nul
        "%PYTHON_EXE%" "%DOWNLOAD_SCRIPT%"
    ) else (
        echo [信息] 模型已存在，直接启动主界面...
        "%PYTHON_EXE%" "%INDEX_SCRIPT%"
        pause
        exit /b 0
    )
)

echo [信息] 下载完成后，按任意键启动主界面...
pause >nul
"%PYTHON_EXE%" "%INDEX_SCRIPT%"
pause
