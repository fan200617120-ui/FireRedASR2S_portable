@echo off
chcp 65001 >nul
title FireRed 模型专用下载器

color 0B
setlocal enabledelayedexpansion

set "PROJECT_ROOT=%~dp0"
set "PROJECT_ROOT=%PROJECT_ROOT:~0,-1%"

set "PYTHON_DIR=%PROJECT_ROOT%\python_embeded"
set "PYTHON_EXE=%PYTHON_DIR%\python.exe"
set "DOWNLOAD_SCRIPT=%PROJECT_ROOT%\download_models_ex.py"

if not exist "%PYTHON_EXE%" (
    echo [错误] 未找到嵌入版 Python，请确保 python_embeded 目录完整。
    pause
    exit /b 1
)

if not exist "%DOWNLOAD_SCRIPT%" (
    echo [错误] 未找到下载脚本 download_models_ex.py，请重新下载完整包。
    pause
    exit /b 1
)

echo 即将启动模型下载工具，请根据提示选择需要下载的模型。
echo.
pause
"%PYTHON_EXE%" "%DOWNLOAD_SCRIPT%"

echo 下载工具已退出。
pause