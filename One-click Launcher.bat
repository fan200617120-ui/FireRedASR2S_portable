@echo off
chcp 65001 >nul
title Speech Subtitle Workstation - Made by Fan
REM FireRedASR2S Project Copyright (c) 2026 FireRedTeam (Original Author)
REM This startup script is a secondary development, released under Apache 2.0 License

color 0B
setlocal enabledelayedexpansion

set "PROJECT_ROOT=%~dp0"
set "PROJECT_ROOT=%PROJECT_ROOT:~0,-1%"

set "PYTHON_DIR=%PROJECT_ROOT%\python_embeded"
set "PYTHON_EXE=%PYTHON_DIR%\python.exe"
set "MODEL_DIR=%PROJECT_ROOT%\pretrained_models"
set "INDEX_SCRIPT=%PROJECT_ROOT%\Index_Public_release.py"
set "DOWNLOAD_SCRIPT=%PROJECT_ROOT%\download_models_en.py"

if not exist "%PYTHON_EXE%" (
    echo [Error] Embedded Python not found. Please make sure the python_embeded directory is complete.
    pause
    exit /b 1
)

if not exist "%INDEX_SCRIPT%" (
    echo [Error] Main interface script Index_Public_release.py not found. Please re-download the full package.
    pause
    exit /b 1
)

if not exist "%MODEL_DIR%" (
    echo [Info] Model folder not found. The model download tool will now open.
    echo Please follow the prompts to select a download source. After downloading, the main interface will start automatically.
    echo Press any key to continue...
    pause >nul
    "%PYTHON_EXE%" "%DOWNLOAD_SCRIPT%"
) else (
    dir "%MODEL_DIR%\FireRedASR2-AED" >nul 2>&1
    if errorlevel 1 (
        echo [Info] Model folder is empty or missing required models. The model download tool will now open.
        echo Press any key to continue...
        pause >nul
        "%PYTHON_EXE%" "%DOWNLOAD_SCRIPT%"
    ) else (
        echo [Info] Models already exist. Starting main interface directly...
        "%PYTHON_EXE%" "%INDEX_SCRIPT%"
        pause
        exit /b 0
    )
)

echo [Info] Download completed. Press any key to start the main interface...
pause >nul
"%PYTHON_EXE%" "%INDEX_SCRIPT%"
pause