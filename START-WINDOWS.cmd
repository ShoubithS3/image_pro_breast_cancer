@echo off
setlocal
cd /d "%~dp0"
where py >nul 2>nul
if %errorlevel% equ 0 (
    py -3 run_local.py
) else (
    where python >nul 2>nul
    if errorlevel 1 (
        echo Install Python 3.12 from python.org, enable PATH, then run this file again.
        pause
        exit /b 1
    )
    python run_local.py
)
if errorlevel 1 (
    echo Setup did not finish. Read the error above and LOCAL-PC.md.
    pause
    exit /b 1
)
