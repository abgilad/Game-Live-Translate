@echo off
setlocal
cd /d "%~dp0"

echo [1/3] Checking virtual environment...
if not exist ".venv\Scripts\activate.bat" (
    echo Creating virtual environment...
    python -m venv .venv
)

echo [2/3] Activating virtual environment and installing dependencies...
call .venv\Scripts\activate.bat
pip install -r requirements.txt --quiet

echo [3/3] Starting Live Translate application...
python main.py

pause
