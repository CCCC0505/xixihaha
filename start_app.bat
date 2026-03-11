@echo off
setlocal

cd /d "%~dp0"

if not exist ".venv\Scripts\streamlit.exe" (
    echo [ERROR] Missing .venv\Scripts\streamlit.exe
    echo Create the virtual environment and install dependencies first:
    echo python -m venv .venv
    echo .venv\Scripts\activate
    echo pip install -r requirements.txt
    pause
    exit /b 1
)

echo Starting Streamlit app...
echo Open in browser: http://localhost:8501

".venv\Scripts\streamlit.exe" run app.py
