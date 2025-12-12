@echo off
cd /d "%~dp0"
call venv\Scripts\activate
python gemini_v15/main_v15.py
pause
