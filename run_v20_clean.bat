@echo off
cd /d "%~dp0"
call .\venv\Scripts\activate
python -m gemini_v20_invest.poc_v20 2>nul
pause
