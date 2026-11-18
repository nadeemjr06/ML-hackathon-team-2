@echo off
REM Laptop 1 (Windows): start the PathSnap API. Usage: deploy\run_api.bat [C:\path\to\eval_trips]
cd /d %~dp0\..
if not "%~1"=="" set PATHSNAP_EVAL_DIR=%~1
ipconfig | findstr IPv4
.venv\Scripts\uvicorn api.main:app --host 0.0.0.0 --port 8000
