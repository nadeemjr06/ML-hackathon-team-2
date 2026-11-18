@echo off
REM Laptop 2 (Windows): serve the UI. Usage: deploy\run_ui.bat http://<laptop1-ip>:8000
cd /d %~dp0\..\ui
echo Open: http://localhost:5500/?api=%~1
python -m http.server 5500
