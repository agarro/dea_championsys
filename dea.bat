REM DEA Championsys en Windows
:: Fijarse si el entorno virtual es 'venv' o '.venv'

@echo off
setlocal
cd /d "%~dp0"

start "Python App" cmd /k "call venv\Scripts\activate.bat && python app.py"
timeout /t 3 /nobreak >nul
start "" "http://localhost:5001"

endlocal
exit /b 0
