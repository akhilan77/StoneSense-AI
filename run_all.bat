@echo off
setlocal enabledelayedexpansion
title StoneSense AI - Full Application Launcher

cd /d "%~dp0"

echo =========================================================
echo  StoneSense AI - Full Stack Launcher
echo =========================================================
echo.

:: 1. Create logs directory if missing
if not exist "logs" mkdir logs

:: 2. Check occupied ports
netstat -ano | findstr LISTENING | findstr ":8000 " >nul
if %ERRORLEVEL% equ 0 (
    echo [WARNING] Port 8000 is occupied (FastAPI Backend port).
)

netstat -ano | findstr LISTENING | findstr ":5173 " >nul
if %ERRORLEVEL% equ 0 (
    echo [WARNING] Port 5173 is occupied (Vite Frontend port).
)

:: 3. Launch Backend in new window
echo [1/3] Launching FastAPI Backend...
start "StoneSense AI Backend" cmd /k "%~dp0run_backend.bat"

:: 4. Wait for backend startup (3 seconds)
echo [2/3] Waiting for Backend initialization...
ping 127.0.0.1 -n 4 >nul

:: 5. Launch Frontend in new window
echo [3/3] Launching React Frontend...
start "StoneSense AI Frontend" cmd /k "%~dp0run_frontend.bat"

:: 6. Open browsers automatically
ping 127.0.0.1 -n 3 >nul
echo.
echo Opening browser interfaces...
start http://localhost:5173
start http://127.0.0.1:8000/docs

echo.
echo =========================================================
echo  [SUCCESS] StoneSense AI application processes launched!
echo  Backend:  http://127.0.0.1:8000/docs
echo  Frontend: http://localhost:5173
echo =========================================================
echo.
pause
