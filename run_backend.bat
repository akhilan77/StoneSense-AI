@echo off
setlocal enabledelayedexpansion
title StoneSense AI - Backend Server

cd /d "%~dp0"

echo =========================================================
echo  StoneSense AI - FastAPI Backend Launcher
echo =========================================================
echo.

:: Check virtual environment
if not exist "backend\.venv\Scripts\activate.bat" (
    echo [WARNING] Backend virtual environment missing. Attempting auto-repair...
    python -m venv backend\.venv
)

if not exist "backend\.venv\Scripts\python.exe" (
    echo [ERROR] Could not create the backend virtual environment.
    pause
    exit /b 1
)

echo Checking backend dependencies...
backend\.venv\Scripts\python.exe -c "import fastapi, uvicorn, joblib, xgboost, torch"
if !ERRORLEVEL! neq 0 (
    echo [WARNING] Backend dependencies are incomplete. Installing requirements...
    backend\.venv\Scripts\python.exe -m pip install -r backend\requirements.txt
    if !ERRORLEVEL! neq 0 (
        echo [ERROR] Backend dependency installation failed.
        pause
        exit /b 1
    )
)

cd backend

echo Starting FastAPI Uvicorn Server...
echo.
echo  - Service URL:     http://127.0.0.1:8000
echo  - Swagger API Docs: http://127.0.0.1:8000/docs
echo  - Redoc API Docs:   http://127.0.0.1:8000/redoc
echo.
echo Press Ctrl+C to stop the server.
echo =========================================================
echo.

.venv\Scripts\python.exe -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8000

if %ERRORLEVEL% neq 0 (
    echo.
    echo [ERROR] Backend failed to start or crashed.
)

pause
