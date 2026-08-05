@echo off
setlocal enabledelayedexpansion
title StoneSense AI - Frontend Development Server

cd /d "%~dp0"

echo =========================================================
echo  StoneSense AI - Vite React Frontend Launcher
echo =========================================================
echo.

cd frontend

if not exist "node_modules" (
    echo [WARNING] node_modules not found. Running npm install...
    call npm install
    if %ERRORLEVEL% neq 0 (
        echo [ERROR] npm install failed.
        pause
        exit /b 1
    )
)

echo Starting Vite Frontend Dev Server...
echo.
echo  - Frontend Web UI: http://localhost:5173
echo.
echo Press Ctrl+C to stop the dev server.
echo =========================================================
echo.

call npm run dev

if %ERRORLEVEL% neq 0 (
    echo.
    echo [ERROR] Frontend failed to start or crashed.
)

pause
