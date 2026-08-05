@echo off
setlocal enabledelayedexpansion
title StoneSense AI - Stop Processes

cd /d "%~dp0"

echo =========================================================
echo  StoneSense AI - Process Termination Tool
echo =========================================================
echo.
echo This action will terminate running instances of:
echo  - node.exe (Frontend server)
echo  - python.exe (Backend/ML/DL scripts)
echo  - uvicorn (FastAPI server)
echo.

set /p confirm="Are you sure you want to stop all processes? (Y/N): "
if /i "%confirm%" neq "Y" (
    echo Operation cancelled.
    pause
    exit /b 0
)

echo.
echo Terminating uvicorn processes...
taskkill /F /IM uvicorn.exe 2>nul

echo Terminating node.exe processes...
taskkill /F /IM node.exe 2>nul

echo Terminating python.exe processes...
taskkill /F /IM python.exe 2>nul

echo.
echo =========================================================
echo  [SUCCESS] All targeted processes have been stopped.
echo =========================================================
echo.
pause
