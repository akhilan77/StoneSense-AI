@echo off
setlocal enabledelayedexpansion
title StoneSense AI Developer Console

cd /d "%~dp0"

:menu
cls
echo =========================================================
echo  StoneSense AI Developer Console
echo =========================================================
echo.
echo   1. Install Everything
echo   2. Start Backend
echo   3. Start Frontend
echo   4. Start ML Console
echo   5. Start DL Console
echo   6. Start Entire Project (Backend + Frontend)
echo   7. Stop Everything
echo   8. Open Swagger API Docs
echo   9. Open Frontend Web App
echo  10. Exit
echo.
echo =========================================================
set /p choice="Select an option (1-10): "

if "%choice%"=="1" (
    call "%~dp0install_all.bat"
    goto menu
)
if "%choice%"=="2" (
    start "StoneSense AI Backend" cmd /k "%~dp0run_backend.bat"
    goto menu
)
if "%choice%"=="3" (
    start "StoneSense AI Frontend" cmd /k "%~dp0run_frontend.bat"
    goto menu
)
if "%choice%"=="4" (
    call "%~dp0run_ml.bat"
    goto menu
)
if "%choice%"=="5" (
    call "%~dp0run_dl.bat"
    goto menu
)
if "%choice%"=="6" (
    call "%~dp0run_all.bat"
    goto menu
)
if "%choice%"=="7" (
    call "%~dp0stop_all.bat"
    goto menu
)
if "%choice%"=="8" (
    start http://127.0.0.1:8000/docs
    goto menu
)
if "%choice%"=="9" (
    start http://localhost:5173
    goto menu
)
if "%choice%"=="10" (
    exit /b 0
)

echo Invalid option selected.
ping 127.0.0.1 -n 2 >nul
goto menu
