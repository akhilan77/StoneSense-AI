@echo off
setlocal enabledelayedexpansion
title Installing StoneSense AI Dependencies

:: Set working directory to project root
cd /d "%~dp0"

echo =========================================================
echo  StoneSense AI - Automated Installation Tool
echo =========================================================
echo.

:: 1. Verify Prerequisites
echo [1/5] Verifying System Prerequisites...

where python >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Python is not installed or not in system PATH.
    echo Please install Python 3.10+ and add it to PATH.
    pause
    exit /b 1
)
for /f "tokens=*" %%v in ('python --version 2^>^&1') do set PYTHON_VERSION=%%v
echo  - Found: !PYTHON_VERSION!

where node >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Node.js is not installed or not in system PATH.
    echo Please install Node.js 18+ and add it to PATH.
    pause
    exit /b 1
)
for /f "tokens=*" %%v in ('node --version 2^>^&1') do set NODE_VERSION=%%v
echo  - Found Node.js: !NODE_VERSION!

where npm >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo [ERROR] npm is not installed or not in system PATH.
    pause
    exit /b 1
)
for /f "tokens=*" %%v in ('npm --version 2^>^&1') do set NPM_VERSION=%%v
echo  - Found npm: !NPM_VERSION!

:: 2. Setup Backend Virtual Environment & Dependencies
echo.
echo [2/5] Setting up Backend Virtual Environment...
if not exist "backend\.venv" (
    echo  - Creating backend\.venv...
    python -m venv backend\.venv
    if %ERRORLEVEL% neq 0 (
        echo [ERROR] Failed to create backend virtual environment.
        pause
        exit /b 1
    )
)
echo  - Installing shared Python requirements...
call backend\.venv\Scripts\activate.bat
python -m pip install --upgrade pip >nul 2>&1
pip install -r requirements.txt
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Failed to install backend dependencies.
    pause
    exit /b 1
)
call deactivate
echo  - Backend environment ready!

:: 3. Setup Frontend Dependencies
echo.
echo [3/5] Setting up Frontend Dependencies...
cd frontend
echo  - Running npm install...
call npm install
if %ERRORLEVEL% neq 0 (
    echo [ERROR] npm install failed in frontend directory.
    cd ..
    pause
    exit /b 1
)
cd ..
echo  - Frontend dependencies ready!

:: 4. Setup ML Virtual Environment & Dependencies
echo.
echo [4/5] Setting up ML Virtual Environment...
if not exist "ml\.venv" (
    echo  - Creating ml\.venv...
    python -m venv ml\.venv
    if %ERRORLEVEL% neq 0 (
        echo [ERROR] Failed to create ML virtual environment.
        pause
        exit /b 1
    )
)
echo  - Installing shared Python requirements in ML environment...
call ml\.venv\Scripts\activate.bat
python -m pip install --upgrade pip >nul 2>&1
pip install -r requirements.txt
if %ERRORLEVEL% neq 0 (
    echo [WARNING] Failed to install some ML dependencies.
)
call deactivate
echo  - ML environment ready!

:: 5. Setup DL Virtual Environment & Dependencies
echo.
echo [5/5] Setting up DL Virtual Environment...
if not exist "dl\.venv" (
    echo  - Creating dl\.venv...
    python -m venv dl\.venv
    if %ERRORLEVEL% neq 0 (
        echo [ERROR] Failed to create DL virtual environment.
        pause
        exit /b 1
    )
)
echo  - Installing shared Python requirements in DL environment...
call dl\.venv\Scripts\activate.bat
python -m pip install --upgrade pip >nul 2>&1
pip install -r requirements.txt
if %ERRORLEVEL% neq 0 (
    echo [WARNING] Failed to install some DL dependencies.
)
call deactivate
echo  - DL environment ready!

echo.
echo =========================================================
echo  [SUCCESS] All StoneSense AI components installed successfully!
echo =========================================================
echo.
pause
exit /b 0
