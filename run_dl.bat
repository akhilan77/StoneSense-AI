@echo off
setlocal enabledelayedexpansion
title StoneSense AI - Deep Learning Console

cd /d "%~dp0"

if not exist "dl\.venv\Scripts\activate.bat" (
    echo [WARNING] DL virtual environment missing. Creating...
    python -m venv dl\.venv
)

call dl\.venv\Scripts\activate.bat

:menu
cls
echo =========================================================
echo  StoneSense AI - Deep Learning Pipeline Console
echo =========================================================
echo.
echo  1. Validate CT Dataset
echo  2. Run Exploratory Data Analysis (EDA)
echo  3. Preprocess Images & Build Data Loaders
echo  4. Train ResNet18 Classifier
echo  5. Evaluate Model & Generate Metrics
echo  6. Run Grad-CAM Saliency Maps
echo  7. Exit
echo.
echo =========================================================
set /p choice="Select an option (1-7): "

if "%choice%"=="1" (
    echo.
    echo Running Image Dataset Validation...
    python dl\preprocessing\validate_dl_dataset.py
    pause
    goto menu
)
if "%choice%"=="2" (
    echo.
    echo Running Image Exploratory Data Analysis...
    python dl\preprocessing\run_dl_eda.py
    pause
    goto menu
)
if "%choice%"=="3" (
    echo.
    echo Preprocessing CT Scan Images...
    python dl\preprocessing\image_preprocessor.py
    pause
    goto menu
)
if "%choice%"=="4" (
    echo.
    echo Training ResNet18 Deep Learning Model...
    python dl\training\train_resnet18.py
    pause
    goto menu
)
if "%choice%"=="5" (
    echo.
    echo Evaluating Deep Learning Model...
    python dl\training\evaluate.py
    pause
    goto menu
)
if "%choice%"=="6" (
    echo.
    echo Generating Grad-CAM Saliency Maps...
    python dl\explainability\gradcam.py
    pause
    goto menu
)
if "%choice%"=="7" (
    call deactivate
    exit /b 0
)

echo.
echo Invalid selection. Please select 1 to 7.
ping 127.0.0.1 -n 2 >nul
goto menu
