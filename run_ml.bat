@echo off
setlocal enabledelayedexpansion
title StoneSense AI - Machine Learning Console

cd /d "%~dp0"

if not exist "ml\.venv\Scripts\activate.bat" (
    echo [WARNING] ML virtual environment missing. Creating...
    python -m venv ml\.venv
)

call ml\.venv\Scripts\activate.bat

:menu
cls
echo =========================================================
echo  StoneSense AI - Machine Learning Pipeline Console
echo =========================================================
echo.
echo  1. Validate Tabular Dataset
echo  2. Run Exploratory Data Analysis (EDA)
echo  3. Preprocess Dataset & Feature Engineering
echo  4. Train XGBoost Risk Model
echo  5. Run SHAP Feature Attribution
echo  6. Exit
echo.
echo =========================================================
set /p choice="Select an option (1-6): "

if "%choice%"=="1" (
    echo.
    echo Running Dataset Validation...
    python ml\preprocessing\validate_ml_dataset.py
    pause
    goto menu
)
if "%choice%"=="2" (
    echo.
    echo Running Exploratory Data Analysis...
    python ml\preprocessing\run_ml_eda.py
    pause
    goto menu
)
if "%choice%"=="3" (
    echo.
    echo Running Data Preprocessing & Feature Engineering...
    python ml\preprocessing\preprocessing_pipeline.py
    pause
    goto menu
)
if "%choice%"=="4" (
    echo.
    echo Training Machine Learning Model...
    python ml\training\train_risk_model.py
    pause
    goto menu
)
if "%choice%"=="5" (
    echo.
    echo Running SHAP Feature Attribution...
    python ml\explainability\shap_explainer.py
    pause
    goto menu
)
if "%choice%"=="6" (
    call deactivate
    exit /b 0
)

echo.
echo Invalid selection. Please select 1 to 6.
ping 127.0.0.1 -n 2 >nul
goto menu
