# =========================================================
# StoneSense AI - PowerShell Developer Automation Suite
# =========================================================

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
Set-Location $ScriptDir

# Setup Logs Directory & Rotation
$LogDir = Join-Path $ScriptDir "logs"
if (!(Test-Path $LogDir)) {
    New-Item -ItemType Directory -Path $LogDir | Out-Null
}

# Rotate log files older than 7 days
Get-ChildItem -Path $LogDir -Filter "*.log" | Where-Object { $_.LastWriteTime -lt (Get-Date).AddDays(-7) } | Remove-Item -Force -ErrorAction SilentlyContinue

$LogFile = Join-Path $LogDir "developer_console_$(Get-Date -Format 'yyyyMMdd').log"

function Write-Log {
    param (
        [string]$Message,
        [string]$Level = "INFO",
        [ConsoleColor]$Color = [ConsoleColor]::White
    )
    $Timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    $FormattedMessage = "[$Timestamp] [$Level] $Message"
    Add-Content -Path $LogFile -Value $FormattedMessage -ErrorAction SilentlyContinue
    Write-Host "[$Level] $Message" -ForegroundColor $Color
}

function Show-Header {
    Clear-Host
    Write-Host "=========================================================" -ForegroundColor Cyan
    Write-Host "         StoneSense AI Developer Console (PowerShell)    " -ForegroundColor Header
    Write-Host "=========================================================" -ForegroundColor Cyan
    Write-Host ""
}

function Test-PortOccupied {
    param ([int]$Port)
    $connection = Test-NetConnection -ComputerName "127.0.0.1" -Port $Port -WarningAction SilentlyContinue
    return $connection.TcpTestSucceeded
}

function Check-And-Handle-Ports {
    foreach ($port in @(8000, 5173)) {
        if (Test-PortOccupied -Port $port) {
            Write-Log "Port $port is currently occupied!" "WARNING" -Color Yellow
            $choice = Read-Host "Port $port is in use. Would you like to terminate processes using it? (Y/N)"
            if ($choice -eq "Y" -or $choice -eq "y") {
                Get-Process -ErrorAction SilentlyContinue | Where-Object { $_.ProcessName -match "python|node|uvicorn" } | Stop-Process -Force -ErrorAction SilentlyContinue
                Write-Log "Terminated processes on port $port." "SUCCESS" -Color Green
            }
        }
    }
}

function Install-All {
    Show-Header
    Write-Log "Starting Complete Environment Setup..." "INFO" -Color Cyan

    # Check prerequisites
    if (!(Get-Command python -ErrorAction SilentlyContinue)) {
        Write-Log "Python is not installed or not in PATH." "ERROR" -Color Red
        Read-Host "Press Enter to return..."
        return
    }
    if (!(Get-Command node -ErrorAction SilentlyContinue)) {
        Write-Log "Node.js is not installed or not in PATH." "ERROR" -Color Red
        Read-Host "Press Enter to return..."
        return
    }

    # Setup backend
    Write-Log "[1/4] Setting up Backend Environment..." "INFO" -Color Yellow
    if (!(Test-Path "backend\.venv")) {
        python -m venv backend\.venv
    }
    & "backend\.venv\Scripts\python.exe" -m pip install --upgrade pip | Out-Null
    & "backend\.venv\Scripts\pip.exe" install -r backend\requirements.txt

    # Setup frontend
    Write-Log "[2/4] Setting up Frontend Environment..." "INFO" -Color Yellow
    Push-Location frontend
    npm install
    Pop-Location

    # Setup ML
    Write-Log "[3/4] Setting up ML Environment..." "INFO" -Color Yellow
    if (!(Test-Path "ml\.venv")) {
        python -m venv ml\.venv
    }
    if (Test-Path "ml\requirements.txt") {
        & "ml\.venv\Scripts\pip.exe" install -r ml\requirements.txt -ErrorAction SilentlyContinue
    }

    # Setup DL
    Write-Log "[4/4] Setting up DL Environment..." "INFO" -Color Yellow
    if (!(Test-Path "dl\.venv")) {
        python -m venv dl\.venv
    }
    if (Test-Path "dl\requirements.txt") {
        & "dl\.venv\Scripts\pip.exe" install -r dl\requirements.txt -ErrorAction SilentlyContinue
    }

    Write-Log "All StoneSense AI components installed successfully!" "SUCCESS" -Color Green
    Read-Host "Press Enter to return..."
}

function Run-Backend {
    Show-Header
    Check-And-Handle-Ports
    Write-Log "Starting FastAPI Backend Server..." "INFO" -Color Cyan
    Start-Process cmd -ArgumentList "/k `"$ScriptDir\run_backend.bat`""
}

function Run-Frontend {
    Show-Header
    Check-And-Handle-Ports
    Write-Log "Starting Vite Frontend Dev Server..." "INFO" -Color Cyan
    Start-Process cmd -ArgumentList "/k `"$ScriptDir\run_frontend.bat`""
}

function Run-ML {
    Start-Process cmd -ArgumentList "/k `"$ScriptDir\run_ml.bat`""
}

function Run-DL {
    Start-Process cmd -ArgumentList "/k `"$ScriptDir\run_dl.bat`""
}

function Run-All {
    Show-Header
    Check-And-Handle-Ports
    Write-Log "Launching Full StoneSense AI Stack..." "INFO" -Color Cyan
    Start-Process cmd -ArgumentList "/k `"$ScriptDir\run_all.bat`""
}

function Stop-All {
    Show-Header
    Write-Log "Stopping all running application processes..." "WARNING" -Color Yellow
    Start-Process cmd -ArgumentList "/c `"$ScriptDir\stop_all.bat`""
}

# Main Interactive Loop
while ($true) {
    Show-Header
    Write-Host "  1. Install Everything" -ForegroundColor White
    Write-Host "  2. Start Backend" -ForegroundColor White
    Write-Host "  3. Start Frontend" -ForegroundColor White
    Write-Host "  4. Start ML Console" -ForegroundColor White
    Write-Host "  5. Start DL Console" -ForegroundColor White
    Write-Host "  6. Start Entire Project (Backend + Frontend)" -ForegroundColor White
    Write-Host "  7. Stop Everything" -ForegroundColor White
    Write-Host "  8. Open Swagger API Docs" -ForegroundColor White
    Write-Host "  9. Open Frontend Web App" -ForegroundColor White
    Write-Host " 10. Exit" -ForegroundColor White
    Write-Host ""
    Write-Host "=========================================================" -ForegroundColor Cyan
    $choice = Read-Host "Select an option (1-10)"

    switch ($choice) {
        "1" { Install-All }
        "2" { Run-Backend }
        "3" { Run-Frontend }
        "4" { Run-ML }
        "5" { Run-DL }
        "6" { Run-All }
        "7" { Stop-All }
        "8" { Start-Process "http://127.0.0.1:8000/docs" }
        "9" { Start-Process "http://localhost:5173" }
        "10" { Exit }
        default { Write-Host "Invalid option. Please select 1-10." -ForegroundColor Red; Start-Sleep -Seconds 1 }
    }
}
