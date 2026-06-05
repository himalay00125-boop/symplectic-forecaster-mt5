@echo off
REM ============================================================
REM  Symplectic Forecaster — MetaTrader 5 Launcher
REM  Uses Python 3.11 (required for MetaTrader5 package)
REM ============================================================

set PYTHON=C:\Users\himal\AppData\Local\Microsoft\WindowsApps\PythonSoftwareFoundation.Python.3.11_qbz5n2kfra8p0\python.exe

REM Check Python exists
if not exist "%PYTHON%" (
    echo [ERROR] Python 3.11 not found at:
    echo         %PYTHON%
    echo         Please install Python 3.11 from Microsoft Store or python.org
    pause
    exit /b 1
)

echo.
echo  Using Python 3.11: %PYTHON%
echo.

REM Pass all command-line arguments through
"%PYTHON%" "%~dp0symplectic_forecaster.py" %*

echo.
pause
