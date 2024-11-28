@echo off
:: Batch file for setting up a virtual environment and installing requirements
:: Make sure this file is in the root directory of your project

:: Open Command Prompt in the script's directory
cd /d "%~dp0"

:: Display starting message
echo ========================================================
echo Activatation of Python Virtual Environment for LungAnalysis
echo ========================================================


:: Check Python installation and list installed versions
echo Checking Python installation...
py -0

:: Check if Python 3.12 is installed
echo Checking for Python 3.12 installation...
py -3.12 --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ... Python 3.12 is not installed or not found in PATH.
    echo ... Please install Python 3.12.x from https://www.python.org/downloads/
    pause
    exit /b 1
) else (
    echo ... Python 3.12 is installed.
)

:: Check pip installation
echo Checking pip installation...
py -3.12 -m pip --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ... pip is not installed or not working with Python 3.12.
    echo ... Please ensure pip is installed for Python 3.12.
    pause
    exit /b 1
) else (
    echo ... pip is installed.
)

:: Activate the virtual environment
echo Activating virtual environment...
call .\.venv\Scripts\activate
if %errorlevel% neq 0 (
    echo ... Failed to activate virtual environment. Please check your setup.
    pause
    exit /b 1
) else (
    echo =================================
    echo Virtual environment activated.
    echo =================================
)

cmd

