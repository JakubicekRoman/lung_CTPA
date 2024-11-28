@echo off
:: Batch file for setting up and activating a virtual environment
:: Make sure this file is in the root directory of your project

:: Open Command Prompt in the script's directory
cd /d "%~dp0"

:: Display starting message
echo ========================================================
echo Activating Python Virtual Environment for LungAnalysis
echo ========================================================

:: Check Python installation and list installed versions
echo Checking Python installation...
py -0
if %errorlevel% neq 0 (
    echo ... Python is not installed or not properly configured.
    echo ... Please install Python and add it to your PATH.
    pause
    exit /b 1
) else (
    echo ... Python is installed. Proceeding...
)

:: Activate the virtual environment
echo Activating virtual environment...
if exist .\.venv\Scripts\activate (
    call .\.venv\Scripts\activate
    if %errorlevel% neq 0 (
        echo ... Failed to activate virtual environment. Please check your setup.
        pause
        exit /b 1
    ) else (
        echo =================================
        echo Virtual environment activated.
        echo You can now run LungAnalysis script
        echo =================================
    )
) else (
    echo ... Virtual environment not found. Please set it up first.
    echo ... Use 'python -m venv .venv' to create the virtual environment.
    pause
    exit /b 1
)

cmd /k
