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

cmd /k

