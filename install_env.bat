@echo off
:: Batch file for setting up a virtual environment and installing requirements
:: Ensure this file is in the root directory of your project

:: Move to the script's directory
cd /d "%~dp0"

:: Display starting message
echo ========================================================
echo Setting up Python Virtual Environment for LungAnalysis
echo ========================================================

:: Check Python installation and list installed versions
echo Checking Python installation...
py -0
if %errorlevel% neq 0 (
    echo ... Python is not installed or not properly configured.
    echo ... Please install Python from https://www.python.org/downloads/
    pause
    exit /b 1
)

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

:: Create a virtual environment if it doesn't already exist
if not exist ".\.venv" (
    echo Creating virtual environment...
    py -3.12 -m venv ".\.venv"
    if %errorlevel% neq 0 (
        echo ... Failed to create virtual environment. Please check your Python 3.12 installation.
        pause
        exit /b 1
    ) else (
        echo ... Virtual environment created successfully.
    )
) else (
    echo ... Virtual environment already exists. Skipping creation.
)

:: Activate the virtual environment
echo Activating virtual environment...
call .\.venv\Scripts\activate
if %errorlevel% neq 0 (
    echo ... Failed to activate virtual environment. Please check your setup.
    pause
    exit /b 1
) else (
    echo ... Virtual environment activated.
)

:: Upgrade pip
echo Upgrading pip...
.\.venv\Scripts\python.exe -m pip install --upgrade pip
if %errorlevel% neq 0 (
    echo ... Failed to upgrade pip. Continuing...
) else (
    echo ... pip upgraded successfully.
)

:: Install requirements
echo Installing required Python packages...
.\.venv\Scripts\python.exe -m pip install -r "%~dp0requirements.txt"
if %errorlevel% neq 0 (
    echo ... Failed to install required packages. Please check your requirements.txt file.
    pause
    exit /b 1
) else (
    echo ... Required packages installed successfully.
)

:: Final message
echo ========================================================
echo Virtual environment setup completed. All checks passed.
echo To activate the environment in the future, run:
echo     .\.venv\Scripts\activate
echo ========================================================

:: Keep the command prompt open for further use
cmd /k
