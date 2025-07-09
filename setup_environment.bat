@echo off
echo Setting up Python environment for Facial Classification Project...
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo Error: Python is not installed or not in PATH
    echo Please install Python 3.8 or higher from https://python.org
    pause
    exit /b 1
)

echo Python found. Creating virtual environment...
python -m venv facial_classification_env

echo Activating virtual environment...
call facial_classification_env\Scripts\activate.bat

echo Installing required packages...
pip install --upgrade pip
pip install -r requirements.txt

echo.
echo Setup completed successfully!
echo.
echo To activate the environment in the future, run:
echo   facial_classification_env\Scripts\activate.bat
echo.
echo To run the main program:
echo   python main.py
echo.
pause 