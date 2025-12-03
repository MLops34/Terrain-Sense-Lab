@echo off
echo Starting House Price Prediction Web Application...
echo.
cd /d "%~dp0"
echo Current directory: %CD%
echo.
echo The application will be available at: http://127.0.0.1:5000
echo Press Ctrl+C to stop the server
echo.

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not found in PATH!
    echo Please make sure Python is installed and added to your system PATH.
    pause
    exit /b 1
)

REM Check if the app.py file exists
if not exist "webapp\app.py" (
    echo ERROR: webapp\app.py not found!
    echo Please make sure you are running this script from the project root directory.
    pause
    exit /b 1
)

echo Starting Flask application...
python webapp\app.py
if errorlevel 1 (
    echo.
    echo ERROR: Failed to start the application!
    pause
    exit /b 1
)
pause