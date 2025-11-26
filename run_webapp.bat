@echo off
echo Starting House Price Prediction Web Application...
echo.
cd %~dp0
echo.
echo The application will be available at: http://127.0.0.1:5000
echo Press Ctrl+C to stop the server
echo.
python webapp\app.py
pause