Write-Host "Starting House Price Prediction Web Application..." -ForegroundColor Green
Write-Host ""
$scriptPath = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $scriptPath
Write-Host ""
Write-Host "The application will be available at: http://127.0.0.1:5000" -ForegroundColor Cyan
Write-Host "Press Ctrl+C to stop the server" -ForegroundColor Yellow
Write-Host ""
python webapp\app.py

