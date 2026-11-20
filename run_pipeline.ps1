$ErrorActionPreference = 'Stop'
$Python = "C:\Users\deban\AppData\Local\Programs\Python\Python312\python.exe"

Clear-Host
Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host "  Deepfake Detector: Full Dataset Processing Pipeline  " -ForegroundColor White
Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "This terminal will monitor the progress of the entire dataset." -ForegroundColor Yellow
Write-Host "Total videos expected: ~20,000+" -ForegroundColor Yellow
Write-Host ""

Write-Host "[STAGE 1/2] Verifying Dataset..." -ForegroundColor Green
& $Python deepfake_detector\main.py --verify
if ($LASTEXITCODE -ne 0) { Write-Host "Error during verification." -ForegroundColor Red; Read-Host; exit }

Write-Host "`n[STAGE 2/2] Preprocessing Features (Frames, Faces, FFT, Audio)..." -ForegroundColor Green
& $Python deepfake_detector\main.py --preprocess
if ($LASTEXITCODE -ne 0) { Write-Host "Error during preprocessing." -ForegroundColor Red; Read-Host; exit }

Write-Host "`n==========================================================" -ForegroundColor Cyan
Write-Host "Processing Complete! You can close this window now." -ForegroundColor Green
Read-Host "Press Enter to exit"
