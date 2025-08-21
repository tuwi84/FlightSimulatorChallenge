param(
  [string]$GameExe = ".\YourGame.exe",
  [string]$Args    = "-NullRHI -game -ResX=640 -ResY=360 -log",
  [string]$Script  = "python intercept.py --config configs\default.json",
  [int]$WaitBootSec = 10
)

Write-Host "Launching UE game headless..."
$proc = Start-Process -FilePath $GameExe -ArgumentList $Args -PassThru
Start-Sleep -Seconds $WaitBootSec

Write-Host "Running Python script..."
cmd /c $Script

Write-Host "Killing UE process..."
Stop-Process -Id $proc.Id -Force
Write-Host "Done."
