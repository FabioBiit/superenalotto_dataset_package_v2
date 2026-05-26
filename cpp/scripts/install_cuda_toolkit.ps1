# Installer per CUDA Toolkit 13.0 (rolling release per RTX 3080 Laptop).
#
# Prerequisiti:
#   - Driver NVIDIA gia' presente (>= 525, idealmente >= 580). Verifica con `nvidia-smi`.
#   - Nessun processo GPU attivo (CARLA, training Python, giochi, ecc.).
#   - ~3.5 GB liberi.
#
# Sicurezza:
#   - Usa il flag --toolkit per installare SOLO toolkit (no driver, no samples).
#   - Se vuoi aggiornare il driver fai un install separato.
#   - Lo script controlla che nessun processo Python/CARLA usi la GPU prima di partire.
#
# Usage:
#   .\scripts\install_cuda_toolkit.ps1                  # Interactive
#   .\scripts\install_cuda_toolkit.ps1 -Force           # Skip GPU-process check
#   .\scripts\install_cuda_toolkit.ps1 -Version 12.6    # Use a specific CUDA version

param(
    [string]$Version = "13.0.2",
    [switch]$Force,
    [switch]$NoDriver = $true
)

$ErrorActionPreference = "Stop"

Write-Host "===== CUDA Toolkit Installer =====" -ForegroundColor Cyan

# 1. Verify NVIDIA driver
Write-Host "[1/5] Checking NVIDIA driver..." -ForegroundColor Yellow
$smi = "C:\Windows\System32\nvidia-smi.exe"
if (-not (Test-Path $smi)) {
    Write-Error "nvidia-smi not found. Install NVIDIA driver first."
}
$gpuInfo = & $smi --query-gpu=name,driver_version,compute_cap --format=csv,noheader
Write-Host "  GPU: $gpuInfo" -ForegroundColor Green

# 2. Check existing CUDA
Write-Host "[2/5] Checking existing CUDA install..." -ForegroundColor Yellow
$cudaRoot = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA"
if (Test-Path $cudaRoot) {
    $existing = Get-ChildItem $cudaRoot -Directory -ErrorAction SilentlyContinue | Select-Object -ExpandProperty Name
    if ($existing) {
        Write-Host "  WARN: CUDA already installed: $($existing -join ', ')" -ForegroundColor DarkYellow
        if (-not $Force) {
            $ans = Read-Host "  Re-install anyway? (y/N)"
            if ($ans -ne 'y') { Write-Host "Aborting." ; exit 0 }
        }
    }
} else {
    Write-Host "  No existing CUDA install." -ForegroundColor Green
}

# 3. Check GPU is idle
Write-Host "[3/5] Checking GPU is idle..." -ForegroundColor Yellow
$procs = & $smi --query-compute-apps=pid,process_name --format=csv,noheader 2>&1
if ($procs -and ($procs -match '\d+')) {
    Write-Host "  WARN: GPU has active compute processes:" -ForegroundColor Red
    Write-Host "    $procs"
    if (-not $Force) {
        Write-Error "Refusing to install while GPU is in use. Stop the processes first, or use -Force."
    }
} else {
    Write-Host "  GPU is idle." -ForegroundColor Green
}

# 4. Download installer (network installer is much smaller, ~50 MB)
Write-Host "[4/5] Downloading CUDA $Version network installer..." -ForegroundColor Yellow
$url = "https://developer.download.nvidia.com/compute/cuda/$Version/network_installers/cuda_${Version}_windows_network.exe"
$exe = Join-Path $env:TEMP "cuda_${Version}_windows_network.exe"
if (-not (Test-Path $exe)) {
    Invoke-WebRequest -Uri $url -OutFile $exe -UseBasicParsing
    Write-Host "  Downloaded: $exe ($([math]::Round((Get-Item $exe).Length / 1MB, 1)) MB)" -ForegroundColor Green
} else {
    Write-Host "  Cached at: $exe" -ForegroundColor Green
}

# 5. Run silent install — toolkit only, no driver
Write-Host "[5/5] Running installer (silent, toolkit only, no driver)..." -ForegroundColor Yellow
# Components: nvcc, cudart, cublas, cufft, curand, cusolver, cusparse, npp, nvjpeg, nvrtc, visual_studio_integration
$components = @(
    "nvcc_$Version",
    "cudart_$Version",
    "curand_$Version",
    "cublas_$Version",
    "cufft_$Version",
    "nvrtc_$Version",
    "visual_studio_integration_$Version"
) -join " "

$args = @("-s") + $components.Split(' ')
Write-Host "  Command: $exe $args"
Write-Host "  This may take 5-10 minutes. Be patient..."

$proc = Start-Process -FilePath $exe -ArgumentList $args -Wait -PassThru -NoNewWindow
if ($proc.ExitCode -eq 0) {
    Write-Host "  Install OK (exit 0)" -ForegroundColor Green
} else {
    Write-Error "Install failed with exit code $($proc.ExitCode). Check $env:TEMP\CUDA_Setup.log"
}

# 6. Verify
Write-Host ""
Write-Host "===== Verification =====" -ForegroundColor Cyan
$nvcc = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v$($Version.Substring(0, 4))\bin\nvcc.exe"
if (Test-Path $nvcc) {
    Write-Host "nvcc found: $nvcc" -ForegroundColor Green
    & $nvcc --version
    Write-Host ""
    Write-Host "Next step: rebuild with CUDA preset:" -ForegroundColor Cyan
    Write-Host "  cd cpp"
    Write-Host "  .\scripts\build.ps1 -Preset cuda"
} else {
    Write-Host "nvcc NOT found at $nvcc - install may have failed or used different path" -ForegroundColor Red
    Get-ChildItem "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA" -ErrorAction SilentlyContinue
}
