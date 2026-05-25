# Build script for SuperEnalotto Engine (Windows PowerShell).
#
# Prerequisites:
#   - CMake 3.24+    (https://cmake.org/download/)
#   - Ninja          (winget install Ninja-build.Ninja)
#   - vcpkg          (https://github.com/microsoft/vcpkg)  + $env:VCPKG_ROOT set
#   - MSVC 2022      (Build Tools for Visual Studio)
#   - Python 3.11+   (for pybind11 module)
#   - CUDA Toolkit   (optional, only if -Preset cuda)
#
# Usage:
#   .\scripts\build.ps1                       # CPU release
#   .\scripts\build.ps1 -Preset cuda          # CUDA release
#   .\scripts\build.ps1 -Configuration Debug

param(
    [ValidateSet("cpu", "cuda")]    [string]$Preset        = "cpu",
    [ValidateSet("Release", "Debug")] [string]$Configuration = "Release"
)

$ErrorActionPreference = "Stop"

if (-not $env:VCPKG_ROOT) {
    Write-Error "VCPKG_ROOT is not set. Install vcpkg and set the env var."
}

Push-Location $PSScriptRoot\..

try {
    Write-Host "[1/3] Configuring (preset=$Preset)..." -ForegroundColor Cyan
    cmake --preset $Preset
    if ($LASTEXITCODE -ne 0) { throw "Configure failed" }

    Write-Host "[2/3] Building (config=$Configuration)..." -ForegroundColor Cyan
    cmake --build --preset "$Preset-$($Configuration.ToLower())"
    if ($LASTEXITCODE -ne 0) { throw "Build failed" }

    Write-Host "[3/3] Running tests..." -ForegroundColor Cyan
    ctest --preset "$Preset-tests"
    if ($LASTEXITCODE -ne 0) { throw "Tests failed" }

    $exe = Join-Path "build" "$Preset" "bin" "$Configuration" "se_cli.exe"
    if (Test-Path $exe) {
        Write-Host "`nCLI available at: $exe" -ForegroundColor Green
        & $exe --version
    }
}
finally {
    Pop-Location
}
