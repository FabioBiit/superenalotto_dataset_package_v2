# Build script for SuperEnalotto Engine (Windows PowerShell).
#
# Prerequisites:
#   - Visual Studio 2022 (Community/Pro/Enterprise/BuildTools) with "Desktop development with C++"
#   - vcpkg installed; $env:VCPKG_ROOT pointing to it
#   - CMake 3.24+ and Ninja (auto-detected: install via `pip install cmake ninja` if missing)
#   - CUDA Toolkit (optional, only if -Preset cuda)
#
# Usage:
#   .\scripts\build.ps1                              # CPU release
#   .\scripts\build.ps1 -Preset cuda                 # CUDA release
#   .\scripts\build.ps1 -Configuration Debug
#   .\scripts\build.ps1 -SkipTests

param(
    [ValidateSet("cpu", "cuda")]      [string]$Preset        = "cpu",
    [ValidateSet("Release", "Debug")] [string]$Configuration = "Release",
    [switch]$SkipTests
)

$ErrorActionPreference = "Stop"

# -------- Locate and load Visual Studio 2022 environment --------
$vswhere = "C:\Program Files (x86)\Microsoft Visual Studio\Installer\vswhere.exe"
if (-not (Test-Path $vswhere)) {
    Write-Error "vswhere.exe not found. Install Visual Studio 2022."
}
$vsInstall = & $vswhere -latest -products '*' `
    -requires Microsoft.VisualCpp.Tools.HostX64.TargetX64 `
    -property installationPath
if (-not $vsInstall) {
    Write-Error "Visual Studio with C++ tools not found. Install 'Desktop development with C++'."
}
$vcvars = Join-Path $vsInstall "VC\Auxiliary\Build\vcvars64.bat"
if (-not (Test-Path $vcvars)) {
    Write-Error "vcvars64.bat not found at $vcvars"
}
Write-Host "Loading VS env from: $vcvars" -ForegroundColor DarkCyan

# Import vcvars64 into the current PowerShell session.
$tmpEnv = [System.IO.Path]::GetTempFileName()
cmd /c "call `"$vcvars`" && set" > $tmpEnv
Get-Content $tmpEnv | ForEach-Object {
    if ($_ -match '^([^=]+)=(.*)$') {
        [System.Environment]::SetEnvironmentVariable($Matches[1], $Matches[2], 'Process')
    }
}
Remove-Item $tmpEnv

if (-not $env:VCPKG_ROOT) {
    Write-Error "VCPKG_ROOT is not set."
}

# -------- Add pip-installed cmake/ninja to PATH if found --------
$pyScripts = Join-Path (Split-Path (Get-Command python).Source -Parent) "Scripts"
if (Test-Path $pyScripts) { $env:PATH = "$pyScripts;$env:PATH" }

Push-Location $PSScriptRoot\..

try {
    Write-Host "[1/3] Configure (preset=$Preset)..." -ForegroundColor Cyan
    cmake --preset $Preset
    if ($LASTEXITCODE -ne 0) { throw "Configure failed (exit $LASTEXITCODE)" }

    Write-Host "[2/3] Build (config=$Configuration)..." -ForegroundColor Cyan
    cmake --build --preset "$Preset-$($Configuration.ToLower())"
    if ($LASTEXITCODE -ne 0) { throw "Build failed (exit $LASTEXITCODE)" }

    if (-not $SkipTests) {
        Write-Host "[3/3] Tests..." -ForegroundColor Cyan
        ctest --preset "$Preset-tests"
        if ($LASTEXITCODE -ne 0) { throw "Tests failed (exit $LASTEXITCODE)" }
    } else {
        Write-Host "[3/3] Skipping tests" -ForegroundColor DarkYellow
    }

    $exe = Join-Path "build" "$Preset" "bin" "$Configuration" "se_cli.exe"
    if (Test-Path $exe) {
        Write-Host "`nBuilt: $exe" -ForegroundColor Green
        & $exe --version
    }
}
finally {
    Pop-Location
}
