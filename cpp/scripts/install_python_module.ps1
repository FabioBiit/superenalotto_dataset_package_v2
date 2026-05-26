# Install the built se_engine.pyd into a location Python can import from.
#
# Usage:
#   .\scripts\install_python_module.ps1                  # CPU release
#   .\scripts\install_python_module.ps1 -Preset cuda
#
# The .pyd is copied to: <project_root>/inference/  (so 'import se_engine'
# works when running scripts from the project root).

param(
    [ValidateSet("cpu", "cuda")]      [string]$Preset        = "cpu",
    [ValidateSet("Release", "Debug")] [string]$Configuration = "Release"
)

$ErrorActionPreference = "Stop"

$cppDir = Split-Path -Parent $PSScriptRoot
$projRoot = Split-Path -Parent $cppDir
$pydPattern = Join-Path $cppDir "build\$Preset\lib\$Configuration\se_engine*.pyd"

$found = @(Get-ChildItem -Path $pydPattern -ErrorAction SilentlyContinue)
if ($found.Count -eq 0) {
    Write-Error "se_engine*.pyd not found at $pydPattern. Run build.ps1 first."
}

$destDir = Join-Path $projRoot "inference"
New-Item -ItemType Directory -Force -Path $destDir | Out-Null

foreach ($f in $found) {
    $dest = Join-Path $destDir $f.Name
    Copy-Item -Force $f.FullName $dest
    Write-Host "Installed: $dest" -ForegroundColor Green
}

# Also copy alongside the demo script for convenience
$demoDir = Join-Path $cppDir "scripts"
foreach ($f in $found) {
    $dest = Join-Path $demoDir $f.Name
    Copy-Item -Force $f.FullName $dest
    Write-Host "Installed: $dest" -ForegroundColor Green
}

Write-Host ""
Write-Host "Try it: python cpp\scripts\demo_from_python.py" -ForegroundColor Cyan
