# Configure + build Glow Effect on Windows (no need to open Visual Studio).
#
#   .\scripts\build.ps1 -TensorRTRoot C:\TensorRT-10.4.0.26 -OpenCVDir C:\opencv\build
#   .\scripts\build.ps1 -CudaArch 86            # RTX 30xx; 89 = RTX 40xx, 75 = RTX 20xx, 120 = RTX 50xx
#   .\scripts\build.ps1 -Generator Ninja        # from an "x64 Native Tools" prompt
#
# Needs: CMake >= 3.21, Visual Studio 2019/2022 Build Tools (C++ workload), CUDA Toolkit, TensorRT, OpenCV.
param(
    [string]$TensorRTRoot = $env:TENSORRT_ROOT,
    [string]$OpenCVDir = $env:OpenCV_DIR,
    [string]$CudaArch = "",
    [string]$Config = "Release",
    [string]$Generator = "Visual Studio 17 2022",
    [string]$BuildDir = ""
)
$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $PSScriptRoot
if (-not $BuildDir) { $BuildDir = Join-Path $Root "build" }

$cmakeArgs = @("-S", $Root, "-B", $BuildDir, "-G", $Generator)
if ($Generator -like "Visual Studio*") { $cmakeArgs += @("-A", "x64") }
else { $cmakeArgs += "-DCMAKE_BUILD_TYPE=$Config" }
if ($TensorRTRoot) { $cmakeArgs += "-DTensorRT_ROOT=$TensorRTRoot" }
if ($OpenCVDir) { $cmakeArgs += "-DOpenCV_DIR=$OpenCVDir" }
if ($CudaArch) { $cmakeArgs += "-DCMAKE_CUDA_ARCHITECTURES=$CudaArch" }

Write-Host "==> Configuring in $BuildDir" -ForegroundColor Cyan
& cmake @cmakeArgs
if ($LASTEXITCODE -ne 0) { throw "CMake configure failed" }

Write-Host "==> Building ($Config)" -ForegroundColor Cyan
& cmake --build $BuildDir --config $Config --parallel
if ($LASTEXITCODE -ne 0) { throw "Build failed" }

$exe = Join-Path $BuildDir "bin\$Config\glow_effect.exe"
if (-not (Test-Path $exe)) { $exe = Join-Path $BuildDir "bin\glow_effect.exe" }
Write-Host ""
Write-Host "Done: $exe" -ForegroundColor Green
Write-Host "Try:  & `"$exe`" --help"
