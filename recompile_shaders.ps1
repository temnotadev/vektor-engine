# Recompile all GLSL compute shaders into SPIR-V (.spv) and refresh the
# per-config output folders used by rt_renderer.exe.
#
# Usage:  .\recompile_shaders.ps1
#
# Flags match the CMake "shaders" target:
#   glslangValidator -V --target-env vulkan1.2 -I<root> <name>.comp -o <name>.comp.spv

$ErrorActionPreference = "Stop"

# Resolve paths relative to this script, so it works from any working directory.
$root    = Split-Path -Parent $MyInvocation.MyCommand.Path
$outDir  = Join-Path $root "build\shaders"

# Prefer the Vulkan SDK, fall back to glslangValidator on PATH.
$compiler = Join-Path $env:VULKAN_SDK "Bin\glslangValidator.exe"
if (-not (Test-Path $compiler)) { $compiler = "glslangValidator" }

# All shaders that main.cpp loads as .spv (the 12 CMake shaders + ssr_half).
$shaders = @(
    "gbuffer", "light_grid", "cache_update", "tile_lights",
    "di_initial", "di_temporal", "di_spatial", "direct", "gi",
    "nrd_prep", "nrd_di_upsample", "composite", "ssr_half", "ssr_blur"
)

New-Item -ItemType Directory -Force -Path $outDir | Out-Null

$failed = 0
foreach ($name in $shaders) {
    $src = Join-Path $root "$name.comp"
    $dst = Join-Path $outDir "$name.comp.spv"
    $log = & $compiler -V --target-env vulkan1.2 -I"$root" "$src" -o "$dst" 2>&1
    if ($LASTEXITCODE -ne 0) {
        Write-Error "FAILED: $name.comp`n$log"
        $failed++
    } else {
        Write-Output ("OK   {0,-26} {1,8} bytes" -f "$name.comp.spv", (Get-Item $dst).Length)
    }
}

if ($failed -gt 0) { throw "$failed shader(s) failed to compile." }

# Mirror the CMake post-build copy so both Debug and Release exes see fresh SPVs.
$spvFiles = Get-ChildItem -Path $outDir -Filter *.spv -File
foreach ($cfg in @("Debug", "Release")) {
    $target = Join-Path $root "build\$cfg\shaders"
    if (-not (Test-Path $target)) { continue }
    foreach ($f in $spvFiles) { Copy-Item -Force $f.FullName (Join-Path $target $f.Name) }
    # Custom shaders include common.glsl from this folder at runtime.
    Copy-Item -Force (Join-Path $root "common.glsl") (Join-Path $target "common.glsl")
    Write-Output ("Synced {0} SPVs -> {1}" -f $spvFiles.Count, $target)
}

Write-Output "Done. $($shaders.Count) shaders compiled successfully."
