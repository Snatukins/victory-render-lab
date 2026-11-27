param(
    [Parameter(Mandatory)][string]$SdkDir,
    [Parameter(Mandatory)][string]$OutputDir
)
$ErrorActionPreference = 'Stop'
$sdk = (Resolve-Path -LiteralPath $SdkDir).Path
$out = [IO.Path]::GetFullPath($OutputDir)
$libs = Join-Path $sdk 'lib\Windows_x86_64'
$runtime = Join-Path $libs 'rel\nvngx_dlss.dll'
$signature = Get-AuthenticodeSignature -LiteralPath $runtime
if ($signature.Status -ne 'Valid' -or $signature.SignerCertificate.Subject -notmatch 'NVIDIA') {
    throw "Expected an NVIDIA-signed DLSS runtime: $runtime"
}
$vswhere = Join-Path ([Environment]::GetFolderPath('ProgramFilesX86')) 'Microsoft Visual Studio\Installer\vswhere.exe'
$vs = & $vswhere -latest -products '*' -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
if (-not $vs) { throw 'Visual Studio C++ Build Tools are required' }
$vcvars = Join-Path $vs 'VC\Auxiliary\Build\vcvars64.bat'
$source = Join-Path $PSScriptRoot 'dlss.cpp'
$include = Join-Path $sdk 'include'
$library = Join-Path $libs 'x64\nvsdk_ngx_s.lib'
New-Item -Path $out -ItemType Directory -Force | Out-Null
Push-Location -LiteralPath $out
try {
    cmd.exe /d /c ('call "{0}" && cl.exe /nologo /std:c++17 /EHsc /O2 /MT /LD /I"{1}" "{2}" "{3}" /link advapi32.lib user32.lib /OUT:victory_dlss.dll' -f $vcvars, $include, $source, $library)
    if ($LASTEXITCODE -ne 0) { throw 'DLSS adapter build failed' }
    Copy-Item -LiteralPath $runtime -Destination $out -Force
    Write-Output "Set VICTORY_DLSS_DIR to $out"
} finally {
    Pop-Location
}
