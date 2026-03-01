param()
$ErrorActionPreference = 'Stop'
$root = Resolve-Path (Join-Path $PSScriptRoot '..')
Set-Location $root

git config core.hooksPath .githooks
Write-Host "Git hooks path set to .githooks" -ForegroundColor Green
Write-Host "pre-commit guard: tools/context_commit_guard.py" -ForegroundColor Green
