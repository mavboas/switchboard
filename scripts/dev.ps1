# Sobe a stack local sem Docker no Windows (PowerShell): conectores MCP e agentes A2A de exemplo, console e router.
# Usa SQLite em .\data por padrão; para PostgreSQL, defina $env:SWITCHBOARD_DATABASE_URL antes.
$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")

if (-not $env:SWITCHBOARD_DATABASE_URL) { $env:SWITCHBOARD_DATABASE_URL = "sqlite:///./data/switchboard.db" }
if (-not $env:SWITCHBOARD_SECRET_KEY) { $env:SWITCHBOARD_SECRET_KEY = "dev-local-troque-em-producao" }
if (-not $env:SWITCHBOARD_ROUTER_URL) { $env:SWITCHBOARD_ROUTER_URL = "http://127.0.0.1:8080" }
# URL pela qual os agentes A2A mandam push notifications ao router
if (-not $env:SWITCHBOARD_PUBLIC_URL) { $env:SWITCHBOARD_PUBLIC_URL = "http://127.0.0.1:8080" }
New-Item -ItemType Directory -Force -Path data | Out-Null

$procs = @()
$procs += Start-Process uv -ArgumentList "run", "switchboard-agent", "credito", "--port", "8101" -NoNewWindow -PassThru
$procs += Start-Process uv -ArgumentList "run", "switchboard-agent", "chamados", "--port", "8102" -NoNewWindow -PassThru
$procs += Start-Process uv -ArgumentList "run", "switchboard-agent", "analise-credito", "--port", "8201" -NoNewWindow -PassThru
$procs += Start-Process uv -ArgumentList "run", "switchboard-agent", "risco", "--port", "8202" -NoNewWindow -PassThru
$env:SWITCHBOARD_PORT = "8000"
$procs += Start-Process uv -ArgumentList "run", "switchboard-console" -NoNewWindow -PassThru
Start-Sleep -Seconds 3
$env:SWITCHBOARD_PORT = "8080"
$procs += Start-Process uv -ArgumentList "run", "switchboard-router" -NoNewWindow -PassThru

Write-Host ""
Write-Host "  console: http://localhost:8000   router: http://localhost:8080/docs"
Write-Host "  Ctrl+C encerra tudo."
try { Wait-Process -Id ($procs | ForEach-Object { $_.Id }) }
finally { $procs | ForEach-Object { Stop-Process -Id $_.Id -ErrorAction SilentlyContinue } }
