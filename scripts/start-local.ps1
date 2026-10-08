$ErrorActionPreference = 'Stop'
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$pythonPath = Join-Path $projectRoot '.venv\Scripts\python.exe'
$runtimePath = Join-Path $projectRoot '.runtime'
if (-not (Test-Path -LiteralPath $pythonPath)) { throw 'Create the project .venv first.' }
foreach ($port in @(8000, 8501)) {
    if (Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue) {
        throw "Port $port is occupied. Keep the existing process or use stop-local.ps1 for this project."
    }
}
if ($env:TICKET_DATABASE_URL -and -not $env:TICKET_DATABASE_URL.StartsWith('mysql+')) {
    throw 'TICKET_DATABASE_URL in this PowerShell session overrides .env. Clear or update that environment variable, then retry.'
}
& $pythonPath (Join-Path $PSScriptRoot 'setup_local_mysql.py')
if ($LASTEXITCODE -ne 0) { throw 'MySQL setup failed.' }
New-Item -ItemType Directory -Force -Path $runtimePath | Out-Null
$apiProcess = Start-Process -FilePath $pythonPath -ArgumentList @('-m', 'ticket_intelligence.api.main') -WorkingDirectory $projectRoot -WindowStyle Hidden -RedirectStandardOutput (Join-Path $runtimePath 'api.stdout.log') -RedirectStandardError (Join-Path $runtimePath 'api.stderr.log') -PassThru
$uiProcess = Start-Process -FilePath $pythonPath -ArgumentList @('-m', 'streamlit', 'run', 'ui/app.py') -WorkingDirectory $projectRoot -WindowStyle Hidden -RedirectStandardOutput (Join-Path $runtimePath 'ui.stdout.log') -RedirectStandardError (Join-Path $runtimePath 'ui.stderr.log') -PassThru
@{root=$projectRoot; python=$pythonPath; api=$apiProcess.Id; ui=$uiProcess.Id} | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $runtimePath 'local_processes.json') -Encoding utf8
Write-Output 'API: http://127.0.0.1:8000/docs'
Write-Output 'Dashboard: http://127.0.0.1:8501'
