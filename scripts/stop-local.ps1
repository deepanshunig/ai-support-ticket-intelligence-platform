$ErrorActionPreference = 'Stop'
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$runtimePath = Join-Path $projectRoot '.runtime'
$processFile = Join-Path $runtimePath 'local_processes.json'
if (Test-Path -LiteralPath $processFile) {
    $savedProcesses = Get-Content -LiteralPath $processFile -Raw | ConvertFrom-Json
    if ($savedProcesses.root -ne $projectRoot) { throw 'Process record belongs to a different project.' }
    foreach ($entry in @(@{id=$savedProcesses.api; module='ticket_intelligence.api.main'}, @{id=$savedProcesses.ui; module='streamlit'})) {
        $processInfo = Get-CimInstance Win32_Process -Filter "ProcessId = $($entry.id)"
        if ($processInfo) {
            if (-not $processInfo.CommandLine.Contains($savedProcesses.python) -or -not $processInfo.CommandLine.Contains($entry.module)) {
                throw 'Process identity changed; preserving it.'
            }
            & taskkill.exe /PID $entry.id /T /F | Out-Null
        }
    }
    Remove-Item -LiteralPath $processFile
}
Write-Output 'Stopped the project API and dashboard. Database servers remain running.'
