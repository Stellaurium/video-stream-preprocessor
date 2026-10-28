param(
    [string]$Python = '',
    [switch]$Dev,
    [switch]$CheckOnly,
    [switch]$SkipModels,
    [switch]$NoPause
)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$probe = "import sys,struct; print('%s|%s|%s|%s' % (sys.version_info[0],sys.version_info[1],struct.calcsize('P')*8,sys.executable))"

function Test-PythonCandidate($candidate) {
    $candidateArgs = @($candidate.Arguments)
    try {
        $answer = & $candidate.File @candidateArgs -c $probe 2>$null
        if ($LASTEXITCODE -ne 0) { return $null }
        $parts = (($answer | Select-Object -Last 1) -as [string]).Trim().Split('|')
        if ($parts.Count -eq 4 -and $parts[0] -eq '3' -and $parts[1] -eq '11' -and $parts[2] -eq '64') {
            return [pscustomobject]@{ File = $parts[3]; Arguments = @() }
        }
    } catch { }
    return $null
}

$setupExit = 0
try {
    # Do not let another Python installation inject modules into this setup process.
    Remove-Item Env:PYTHONHOME -ErrorAction SilentlyContinue
    $env:PYTHONPATH = ''
    $env:PYTHONUTF8 = '1'
    $candidates = [Collections.Generic.List[object]]::new()
    if (-not $Python -and $env:APP_PYTHON) { $Python = $env:APP_PYTHON }
    if ($Python) {
        $command = Get-Command -Name $Python -CommandType Application -ErrorAction SilentlyContinue | Select-Object -First 1
        if (-not $command) { throw 'The selected Python executable was not found.' }
        $candidates.Add([pscustomobject]@{ File = $command.Source; Arguments = @() })
    } else {
        $launcher = Get-Command py.exe -CommandType Application -ErrorAction SilentlyContinue
        if ($launcher) { $candidates.Add([pscustomobject]@{ File = $launcher.Source; Arguments = @('-3.11') }) }
        foreach ($name in @('python.exe','python3.exe')) {
            foreach ($command in @(Get-Command $name -All -CommandType Application -ErrorAction SilentlyContinue)) {
                if ($command.Source -notlike '*\WindowsApps\*') {
                    $candidates.Add([pscustomobject]@{ File = $command.Source; Arguments = @() })
                }
            }
        }
    }
    $selected = $null
    foreach ($candidate in $candidates) {
        $selected = Test-PythonCandidate $candidate
        if ($selected) { break }
    }
    if (-not $selected) {
        throw 'Python 3.11 x64 is required. Retry with -Python "C:\Path\To\Python311\python.exe". Python 2.x or other versions will not be used.'
    }
    if ($CheckOnly) {
        [pscustomobject]@{ python = $selected.File; version = '3.11'; bits = 64; project = $projectRoot } | ConvertTo-Json -Compress
        exit 0
    }
    Set-Location -LiteralPath $projectRoot
    $environment = Join-Path $projectRoot '.venv\Scripts\python.exe'
    if (-not (Test-Path -LiteralPath $environment)) {
        Write-Host ('Creating environment with: ' + $selected.File)
        & $selected.File -m venv (Join-Path $projectRoot '.venv')
        if ($LASTEXITCODE -ne 0) { throw 'Could not create the virtual environment.' }
    }
    if (-not (Test-PythonCandidate ([pscustomobject]@{ File = $environment; Arguments = @() }))) {
        throw 'The existing .venv is not Python 3.11 x64. Recreate it with the selected interpreter.'
    }
    $requirements = if ($Dev -and (Test-Path -LiteralPath 'requirements-dev.txt')) { 'requirements-dev.txt' } else { 'requirements.txt' }
    & $environment -m pip install --disable-pip-version-check --require-virtualenv -r $requirements
    if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed. Check the network and pip output.' }
    $modelTool = Join-Path $PSScriptRoot 'prepare_models.py'
    if ((Test-Path -LiteralPath $modelTool) -and -not $SkipModels) {
        & $environment $modelTool
        if ($LASTEXITCODE -ne 0) { throw 'Model preparation failed. Rerun setup to resume verified downloads.' }
    }
    $ffmpeg = Join-Path $projectRoot 'bin\ffmpeg.exe'
    if (-not (Test-Path -LiteralPath $ffmpeg)) {
        $found = Get-Command ffmpeg.exe -CommandType Application -ErrorAction SilentlyContinue
        $ffmpeg = if ($found) { $found.Source } else { $null }
    }
    if ($ffmpeg) {
        Write-Host ('FFmpeg found: ' + $ffmpeg)
    } else {
        Write-Warning 'Camera discovery needs FFmpeg. Place ffmpeg.exe in bin/ or add it to PATH; see bin/README.md.'
    }
    Write-Host 'Setup completed. Use the launch CMD in the project root.'
} catch {
    [Console]::Error.WriteLine('Setup failed: ' + $_.Exception.Message)
    $setupExit = 1
}
if (-not $NoPause -and -not $CheckOnly) { Read-Host 'Press Enter to close' | Out-Null }
exit $setupExit
