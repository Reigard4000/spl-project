# Запуск веб-клиента SPL на Windows 10/11.

[CmdletBinding()]
param(
    [int]$Port = 3000,
    [string]$HostName = "127.0.0.1",
    [switch]$NoBrowser,
    [string]$PythonBin = "",
    [switch]$Help
)

$ErrorActionPreference = "Stop"
$env:PYTHONUTF8 = "1"

$Root = Split-Path $PSScriptRoot -Parent
$ServerScript = Join-Path $PSScriptRoot "server.py"

function Show-Usage {
    Write-Host @"
Использование: .\run_client.ps1 [-Port 3000] [-HostName 127.0.0.1] [-NoBrowser] [-PythonBin <path>]

  -Port       порт клиента (по умолчанию 3000)
  -HostName   адрес прослушивания (по умолчанию 127.0.0.1)
  -NoBrowser  не открывать браузер автоматически
  -PythonBin  путь к python.exe или имя команды Python
  -Help       показать эту справку

API должен быть запущен отдельно: ..\run.cmd api
"@
}

if ($Help) {
    Show-Usage
    exit 0
}

function Find-Python {
    if ($PythonBin) {
        if (Test-Path $PythonBin) {
            return [pscustomobject]@{ Exe = (Resolve-Path $PythonBin).Path; Args = @() }
        }
        $command = Get-Command $PythonBin -ErrorAction SilentlyContinue
        if ($command) {
            return [pscustomobject]@{ Exe = $command.Source; Args = @() }
        }
        throw "Python не найден: $PythonBin"
    }

    $venvCandidates = @(
        (Join-Path $Root "venv\Scripts\python.exe"),
        (Join-Path $Root ".venv\Scripts\python.exe")
    )
    foreach ($candidate in $venvCandidates) {
        if (Test-Path $candidate) {
            return [pscustomobject]@{ Exe = $candidate; Args = @() }
        }
    }

    $py = Get-Command py -ErrorAction SilentlyContinue
    if ($py) {
        foreach ($version in @("-3.12", "-3.11", "-3.10", "-3")) {
            & $py.Source $version -c "import sys" 2>$null | Out-Null
            if ($LASTEXITCODE -eq 0) {
                return [pscustomobject]@{ Exe = $py.Source; Args = @($version) }
            }
        }
    }

    $python = Get-Command python -ErrorAction SilentlyContinue
    if ($python) {
        return [pscustomobject]@{ Exe = $python.Source; Args = @() }
    }

    throw "Python не найден. Установи Python 3.10+ и повтори запуск."
}

$pythonInfo = Find-Python
$PythonExe = $pythonInfo.Exe
$PythonArgs = $pythonInfo.Args

& $PythonExe @PythonArgs -c "import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)"
if ($LASTEXITCODE -ne 0) {
    throw "Нужен Python 3.10 или новее."
}

$arguments = @()
$arguments += $PythonArgs
$arguments += @(
    $ServerScript,
    "--host", $HostName,
    "--port", "$Port"
)
if ($NoBrowser) {
    $arguments += "--no-browser"
}

& $PythonExe @arguments
exit $LASTEXITCODE
