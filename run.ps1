# Запуск SPL в Windows: подготовка окружения, данных, детекторов, API и Streamlit.

[CmdletBinding()]
param(
    [Parameter(Position = 0)]
    [ValidateSet("all", "api", "ui")]
    [string]$Mode = "all",

    [switch]$NoInstall,
    [switch]$NoSetup,
    [switch]$NoGeracl,
    [switch]$Retrain,
    [string]$PythonBin = "",
    [switch]$Help
)

$ErrorActionPreference = "Stop"
$env:PYTHONUTF8 = "1"

$Root = $PSScriptRoot
Set-Location $Root

$VenvDir = if ($env:VENV_DIR) { $env:VENV_DIR } else { "venv" }
$ApiHost = if ($env:API_HOST) { $env:API_HOST } else { "127.0.0.1" }
$ApiPort = if ($env:API_PORT) { $env:API_PORT } else { "8000" }
$UiHost = if ($env:UI_HOST) { $env:UI_HOST } else { "127.0.0.1" }
$UiPort = if ($env:UI_PORT) { $env:UI_PORT } else { "8501" }
$script:ApiProcess = $null

$ModelCheckScript = @'
import pickle

for path in ("models/adversarial_detector.pkl", "models/anomaly_detector.pkl"):
    with open(path, "rb") as file:
        artifacts = pickle.load(file)
    if "model" not in artifacts:
        raise SystemExit(1)
'@

function Show-Usage {
    Write-Host @"
Использование: .\run.ps1 [all|api|ui] [-NoInstall] [-NoSetup] [-NoGeracl] [-Retrain]

  all          запустить API и Streamlit (по умолчанию)
  api          запустить только API
  ui           запустить только Streamlit
  -NoInstall   не устанавливать зависимости
  -NoSetup     не генерировать данные и не обучать детекторы
  -NoGeracl    не устанавливать официальный backend geracl
  -Retrain     переобучить детекторы даже при наличии pkl
  -PythonBin   путь к python.exe или имя команды Python
  -Help        показать эту справку

Переменные окружения: VENV_DIR, API_HOST, API_PORT, UI_HOST, UI_PORT.
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

    throw "Python не найден. Установи Python 3.10-3.12 и повтори запуск."
}

$pythonInfo = Find-Python
$PythonExe = $pythonInfo.Exe
$PythonArgs = $pythonInfo.Args

$VenvPath = Join-Path $Root $VenvDir
$VenvPython = Join-Path $VenvPath "Scripts\python.exe"

if (-not (Test-Path $VenvPython)) {
    Write-Host "Создаю виртуальное окружение $VenvDir..."
    & $PythonExe @PythonArgs -m venv $VenvPath
    if ($LASTEXITCODE -ne 0) {
        throw "Не удалось создать виртуальное окружение."
    }
}

& $VenvPython -c "import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)"
if ($LASTEXITCODE -ne 0) {
    throw "Нужен Python 3.10 или новее."
}

if (-not $NoInstall) {
    $Requirements = Join-Path $Root "requirements.txt"
    $Marker = Join-Path $VenvPath ".spl_requirements"
    $NeedInstall = $true
    if (Test-Path $Marker) {
        $NeedInstall = (Get-Item $Requirements).LastWriteTime -gt (Get-Item $Marker).LastWriteTime
    }

    if ($NeedInstall) {
        Write-Host "Устанавливаю зависимости..."
        & $VenvPython -m pip install --upgrade pip
        if ($LASTEXITCODE -ne 0) {
            throw "Не удалось обновить pip."
        }
        & $VenvPython -m pip install -r $Requirements
        if ($LASTEXITCODE -ne 0) {
            throw "Не удалось установить зависимости."
        }
        Set-Content -Path $Marker -Value (Get-Date -Format "o") -Encoding UTF8
    }
}

function Test-Geracl {
    & $VenvPython -c "import geracl" 2>$null | Out-Null
    return ($LASTEXITCODE -eq 0)
}

function Test-Models {
    $adversarial = Join-Path $Root "models\adversarial_detector.pkl"
    $anomaly = Join-Path $Root "models\anomaly_detector.pkl"
    if (-not ((Test-Path $adversarial) -and (Test-Path $anomaly))) {
        return $false
    }
    & $VenvPython -c $ModelCheckScript 2>$null | Out-Null
    return ($LASTEXITCODE -eq 0)
}

if (-not (Test-Geracl)) {
    if ($NoInstall) {
        Write-Host "Подсказка: geracl не установлен, чистые запросы вернут ModelError."
        Write-Host "Установи backend: .\venv\Scripts\python.exe -m pip install -r requirements-geracl.txt"
    } elseif ($NoGeracl) {
        Write-Host "Backend geracl пропущен (-NoGeracl). Чистые запросы вернут ModelError."
    } else {
        $git = Get-Command git -ErrorAction SilentlyContinue
        if (-not $git) {
            Write-Warning "Для установки geracl нужен git. Чистые запросы вернут ModelError."
        } else {
            Write-Host "Устанавливаю geracl (это может занять время)..."
            $GeraclRequirements = Join-Path $Root "requirements-geracl.txt"
            & $VenvPython -m pip install -r $GeraclRequirements
            if ($LASTEXITCODE -ne 0) {
                Write-Warning "Не удалось установить geracl. Чистые запросы вернут ModelError."
            } else {
                Write-Host "geracl установлен."
            }
        }
    }
}

$ModelsReady = Test-Models
if ($Retrain) {
    $ModelsReady = $false
}

if ($NoSetup) {
    if ($ModelsReady) {
        Write-Host "Использую существующие обученные детекторы."
    } else {
        Write-Warning "Указан -NoSetup, но детекторы отсутствуют или не загружаются."
    }
} elseif ($ModelsReady) {
    Write-Host "Обученные детекторы найдены, обучение пропущено."
} else {
    $adversarial = Join-Path $Root "models\adversarial_detector.pkl"
    $anomaly = Join-Path $Root "models\anomaly_detector.pkl"
    if ($Retrain) {
        Write-Host "Принудительное переобучение детекторов (-Retrain)..."
    } elseif ((Test-Path $adversarial) -or (Test-Path $anomaly)) {
        Write-Host "Сохранённые детекторы не загрузились, переобучаю..."
    }

    $trainCsv = Join-Path $Root "data\detector_train.csv"
    $anomalyTexts = Join-Path $Root "data\anomaly_texts.txt"
    if (-not ((Test-Path $trainCsv) -and (Test-Path $anomalyTexts))) {
        Write-Host "Генерирую данные..."
        & $VenvPython "data/generate_data.py"
        if ($LASTEXITCODE -ne 0) {
            throw "Не удалось сгенерировать данные."
        }
    }

    Write-Host "Обучаю детекторы..."
    & $VenvPython "scripts/train_detectors.py"
    if ($LASTEXITCODE -ne 0) {
        throw "Не удалось обучить детекторы."
    }
    if (-not (Test-Models)) {
        throw "Обученные детекторы не загружаются."
    }
}

Write-Host "Модель GeRaCl загрузится при первом чистом запросе (кэш около 600 МБ)."

$HealthHost = $ApiHost
if ($ApiHost -eq "0.0.0.0" -or $ApiHost -eq "::") {
    $HealthHost = "127.0.0.1"
}

function Wait-Api {
    for ($attempt = 0; $attempt -lt 40; $attempt++) {
        try {
            $response = Invoke-WebRequest -Uri "http://${HealthHost}:${ApiPort}/health" -UseBasicParsing -TimeoutSec 1
            if ($response.StatusCode -eq 200) {
                return $true
            }
        } catch {
        }
        if ($script:ApiProcess -and $script:ApiProcess.HasExited) {
            return $false
        }
        Start-Sleep -Milliseconds 500
    }
    return $false
}

function Start-Api {
    Write-Host "Запускаю API на http://${ApiHost}:${ApiPort}..."
    $env:PYTHONPATH = $Root
    $arguments = @(
        "-m", "uvicorn",
        "src.presentation.api.app:create_app",
        "--factory",
        "--host", $ApiHost,
        "--port", $ApiPort
    )
    $script:ApiProcess = Start-Process -FilePath $VenvPython -ArgumentList $arguments -PassThru -NoNewWindow
    if (-not (Wait-Api)) {
        throw "API не запустился. Проверь лог выше."
    }
    Write-Host "API готов: http://${ApiHost}:${ApiPort}"
    Write-Host "Swagger: http://${ApiHost}:${ApiPort}/docs"
}

function Start-Ui {
    Write-Host "Запускаю Streamlit на http://${UiHost}:${UiPort}..."
    $env:PYTHONPATH = $Root
    & $VenvPython -m streamlit run "src/presentation/ui/streamlit_app.py" `
        --server.address $UiHost `
        --server.port $UiPort `
        --server.headless true
}

try {
    switch ($Mode) {
        "all" {
            Start-Api
            Start-Ui
        }
        "api" {
            Start-Api
            Write-Host "Нажми Ctrl+C для остановки."
            Wait-Process -Id $script:ApiProcess.Id
        }
        "ui" {
            Start-Ui
        }
    }
} finally {
    if ($script:ApiProcess -and -not $script:ApiProcess.HasExited) {
        Write-Host "Останавливаю API..."
        Stop-Process -Id $script:ApiProcess.Id -Force -ErrorAction SilentlyContinue
    }
}
