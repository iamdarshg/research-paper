param(
    [switch]$BootstrapVenv,
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$PytestArgs = @()
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSCommandPath
$WindowsVenvPython = Join-Path $RepoRoot ".venv-win\Scripts\python.exe"
$WindowsProjectVenvPython = Join-Path $RepoRoot ".venv\Scripts\python.exe"
$UnixProjectVenvConfig = Join-Path $RepoRoot ".venv\pyvenv.cfg"
$DevRequirements = Join-Path $RepoRoot "requirements-dev.txt"
$RequiredModules = @("pytest", "torch", "numpy", "scipy", "skimage", "click", "yaml", "trimesh", "PyQt5")

function Test-PythonModules {
    param(
        [Parameter(Mandatory = $true)]
        [string]$PythonPath,
        [Parameter(Mandatory = $true)]
        [string[]]$Modules
    )

    if (-not (Test-Path $PythonPath)) {
        return $false
    }

    $moduleList = ($Modules | ForEach-Object { "'$_'" }) -join ","
    $probeCode = @"
import importlib.util
import sys

missing = [name for name in [$moduleList] if importlib.util.find_spec(name) is None]
sys.exit(1 if missing else 0)
"@

    $psi = New-Object System.Diagnostics.ProcessStartInfo
    $psi.FileName = $PythonPath
    $psi.Arguments = "-c `"$probeCode`""
    $psi.WorkingDirectory = $RepoRoot
    $psi.RedirectStandardOutput = $true
    $psi.RedirectStandardError = $true
    $psi.UseShellExecute = $false

    $process = [System.Diagnostics.Process]::Start($psi)
    $null = $process.StandardOutput.ReadToEnd()
    $null = $process.StandardError.ReadToEnd()
    $process.WaitForExit()

    return $process.ExitCode -eq 0
}

function Get-PythonCandidates {
    $candidates = New-Object System.Collections.Generic.List[string]

    foreach ($path in @(
        $WindowsVenvPython,
        $WindowsProjectVenvPython
    )) {
        if ($path -and (Test-Path $path) -and -not $candidates.Contains($path)) {
            [void]$candidates.Add($path)
        }
    }

    $pythonCommand = Get-Command python -ErrorAction SilentlyContinue
    if ($pythonCommand -and $pythonCommand.Source -and -not $candidates.Contains($pythonCommand.Source)) {
        [void]$candidates.Add($pythonCommand.Source)
    }

    foreach ($path in (Get-ChildItem @(
        "C:\Program Files\Python*\python.exe",
        "C:\Users\$env:USERNAME\AppData\Local\Programs\Python\Python*\python.exe"
    ) -ErrorAction SilentlyContinue | Sort-Object FullName -Descending | Select-Object -ExpandProperty FullName)) {
        if ($path -and -not $candidates.Contains($path)) {
            [void]$candidates.Add($path)
        }
    }

    return $candidates
}

function New-WindowsVenv {
    param(
        [Parameter(Mandatory = $true)]
        [string]$BasePython
    )

    if (-not (Test-Path $DevRequirements)) {
        throw "Missing dev requirements file: $DevRequirements"
    }

    Write-Host "Creating Windows virtualenv at .venv-win using $BasePython"
    & $BasePython -m venv (Join-Path $RepoRoot ".venv-win")
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to create .venv-win"
    }

    Write-Host "Installing dev dependencies from requirements-dev.txt"
    & $WindowsVenvPython -m pip install --upgrade pip
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to upgrade pip in .venv-win"
    }

    & $WindowsVenvPython -m pip install -r $DevRequirements
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to install dev dependencies into .venv-win"
    }
}

if (Test-Path $UnixProjectVenvConfig) {
    $venvConfigText = Get-Content $UnixProjectVenvConfig -Raw
    if ($venvConfigText -match "home = /usr/bin") {
        Write-Host "Ignoring Unix-style .venv on Windows; use .venv-win or a Windows Python install."
    }
}

$selectedPython = $null
foreach ($candidate in Get-PythonCandidates) {
    if (Test-PythonModules -PythonPath $candidate -Modules $RequiredModules) {
        $selectedPython = $candidate
        break
    }
}

if (-not $selectedPython -and $BootstrapVenv) {
    $basePython = (Get-PythonCandidates | Where-Object { $_ -ne $WindowsVenvPython } | Select-Object -First 1)
    if (-not $basePython) {
        throw "No Windows Python installation was found to bootstrap .venv-win."
    }

    New-WindowsVenv -BasePython $basePython

    if (Test-PythonModules -PythonPath $WindowsVenvPython -Modules $RequiredModules) {
        $selectedPython = $WindowsVenvPython
    } else {
        throw "Bootstrapped .venv-win, but required modules are still missing."
    }
}

if (-not $selectedPython) {
    throw @"
No suitable Windows Python test environment was found.

Try one of:
  1. .\run_tests.ps1 -BootstrapVenv -q
  2. Install dev dependencies into a Windows Python:
     pip install -r requirements-dev.txt
"@
}

Write-Host "Using Python: $selectedPython"
Push-Location $RepoRoot
try {
    & $selectedPython -m pytest @PytestArgs
    exit $LASTEXITCODE
}
finally {
    Pop-Location
}
