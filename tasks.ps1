# tasks.ps1 — Windows-native equivalent of the Makefile.
#
# Usage from a PowerShell prompt at the repo root:
#
#   .\tasks.ps1 help
#   .\tasks.ps1 prepare
#   .\tasks.ps1 train
#   .\tasks.ps1 evaluate
#   .\tasks.ps1 serve
#
# If PowerShell blocks the script with an execution-policy error, run this
# ONCE (as your normal user, not as admin):
#
#   Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
#
# That allows local scripts you wrote yourself while still blocking
# unsigned ones from the internet.

param(
    [Parameter(Position = 0)]
    [string]$Task = "help",

    [Parameter(Position = 1)]
    [string]$Arg = ""
)

$ErrorActionPreference = "Stop"

function Show-Help {
    Write-Host ""
    Write-Host "RetailGenius Churn - PowerShell task runner" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "  Setup"
    Write-Host "    install         Install runtime dependencies"
    Write-Host "    install-dev     Install runtime + dev dependencies + pre-commit hooks"
    Write-Host ""
    Write-Host "  Quality"
    Write-Host "    format          Run black and isort"
    Write-Host "    lint            Run flake8"
    Write-Host "    test            Run pytest"
    Write-Host ""
    Write-Host "  Pipeline"
    Write-Host "    prepare         Build train/val/test splits + reference snapshot"
    Write-Host "    train           Train all three models, log to MLflow"
    Write-Host "    evaluate        Evaluate Staging on test set, promote to Production"
    Write-Host "    serve           Serve Production model locally on port 5001"
    Write-Host "    mlflow-ui       Open MLflow UI on port 5000"
    Write-Host ""
    Write-Host "  Monitoring & operations"
    Write-Host "    drift <path>    Run drift report on the parquet at <path>"
    Write-Host "    rollback-list   List every registered model version"
    Write-Host ""
    Write-Host "  Docs & deploy"
    Write-Host "    docs            Build Sphinx HTML docs"
    Write-Host "    docker-build    Build the serving Docker image"
    Write-Host "    docker-run      Run the serving container on port 8080"
    Write-Host ""
    Write-Host "    clean           Remove caches"
    Write-Host ""
}

switch ($Task.ToLower()) {
    "help"            { Show-Help }

    "install"         { pip install -r requirements.txt; pip install -e . }
    "install-dev"     { pip install -r requirements-dev.txt; pip install -e .; pre-commit install }

    "format"          { black src tests; isort src tests }
    "lint"            { flake8 src tests }
    "test"            { pytest }

    "prepare"         { python -m churn.data.make_dataset }
    "train"           { python -m churn.models.train }
    "evaluate"        { python -m churn.models.evaluate }
    "serve"           { mlflow models serve -m "models:/churn_classifier/Production" -p 5001 --no-conda }
    "mlflow-ui"       { mlflow ui --port 5000 }

    "drift" {
        if ([string]::IsNullOrWhiteSpace($Arg)) {
            Write-Host "Usage: .\tasks.ps1 drift <path-to-current.parquet>" -ForegroundColor Yellow
            exit 1
        }
        python -m churn.monitoring.report --current $Arg
    }

    "rollback-list"   { python -m churn.models.rollback --list }

    "docs"            { sphinx-build -b html docs docs\_build\html }
    "docker-build"    { docker build -t retailgenius-churn:latest . }
    "docker-run"      { docker run -p 8080:8080 retailgenius-churn:latest }

    "clean" {
        Get-ChildItem -Path . -Recurse -Force -Directory `
            | Where-Object { $_.Name -in @("__pycache__", ".pytest_cache") } `
            | Remove-Item -Recurse -Force
        if (Test-Path .coverage)        { Remove-Item .coverage }
        if (Test-Path htmlcov)          { Remove-Item htmlcov -Recurse -Force }
        if (Test-Path docs\_build)      { Remove-Item docs\_build -Recurse -Force }
        Write-Host "Cleaned."
    }

    default {
        Write-Host "Unknown task: $Task" -ForegroundColor Red
        Show-Help
        exit 1
    }
}
