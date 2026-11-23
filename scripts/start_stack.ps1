
$ErrorActionPreference = "Stop"

$composeFiles = @(
    "-f", "compose.db.yaml",
    "-f", "compose.monitoring.yaml",
    "-f", "compose.api.yaml",
    "-f", "compose.unified.yaml"
)

if (-not $env:POSTGRES_PASSWORD) {
    throw "POSTGRES_PASSWORD is not set."
}

if (-not $env:GRAFANA_ADMIN_PASSWORD) {
    throw "GRAFANA_ADMIN_PASSWORD is not set."
}

docker compose @composeFiles config --quiet

if ($LASTEXITCODE -ne 0) {
    throw "Docker Compose configuration validation failed."
}

docker compose @composeFiles up -d --build

if ($LASTEXITCODE -ne 0) {
    throw "Docker Compose deployment failed."
}

docker compose @composeFiles ps
