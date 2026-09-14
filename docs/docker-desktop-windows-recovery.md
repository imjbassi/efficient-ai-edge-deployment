# Recover Docker Desktop from stale Windows sockets

Docker Desktop can fail after an unclean shutdown because stale AF_UNIX socket reparse points under `%LOCALAPPDATA%` become inaccessible. The characteristic error says that `dockerInference`, `sailor-ingest.sock`, or `engine.sock` "cannot be accessed by the system." This is tracked upstream as [docker/desktop-feedback#554](https://github.com/docker/desktop-feedback/issues/554).

Deleting or renaming the socket files themselves does not work. Preserve and rename their parent runtime directories while Docker Desktop is completely stopped, then let Docker recreate them:

```powershell
Get-Process | Where-Object {$_.ProcessName -match '^docker|^com\.docker'} | Stop-Process -Force

Rename-Item "$env:LOCALAPPDATA\Docker\run" "run.corrupt-$(Get-Date -Format yyyyMMdd-HHmmss)"
New-Item -ItemType Directory "$env:LOCALAPPDATA\Docker\run"

if (Test-Path "$env:LOCALAPPDATA\docker-secrets-engine") {
  Rename-Item "$env:LOCALAPPDATA\docker-secrets-engine" "docker-secrets-engine.corrupt-$(Get-Date -Format yyyyMMdd-HHmmss)"
}

Start-Process "$env:LOCALAPPDATA\Programs\DockerDesktop\Docker Desktop.exe"
```

This recovery does not modify Docker's VM disk, images, containers, or volumes. The renamed directories may remain inaccessible because they contain the corrupted reparse points.

To reduce recurrence, avoid force-killing Docker or putting Windows to sleep while Docker Desktop is starting. Before controlled host benchmarks, use `docker desktop stop` and wait for `docker desktop status` to report that the application has stopped.
