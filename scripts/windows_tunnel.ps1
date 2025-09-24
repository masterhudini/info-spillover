# MLFlow SSH Tunnel PowerShell Script
# Update the variables below with your actual values

# Configuration - UPDATE THESE VALUES
$HOST_IP = "34.118.75.91"
$HOST_USER = "Hudini"
$LOCAL_PORT = 5000
$REMOTE_PORT = 5000

Write-Host "====================================" -ForegroundColor Cyan
Write-Host "MLFlow SSH Tunnel Setup" -ForegroundColor Cyan
Write-Host "====================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Host IP: $HOST_IP" -ForegroundColor Yellow
Write-Host "User: $HOST_USER" -ForegroundColor Yellow
Write-Host "Local Port: $LOCAL_PORT" -ForegroundColor Yellow
Write-Host "Remote Port: $REMOTE_PORT" -ForegroundColor Yellow
Write-Host ""
Write-Host "This will create SSH tunnel to access MLFlow UI" -ForegroundColor Green
Write-Host "After connection, open: http://localhost:$LOCAL_PORT" -ForegroundColor Green
Write-Host ""

# Check if SSH is available
try {
    $null = Get-Command ssh -ErrorAction Stop
    Write-Host "✓ SSH client found" -ForegroundColor Green
} catch {
    Write-Host "✗ SSH client not found" -ForegroundColor Red
    Write-Host ""
    Write-Host "Install options:" -ForegroundColor Yellow
    Write-Host "1. Windows 10/11: Enable OpenSSH Client in Windows Features"
    Write-Host "2. Install Git for Windows (includes SSH)"
    Write-Host "3. Use PuTTY instead"
    Write-Host ""
    Read-Host "Press Enter to exit"
    exit 1
}

# Check if port is already in use
$portInUse = Get-NetTCPConnection -LocalPort $LOCAL_PORT -ErrorAction SilentlyContinue
if ($portInUse) {
    Write-Host "⚠️  Port $LOCAL_PORT is already in use" -ForegroundColor Yellow
    Write-Host "Close the application using this port or change LOCAL_PORT variable"
    Write-Host ""
}

Write-Host "Connecting via SSH tunnel..." -ForegroundColor Green
Write-Host "Press Ctrl+C to disconnect" -ForegroundColor Yellow
Write-Host ""

# Create SSH tunnel
try {
    ssh -L "${LOCAL_PORT}:localhost:${REMOTE_PORT}" "${HOST_USER}@${HOST_IP}"
} catch {
    Write-Host ""
    Write-Host "SSH connection failed: $($_.Exception.Message)" -ForegroundColor Red
}

Write-Host ""
Write-Host "SSH tunnel disconnected." -ForegroundColor Yellow
Read-Host "Press Enter to exit"