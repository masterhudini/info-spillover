@echo off
REM MLFlow SSH Tunnel for Windows
REM Update the variables below with your actual values

REM Configuration - UPDATE THESE VALUES
set HOST_IP=34.118.75.91
set HOST_USER=Hudini
set LOCAL_PORT=5000
set REMOTE_PORT=5000

echo ====================================
echo MLFlow SSH Tunnel Setup
echo ====================================
echo.
echo Host IP: %HOST_IP%
echo User: %HOST_USER%
echo Local Port: %LOCAL_PORT%
echo Remote Port: %REMOTE_PORT%
echo.
echo This will create SSH tunnel to access MLFlow UI
echo After connection, open: http://localhost:%LOCAL_PORT%
echo.

REM Check if ssh is available
ssh -V >nul 2>&1
if errorlevel 1 (
    echo ERROR: SSH not found in PATH
    echo.
    echo Install options:
    echo 1. Windows 10/11: Enable OpenSSH Client in Windows Features
    echo 2. Install Git for Windows (includes SSH)
    echo 3. Install PuTTY and use putty_tunnel.bat instead
    echo.
    pause
    exit /b 1
)

echo Connecting via SSH tunnel...
echo Press Ctrl+C to disconnect
echo.

REM Create SSH tunnel
ssh -L %LOCAL_PORT%:localhost:%REMOTE_PORT% %HOST_USER%@%HOST_IP%

echo.
echo SSH tunnel disconnected.
pause