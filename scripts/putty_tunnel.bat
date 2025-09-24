@echo off
REM MLFlow SSH Tunnel using PuTTY for Windows
REM Download PuTTY from: https://www.putty.org/

REM Configuration - UPDATE THESE VALUES
set HOST_IP=34.118.75.91
set HOST_USER=Hudini
set LOCAL_PORT=5000
set REMOTE_PORT=5000

echo ====================================
echo MLFlow SSH Tunnel via PuTTY
echo ====================================
echo.
echo Host IP: %HOST_IP%
echo User: %HOST_USER%
echo Local Port: %LOCAL_PORT%
echo Remote Port: %REMOTE_PORT%
echo.

REM Check if PuTTY is available
where plink >nul 2>&1
if errorlevel 1 (
    echo ERROR: PuTTY plink.exe not found in PATH
    echo.
    echo Download and install PuTTY from: https://www.putty.org/
    echo Make sure plink.exe is in your PATH or place this script in PuTTY directory
    echo.
    pause
    exit /b 1
)

echo Creating SSH tunnel with PuTTY...
echo After connection, open: http://localhost:%LOCAL_PORT%
echo Press Ctrl+C to disconnect
echo.

REM Create SSH tunnel using PuTTY's plink
plink -ssh -L %LOCAL_PORT%:localhost:%REMOTE_PORT% %HOST_USER%@%HOST_IP%

echo.
echo SSH tunnel disconnected.
pause