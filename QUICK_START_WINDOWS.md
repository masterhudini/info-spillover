# 🚀 Quick Start - Windows Access to MLFlow

## Szybki dostęp do MLFlow UI z Windows

### 1. Na Linux host (gdzie uruchomiłeś tę konfigurację):

```bash
# Start MLFlow server w tle
make mlflow-bg
```

### 2. Na Windows - Opcja A: PowerShell/CMD

```bash
# SSH tunnel command
ssh -L 5000:localhost:5000 Hudini@34.118.75.91
```

**Po połączeniu otwórz w przeglądarce:** http://localhost:5000

### 3. Na Windows - Opcja B: Gotowe skrypty

**Pobierz z Linux hosta:**
- `scripts/windows_tunnel.bat` - Windows Batch
- `scripts/windows_tunnel.ps1` - PowerShell
- `scripts/putty_tunnel.bat` - Dla PuTTY

**Uruchom wybrany skrypt na Windows**

### 4. Testuj eksperyment

**Na Linux host:**
```bash
# Uruchom przykładowy eksperyment
make sample-experiment
```

**Na Windows:**
- Otwórz http://localhost:5000
- Znajdź experiment: "info_spillover_experiment"
- Zobacz metryki i artifacts

## 🔧 Rozwiązywanie problemów

### SSH nie działa
```bash
# Test połączenia SSH
ssh Hudini@34.118.75.91

# Jeśli nie masz SSH na Windows:
# 1. Windows 10/11: Włącz "OpenSSH Client" w Windows Features
# 2. Lub zainstaluj Git for Windows
# 3. Lub użyj PuTTY
```

### Port zajęty na Windows
```powershell
# Sprawdź co używa portu 5000
netstat -an | findstr :5000

# Zabij proces lub użyj innego portu
ssh -L 5001:localhost:5000 Hudini@34.118.75.91
# Wtedy: http://localhost:5001
```

### MLFlow nie działa na Linux
```bash
# Sprawdź czy MLFlow działa
ps aux | grep mlflow

# Restart jeśli potrzeba
make mlflow-stop
make mlflow-bg
```

## 📊 Co zobaczysz w MLFlow UI

- **Experiments:** info_spillover_experiment
- **Runs:** sample_spillover_analysis (jeśli uruchomiono)
- **Metrics:** accuracy, f1_score, transfer_entropy
- **Parameters:** model configs, data info
- **Artifacts:** plots, models

## 🎯 Gotowe do eksperymentów!

Po ustawieniu tunnel można pracować z MLFlow tak jakby był lokalny na Windows!