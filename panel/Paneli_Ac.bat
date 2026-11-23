@echo off
rem Kontrol panelini env_fer ortamindaki Python ile acar.
cd /d "%~dp0"
if exist "..\env_fer\Scripts\python.exe" (
  start "Duygu Tanima Paneli" "..\env_fer\Scripts\python.exe" panel.py
) else (
  echo env_fer bulunamadi. README.md dosyasindaki kurulum adimlarina bakin.
  pause
)
