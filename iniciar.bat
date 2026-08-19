@echo off
cd /d "%~dp0"
if exist ".venv\Scripts\python.exe" (
	".venv\Scripts\python.exe" app.py
) else (
	"C:\Users\nilbe\AppData\Local\Programs\Python\Python313\python.exe" app.py
)
if errorlevel 1 pause