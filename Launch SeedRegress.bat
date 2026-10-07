@echo off
setlocal
cd /d "%~dp0"

where python >nul 2>&1
if errorlevel 1 goto missing

python -c "import sys; raise SystemExit(0 if sys.version_info >= (3, 11) else 1)"
if errorlevel 1 goto old

if not exist .venv (
  python -m venv .venv
)
call .venv\Scripts\activate.bat

python -c "import pathlib,sys; s=pathlib.Path('.venv/.requirements.stamp'); r=pathlib.Path('requirements.txt'); sys.exit(0 if s.is_file() and s.stat().st_mtime >= r.stat().st_mtime else 1)"
if errorlevel 1 (
  python -m pip install --upgrade pip
  python -m pip install -r requirements.txt
  echo ok>.venv\.requirements.stamp
)

python -m seedregress serve
if errorlevel 1 pause
exit /b %errorlevel%

:missing
echo Python 3.11 or newer is required, and python was not found.
echo Install it from https://www.python.org/downloads/ and then double-click this launcher again.
pause
exit /b 1

:old
echo Python 3.11 or newer is required.
echo Install a newer Python from https://www.python.org/downloads/ and then double-click this launcher again.
pause
exit /b 1
