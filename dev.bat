@echo off
setlocal enabledelayedexpansion

:: Ensure uv and nodejs are on PATH (in case not in system PATH)
set "PATH=D:\uv;D:\nodejs;%PATH%"

:: Use current script directory as project root (portable across machines)
set "PROJECT_DIR=%~dp0"

echo ==========================================
echo   EHT Analytica - Dev Mode
echo ==========================================
echo.
echo  Backend:  http://127.0.0.1:9876  (auto-reload)
echo  Frontend: http://localhost:5173  (hot reload, open this)
echo.
echo  Close each window to stop.
echo ==========================================
echo.

echo Killing any process on port 9876 ...
for /f "tokens=5" %%a in ('netstat -ano ^| findstr /R /C:"127.0.0.1:9876.*LISTENING"') do (
    echo   Killing PID %%a
    taskkill /F /PID %%a >nul 2>&1
)
for /f "tokens=5" %%a in ('netstat -ano ^| findstr /R /C:"0.0.0.0:5173.*LISTENING"') do (
    echo   Killing PID %%a
    taskkill /F /PID %%a >nul 2>&1
)

:: Ensure Python virtual environment exists
if not exist "%PROJECT_DIR%.venv\" (
    echo.
    echo Creating Python virtual environment with uv...
    cd /d "%PROJECT_DIR%"
    uv sync
    if errorlevel 1 (
        echo ERROR: uv sync failed. Is uv installed? Check D:\uv is on PATH.
        pause
        exit /b 1
    )
    echo Environment ready.
)

echo.
echo Starting backend ...
start "EHT Backend" cmd /k "cd /d "%PROJECT_DIR%" && uv run uvicorn backend.main:app --host 127.0.0.1 --port 9876 --reload"

echo Starting frontend dev server ...
start "EHT Frontend" cmd /k "cd /d "%PROJECT_DIR%\frontend" && npm run dev"

echo.
echo Waiting for frontend to be ready...
timeout /t 3 /nobreak >nul
echo Opening http://localhost:5173 ...
start http://localhost:5173

echo.
echo Both servers started. Close each cmd window to stop.
echo.
pause
