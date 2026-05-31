@echo off
set "PROJECT_DIR=d:\Desktop\EHT\EHT_Analytica"
set "CONDA_PY=C:\Users\WC\.conda\envs\eht\python.exe"

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

echo Starting backend...
start "EHT Backend" cmd /c "cd /d %PROJECT_DIR% && %CONDA_PY% -m uvicorn backend.main:app --host 127.0.0.1 --port 9876 --reload"

echo Starting frontend dev server...
start "EHT Frontend" cmd /c "cd /d %PROJECT_DIR%\frontend && npm run dev"

echo.
echo Both servers started. Open http://localhost:5173 in your browser.
echo.
pause
