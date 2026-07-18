@echo off
setlocal
cd /d "%~dp0"

where py >nul 2>nul
if %errorlevel% equ 0 (
    py -3 -m mesh_review.server --images "%~dp0images" --state-dir "%~dp0review_state" --target 200 --port 0
) else (
    python -m mesh_review.server --images "%~dp0images" --state-dir "%~dp0review_state" --target 200 --port 0
)

echo.
echo Decisions and exports are stored in %~dp0review_state
pause
