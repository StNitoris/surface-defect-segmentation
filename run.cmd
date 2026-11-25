@echo off
setlocal
set "PROJECT_PY="
for %%P in ("%~dp0.venv\Scripts\python.exe" "%CONDA_PREFIX%\python.exe" "%USERPROFILE%\miniconda3\envs\OpenCV\python.exe") do if exist "%%~P" call :probe "%%~P"
for /f "delims=" %%P in ('where python.exe 2^>nul ^| findstr /i /v "WindowsApps"') do if not defined PROJECT_PY call :probe "%%P"
if not defined PROJECT_PY (
  echo Install requirements.txt in a Python environment and run again.
  exit /b 1
)
"%PROJECT_PY%" "%~dp0run_project.py" %*
exit /b %errorlevel%

:probe
if defined PROJECT_PY exit /b
setlocal
set "PATH=%~dp1;%~dp1Library\bin;%PATH%"
"%~1" -c "import cv2,numpy" >nul 2>nul
if errorlevel 1 (endlocal & exit /b)
endlocal & set "PROJECT_PY=%~1" & set "PATH=%~dp1;%~dp1Library\bin;%PATH%"
exit /b
