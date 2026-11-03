@echo off
chcp 65001 >nul 2>&1
cd /d "%~dp0"
title Vision Pilot - dashcam

set "PY="
where py >nul 2>&1 && set "PY=py -3"
if not defined PY (
    where python >nul 2>&1 && set "PY=python"
)
if not defined PY goto nopython

echo Checking dependencies...
%PY% -c "import cv2, numpy, onnxruntime" >nul 2>&1
if errorlevel 1 (
    echo.
    echo First run: installing three packages. This takes a few minutes.
    echo Do not close this window.
    echo.
    %PY% -m pip install --disable-pip-version-check -r requirements.txt
    if errorlevel 1 goto installfailed
    %PY% -c "import cv2, numpy, onnxruntime" >nul 2>&1
    if errorlevel 1 goto installfailed
)

echo Opening the window...
%PY% gui.py
if errorlevel 1 goto runfailed
exit /b 0

:nopython
echo.
echo  PYTHON WAS NOT FOUND ON THIS MACHINE.
echo.
echo  Download it from:  https://www.python.org/downloads/
echo  On the first installer screen, TICK "Add python.exe to PATH".
echo  Then close this window and run RUN.bat again.
echo.
pause
exit /b 1

:installfailed
echo.
echo  INSTALLING THE PACKAGES FAILED. Read the error above.
echo  Usually this means no network access, or a Python older than 3.9.
echo.
pause
exit /b 1

:runfailed
echo.
echo  The window exited with an error. Read the message above.
echo.
pause
exit /b 1
