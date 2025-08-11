@echo off
echo Starting Rat Control GUI...
echo.

REM Change to the parent directory (root)
cd /d "%~dp0.."
echo Current directory: %CD%

REM Set your conda environment name here
set CONDA_ENV_NAME=AxonSurveyEnv

REM Check if conda is installed
call conda --version
if errorlevel 1 (
    echo ERROR: Conda is not installed or not in PATH
    echo Please install Anaconda/Miniconda or add it to system path and try again 
    pause
    exit /b 1
) else (
    echo Conda check passed
)

REM Activate conda environment
echo Activating conda environment: %CONDA_ENV_NAME%
call conda activate %CONDA_ENV_NAME%
if errorlevel 1 (
    echo ERROR: Failed to activate conda environment '%CONDA_ENV_NAME%'
    echo Please make sure the environment exists
    pause
    exit /b 1
)

REM Check if Flask is installed in the environment
python -c "import flask" >nul 2>&1
if errorlevel 1 (
    echo Installing Flask in conda environment...
    conda install flask -y
    if errorlevel 1 (
        echo ERROR: Failed to install Flask
        pause
        exit /b 1
    )
)

REM Start the Flask application
echo Starting Flask server...
echo Please wait 10-15 seconds, the browser will open when server has started
echo.
echo ====================================================
echo  IMPORTANT: To stop the server when you're done:
echo  1. Click in this black window
echo  2. Press Ctrl+C (hold Ctrl, then press C)
echo  3. Or simply close this window
echo ====================================================
echo.

start /b python gui/app.py

REM Start Flask and open browser
echo Starting server and opening browser... Forgive the delay, the backend is python
echo If there's an error when browser opens, try refreshing a couple times
timeout /t 10 /nobreak
start "" "firefox.exe" --new-window "http://localhost:5000"

REM This runs after the server stops (Ctrl+C or app closes)
echo.
echo Server stopped.
echo You can now close this window.
pause