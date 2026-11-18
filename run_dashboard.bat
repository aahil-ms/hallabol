@echo off
cd /d "%~dp0"
echo ============================================
echo   SupeRR Selector -- Setup ^& Launch
echo ============================================

echo.
echo Checking environment...
python --version
if errorlevel 1 (
    echo [ERROR] Python is not found in your system PATH!
    echo Please install Python 3.10+ and check "Add Python to PATH".
    pause
    exit /b 1
)

echo.
echo [1/3] Checking processed data...
if not exist "processed\batting_scored.csv" (
    echo Processing raw CSV files...
    python data_processor.py
    if errorlevel 1 (
        echo [ERROR] data_processor.py failed. Check that data\ folder has all CSVs.
        pause
        exit /b 1
    )
    echo Training ML models...
    python train_model.py
    if errorlevel 1 (
        echo [ERROR] train_model.py failed.
        pause
        exit /b 1
    )
) else (
    echo Processed data and trained models already exist! Skipping reprocessing.
)

echo.
echo [2/3] Launching Streamlit dashboard...
echo Open your browser at: http://localhost:8501
python -m streamlit run app.py --server.headless false

if errorlevel 1 (
    echo.
    echo Streamlit exited with an error. Trying with direct streamlit command...
    streamlit run app.py
)
pause
