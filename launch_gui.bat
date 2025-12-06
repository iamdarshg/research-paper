@echo off
REM Streamlit GUI Launcher
REM Run this from the research-paper directory

cd /d %~dp0
echo.
echo ========================================
echo  Paper Airplane AI Optimizer - GUI
echo ========================================
echo.
echo Starting Streamlit app...
echo Opening browser at http://localhost:8502
echo.
python -m streamlit run src/gui/app.py
