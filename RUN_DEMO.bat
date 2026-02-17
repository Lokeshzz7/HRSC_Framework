@echo off
REM Quick Demo Script for HRSC Framework
REM This sets the encoding and runs the demo

echo ================================================================================
echo HRSC FRAMEWORK - DEMO LAUNCHER
echo ================================================================================
echo.

set PYTHONIOENCODING=utf-8

echo [1] Choose demo mode:
echo     1 - Quick Start (automated demo with sample queries)
echo     2 - Interactive Mode (type your own queries)
echo     3 - Comparison Mode (see cache performance)
echo.

set /p choice="Enter choice (1-3): "

if "%choice%"=="1" (
    echo.
    echo Running Quick Start Demo...
    echo.
    python quickstart_hrsc.py
) else if "%choice%"=="2" (
    echo.
    echo Starting Interactive Mode...
    echo.
    python hrsc_demo.py --interactive
) else if "%choice%"=="3" (
    echo.
    echo Running Comparison Demo...
    echo.
    python hrsc_demo.py --compare --query "What is HiRAG?"
) else (
    echo Invalid choice. Please run again and choose 1, 2, or 3.
)

echo.
echo Demo complete!
pause
