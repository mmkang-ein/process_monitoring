@echo off
chcp 65001 > nul
title 공정제어 상시모니터링 시스템 v1.0

:: ============================================================
::  공정제어 상시모니터링 시스템 v1.0
::  철강/금속 압연·코일 생산라인 이상탐지 대시보드
:: ============================================================

color 0B
cls

echo.
echo  ╔══════════════════════════════════════════════════════╗
echo  ║      공정제어 상시모니터링 시스템  v1.0             ║
echo  ║      철강/금속 압연·코일 생산라인                   ║
echo  ╚══════════════════════════════════════════════════════╝
echo.
echo   [분석 모듈]  Z-Score / IQR / Isolation Forest
echo               SPC 관리도 / Cp·Cpk 공정능력지수
echo.
echo  ──────────────────────────────────────────────────────
echo.

:: ── 작업 디렉토리를 bat 파일 위치로 고정 ──────────────────────
set "APPDIR=%~dp0"
cd /d "%APPDIR%"

:: ── Python 설치 확인 ───────────────────────────────────────────
echo   [1/3] Python 환경 확인 중...
python --version > nul 2>&1
if errorlevel 1 (
    echo.
    echo   [오류] Python이 설치되어 있지 않습니다.
    echo          https://www.python.org 에서 Python 3.10 이상을 설치하세요.
    echo.
    pause
    exit /b 1
)
for /f "tokens=*" %%v in ('python --version 2^>^&1') do set PYVER=%%v
echo          %PYVER% 확인 완료

:: ── 필수 패키지 설치 ──────────────────────────────────────────
echo.
echo   [2/3] 필수 패키지 확인 및 설치 중...
echo          (최초 실행 시 1~2분 소요될 수 있습니다)
echo.
python -m pip install -r requirements.txt --quiet --disable-pip-version-check
if errorlevel 1 (
    echo.
    echo   [오류] 패키지 설치에 실패했습니다.
    echo          인터넷 연결을 확인하고 다시 시도하세요.
    echo.
    pause
    exit /b 1
)
echo          패키지 설치 완료

:: ── 기존 프로세스 포트 충돌 방지 (8502 사용 중이면 해제) ───────
echo.
echo   [3/3] 대시보드 시작 중... (포트 8502)
echo.

for /f "tokens=5" %%p in ('netstat -ano 2^>nul ^| findstr ":8502 "') do (
    taskkill /PID %%p /F > nul 2>&1
)
timeout /t 1 /nobreak > nul

:: ── Streamlit 실행 ─────────────────────────────────────────────
echo  ┌──────────────────────────────────────────────────────┐
echo  │  브라우저가 자동으로 열립니다                        │
echo  │                                                      │
echo  │  로컬 접속  : http://localhost:8502                  │
echo  │  클라우드   : https://processmonitoring-             │
echo  │               h3pqqit76nx2yhexzuuakx.streamlit.app   │
echo  │                                                      │
echo  │  ※ 종료하려면 이 창을 닫으세요                      │
echo  └──────────────────────────────────────────────────────┘
echo.

python -m streamlit run "%APPDIR%app.py" ^
    --server.port 8502 ^
    --server.headless false ^
    --browser.gatherUsageStats false ^
    --theme.primaryColor "#00d4ff" ^
    --theme.backgroundColor "#070b14" ^
    --theme.secondaryBackgroundColor "#0d1422" ^
    --theme.textColor "#e8f0ff"

echo.
echo   대시보드가 종료되었습니다.
pause
