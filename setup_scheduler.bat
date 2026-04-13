@echo off
echo 모멘텀 로테이션 봇 자동 실행 설정 중...

:: Python 경로 자동 감지
for /f "delims=" %%i in ('where python') do set PYTHON_PATH=%%i

:: 현재 디렉토리
set BOT_DIR=%~dp0
set BOT_SCRIPT=%BOT_DIR%momentum_bot.py
set LOG_FILE=%BOT_DIR%data\momentum_bot.log

:: 작업 스케줄러 등록 (매일 오전 9시)
schtasks /create /tn "MomentumBot" /tr "\"%PYTHON_PATH%\" -X utf8 \"%BOT_SCRIPT%\" >> \"%LOG_FILE%\" 2>&1" /sc daily /st 09:00 /f

if %errorlevel% == 0 (
    echo.
    echo 설정 완료!
    echo 매일 오전 9시에 자동 실행됩니다.
    echo 로그 파일: %LOG_FILE%
) else (
    echo.
    echo 설정 실패. 관리자 권한으로 다시 실행해주세요.
)

pause
