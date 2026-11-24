@echo off
chcp 65001 > nul
title OptoBind Studio - Interface Web
cd /d "%~dp0"
python web_server.py --port 8080 --open-browser
pause
