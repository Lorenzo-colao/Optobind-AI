@echo off
chcp 65001 > nul
title OptoBind-AI - Pipeline de Design de Biosenseurs Optiques

echo ======================================================================
echo   OptoBind-AI - Pipeline de Design De Novo de Biosenseurs Optiques
echo   Cible: Zn2+  ^|  Fluorophore: DFHBI-1T  ^|  pH: 7.4
echo ======================================================================
echo.

wsl -d Ubuntu -e bash -c "ln -sfn '/mnt/c/Users/colao/Desktop/Projet protéine' /tmp/optobind_project && cd /tmp/optobind_project && PYTHONIOENCODING=utf-8 /home/colao/miniconda3/envs/optobind-ai/bin/python orchestrator.py %*"

echo.
echo ======================================================================
echo   Execution terminee.
echo ======================================================================
pause
