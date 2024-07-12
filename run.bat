@echo off

cd /d "C:\Users\flore\Downloads\Django_Mal_Pred_Caps\my_project"
start cmd /k npm run dev
start cmd /k py manage.py runserver

pause