@echo off

cd /d "C:\Users\flore\Downloads\Django_Mal_Pred_Caps\my_project"
start cmd /k npm run dev

cd /d "C:\Users\flore\Downloads\Django_Mal_Pred_Caps\env\Scripts"
start cmd /k "activate.bat & cd ..\.. & cd my_project & py manage.py runserver" 

pause