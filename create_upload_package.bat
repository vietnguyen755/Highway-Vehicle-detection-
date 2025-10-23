@echo off
echo Creating clean package for GitHub web upload...

REM Create a temporary directory for the clean package
mkdir github_upload_temp

REM Copy essential files
copy main.py github_upload_temp\
copy README.md github_upload_temp\
copy PROJECT_REPORT.md github_upload_temp\
copy requirements.txt github_upload_temp\
copy test_improved_model.bat github_upload_temp\
copy .gitignore github_upload_temp\

REM Copy dataset structure (without images)
mkdir github_upload_temp\dataset
mkdir github_upload_temp\dataset\train
mkdir github_upload_temp\dataset\train\labels
copy dataset\data.yaml github_upload_temp\dataset\
xcopy dataset\train\labels\* github_upload_temp\dataset\train\labels\ /E /I

REM Copy new_finetunedata structure (without images)
mkdir github_upload_temp\new_finetunedata
mkdir github_upload_temp\new_finetunedata\train
mkdir github_upload_temp\new_finetunedata\train\labels
copy new_finetunedata\data.yaml github_upload_temp\new_finetunedata\
copy new_finetunedata\README.* github_upload_temp\new_finetunedata\
xcopy new_finetunedata\train\labels\* github_upload_temp\new_finetunedata\train\labels\ /E /I

echo.
echo Package created in 'github_upload_temp' folder
echo You can now ZIP this folder and upload to GitHub
echo.
pause
