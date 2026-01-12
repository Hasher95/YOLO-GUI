@echo off
echo Installing dependencies...
python -m pip install --upgrade pip
python -m pip install ultralytics customtkinter pyinstaller matplotlib pandas pyyaml

echo.
echo Cleaning previous build...
if exist "build" rmdir /s /q "build"
if exist "dist" rmdir /s /q "dist"

echo.
echo Building executable...
echo Using --onedir for faster startup times.
echo This may take a few minutes.
python -m PyInstaller --noconfirm --onedir --windowed --name "YoloTrainer" --collect-all ultralytics --collect-all customtkinter train_yolo_gui.py

echo.
if exist "dist\YoloTrainer\YoloTrainer.exe" (
    echo Build successful! File is in 'dist\YoloTrainer\YoloTrainer.exe'.
    start dist\YoloTrainer
) else (
    echo Build failed. Please check the error messages above.
    pause
)
pause
