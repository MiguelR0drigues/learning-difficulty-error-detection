@echo off
echo Cleaning old build files...
if exist main.pdf del /q main.pdf
if exist build rmdir /s /q build

echo Building PDF (outputs in build/)...
latexmk -pdf -outdir=build -auxdir=build -pdflatex="pdflatex -interaction=nonstopmode" main.tex

if %ERRORLEVEL% EQU 0 (
    echo Copying PDF to root...
    copy build\main.pdf .
    echo Build successful!
) else (
    echo Build failed with exit code %ERRORLEVEL%
    exit /b %ERRORLEVEL%
)
