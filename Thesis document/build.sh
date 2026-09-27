#!/bin/bash
# Build script for LaTeX dissertation

echo "Cleaning old build files..."
rm -f main.pdf
rm -rf build/

echo "Building PDF (outputs in build/)..."
# Use latexmk with outdir to keep root clean
latexmk -pdf -outdir=build -auxdir=build -pdflatex="pdflatex -interaction=nonstopmode" main.tex

if [ $? -eq 0 ]; then
    echo "Copying PDF to root..."
    cp build/main.pdf .
    echo "Build successful!"
else
    echo "Build failed with exit code $?"
    exit 1
fi
