# This makefile assumes latexmk location is in the PATH
.PHONY: main.pdf all clean clean-all watch preview

all: main.pdf

main.pdf: main.tex
	latexmk -pdf -outdir=build -auxdir=build -pdflatex="pdflatex -interaction=nonstopmode" main.tex
	@cp build/main.pdf . 2>/dev/null || copy build\\main.pdf .

clean:
	latexmk -outdir=build -auxdir=build -C

clean-all:
	latexmk -outdir=build -auxdir=build -C
	rm -rf build/
	rm -f main.pdf

# Preview mode
watch: preview

preview:
	latexmk -pdf -outdir=build -auxdir=build -pdflatex="pdflatex -interaction=nonstopmode" -pvc main.tex
