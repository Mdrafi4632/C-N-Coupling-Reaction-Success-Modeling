PDF      := main.pdf
TEX      := main.tex
FIG_PY   := make_figures.py

.PHONY: all figures clean

all: figures $(PDF)

figures:
	cd .. && uv run python report/$(FIG_PY)

$(PDF): $(TEX)
	pdflatex -interaction=nonstopmode $(TEX) > /dev/null
	pdflatex -interaction=nonstopmode $(TEX) > /dev/null

clean:
	rm -f *.aux *.log *.out *.toc $(PDF)
