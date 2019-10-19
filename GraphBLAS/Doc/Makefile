GraphBLAS_UserGuide.pdf: GraphBLAS_UserGuide.tex GraphBLAS_UserGuide.bib
	pdflatex GraphBLAS_UserGuide.tex
	bibtex GraphBLAS_UserGuide
	pdflatex GraphBLAS_UserGuide.tex
	pdflatex GraphBLAS_UserGuide.tex

clean:
	- $(RM) -r *.out *.aux *.log *.bbl *.blg *.toc

purge: clean

distclean: clean

