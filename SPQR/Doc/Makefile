spqr_user_guide.pdf: spqr_user_guide.tex spqr_user_guide.bib Makefile
	pdflatex spqr_user_guide.tex
	bibtex spqr_user_guide
	pdflatex spqr_user_guide.tex
	pdflatex spqr_user_guide.tex
	- $(RM) *.aux *.bbl *.blg *.log *.out

clean: purge

purge:
	- $(RM) *.aux *.bbl *.blg *.log *.out
