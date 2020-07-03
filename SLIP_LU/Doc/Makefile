#-------------------------------------------------------------------------------
# SLIP_LU/Doc/Makefile
#-------------------------------------------------------------------------------

# SLIP_LU: (c) 2019-2020, Chris Lourenco, Jinhao Chen, Erick Moreno-Centeno,
# Timothy A. Davis, Texas A&M University.  All Rights Reserved.  See
# SLIP_LU/License for the license.

#-------------------------------------------------------------------------------

# creates the user guide

SLIP_LU_UserGuide.pdf: SLIP_LU_UserGuide.tex SLIP_LU_UserGuide.bib
	pdflatex SLIP_LU_UserGuide.tex
	bibtex SLIP_LU_UserGuide
	pdflatex SLIP_LU_UserGuide.tex
	pdflatex SLIP_LU_UserGuide.tex
	pdflatex SLIP_LU_UserGuide.tex
	- $(RM) -r *.out *.aux *.log *.bbl *.blg *.toc *.cpt

clean:
	- $(RM) -r *.out *.aux *.log *.bbl *.blg *.toc *.cpt

purge: clean

distclean: clean

