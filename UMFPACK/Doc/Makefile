#-------------------------------------------------------------------------------
# UMFPACK Makefile for compiling on Unix systems
#-------------------------------------------------------------------------------

# UMFPACK, Copyright (c) 2005-2023 by Timothy A. Davis.
# All Rights Reserved.  See ../Doc/License.txt for License.

dist:  UMFPACK_QuickStart.pdf UMFPACK_UserGuide.pdf
	- $(RM) *.aux *.bbl *.blg *.log *.toc *.out

UMFPACK_UserGuide.pdf:  UMFPACK_UserGuide.tex UserGuide.bib umfpack_version.tex
	pdflatex UMFPACK_UserGuide
	bibtex UMFPACK_UserGuide
	pdflatex UMFPACK_UserGuide
	pdflatex UMFPACK_UserGuide

UMFPACK_QuickStart.pdf: UMFPACK_QuickStart.tex umfpack_version.tex
	pdflatex UMFPACK_QuickStart
	pdflatex UMFPACK_QuickStart

purge: clean

clean:
	- $(RM) *.aux *.bbl *.blg *.log *.toc *.out

