#-------------------------------------------------------------------------------
# UMFPACK Makefile for compiling on Unix systems
#-------------------------------------------------------------------------------

# UMFPACK Version 4.4, Copyright (c) 2005 by Timothy A. Davis.
# All Rights Reserved.  See ../Doc/License.txt for License.

default: dist

include ../../SuiteSparse_config/SuiteSparse_config.mk

#-------------------------------------------------------------------------------
# Remove all but the files in the original distribution
#-------------------------------------------------------------------------------

purge: clean
	- $(RM) -r $(PURGE)

clean:
	- $(RM) -r $(CLEAN)

#-------------------------------------------------------------------------------
# Create the User Guide and Quick Start Guide
#-------------------------------------------------------------------------------

UMFPACK_UserGuide.pdf:  UMFPACK_UserGuide.tex UserGuide.bib
	pdflatex UMFPACK_UserGuide
	bibtex UMFPACK_UserGuide
	pdflatex UMFPACK_UserGuide
	pdflatex UMFPACK_UserGuide

UMFPACK_QuickStart.pdf: UMFPACK_QuickStart.tex
	pdflatex UMFPACK_QuickStart
	pdflatex UMFPACK_QuickStart

dist:  UMFPACK_QuickStart.pdf UMFPACK_UserGuide.pdf
	- $(RM) *.aux *.bbl *.blg *.log *.toc

