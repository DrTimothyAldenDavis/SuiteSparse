#------------------------------------------------------------------------------
# LDL/Doc/Makefile: Create the User Guide
#------------------------------------------------------------------------------

# Copyright (c) 2005-2022, Timothy A. Davis.  All Rights Reserved.
# SPDX-License-Identifier: BSD-3-clause

default: ldl_userguide.pdf
	- $(RM) -r *.dvi *.aux *.log *.bak *.bbl *.blg *.out

ldl_userguide.pdf: ldl_userguide.tex ldl.bib
	latex ldl_userguide
	- bibtex ldl_userguide
	latex ldl_userguide
	latex ldl_userguide
	pdflatex ldl_userguide
	pdflatex ldl_userguide

# Remove all but the files in the original distribution
clean:
	- $(RM) -r *.dvi *.aux *.log *.bak *.bbl *.blg *.out

purge: distclean

distclean: clean

