#-------------------------------------------------------------------------------
# SPEX/Doc/Makefile
#-------------------------------------------------------------------------------

# SPEX: (c) 2019-2022, Chris Lourenco (US Naval Academy), Jinhao Chen,
# Erick Moreno-Centeno, Timothy A. Davis, Texas A&M.  All Rights Reserved.
# SPDX-License-Identifier: GPL-2.0-or-later or LGPL-3.0-or-later

#-------------------------------------------------------------------------------

# creates the user guide

SPEX_UserGuide.pdf: SPEX_UserGuide.tex SPEX_UserGuide.bib Makefile SPEX_version.tex
	pdflatex SPEX_UserGuide.tex
	bibtex SPEX_UserGuide
	pdflatex SPEX_UserGuide.tex
	pdflatex SPEX_UserGuide.tex
	pdflatex SPEX_UserGuide.tex
	- $(RM) -r *.out *.aux *.log *.bbl *.blg *.toc *.cpt

clean:
	- $(RM) -r *.out *.aux *.log *.bbl *.blg *.toc *.cpt

purge: clean

distclean: clean

