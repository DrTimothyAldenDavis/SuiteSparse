#------------------------------------------------------------------------------
# CAMD/Doc/Makefile: Create the User Guide and Quick Start Guide
#------------------------------------------------------------------------------

# Copyright (c) 2007-2022, Timothy A. Davis, Yanqing Chen, Patrick Amestoy,
# Iain Duff.  All Rights Reserved.
# SPDX-License-Identifier: BSD-3-clause

default:  CAMD_UserGuide.pdf
	- $(RM) *.aux *.bbl *.blg *.log *.toc camp_temp camd_h.tex

CAMD_UserGuide.pdf: CAMD_UserGuide.tex CAMD_UserGuide.bib ../Include/camd.h \
    camd_version.tex
	printf '\\begin{verbatim}\n' > camd_h.tex
	expand -8 ../Include/camd.h >> camd_h.tex
	printf '\\end{verbatim}\n' >> camd_h.tex
	pdflatex CAMD_UserGuide
	bibtex CAMD_UserGuide
	pdflatex CAMD_UserGuide
	pdflatex CAMD_UserGuide

# Remove all but the files in the original distribution
clean:
	- $(RM) -r *.out *.aux *.log *.bbl *.blg *.toc camd_temp camd_h.tex

purge: distclean

distclean: clean

