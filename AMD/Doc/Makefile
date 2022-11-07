#------------------------------------------------------------------------------
# AMD/Doc/Makefile: Create the User Guide and Quick Start Guide
#------------------------------------------------------------------------------

# Copyright (c) 1996-2022, Timothy A. Davis, Patrick Amestoy, Iain Duff.
# All Rights Reserved.
# SPDX-License-Identifier: BSD-3-clause

default:  AMD_UserGuide.pdf
	- $(RM) *.aux *.bbl *.blg *.log *.toc amd_h.tex

AMD_UserGuide.pdf: AMD_UserGuide.tex AMD_UserGuide.bib ../Include/amd.h \
    amd_version.tex
	printf '\\begin{verbatim}\n' > amd_h.tex
	expand -8 ../Include/amd.h >> amd_h.tex
	printf '\\end{verbatim}\n' >> amd_h.tex
	pdflatex AMD_UserGuide
	bibtex AMD_UserGuide
	pdflatex AMD_UserGuide
	pdflatex AMD_UserGuide

# Remove all but the files in the original distribution
clean:
	- $(RM) -r *.out *.aux *.log *.bbl *.blg *.toc amd_h.tex

purge: distclean

distclean: clean

