# SPQR/Doc/Makefile
# SPQR, Copyright (c) 2008-2022, Timothy A Davis. All Rights Reserved.
# SPDX-License-Identifier: GPL-2.0+

default: spqr_user_guide.pdf

spqr_user_guide.pdf: spqr_user_guide.tex spqr_user_guide.bib Makefile spqr_version.tex
	pdflatex spqr_user_guide.tex
	bibtex spqr_user_guide
	pdflatex spqr_user_guide.tex
	pdflatex spqr_user_guide.tex
	- $(RM) -r *.out *.aux *.bbl *.log *.blg

distclean: purge

clean:
	- $(RM) -r *.out *.aux *.bbl *.log *.blg

purge: clean
	- $(RM) -r $(PURGE)
