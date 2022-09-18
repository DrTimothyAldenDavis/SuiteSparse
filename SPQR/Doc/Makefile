# SPQR/Doc/Makefile
# SPQR, Copyright (c) 2008-2022, Timothy A Davis. All Rights Reserved.
# SPDX-License-Identifier: GPL-2.0+

default: spqr_user_guide.pdf

include ../../SuiteSparse_config/SuiteSparse_config.mk

spqr_user_guide.pdf: spqr_user_guide.tex spqr_user_guide.bib Makefile
	pdflatex spqr_user_guide.tex
	bibtex spqr_user_guide
	pdflatex spqr_user_guide.tex
	pdflatex spqr_user_guide.tex
	- $(RM) -r $(CLEAN) *.out

distclean: purge

clean:
	- $(RM) -r $(CLEAN) *.out

purge: clean
	- $(RM) -r $(PURGE)
