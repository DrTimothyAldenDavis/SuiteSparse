# package minted is need for ParU user guide therefore python3-pygments must be
# installed

# ParU, Copyright (c) 2022-2024, Mohsen Aznaveh and Timothy A. Davis,
# All Rights Reserved.
# SPDX-License-Identifier: GPL-3.0-or-later

default: paru_user_guide.pdf

paru_user_guide.pdf: paru_user_guide.tex paru_user_guide.bib Makefile paru_version.tex
	pdflatex --shell-escape paru_user_guide.tex
	bibtex paru_user_guide
	pdflatex --shell-escape paru_user_guide.tex
	pdflatex --shell-escape paru_user_guide.tex
	- $(RM) -r $(PURGE) *.out *.aux *.log *.bbl *.toc *.blg _minted-paru_*

distclean: purge

clean:
	- $(RM) -r $(CLEAN)

purge: clean
	- $(RM) -r $(PURGE) *.out *.aux *.log *.bbl *.toc *.blg _minted-paru_*

