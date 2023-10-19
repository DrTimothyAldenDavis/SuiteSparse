#package minted is need for ParU user guide therfore python3-pygments must 
#  be installed

# ParU, Copyright (c) 2022, Mohsen Aznaveh and Timothy A. Davis,
# All Rights Reserved.
# SPDX-License-Identifier: GNU GPL 3.0

default: paru_user_guide.pdf


paru_user_guide.pdf: paru_user_guide.tex paru_user_guide.bib Makefile paru_version.tex
	pdflatex --shell-escape  paru_user_guide.tex
	bibtex paru_user_guide
	pdflatex --shell-escape paru_user_guide.tex
	pdflatex --shell-escape paru_user_guide.tex
	- $(RM) -r $(CLEAN) *.out

distclean: purge

clean:
	- $(RM) -r $(CLEAN) 

purge: clean
	- $(RM) -r $(PURGE) *.out paru_user_guide.pdf _minted-paru_*

