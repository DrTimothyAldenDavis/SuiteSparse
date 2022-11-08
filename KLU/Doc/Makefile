#------------------------------------------------------------------------------
# KLU Makefile for creating the user guide
#------------------------------------------------------------------------------

dist:  KLU_UserGuide.pdf
	- $(RM) *.aux *.bbl *.blg *.log *.toc
	- $(RM) klu_simple_c.tex klu_h.tex btf_h.tex

KLU_UserGuide.pdf: KLU_UserGuide.tex KLU_UserGuide.bib \
    ../Include/klu.h ../../BTF/Include/btf.h Makefile klu_version.tex
	printf '\\begin{verbatim}\n' > klu_h.tex
	expand -8 ../Include/klu.h >> klu_h.tex
	printf '\\end{verbatim}\n' >> klu_h.tex
	printf '\\begin{verbatim}\n' > btf_h.tex
	expand -8 ../../BTF/Include/btf.h >> btf_h.tex
	printf '\\end{verbatim}\n' >> btf_h.tex
	printf '\\begin{verbatim}\n' > klu_simple_c.tex
	expand -8 ../Demo/klu_simple.c >> klu_simple_c.tex
	printf '\\end{verbatim}\n' >> klu_simple_c.tex
	pdflatex KLU_UserGuide
	bibtex KLU_UserGuide
	pdflatex KLU_UserGuide
	pdflatex KLU_UserGuide

# Remove all but the files in the original distribution
clean: 
	- $(RM) *.aux *.bbl *.blg *.log *.toc
	- $(RM) klu_simple_c.tex klu_h.tex btf_h.tex

purge: clean

distclean: clean

