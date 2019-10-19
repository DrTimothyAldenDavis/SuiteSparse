#------------------------------------------------------------------------------
# AMD Makefile for compiling on Unix systems (for GNU or original make)
#------------------------------------------------------------------------------

default: dist

include ../../UFconfig/UFconfig.mk

#------------------------------------------------------------------------------
# Remove all but the files in the original distribution
#------------------------------------------------------------------------------

clean:
	- $(RM) $(CLEAN)

purge: distclean

distclean: clean
	- $(RM) *.aux *.bbl *.blg *.log *.toc

#------------------------------------------------------------------------------
# Create the User Guide and Quick Start Guide
#------------------------------------------------------------------------------

AMD_UserGuide.pdf: AMD_UserGuide.tex AMD_UserGuide.bib
	pdflatex AMD_UserGuide
	bibtex AMD_UserGuide
	pdflatex AMD_UserGuide
	pdflatex AMD_UserGuide

dist:  AMD_UserGuide.pdf
	- $(RM) *.aux *.bbl *.blg *.log *.toc

