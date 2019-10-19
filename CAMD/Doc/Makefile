#------------------------------------------------------------------------------
# CAMD Makefile for compiling on Unix systems (for GNU or original make)
#------------------------------------------------------------------------------

default: dist

include ../../UFconfig/UFconfig.mk

#------------------------------------------------------------------------------
# Remove all but the files in the original distribution
#------------------------------------------------------------------------------

clean:
	- $(RM) -r $(CLEAN)
	- $(RM) camd_temp

purge: distclean

distclean: clean
	- $(RM) *.aux *.bbl *.blg *.log *.toc
	- $(RM) -r *.dSYM

#------------------------------------------------------------------------------
# Create the User Guide and Quick Start Guide
#------------------------------------------------------------------------------

CAMD_UserGuide.pdf: CAMD_UserGuide.tex CAMD_UserGuide.bib
	pdflatex CAMD_UserGuide
	bibtex CAMD_UserGuide
	pdflatex CAMD_UserGuide
	pdflatex CAMD_UserGuide

dist:  CAMD_UserGuide.pdf
	- $(RM) *.aux *.bbl *.blg *.log *.toc
	- $(RM) camd_temp

