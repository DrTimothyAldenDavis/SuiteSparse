#-------------------------------------------------------------------------------
# UMFPACK Makefile for compiling on Unix systems (for GNU or original make)
#-------------------------------------------------------------------------------

# UMFPACK Version 4.4, Copyright (c) 2005 by Timothy A. Davis.
# All Rights Reserved.  See ../Doc/License for License.

default: dist

include ../../UFconfig/UFconfig.mk

#-------------------------------------------------------------------------------
# Remove all but the files in the original distribution
#-------------------------------------------------------------------------------

# Note that UserGuide.tex is created from UserGuide.stex, the files in 
# the ../Include directory, and the ../Demo/umfpack_simple.c file.
purge: clean
	- $(RM) *.aux *.bbl *.blg *.log *.toc
	- $(RM) UserGuide.tex

clean:
	- $(RM) $(CLEAN)

#-------------------------------------------------------------------------------
# Create the User Guide and Quick Start Guide
#-------------------------------------------------------------------------------

UMFPACK = umfpack_col_to_triplet umfpack_defaults umfpack_free_numeric \
	umfpack_free_symbolic umfpack_get_numeric umfpack_get_lunz \
	umfpack_get_determinant \
	umfpack_get_symbolic umfpack_numeric umfpack_qsymbolic \
	umfpack_report_control umfpack_report_info umfpack_report_matrix \
	umfpack_report_numeric umfpack_report_perm umfpack_report_status \
	umfpack_report_symbolic umfpack_report_triplet \
	umfpack_report_vector umfpack_solve umfpack_symbolic \
	umfpack_transpose umfpack_triplet_to_col umfpack_scale

UMFPACKW = umfpack_wsolve

USER = $(UMFPACKW) $(UMFPACK)

SRC = $(addprefix ../Include/, $(addsuffix .h,$(USER))) ../Demo/umfpack_simple.c

UserGuide.pdf:  UserGuide.stex UserGuide.sed1 UserGuide.sed2 $(SRC) UserGuide.bib
	sed -f UserGuide.sed1 < UserGuide.stex | sed -f UserGuide.sed2 \
	    | expand -8 > UserGuide.tex
	pdflatex UserGuide
	bibtex UserGuide
	pdflatex UserGuide
	pdflatex UserGuide

QuickStart.pdf: QuickStart.tex
	pdflatex QuickStart
	pdflatex QuickStart

dist:  QuickStart.pdf UserGuide.pdf
	- $(RM) *.aux *.bbl *.blg *.log *.toc
	- $(RM) UserGuide.tex
