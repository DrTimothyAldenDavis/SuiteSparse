#-------------------------------------------------------------------------------
# Makefile for the LDL mexFunction
#-------------------------------------------------------------------------------

default: all

include ../UFconfig/UFconfig.mk

C = $(CC) $(CFLAGS)

all: intro libldl.a ldlsimple ldlmain ldlamd

intro:
	@echo " "
	@echo "    Now compiling the LDL library (libldl.a), and three stand-"
	@echo "    alone C programs (ldlmain, ldlsimple, and ldlamd).  ldlamd"
	@echo "    is ldlmain compiled to use with AMD, which will be skipped"
	@echo "    if you do not have AMD installed in the ../AMD directory."
	@echo "    'make mex' compiles the LDL mexFunction for use in MATLAB,"
	@echo "    and two MATLAB test programs."
	@echo " "

#-------------------------------------------------------------------------------
# the ldl library:
#-------------------------------------------------------------------------------

libldl.a: ldl.c ldl.h
	$(C) -c ldl.c -o ldl.o
	$(AR) libldl.a ldl.o
	- $(RANLIB) libldl.a

#-------------------------------------------------------------------------------
# stand-alone C programs:
#-------------------------------------------------------------------------------

ldlmain:  ldlmain.c libldl.a
	$(C) ldlmain.c libldl.a -o ldlmain -lm
	- ./ldlmain > my_ldlmain.out
	- diff ldlmain.out my_ldlmain.out

ldlsimple:  ldlsimple.c libldl.a
	$(C) ldlsimple.c libldl.a -o ldlsimple -lm
	- ./ldlsimple > my_ldlsimple.out
	- diff ldlsimple.out my_ldlsimple.out

ldlamd:  ldlmain.c libldl.a
	- (cd ../AMD ; $(MAKE))
	- $(C) -I../AMD/Include -I../UFconfig -L../AMD/Lib -DUSE_AMD ldlmain.c -lamd libldl.a -o ldlamd -lm
	- ./ldlamd > my_ldlamd.out
	- diff ldlamd.out my_ldlamd.out

run:
	- ./ldlamd > my_ldlamd.out
	- diff ldlamd.out my_ldlamd.out
	- ./ldlsimple > my_ldlsimple.out
	- diff ldlsimple.out my_ldlsimple.out
	- ./ldlmain > my_ldlmain.out
	- diff ldlmain.out my_ldlmain.out

#-------------------------------------------------------------------------------
# LDL mexFunctions for use in MATLAB:
#-------------------------------------------------------------------------------

mex: ldl_mexfunction

ldl_mexfunction: ldlmex.c ldl.c ldl.h ldlsymbolmex.c
	- $(MEX) ldl.c ldlmex.c
	- $(MEX) -output ldlsymbol ldlsymbolmex.c ldl.c

#-------------------------------------------------------------------------------
# clean-up:
#-------------------------------------------------------------------------------

distclean: purge

purge: clean
	- $(RM) ldlsparse.mex* ldl.dll libldl.a ldlmain ldlamd ldlsimple
	- $(RM) my_ldlmain.out my_ldlamd.out my_ldlsimple.out
	- $(RM) ldlmain.mex* ldlamd.mex* ldlmain.dll ldlamd.dll
	- $(RM) ldlsymbol.mex* ldlsymbol.dll
	- $(RM) *.dvi *.aux *.log *.bak *.bbl *.blg

clean:
	- $(RM) $(CLEAN)

#-------------------------------------------------------------------------------
# user guide:
#-------------------------------------------------------------------------------

doc: ldl_userguide.tex ldl.bib
	latex ldl_userguide
	- bibtex ldl_userguide
	latex ldl_userguide
	latex ldl_userguide
	dvips ldl_userguide -o ldl_userguide.ps
	pdflatex ldl_userguide
	pdflatex ldl_userguide

