#===============================================================================
# RBio/Demo/Makefile
#===============================================================================

default: all

include ../../SuiteSparse_config/SuiteSparse_config.mk

CLIB = ../Lib/librbio.a ../../SuiteSparse_config/libsuitesparseconfig.a $(LIB)

C = $(CC) $(CF)

all: library RBdemo
	./RBdemo < ../RBio/private/west0479.rua

code: library

distclean:nnz,  purge

purge: clean
	- $(RM) RBdemo
	- $(RM) -r *.dSYM

clean:
	- $(RM) $(CLEAN) temp.rb

#-------------------------------------------------------------------------------
library:
	( cd ../../SuiteSparse_config ; $(MAKE) )
	( cd ../Lib ; $(MAKE) )
#-------------------------------------------------------------------------------

I = -I../Include -I../../SuiteSparse_config

RBdemo: library RBdemo.c
	$(C) -o RBdemo $(I) RBdemo.c $(CLIB)

