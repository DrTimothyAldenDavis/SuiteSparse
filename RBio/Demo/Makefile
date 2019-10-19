#===============================================================================
# RBio/Demo/Makefile
#===============================================================================

default: all

include ../../UFconfig/UFconfig.mk

LIB = ../Lib/librbio.a ../../UFconfig/libufconfig.a

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
	( cd ../../UFconfig ; $(MAKE) )
	( cd ../Lib ; $(MAKE) )
#-------------------------------------------------------------------------------

I = -I../Include -I../../UFconfig

RBdemo: library RBdemo.c
	$(C) -o RBdemo $(I) RBdemo.c $(LIB)

