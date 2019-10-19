#-------------------------------------------------------------------------------
# SuiteSparseQR Makefile
#-------------------------------------------------------------------------------

include ../UFconfig/UFconfig.mk

default: all

all:
	( cd Lib ; $(MAKE) )
	( cd Demo ; $(MAKE) )

library:
	( cd Lib ; $(MAKE) )

demo:
	( cd Demo ; $(MAKE) )

cov:
	( cd Tcov ; $(MAKE) )

mex:
	( cd MATLAB ; $(MAKE) )

purge:
	( cd MATLAB ; $(MAKE) purge )
	( cd Lib ; $(MAKE) purge )
	( cd Demo ; $(MAKE) purge )
	( cd Tcov ; $(MAKE) purge )

clean:
	( cd MATLAB ; $(MAKE) clean )
	( cd Lib ; $(MAKE) clean )
	( cd Demo ; $(MAKE) clean )

distclean: purge

ccode: demo
