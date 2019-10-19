default: library

include ../UFconfig/UFconfig.mk

library:
	( cd Lib ; $(MAKE) )

clean:
	( cd Lib ; $(MAKE) clean )

distclean:
	( cd Lib ; $(MAKE) distclean )
	( cd MATLAB ; $(MAKE) distclean )

mex:
	( cd MATLAB ; $(MAKE) )

purge: distclean
