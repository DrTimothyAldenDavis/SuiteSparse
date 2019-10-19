default: library

include ../UFconfig/UFconfig.mk

library:
	( cd Lib ; $(MAKE) )
	( cd Demo ; $(MAKE) )

clean:
	( cd Demo ; $(MAKE) clean )
	( cd Lib ; $(MAKE) clean )
	( cd Tcov ; $(MAKE) clean )

distclean:
	( cd Demo ; $(MAKE) distclean )
	( cd Lib ; $(MAKE) distclean )
	( cd Tcov ; $(MAKE) distclean )
	( cd User ; $(MAKE) distclean )
	( cd MATLAB ; $(MAKE) distclean )

mex:
	( cd MATLAB ; $(MAKE) )

purge: distclean
