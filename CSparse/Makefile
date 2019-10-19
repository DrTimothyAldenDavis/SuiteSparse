# CSparse Makefile

C:
	( cd Lib ; $(MAKE) )
	( cd Demo ; $(MAKE) )

all: C tcov

library:
	( cd Lib ; $(MAKE) )

tcov:
	( cd Tcov ; $(MAKE) )

mex:
	( cd MATLAB ; $(MAKE) )

clean:
	( cd Lib ; $(MAKE) clean )
	( cd Demo ; $(MAKE) clean )
	( cd Tcov ; $(MAKE) clean )
	( cd MATLAB ; $(MAKE) clean )

purge:
	( cd Lib ; $(MAKE) purge )
	( cd Demo ; $(MAKE) purge )
	( cd Tcov ; $(MAKE) purge )
	( cd MATLAB ; $(MAKE) purge )

distclean: purge
