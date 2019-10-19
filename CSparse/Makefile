# CSparse Makefile

C:
	( cd Source ; $(MAKE) )
	( cd Demo ; $(MAKE) )

all: C tcov

library:
	( cd Source ; $(MAKE) )

tcov:
	( cd Tcov ; $(MAKE) )

mex:
	( cd MATLAB ; $(MAKE) )

clean:
	( cd Source ; $(MAKE) clean )
	( cd Demo ; $(MAKE) clean )
	( cd Tcov ; $(MAKE) clean )
	( cd MATLAB ; $(MAKE) clean )

purge:
	( cd Source ; $(MAKE) purge )
	( cd Demo ; $(MAKE) purge )
	( cd Tcov ; $(MAKE) purge )
	( cd MATLAB ; $(MAKE) purge )

distclean: purge
