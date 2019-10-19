# CXSparse Makefile

VERSION = 2.2.4

default: C

include ../UFconfig/UFconfig.mk

C:
	( cd Lib ; $(MAKE) )
	( cd Demo ; $(MAKE) )

all: C cov

library:
	( cd Lib ; $(MAKE) )

cov:
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

# install CSparse
install:
	$(CP) Lib/libcxsparse.a $(INSTALL_LIB)/libcxsparse.$(VERSION).a
	( cd $(INSTALL_LIB) ; ln -s libcxsparse.$(VERSION).a libcxsparse.a )
	$(CP) Include/cs.h $(INSTALL_INCLUDE)

# uninstall CSparse
uninstall:
	$(RM) $(INSTALL_LIB)/libcxsparse.a
	$(RM) $(INSTALL_INCLUDE)/cs.h

