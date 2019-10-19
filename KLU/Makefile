#------------------------------------------------------------------------------
# KLU Makefile
#------------------------------------------------------------------------------

VERSION = 1.2.1

default: all

include ../SuiteSparse_config/SuiteSparse_config.mk

demos: all

all:
	( cd Lib ; $(MAKE) )
	( cd Demo ; $(MAKE) )

library:
	( cd Lib ; $(MAKE) )

clean:
	( cd Demo ; $(MAKE) clean )
	( cd Lib ; $(MAKE) clean )
	( cd Tcov ; $(MAKE) clean )
	( cd MATLAB ; $(RM) $(CLEAN) rename.h )

distclean:
	( cd Demo ; $(MAKE) distclean )
	( cd Lib ; $(MAKE) distclean )
	( cd Tcov ; $(MAKE) distclean )
	( cd User ; $(MAKE) distclean )
	( cd MATLAB ; $(RM) $(CLEAN) rename.h *.mex* )

purge: distclean

cov: library
	( cd Tcov ; $(MAKE) )

# create PDF documents for the original distribution
docs:
	( cd Doc    ; $(MAKE) )

# install KLU
install:
	$(CP) Lib/libklu.a $(INSTALL_LIB)/libklu.$(VERSION).a
	( cd $(INSTALL_LIB) ; ln -sf libklu.$(VERSION).a libklu.a )
	$(CP) Include/klu.h $(INSTALL_INCLUDE)
	chmod 644 $(INSTALL_LIB)/libklu*.a
	chmod 644 $(INSTALL_INCLUDE)/klu.h

# uninstall KLU
uninstall:
	$(RM) $(INSTALL_LIB)/libklu*.a
	$(RM) $(INSTALL_INCLUDE)/klu.h

