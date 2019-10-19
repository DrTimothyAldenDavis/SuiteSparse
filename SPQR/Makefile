#-------------------------------------------------------------------------------
# SuiteSparseQR Makefile
#-------------------------------------------------------------------------------

VERSION = 1.2.1

include ../UFconfig/UFconfig.mk

default: all

all:
	( cd Lib ; $(MAKE) )
	( cd Demo ; $(MAKE) )

library:
	( cd Lib ; $(MAKE) )

demos:
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

ccode: demos

# install SPQR
install:
	$(CP) Lib/libspqr.a $(INSTALL_LIB)/libspqr.$(VERSION).a
	( cd $(INSTALL_LIB) ; ln -sf libspqr.$(VERSION).a libspqr.a )
	$(CP) Include/SuiteSparseQR.hpp $(INSTALL_INCLUDE)
	$(CP) Include/SuiteSparseQR_C.h $(INSTALL_INCLUDE)
	$(CP) Include/SuiteSparseQR_definitions.h $(INSTALL_INCLUDE)
	$(CP) Include/spqr.hpp $(INSTALL_INCLUDE)

# uninstall SPQR
uninstall:
	$(RM) $(INSTALL_LIB)/libspqr*.a
	$(RM) $(INSTALL_INCLUDE)/SuiteSparseQR.hpp
	$(RM) $(INSTALL_INCLUDE)/SuiteSparseQR_C.h
	$(RM) $(INSTALL_INCLUDE)/SuiteSparseQR_definitions.h
	$(RM) $(INSTALL_INCLUDE)/spqr.hpp

