#-------------------------------------------------------------------------------
# SuiteSparseQR Makefile
#-------------------------------------------------------------------------------

VERSION = 1.3.1

include ../SuiteSparse_config/SuiteSparse_config.mk

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

purge:
	( cd Lib ; $(MAKE) purge )
	( cd Demo ; $(MAKE) purge )
	( cd Tcov ; $(MAKE) purge )
	( cd Doc ; $(MAKE) purge )
	( cd MATLAB ; $(RM) $(CLEAN) rename.h *.mex* )

clean:
	( cd Lib ; $(MAKE) clean )
	( cd Demo ; $(MAKE) clean )
	( cd MATLAB ; $(RM) $(CLEAN) rename.h )

distclean: purge

ccode: demos

# create PDF documents for the original distribution
docs:
	( cd Doc    ; $(MAKE) )

# install SPQR
install:
	$(CP) Lib/libspqr.a $(INSTALL_LIB)/libspqr.$(VERSION).a
	( cd $(INSTALL_LIB) ; ln -sf libspqr.$(VERSION).a libspqr.a )
	$(CP) Include/SuiteSparseQR.hpp $(INSTALL_INCLUDE)
	$(CP) Include/SuiteSparseQR_C.h $(INSTALL_INCLUDE)
	$(CP) Include/SuiteSparseQR_definitions.h $(INSTALL_INCLUDE)
	$(CP) Include/spqr.hpp $(INSTALL_INCLUDE)
	chmod 644 $(INSTALL_LIB)/libspqr*.a
	chmod 644 $(INSTALL_INCLUDE)/SuiteSparseQR.hpp
	chmod 644 $(INSTALL_INCLUDE)/SuiteSparseQR_C.h
	chmod 644 $(INSTALL_INCLUDE)/SuiteSparseQR_definitions.h
	chmod 644 $(INSTALL_INCLUDE)/spqr.hpp

# uninstall SPQR
uninstall:
	$(RM) $(INSTALL_LIB)/libspqr*.a
	$(RM) $(INSTALL_INCLUDE)/SuiteSparseQR.hpp
	$(RM) $(INSTALL_INCLUDE)/SuiteSparseQR_C.h
	$(RM) $(INSTALL_INCLUDE)/SuiteSparseQR_definitions.h
	$(RM) $(INSTALL_INCLUDE)/spqr.hpp

