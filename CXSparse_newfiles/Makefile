#------------------------------------------------------------------------------
# CXSparse Makefile
#------------------------------------------------------------------------------

VERSION = 3.1.1

default: C

include ../SuiteSparse_config/SuiteSparse_config.mk

C:
	( cd Lib ; $(MAKE) )
	( cd Demo ; $(MAKE) )

all: C cov

library:
	( cd Lib ; $(MAKE) )

cov:
	( cd Tcov ; $(MAKE) )

clean:
	( cd Lib ; $(MAKE) clean )
	( cd Demo ; $(MAKE) clean )
	( cd Tcov ; $(MAKE) clean )
	( cd MATLAB/CSparse ; $(RM) *.o cs_cl_*.c )
	( cd MATLAB/Test    ; $(RM) *.o cs_cl_*.c )

purge:
	( cd Lib ; $(MAKE) purge )
	( cd Demo ; $(MAKE) purge )
	( cd Tcov ; $(MAKE) purge )
	( cd MATLAB/CSparse ; $(RM) *.o cs_cl_*.c *.mex* )
	( cd MATLAB/Test    ; $(RM) *.o cs_cl_*.c *.mex* )

distclean: purge

# install CSparse
install:
	$(CP) Lib/libcxsparse.a $(INSTALL_LIB)/libcxsparse.$(VERSION).a
	( cd $(INSTALL_LIB) ; ln -sf libcxsparse.$(VERSION).a libcxsparse.a )
	$(CP) Include/cs.h $(INSTALL_INCLUDE)
	chmod 644 $(INSTALL_LIB)/libcxsparse*.a
	chmod 644 $(INSTALL_INCLUDE)/cs.h

# uninstall CSparse
uninstall:
	$(RM) $(INSTALL_LIB)/libcxsparse*.a
	$(RM) $(INSTALL_INCLUDE)/cs.h

