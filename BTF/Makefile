#------------------------------------------------------------------------------
# BTF Makefile
#------------------------------------------------------------------------------

VERSION = 1.2.0

default: library

include ../SuiteSparse_config/SuiteSparse_config.mk

library:
	( cd Lib ; $(MAKE) )

clean:
	( cd Lib ; $(MAKE) clean )
	( cd MATLAB ; $(RM) $(CLEAN) )

distclean:
	( cd Lib ; $(MAKE) distclean )
	( cd MATLAB ; $(RM) $(CLEAN) ; $(RM) *.mex* )

purge: distclean

# install BTF
install:
	$(CP) Lib/libbtf.a $(INSTALL_LIB)/libbtf.$(VERSION).a
	( cd $(INSTALL_LIB) ; ln -sf libbtf.$(VERSION).a libbtf.a )
	$(CP) Include/btf.h $(INSTALL_INCLUDE)
	chmod 644 $(INSTALL_LIB)/libbtf*.a
	chmod 644 $(INSTALL_INCLUDE)/btf.h

# uninstall BTF
uninstall:
	$(RM) $(INSTALL_LIB)/libbtf*.a
	$(RM) $(INSTALL_INCLUDE)/btf.h

