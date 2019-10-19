VERSION = 1.1.1

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

# install BTF
install:
	$(CP) Lib/libbtf.a $(INSTALL_LIB)/libbtf.$(VERSION).a
	( cd $(INSTALL_LIB) ; ln -s libbtf.$(VERSION).a libbtf.a )
	$(CP) Include/btf.h $(INSTALL_INCLUDE)

# uninstall BTF
uninstall:
	$(RM) $(INSTALL_LIB)/libbtf*.a
	$(RM) $(INSTALL_INCLUDE)/btf.h

