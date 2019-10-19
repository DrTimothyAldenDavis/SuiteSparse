#-------------------------------------------------------------------------------
# UFconfig Makefile
#-------------------------------------------------------------------------------

VERSION = 3.7.0

default: ccode

include UFconfig.mk

ccode: libufconfig.a

all: libufconfig.a

libufconfig.a: UFconfig.c UFconfig.h
	$(CC) $(CF) -c UFconfig.c
	$(ARCHIVE) libufconfig.a UFconfig.o
	$(RANLIB) libufconfig.a
	- $(RM) UFconfig.o

distclean: purge

purge: clean
	- $(RM) *.o *.a

clean:
	- $(RM) -r $(CLEAN)

# install UFconfig
install:
	$(CP) libufconfig.a $(INSTALL_LIB)/libufconfig.$(VERSION).a
	( cd $(INSTALL_LIB) ; ln -sf libufconfig.$(VERSION).a libufconfig.a )
	$(CP) UFconfig.h $(INSTALL_INCLUDE)
	chmod 644 $(INSTALL_LIB)/libufconfig*.a
	chmod 644 $(INSTALL_INCLUDE)/UFconfig.h

# uninstall UFconfig
uninstall:
	$(RM) $(INSTALL_LIB)/libufconfig*.a
	$(RM) $(INSTALL_INCLUDE)/UFconfig.h

