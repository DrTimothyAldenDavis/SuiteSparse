#-------------------------------------------------------------------------------
# GPUQREngine Makefile
#-------------------------------------------------------------------------------

VERSION = 1.0.0

# default is 'library' since there is no demo program
default: library

include ../SuiteSparse_config/SuiteSparse_config.mk

# Compile the C-callable libraries
all: library

# Compile the C-callable libraries only.
library:
	( cd Lib ; $(MAKE) )

# Remove all files not in the original distribution
purge:
	( cd Lib ; $(MAKE) purge )

# Remove all files not in the original distribution, except keep the 
# compiled libraries.
clean:
	( cd Lib ; $(MAKE) clean )

distclean: purge

ccode: all

# install GPUQREngine.  Note that the include files are not installed.
install:
	$(CP) Lib/libGPUQREngine.a $(INSTALL_LIB)/libGPUQREngine.$(VERSION).a
	( cd $(INSTALL_LIB) ; ln -sf libGPUQREngine.$(VERSION).a libGPUQREngine.a )
	chmod 644 $(INSTALL_LIB)/libGPUQREngine*.a

# uninstall GPUQREngine
uninstall:
	$(RM) $(INSTALL_LIB)/libGPUQREngine*.a

