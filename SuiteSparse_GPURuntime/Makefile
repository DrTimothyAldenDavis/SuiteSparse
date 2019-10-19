#-------------------------------------------------------------------------------
# SuiteSparse_GPURuntime Makefile
#-------------------------------------------------------------------------------

VERSION = 1.0.0

default: all

include ../SuiteSparse_config/SuiteSparse_config.mk

# Compile the C-callable libraries only (there is no Demo for this package)
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

# install SuiteSparse_GPURuntime (just the library, not the include files)
install:
	$(CP) Lib/libSuiteSparse_GPURuntime.a $(INSTALL_LIB)/libSuiteSparse_GPURuntime.$(VERSION).a
	( cd $(INSTALL_LIB) ; ln -sf libSuiteSparse_GPURuntime.$(VERSION).a libSuiteSparse_GPURuntime.a )
	chmod 644 $(INSTALL_LIB)/libSuiteSparse_GPURuntime*.a

# uninstall SuiteSparse_GPURuntime
uninstall:
	$(RM) $(INSTALL_LIB)/libSuiteSparse_GPURuntime*.a

