#------------------------------------------------------------------------------
# CCOLAMD Makefile
#------------------------------------------------------------------------------

VERSION = 2.8.0

default: all

include ../SuiteSparse_config/SuiteSparse_config.mk

demos: all

# Compile all C code
all:
	( cd Lib    ; $(MAKE) )
	( cd Demo   ; $(MAKE) )

# compile just the C-callable libraries (not Demos)
library:
	( cd Lib    ; $(MAKE) )

# remove object files, but keep the compiled programs and library archives
clean:
	( cd Lib    ; $(MAKE) clean )
	( cd Demo   ; $(MAKE) clean )
	( cd MATLAB ; $(RM) $(CLEAN) )

# clean, and then remove compiled programs and library archives
purge:
	( cd Lib    ; $(MAKE) purge )
	( cd Demo   ; $(MAKE) purge )
	( cd MATLAB ; $(RM) $(CLEAN) ; $(RM) *.mex* )

distclean: purge

# get ready for distribution
dist: purge
	( cd Demo   ; $(MAKE) dist )

ccode: library

lib: library

# install CCOLAMD
install:
	$(CP) Lib/libccolamd.a $(INSTALL_LIB)/libccolamd.$(VERSION).a
	( cd $(INSTALL_LIB) ; ln -sf libccolamd.$(VERSION).a libccolamd.a )
	$(CP) Include/ccolamd.h $(INSTALL_INCLUDE)
	chmod 644 $(INSTALL_LIB)/libccolamd*.a
	chmod 644 $(INSTALL_INCLUDE)/ccolamd.h

# uninstall CCOLAMD
uninstall:
	$(RM) $(INSTALL_LIB)/libccolamd*.a
	$(RM) $(INSTALL_INCLUDE)/ccolamd.h

