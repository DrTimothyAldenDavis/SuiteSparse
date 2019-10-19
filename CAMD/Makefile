#------------------------------------------------------------------------------
# CAMD Makefile
#------------------------------------------------------------------------------

VERSION = 2.3.1

default: demos

include ../SuiteSparse_config/SuiteSparse_config.mk

demos: all

# Compile all C code.  Do not compile the FORTRAN versions.
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
	( cd Doc    ; $(MAKE) clean )

# clean, and then remove compiled programs and library archives
purge:
	( cd Lib    ; $(MAKE) purge )
	( cd Demo   ; $(MAKE) purge )
	( cd MATLAB ; $(RM) $(CLEAN) ; $(RM) *.mex* )
	( cd Doc    ; $(MAKE) purge )

distclean: purge

# create PDF documents for the original distribution
docs:
	( cd Doc    ; $(MAKE) )

# get ready for distribution
dist: purge
	( cd Demo   ; $(MAKE) dist )
	( cd Doc    ; $(MAKE) )

ccode: library

lib: library

# install CAMD
install:
	$(CP) Lib/libcamd.a $(INSTALL_LIB)/libcamd.$(VERSION).a
	( cd $(INSTALL_LIB) ; ln -sf libcamd.$(VERSION).a libcamd.a )
	$(CP) Include/camd.h $(INSTALL_INCLUDE)
	chmod 644 $(INSTALL_LIB)/libcamd*.a
	chmod 644 $(INSTALL_INCLUDE)/camd.h

# uninstall CAMD
uninstall:
	$(RM) $(INSTALL_LIB)/libcamd*.a
	$(RM) $(INSTALL_INCLUDE)/camd.h

