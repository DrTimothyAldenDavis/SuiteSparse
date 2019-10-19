#------------------------------------------------------------------------------
# CAMD Makefile
#------------------------------------------------------------------------------

VERSION = 2.2.3

default: demos

include ../UFconfig/UFconfig.mk

# Compile all C code, including the C-callable routine and the mexFunctions.
# Do not compile the FORTRAN versions, or MATLAB interface.
demos:
	( cd Lib    ; $(MAKE) )
	( cd Demo   ; $(MAKE) )

# Compile all C code, including the C-callable routine and the mexFunctions.
# Do not compile the FORTRAN versions.
all:
	( cd Lib    ; $(MAKE) )
	( cd Demo   ; $(MAKE) )
	( cd MATLAB ; $(MAKE) )

# compile just the C-callable libraries (not mexFunctions or Demos)
library:
	( cd Lib    ; $(MAKE) )

# remove object files, but keep the compiled programs and library archives
clean:
	( cd Lib    ; $(MAKE) clean )
	( cd Demo   ; $(MAKE) clean )
	( cd MATLAB ; $(MAKE) clean )
	( cd Doc    ; $(MAKE) clean )

# clean, and then remove compiled programs and library archives
purge:
	( cd Lib    ; $(MAKE) purge )
	( cd Demo   ; $(MAKE) purge )
	( cd MATLAB ; $(MAKE) purge )
	( cd Doc    ; $(MAKE) purge )

distclean: purge

# create PDF documents for the original distribution
dodoc:
	( cd Doc    ; $(MAKE) )

# get ready for distribution
dist: purge
	( cd Demo   ; $(MAKE) dist )
	( cd Doc    ; $(MAKE) )

ccode: library

lib: library

# compile the MATLAB mexFunction
mex:
	( cd MATLAB ; $(MAKE) )

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

