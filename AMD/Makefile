#------------------------------------------------------------------------------
# AMD Makefile (for GNU Make or original make)
#------------------------------------------------------------------------------

VERSION = 2.2.3

default: demos

include ../UFconfig/UFconfig.mk

# Compile all C code, including the C-callable routines.
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

# compile the FORTRAN libraries and demo programs (not compiled by "make all")
fortran:
	( cd Lib    ; $(MAKE) fortran )
	( cd Demo   ; $(MAKE) fortran )

# compile a FORTRAN demo program that calls the C version of AMD
# (not compiled by "make all")
cross:
	( cd Demo   ; $(MAKE) cross )

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
docs:
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

# install AMD
install:
	$(CP) Lib/libamd.a $(INSTALL_LIB)/libamd.$(VERSION).a
	( cd $(INSTALL_LIB) ; ln -sf libamd.$(VERSION).a libamd.a )
	$(CP) Include/amd.h $(INSTALL_INCLUDE)
	chmod 644 $(INSTALL_LIB)/libamd*
	chmod 644 $(INSTALL_INCLUDE)/amd.h

# uninstall AMD
uninstall:
	$(RM) $(INSTALL_LIB)/libamd*.a
	$(RM) $(INSTALL_INCLUDE)/amd.h

