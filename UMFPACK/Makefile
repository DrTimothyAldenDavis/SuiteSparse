#-------------------------------------------------------------------------------
# UMFPACK Makefile
#-------------------------------------------------------------------------------

VERSION = 5.5.2

# UMFPACK requires the AMD package to be in ../AMD

default: all

include ../UFconfig/UFconfig.mk

# compile all C code (except hb, fortran, and fortran64), including AMD and the
# demos, but not the MATLAB mexFunctions
all:
	( cd Demo   ; $(MAKE) )

# compile just the C-callable UMFPACK library
library:
	( cd Lib    ; $(MAKE) )

# compile the FORTRAN interface and demo program
fortran:
	( cd Demo   ; $(MAKE) fortran )

# compile the 64-bit FORTRAN interface and demo program
fortran64:
	( cd Demo   ; $(MAKE) fortran64 )

# compile the Harwell/Boeing demo program
hb:
	( cd Demo   ; $(MAKE) hb )

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

# create PDF documents for the original distribution
docs:
	( cd Doc    ; $(MAKE) )

# get ready for distribution
dist: purge
	( cd Demo   ; $(MAKE) dist )
	( cd Doc    ; $(MAKE) )

distclean: purge

ccode: library

# statement coverage (requires Linux; takes a lot of time and disk space)
cov: purge
	( cd Tcov ; ./DO.linux )

# install UMFPACK
install:
	$(CP) Lib/libumfpack.a $(INSTALL_LIB)/libumfpack.$(VERSION).a
	( cd $(INSTALL_LIB) ; ln -sf libumfpack.$(VERSION).a libumfpack.a )
	$(CP) Include/*.h $(INSTALL_INCLUDE)
	chmod 644 $(INSTALL_LIB)/libumfpack*.a
	chmod 644 $(INSTALL_INCLUDE)/umfpack*.h

# uninstall UMFPACK
uninstall:
	$(RM) $(INSTALL_LIB)/libumfpack*.a
	$(RM) $(INSTALL_INCLUDE)/umfpack*.h

