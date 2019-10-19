#-------------------------------------------------------------------------------
# UMFPACK makefile (for GNU make or original make)
#-------------------------------------------------------------------------------

# UMFPACK requires the AMD package to be in ../AMD

default: library

include ../UFconfig/UFconfig.mk

# compile all C code (except hb, fortran, and fortran64), including AMD and the
# MATLAB mexFunctions
all:
	( cd ../AMD ; $(MAKE) library )
	( cd ../AMD ; $(MAKE) mex )
	( cd Lib    ; $(MAKE) )
	( cd Demo   ; $(MAKE) )
	( cd MATLAB ; $(MAKE) )
	- cat Doc/License

# compile just the C-callable libraries and demo programs (not mexFunctions)
library:
	( cd ../AMD ; $(MAKE) library )
	( cd Lib    ; $(MAKE) )
	( cd Demo   ; $(MAKE) )
	- cat Doc/License

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
	( cd ../AMD ; $(MAKE) clean )
	( cd Lib    ; $(MAKE) clean )
	( cd Demo   ; $(MAKE) clean )
	( cd MATLAB ; $(MAKE) clean )
	( cd Doc    ; $(MAKE) clean )

# clean, and then remove compiled programs and library archives
purge:
	( cd ../AMD ; $(MAKE) purge )
	( cd Lib    ; $(MAKE) purge )
	( cd Demo   ; $(MAKE) purge )
	( cd MATLAB ; $(MAKE) purge )
	( cd Doc    ; $(MAKE) purge )

# create PDF documents for the original distribution
doc:
	( cd ../AMD ; $(MAKE) doc )
	( cd Doc    ; $(MAKE) )

# get ready for distribution
dist: purge
	( cd ../AMD ; $(MAKE) dist )
	( cd Demo   ; $(MAKE) dist )
	( cd Doc    ; $(MAKE) )

distclean: purge

ccode: library

# compile the MATLAB mexFunction
mex:
	( cd ../AMD/MATLAB ; $(MAKE) )
	( cd MATLAB ; $(MAKE) )

# statement coverage (requires Linux; takes a lot of time and disk space)
cov: purge
	( cd Tcov ; ./DO.linux ; ./covall )

