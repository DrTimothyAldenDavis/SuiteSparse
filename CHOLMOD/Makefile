#-------------------------------------------------------------------------------
# CHOLMOD Makefile
#-------------------------------------------------------------------------------

# Note: If you do not have METIS, or do not wish to use it in CHOLMOD, you must
# compile CHOLMOD with the -DNPARTITION flag.  See ../UFconfig/UFconfig.mk.

default: all

include ../UFconfig/UFconfig.mk

# Compile the C-callable libraries and the Demo programs.
all:
	( cd Lib ; $(MAKE) )
	( cd Demo ; $(MAKE) )

# Compile the C-callable libraries only.
library:
	( cd Lib ; $(MAKE) )

# Remove all files not in the original distribution
purge:
	( cd MATLAB ; $(MAKE) purge )
	( cd Tcov ; $(MAKE) purge )
	( cd Lib ; $(MAKE) purge )
	( cd Valgrind ; $(MAKE) dopurge )
	( cd Demo ; $(MAKE) purge )
	( cd Doc ; $(MAKE) purge )

# Remove all files not in the original distribution, except keep the 
# compiled libraries.
clean:
	( cd MATLAB ; $(MAKE) clean )
	( cd Tcov ; $(MAKE) clean )
	( cd Lib ; $(MAKE) clean )
	( cd Valgrind ; $(MAKE) clean )
	( cd Demo ; $(MAKE) clean )

distclean: purge

ccode: all

# Compile the MATLAB mexFunctions (you can also use cholmod_make.m in MATLAB)
mex:
	( cd MATLAB ; $(MAKE) )

# Run the test coverage suite.  Takes about 40 minutes on a 3.2GHz Pentium.
# Requires Linux (gcc, gcov).
cov:
	( cd Tcov ; $(MAKE) go )

# Run the test coverage suite using Valgrind.  This takes a *** long *** time.
valgrind:
	( cd Valgrind ; $(MAKE) )

# Compile the C-callable libraries and the Demo programs.
demo:
	( cd Demo ; $(MAKE) )

