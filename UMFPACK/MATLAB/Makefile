#-------------------------------------------------------------------------------
# UMFPACK Makefile just for cleanup (use umfpack_make.m to compile, in MATLAB)
#-------------------------------------------------------------------------------

default:
	- echo "Use umfpack_make.m in MATLAB to compile the UMFPACK mexFunction"

# Remove all but the files in the original distribution
distclean:
	- $(RM) -r rename.h GKlib_util.c GKlib_graph.c *.o *.mex*

purge: distclean

clean: distclean

