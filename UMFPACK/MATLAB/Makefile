#-------------------------------------------------------------------------------
# UMFPACK Makefile just for cleanup (use umfpack_make.m to compile, in MATLAB)
#-------------------------------------------------------------------------------

default:
	- echo "Use umfpack_make.m in MATLAB to compile the UMFPACK mexFunction"

include ../../SuiteSparse_config/SuiteSparse_config.mk

#-------------------------------------------------------------------------------
# Remove all but the files in the original distribution
#-------------------------------------------------------------------------------

purge: clean
	- $(RM) *.mex* rename.h

clean:
	- $(RM) $(CLEAN)
