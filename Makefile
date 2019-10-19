#-------------------------------------------------------------------------------
# Makefile for all UF sparse matrix packages
#-------------------------------------------------------------------------------

include UFconfig/UFconfig.mk

# Compile the default rules for each package
default:
	( cd UFconfig/xerbla ; $(MAKE) )
#	( cd metis-4.0 ; $(MAKE) )
	( cd AMD ; $(MAKE) )
	( cd CAMD ; $(MAKE) )
	( cd COLAMD ; $(MAKE) )
	( cd BTF ; $(MAKE) )
	( cd KLU ; $(MAKE) )
	( cd LDL ; $(MAKE) )
	( cd CCOLAMD ; $(MAKE) )
	( cd UMFPACK ; $(MAKE) )
	( cd CHOLMOD ; $(MAKE) )
	( cd CSparse ; $(MAKE) )
	( cd CXSparse ; $(MAKE) )
	( cd SPQR ; $(MAKE) )
#	( cd LPDASA ; $(MAKE) )
#	( cd PARAKLETE ; $(MAKE) )

library: default

# Compile the MATLAB mexFunctions (except RBio and UFcollection)
# CHOLMOD and KLU will fail if you don't have METIS (use SuiteSparse_install.m
# in the MATLAB Command Window instead to compile them without METIS)
mex:
	( cd AMD ; $(MAKE) mex )
	( cd CAMD ; $(MAKE) mex )
	( cd COLAMD ; $(MAKE) mex )
	( cd BTF ; $(MAKE) mex )
	( cd LDL ; $(MAKE) mex )
	( cd CCOLAMD ; $(MAKE) mex )
	( cd CXSparse ; $(MAKE) mex )
	( cd CSparse ; $(MAKE) mex )
	( cd UMFPACK ; $(MAKE) mex )
	( cd SPQR ; $(MAKE) mex )
	( cd CHOLMOD ; $(MAKE) mex )
	( cd KLU ; $(MAKE) mex )

# Remove all files not in the original distribution
purge:
	( cd UFconfig/xerbla ; $(MAKE) purge )
#	( cd metis-4.0 ; $(MAKE) realclean )
	( cd AMD ; $(MAKE) purge )
	( cd CAMD ; $(MAKE) purge )
	( cd COLAMD ; $(MAKE) purge )
	( cd BTF ; $(MAKE) purge )
	( cd KLU ; $(MAKE) purge )
	( cd LDL ; $(MAKE) purge )
	( cd CCOLAMD ; $(MAKE) purge )
	( cd UMFPACK ; $(MAKE) purge )
	( cd CHOLMOD ; $(MAKE) purge )
	( cd CSparse ; $(MAKE) purge )
	( cd CXSparse ; $(MAKE) purge )
	( cd RBio ; $(RM) *.mex* )
	( cd UFcollection ; $(RM) *.mex* )
	( cd SSMULT ; $(RM) *.mex* )
	( cd SPQR ; $(MAKE) purge )
	- $(RM) MATLAB_Tools/spok/*.mex* MATLAB_Tools/spok/private/*.mex*
#	( cd LPDASA ; $(MAKE) purge )
#	( cd PARAKLETE ; $(MAKE) purge )

# Remove all files not in the original distribution, but keep the libraries
clean:
	( cd UFconfig/xerbla ; $(MAKE) clean )
#	( cd metis-4.0 ; $(MAKE) clean )
	( cd AMD ; $(MAKE) clean )
	( cd CAMD ; $(MAKE) clean )
	( cd COLAMD ; $(MAKE) clean )
	( cd BTF ; $(MAKE) clean )
	( cd KLU ; $(MAKE) clean )
	( cd LDL ; $(MAKE) clean )
	( cd CCOLAMD ; $(MAKE) clean )
	( cd UMFPACK ; $(MAKE) clean )
	( cd CHOLMOD ; $(MAKE) clean )
	( cd CSparse ; $(MAKE) clean )
	( cd CXSparse ; $(MAKE) clean )
	( cd SPQR ; $(MAKE) clean )
#	( cd LPDASA ; $(MAKE) clean )
#	( cd PARAKLETE ; $(MAKE) clean )

distclean: purge

# Create CXSparse from CSparse.  Note that the CXSparse directory should
# initially not exist.
cx:
	( cd CSparse ; $(MAKE) purge )
	( cd CXSparse_newfiles ; tar cfv - * | gzip -9 > ../CXSparse_newfiles.tar.gz )
	./CSparse_to_CXSparse CSparse CXSparse CXSparse_newfiles.tar.gz
	( cd CXSparse/Demo ; $(MAKE) )
	( cd CXSparse/Demo ; $(MAKE) > cs_demo.out )
	( cd CXSparse ; $(MAKE) purge )

# statement coverage (Linux only); this requires a lot of time.
# The umfpack tcov requires a lot of disk space
cov:
	( cd CXSparse ; $(MAKE) cov )
	( cd CSparse ; $(MAKE) cov )
	( cd KLU ; $(MAKE) cov )
	( cd CHOLMOD ; $(MAKE) cov )
	( cd UMFPACK ; $(MAKE) cov )
	( cd SPQR ; $(MAKE) cov )
