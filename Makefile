#-------------------------------------------------------------------------------
# Makefile for all SuiteSparse packages
#-------------------------------------------------------------------------------

include SuiteSparse_config/SuiteSparse_config.mk

# Compile the default rules for each package
default:
	- ( cd SuiteSparse_config/xerbla && $(MAKE) )
	- ( cd SuiteSparse_config && $(MAKE) )
#	- ( cd metis-4.0 && $(MAKE) )
	- ( cd AMD && $(MAKE) )
	- ( cd CAMD && $(MAKE) )
	- ( cd COLAMD && $(MAKE) )
	- ( cd BTF && $(MAKE) )
	- ( cd KLU && $(MAKE) )
	- ( cd LDL && $(MAKE) )
	- ( cd CCOLAMD && $(MAKE) )
	- ( cd UMFPACK && $(MAKE) )
	- ( cd CHOLMOD && $(MAKE) )
	- ( cd CSparse && $(MAKE) )
	- ( cd CXSparse && $(MAKE) )
	- ( cd SPQR && $(MAKE) )
	- ( cd RBio && $(MAKE) )

# install all packages in /usr/local/lib and /usr/local/include
install:
	- ( cd SuiteSparse_config && $(MAKE) install )
	- ( cd AMD && $(MAKE) install )
	- ( cd CAMD && $(MAKE) install )
	- ( cd COLAMD && $(MAKE) install )
	- ( cd BTF && $(MAKE) install )
	- ( cd KLU && $(MAKE) install )
	- ( cd LDL && $(MAKE) install )
	- ( cd CCOLAMD && $(MAKE) install )
	- ( cd UMFPACK && $(MAKE) install )
	- ( cd CHOLMOD && $(MAKE) install )
	- ( cd CXSparse && $(MAKE) install )
	- ( cd SPQR && $(MAKE) install )
	- ( cd RBio && $(MAKE) install )

# uninstall all packages
uninstall:
	- ( cd SuiteSparse_config && $(MAKE) uninstall )
	- ( cd AMD && $(MAKE) uninstall )
	- ( cd CAMD && $(MAKE) uninstall )
	- ( cd COLAMD && $(MAKE) uninstall )
	- ( cd BTF && $(MAKE) uninstall )
	- ( cd KLU && $(MAKE) uninstall )
	- ( cd LDL && $(MAKE) uninstall )
	- ( cd CCOLAMD && $(MAKE) uninstall )
	- ( cd UMFPACK && $(MAKE) uninstall )
	- ( cd CHOLMOD && $(MAKE) uninstall )
	- ( cd CXSparse && $(MAKE) uninstall )
	- ( cd SPQR && $(MAKE) uninstall )
	- ( cd RBio && $(MAKE) uninstall )

library:
	- ( cd SuiteSparse_config/xerbla && $(MAKE) )
	- ( cd SuiteSparse_config && $(MAKE) )
#	- ( cd metis-4.0 && $(MAKE) )
	- ( cd AMD && $(MAKE) library )
	- ( cd BTF && $(MAKE) library )
	- ( cd CAMD && $(MAKE) library )
	- ( cd CCOLAMD && $(MAKE) library )
	- ( cd COLAMD && $(MAKE) library )
	- ( cd CHOLMOD && $(MAKE) library )
	- ( cd KLU && $(MAKE) library )
	- ( cd LDL && $(MAKE) library )
	- ( cd UMFPACK && $(MAKE) library )
	- ( cd CSparse && $(MAKE) library )
	- ( cd CXSparse && $(MAKE) library )
	- ( cd SPQR && $(MAKE) library )
	- ( cd RBio && $(MAKE) library )

# Remove all files not in the original distribution
purge:
	- ( cd SuiteSparse_config/xerbla && $(MAKE) purge )
	- ( cd SuiteSparse_config && $(MAKE) purge )
#	- ( cd metis-4.0 && $(MAKE) realclean )
	- ( cd AMD && $(MAKE) purge )
	- ( cd CAMD && $(MAKE) purge )
	- ( cd COLAMD && $(MAKE) purge )
	- ( cd BTF && $(MAKE) purge )
	- ( cd KLU && $(MAKE) purge )
	- ( cd LDL && $(MAKE) purge )
	- ( cd CCOLAMD && $(MAKE) purge )
	- ( cd UMFPACK && $(MAKE) purge )
	- ( cd CHOLMOD && $(MAKE) purge )
	- ( cd CSparse && $(MAKE) purge )
	- ( cd CXSparse && $(MAKE) purge )
	- ( cd RBio && $(MAKE) purge )
	- ( cd MATLAB_Tools/UFcollection && $(RM) *.mex* )
	- ( cd MATLAB_Tools/SSMULT && $(RM) *.mex* )
	- ( cd SPQR && $(MAKE) purge )
	- $(RM) MATLAB_Tools/*/*.mex* MATLAB_Tools/spok/private/*.mex*

# Remove all files not in the original distribution, but keep the libraries
clean:
	- ( cd SuiteSparse_config/xerbla && $(MAKE) clean )
	- ( cd SuiteSparse_config && $(MAKE) clean )
#	- ( cd metis-4.0 && $(MAKE) clean )
	- ( cd AMD && $(MAKE) clean )
	- ( cd CAMD && $(MAKE) clean )
	- ( cd COLAMD && $(MAKE) clean )
	- ( cd BTF && $(MAKE) clean )
	- ( cd KLU && $(MAKE) clean )
	- ( cd LDL && $(MAKE) clean )
	- ( cd CCOLAMD && $(MAKE) clean )
	- ( cd UMFPACK && $(MAKE) clean )
	- ( cd CHOLMOD && $(MAKE) clean )
	- ( cd CSparse && $(MAKE) clean )
	- ( cd CXSparse && $(MAKE) clean )
	- ( cd SPQR && $(MAKE) clean )
	- ( cd RBio && $(MAKE) clean )

# Create the PDF documentation
docs:
	( cd AMD && $(MAKE) docs )
	( cd CAMD && $(MAKE) docs )
	( cd KLU && $(MAKE) docs )
	( cd LDL && $(MAKE) docs )
	( cd UMFPACK && $(MAKE) docs )
	( cd CHOLMOD && $(MAKE) docs )
	( cd SPQR && $(MAKE) docs )

distclean: purge

# Create CXSparse from CSparse
# Note that the CXSparse directory should initially not exist.
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
	- ( cd CXSparse && $(MAKE) cov )
	- ( cd CSparse && $(MAKE) cov )
	- ( cd KLU && $(MAKE) cov )
	- ( cd CHOLMOD && $(MAKE) cov )
	- ( cd UMFPACK && $(MAKE) cov )
	- ( cd SPQR && $(MAKE) cov )
