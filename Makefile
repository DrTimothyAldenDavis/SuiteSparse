#-------------------------------------------------------------------------------
# Makefile for all SuiteSparse packages
#-------------------------------------------------------------------------------

SUITESPARSE = $(CURDIR)
export SUITESPARSE

default: go

include SuiteSparse_config/SuiteSparse_config.mk

# Compile the default rules for each package.
# "make install" will install all libraries in /usr/local/lib
# and include files in /usr/local/include.
go: metis
	( cd SuiteSparse_config && $(MAKE) )
	( cd Mongoose && $(MAKE) CMAKE_OPTIONS='$(CMAKE_OPTIONS)' )
	( cd AMD && $(MAKE) )
	( cd BTF && $(MAKE) )
	( cd CAMD && $(MAKE) )
	( cd CCOLAMD && $(MAKE) )
	( cd COLAMD && $(MAKE) )
	( cd CHOLMOD && $(MAKE) )
	( cd CSparse && $(MAKE) )
	( cd CXSparse && $(MAKE) )
	( cd LDL && $(MAKE) )
	( cd KLU && $(MAKE) )
	( cd UMFPACK && $(MAKE) )
	( cd RBio && $(MAKE) )
	( cd GraphBLAS && $(MAKE) JOBS=$(JOBS) CMAKE_OPTIONS='$(CMAKE_OPTIONS)' )
ifneq ($(GPU_CONFIG),)
	( cd SuiteSparse_GPURuntime && $(MAKE) )
	( cd GPUQREngine && $(MAKE) )
endif
	( cd SPQR && $(MAKE) )
	( cd SLIP_LU && $(MAKE) )

# compile and install in SuiteSparse/lib and SuiteSparse/include
local: metis
	( cd SuiteSparse_config && $(MAKE) local && $(MAKE) install )
	( cd Mongoose && $(MAKE) CMAKE_OPTIONS='$(CMAKE_OPTIONS)' local && $(MAKE) install )
	( cd AMD && $(MAKE) local && $(MAKE) install )
	( cd BTF && $(MAKE) local && $(MAKE) install )
	( cd CAMD && $(MAKE) local && $(MAKE) install )
	( cd CCOLAMD && $(MAKE) local && $(MAKE) install )
	( cd COLAMD && $(MAKE) local && $(MAKE) install )
	( cd CHOLMOD && $(MAKE) local && $(MAKE) install )
	( cd CSparse && $(MAKE) )
	( cd CXSparse && $(MAKE) local && $(MAKE) install )
	( cd LDL && $(MAKE) local && $(MAKE) install )
	( cd KLU && $(MAKE) local && $(MAKE) install )
	( cd UMFPACK && $(MAKE) local && $(MAKE) install )
	( cd RBio && $(MAKE) local && $(MAKE) install )
	( cd GraphBLAS && $(MAKE) JOBS=$(JOBS) CMAKE_OPTIONS='$(CMAKE_OPTIONS)' local && $(MAKE) install )
ifneq ($(GPU_CONFIG),)
	( cd SuiteSparse_GPURuntime && $(MAKE) )
	( cd GPUQREngine && $(MAKE) )
endif
	( cd SPQR && $(MAKE) library && $(MAKE) install )
	( cd SLIP_LU && $(MAKE) )

# install all packages in SuiteSparse/lib and SuiteSparse/include.  Use the
# following command to install in /usr/local/lib and /usr/local/include:
#       sudo make install INSTALL=/usr/local
# See SuiteSparse/README.md for more details.
# (note that CSparse is not installed; CXSparse is installed instead)
install: metisinstall gbinstall moninstall
	( cd SuiteSparse_config && $(MAKE) install )
	# ( cd Mongoose  && $(MAKE) CMAKE_OPTIONS='$(CMAKE_OPTIONS)' install )
	( cd AMD && $(MAKE) install )
	( cd BTF && $(MAKE) install )
	( cd CAMD && $(MAKE) install )
	( cd CCOLAMD && $(MAKE) install )
	( cd COLAMD && $(MAKE) install )
	( cd CHOLMOD && $(MAKE) install )
	( cd CXSparse && $(MAKE) install )
	( cd LDL && $(MAKE) install )
	( cd KLU && $(MAKE) install )
	( cd UMFPACK && $(MAKE) install )
	( cd RBio && $(MAKE) install )
	( cd GraphBLAS && $(MAKE) JOBS=$(JOBS) CMAKE_OPTIONS='$(CMAKE_OPTIONS)' install )
ifneq (,$(GPU_CONFIG))
	( cd SuiteSparse_GPURuntime && $(MAKE) install )
	( cd GPUQREngine && $(MAKE) install )
endif
	( cd SPQR && $(MAKE) install )
	( cd SLIP_LU && $(MAKE) install )

metisinstall: metis
ifeq (,$(MY_METIS_LIB))
        # install METIS from SuiteSparse/metis-5.1.0
	@mkdir -p $(INSTALL_LIB)
	@mkdir -p $(INSTALL_INCLUDE)
	- $(CP) lib/libmetis.* $(INSTALL_LIB)
        # the following is needed only on the Mac, so *.dylib is hardcoded:
	$(SO_INSTALL_NAME) $(INSTALL_LIB)/libmetis.dylib $(INSTALL_LIB)/libmetis.dylib
	- $(CP) include/metis.h $(INSTALL_INCLUDE)
	chmod 755 $(INSTALL_LIB)/libmetis.*
	chmod 644 $(INSTALL_INCLUDE)/metis.h
endif

# uninstall all packages
uninstall:
	( cd SuiteSparse_config && $(MAKE) uninstall )
	- ( cd metis-5.1.0 && $(MAKE) uninstall )
	- ( cd GraphBLAS && $(MAKE) uninstall )
	- ( cd Mongoose  && $(MAKE) uninstall )
	( cd AMD && $(MAKE) uninstall )
	( cd CAMD && $(MAKE) uninstall )
	( cd COLAMD && $(MAKE) uninstall )
	( cd BTF && $(MAKE) uninstall )
	( cd KLU && $(MAKE) uninstall )
	( cd LDL && $(MAKE) uninstall )
	( cd CCOLAMD && $(MAKE) uninstall )
	( cd UMFPACK && $(MAKE) uninstall )
	( cd CHOLMOD && $(MAKE) uninstall )
	( cd CSparse && $(MAKE) uninstall )
	( cd CXSparse && $(MAKE) uninstall )
	( cd RBio && $(MAKE) uninstall )
	( cd SuiteSparse_GPURuntime && $(MAKE) uninstall )
	( cd GPUQREngine && $(MAKE) uninstall )
	( cd SPQR && $(MAKE) uninstall )
	( cd SLIP_LU && $(MAKE) uninstall )
ifeq (,$(MY_METIS_LIB))
        # uninstall METIS, which came from SuiteSparse/metis-5.1.0
	$(RM) $(INSTALL_LIB)/libmetis.*
	$(RM) $(INSTALL_INCLUDE)/metis.h
endif

# compile the dynamic libraries.  For GraphBLAS and Mongoose, this also builds
# the static library
library: metis
	( cd SuiteSparse_config && $(MAKE) )
	( cd Mongoose  && $(MAKE) CMAKE_OPTIONS='$(CMAKE_OPTIONS)' library )
	( cd AMD && $(MAKE) library )
	( cd BTF && $(MAKE) library )
	( cd CAMD && $(MAKE) library )
	( cd CCOLAMD && $(MAKE) library )
	( cd COLAMD && $(MAKE) library )
	( cd CHOLMOD && $(MAKE) library )
	( cd KLU && $(MAKE) library )
	( cd LDL && $(MAKE) library )
	( cd UMFPACK && $(MAKE) library )
	( cd CSparse && $(MAKE) library )
	( cd CXSparse && $(MAKE) library )
	( cd RBio && $(MAKE) library )
	( cd GraphBLAS && $(MAKE) JOBS=$(JOBS) CMAKE_OPTIONS='$(CMAKE_OPTIONS)' library )
ifneq (,$(GPU_CONFIG))
	( cd SuiteSparse_GPURuntime && $(MAKE) library )
	( cd GPUQREngine && $(MAKE) library )
endif
	( cd SPQR && $(MAKE) library )
	( cd SLIP_LU && $(MAKE) library )

# Remove all files not in the original distribution
purge:
	- ( cd SuiteSparse_config && $(MAKE) purge )
	- ( cd metis-5.1.0 && $(MAKE) distclean )
	- ( cd AMD && $(MAKE) purge )
	- ( cd GraphBLAS && $(MAKE) purge )
	- ( cd Mongoose  && $(MAKE) purge )
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
	- ( cd MATLAB_Tools/SuiteSparseCollection && $(RM) *.mex* )
	- ( cd MATLAB_Tools/SSMULT && $(RM) *.mex* )
	- ( cd SuiteSparse_GPURuntime && $(MAKE) purge )
	- ( cd GPUQREngine && $(MAKE) purge )
	- ( cd SPQR && $(MAKE) purge )
	- ( cd SLIP_LU && $(MAKE) purge )
	- $(RM) MATLAB_Tools/*/*.mex* MATLAB_Tools/spok/private/*.mex*
	- $(RM) -r include/* bin/* lib/* share/*

# Remove all files not in the original distribution, but keep the libraries
clean:
	- ( cd SuiteSparse_config && $(MAKE) clean )
	- ( cd metis-5.1.0 && $(MAKE) clean )
	- ( cd GraphBLAS && $(MAKE) clean )
	- ( cd Mongoose  && $(MAKE) clean )
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
	- ( cd RBio && $(MAKE) clean )
	- ( cd SuiteSparse_GPURuntime && $(MAKE) clean )
	- ( cd GPUQREngine && $(MAKE) clean )
	- ( cd SPQR && $(MAKE) clean )
	- ( cd SLIP_LU && $(MAKE) clean )

# Create the PDF documentation
docs:
	( cd GraphBLAS && $(MAKE) docs )
#	( cd Mongoose  && $(MAKE) docs )
	( cd AMD && $(MAKE) docs )
	( cd CAMD && $(MAKE) docs )
	( cd KLU && $(MAKE) docs )
	( cd LDL && $(MAKE) docs )
	( cd UMFPACK && $(MAKE) docs )
	( cd CHOLMOD && $(MAKE) docs )
	( cd SPQR && $(MAKE) docs )
	( cd SLIP_LU && $(MAKE) docs )

distclean: purge

# statement coverage (Linux only); this requires a lot of time.
cov: purge
	( cd CXSparse && $(MAKE) cov )
	( cd CSparse && $(MAKE) cov )
	( cd CHOLMOD && $(MAKE) cov )
	( cd KLU && $(MAKE) cov )
	( cd SPQR && $(MAKE) cov )
	( cd UMFPACK && $(MAKE) cov )
	( cd SLIP_LU && $(MAKE) cov )

# configure and compile METIS, placing the libmetis.* library in
# SuiteSparse/lib and the metis.h include file in SuiteSparse/include.
metis: include/metis.h

# Install the shared version of METIS in SuiteSparse/lib.
# The SO_INSTALL_NAME commmand is only needed on the Mac, so *.dylib is
# hardcoded below.
include/metis.h:
ifeq (,$(MY_METIS_LIB))
	- ( cd metis-5.1.0 && $(MAKE) config shared=1 prefix=$(SUITESPARSE) cc=$(CC) )
	- ( cd metis-5.1.0 && $(MAKE) )
	- ( cd metis-5.1.0 && $(MAKE) install )
	- $(SO_INSTALL_NAME) $(SUITESPARSE)/lib/libmetis.dylib \
                             $(SUITESPARSE)/lib/libmetis.dylib
else
	@echo 'Using pre-installed METIS 5.1.0 library at ' '[$(MY_METIS_LIB)]'
endif

# just compile GraphBLAS
gb:
	echo $(CMAKE_OPTIONS)
	( cd GraphBLAS && $(MAKE) JOBS=$(JOBS) CMAKE_OPTIONS='$(CMAKE_OPTIONS)' )

# compile and install GraphBLAS
gbinstall: gb
	echo $(CMAKE_OPTIONS)
	( cd GraphBLAS && $(MAKE) JOBS=$(JOBS) CMAKE_OPTIONS='$(CMAKE_OPTIONS)' install )

# compile and install GraphBLAS libgraphblas_renamed, for MATLAB
gbrenamed:
	echo $(CMAKE_OPTIONS)
	( cd GraphBLAS/GraphBLAS && $(MAKE) JOBS=$(JOBS) CMAKE_OPTIONS='$(CMAKE_OPTIONS)' )
	( cd GraphBLAS/GraphBLAS && $(MAKE) JOBS=$(JOBS) CMAKE_OPTIONS='$(CMAKE_OPTIONS)' install )

# just compile Mongoose
mon:
	( cd Mongoose && $(MAKE) CMAKE_OPTIONS='$(CMAKE_OPTIONS)' )

# compile and install Mongoose
moninstall: mon
	( cd Mongoose  && $(MAKE) CMAKE_OPTIONS='$(CMAKE_OPTIONS)' install )

