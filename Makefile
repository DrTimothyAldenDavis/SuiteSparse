#-------------------------------------------------------------------------------
# Makefile for all SuiteSparse packages
#-------------------------------------------------------------------------------

SUITESPARSE = $(CURDIR)
export SUITESPARSE

default: go

include SuiteSparse_config/SuiteSparse_config.mk

# Compile the default rules for each package
go: metis
	( cd SuiteSparse_config && $(MAKE) )
	( cd Mongoose  && $(MAKE) CMAKE_OPTIONS='$(CMAKE_OPTIONS)' )
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
ifneq ($(GPU_CONFIG),)
	( cd SuiteSparse_GPURuntime && $(MAKE) )
	( cd GPUQREngine && $(MAKE) )
endif
	( cd SPQR && $(MAKE) )
	( cd GraphBLAS && $(MAKE) JOBS=$(JOBS) CMAKE_OPTIONS='$(CMAKE_OPTIONS)' )
#	( cd PIRO_BAND && $(MAKE) )
#	( cd SKYLINE_SVD && $(MAKE) )

# install all packages in /usr/local/lib and /usr/local/include
# (note that CSparse is not installed; CXSparse is installed instead)
install: metisinstall
	( cd SuiteSparse_config && $(MAKE) install )
	( cd Mongoose  && $(MAKE) CMAKE_OPTIONS='$(CMAKE_OPTIONS)' install )
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
ifneq (,$(GPU_CONFIG))
	( cd SuiteSparse_GPURuntime && $(MAKE) install )
	( cd GPUQREngine && $(MAKE) install )
endif
	( cd SPQR && $(MAKE) install )
	( cd GraphBLAS && $(MAKE) JOBS=$(JOBS) CMAKE_OPTIONS='$(CMAKE_OPTIONS)' install )
#	( cd PIRO_BAND && $(MAKE) install )
#	( cd SKYLINE_SVD && $(MAKE) install )
	$(CP) README.md $(INSTALL_DOC)/SuiteSparse_README.txt
	chmod 644 $(INSTALL_DOC)/SuiteSparse_README.txt

metisinstall: metis
ifeq (,$(MY_METIS_LIB))
        # install METIS from SuiteSparse/metis-5.1.0
	@mkdir -p $(INSTALL_LIB)
	@mkdir -p $(INSTALL_INCLUDE)
	@mkdir -p $(INSTALL_DOC)
	- $(CP) lib/libmetis.* $(INSTALL_LIB)
	- $(CP) metis-5.1.0/manual/manual.pdf $(INSTALL_DOC)/METIS_manual.pdf
	- $(CP) metis-5.1.0/README.txt $(INSTALL_DOC)/METIS_README.txt
        # the following is needed only on the Mac, so *.dylib is hardcoded:
	$(SO_INSTALL_NAME) $(INSTALL_LIB)/libmetis.dylib $(INSTALL_LIB)/libmetis.dylib
	- $(CP) include/metis.h $(INSTALL_INCLUDE)
	chmod 755 $(INSTALL_LIB)/libmetis.*
	chmod 644 $(INSTALL_INCLUDE)/metis.h
	chmod 644 $(INSTALL_DOC)/METIS_manual.pdf
	chmod 644 $(INSTALL_DOC)/METIS_README.txt
endif

# uninstall all packages
uninstall:
	$(RM) $(INSTALL_DOC)/SuiteSparse_README.txt
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
#	( cd PIRO_BAND && $(MAKE) uninstall )
#	( cd SKYLINE_SVD && $(MAKE) uninstall )
ifeq (,$(MY_METIS_LIB))
        # uninstall METIS, which came from SuiteSparse/metis-5.1.0
	$(RM) $(INSTALL_LIB)/libmetis.*
	$(RM) $(INSTALL_INCLUDE)/metis.h
	$(RM) $(INSTALL_DOC)/METIS_manual.pdf
	$(RM) $(INSTALL_DOC)/METIS_README.txt
endif
	$(RM) -r $(INSTALL_DOC)

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
ifneq (,$(GPU_CONFIG))
	( cd SuiteSparse_GPURuntime && $(MAKE) library )
	( cd GPUQREngine && $(MAKE) library )
endif
	( cd SPQR && $(MAKE) library )
	( cd GraphBLAS && $(MAKE) JOBS=$(JOBS) CMAKE_OPTIONS='$(CMAKE_OPTIONS)' library )
#	( cd PIRO_BAND && $(MAKE) library )
#	( cd SKYLINE_SVD && $(MAKE) library )

# compile the static libraries (except for metis, GraphBLAS, and Mongoose).
# metis is only dynamic, and the 'make static' for GraphBLAS and Mongoose makes
# both the dynamic and static libraries.
static: metis
	( cd SuiteSparse_config && $(MAKE) static )
	( cd Mongoose  && $(MAKE) CMAKE_OPTIONS='$(CMAKE_OPTIONS)' static )
	( cd AMD && $(MAKE) static )
	( cd BTF && $(MAKE) static )
	( cd CAMD && $(MAKE) static )
	( cd CCOLAMD && $(MAKE) static )
	( cd COLAMD && $(MAKE) static )
	( cd CHOLMOD && $(MAKE) static )
	( cd KLU && $(MAKE) static )
	( cd LDL && $(MAKE) static )
	( cd UMFPACK && $(MAKE) static )
	( cd CSparse && $(MAKE) static )
	( cd CXSparse && $(MAKE) static )
	( cd RBio && $(MAKE) static )
ifneq (,$(GPU_CONFIG))
	( cd SuiteSparse_GPURuntime && $(MAKE) static )
	( cd GPUQREngine && $(MAKE) static )
endif
	( cd SPQR && $(MAKE) static )
	( cd GraphBLAS && $(MAKE) JOBS=$(JOBS) CMAKE_OPTIONS='$(CMAKE_OPTIONS)' static )
#	( cd PIRO_BAND && $(MAKE) static )
#	( cd SKYLINE_SVD && $(MAKE) static )

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
#	- ( cd PIRO_BAND && $(MAKE) purge )
#	- ( cd SKYLINE_SVD && $(MAKE) purge )
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
#	- ( cd PIRO_BAND && $(MAKE) clean )
#	- ( cd SKYLINE_SVD && $(MAKE) clean )

# Create the PDF documentation
docs:
	( cd GraphBLAS && $(MAKE) docs )
	( cd Mongoose  && $(MAKE) docs )
	( cd AMD && $(MAKE) docs )
	( cd CAMD && $(MAKE) docs )
	( cd KLU && $(MAKE) docs )
	( cd LDL && $(MAKE) docs )
	( cd UMFPACK && $(MAKE) docs )
	( cd CHOLMOD && $(MAKE) docs )
	( cd SPQR && $(MAKE) docs )
#	( cd PIRO_BAND && $(MAKE) docs )
#	( cd SKYLINE_SVD && $(MAKE) docs )

distclean: purge

# Create CXSparse from CSparse
# Note that the CXSparse directory should initially not exist.
cx:
	( cd CSparse ; $(MAKE) purge )
	( cd SuiteSparse_config && $(MAKE) )
	( cd CXSparse_newfiles ; tar cfv - * | gzip -9 > ../CXSparse_newfiles.tar.gz )
	./CSparse_to_CXSparse CSparse CXSparse CXSparse_newfiles.tar.gz
	( cd CXSparse/Demo ; $(MAKE) )
	( cd CXSparse/Demo ; $(MAKE) > cs_demo.out )
	( cd CXSparse ; $(MAKE) purge )
	$(RM) -f CXSparse_newfiles.tar.gz

# statement coverage (Linux only); this requires a lot of time.
# The umfpack tcov requires a lot of disk space in /tmp
cov: purge
	( cd CXSparse && $(MAKE) cov )
	( cd CSparse && $(MAKE) cov )
	( cd CHOLMOD && $(MAKE) cov )
	( cd KLU && $(MAKE) cov )
	( cd SPQR && $(MAKE) cov )
	( cd UMFPACK && $(MAKE) cov )
#	( cd PIRO_BAND && $(MAKE) cov )
#	( cd SKYLINE_SVD && $(MAKE) cov )

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

# just install GraphBLAS
gbinstall:
	echo $(CMAKE_OPTIONS)
	( cd GraphBLAS && $(MAKE) JOBS=$(JOBS) CMAKE_OPTIONS='$(CMAKE_OPTIONS)' install )

