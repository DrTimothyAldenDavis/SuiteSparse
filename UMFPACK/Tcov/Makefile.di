KIND = -DDINT

all: go

include SuiteSparse_config/SuiteSparse_config.mk

go: run
	- ( cd UMFPACK/Source ; ./ucov.di )
	- ( cd AMD/Source     ; ./acov.di )

run: prog
	- ./ut > ut.out
	- tail ut.out
	#- $(RM) ut.out

prog:
	( cd CHOLMOD ; $(MAKE) library TCOV=yes )
	( cd CAMD ; $(MAKE) library TCOV=yes )
	( cd CCOLAMD ; $(MAKE) library TCOV=yes )
	( cd COLAMD ; $(MAKE) library TCOV=yes )
	( cd SuiteSparse_config ; $(MAKE) TCOV=yes )
	( cd metis-4.0/Lib ; $(MAKE) )
	( cd UMFPACK ; $(MAKE) library TCOV=yes )
	( cd AMD ; $(MAKE) library TCOV=yes )
	$(CC) $(KIND) $(CF) $(UMFPACK_CONFIG) -IUMFPACK/Source -IUMFPACK/Include -IAMD/Source -IAMD/Include -ISuiteSparse_config -o ut ut.c UMFPACK/Lib/libumfpack.a AMD/Lib/libamd.a CHOLMOD/Lib/libcholmod.a CAMD/Lib/libcamd.a COLAMD/Lib/libcolamd.a metis-4.0/libmetis.a CCOLAMD/Lib/libccolamd.a SuiteSparse_config/libsuitesparseconfig.a $(LIB)

utcov:
	- ( cd UMFPACK/Source ; ./ucov.di )
	- ( cd AMD/Source     ; ./acov.di )


purge:
	( cd UMFPACK ; $(MAKE) purge )
	( cd AMD ; $(MAKE) purge )
