
default: linux

include ../../SuiteSparse_config/SuiteSparse_config.mk

linux: distclean
	# ( cd ../..   	   && $(MAKE) purge )
	( cd ../..   	   && $(MAKE) metis )
	( cd ../../CAMD    && $(MAKE) TCOV=yes )
	( cd ../../COLAMD  && $(MAKE) TCOV=yes )
	( cd ../../CCOLAMD && $(MAKE) TCOV=yes )
	( cd ../../CHOLMOD && $(MAKE) TCOV=yes )
	./DO.all $(TCOV_TMP)

clean:
	- $(RM) -r $(TCOV_TMP)/UMFPACK_TCOV_TMP

distclean: clean
	- $(RM) -r cover.out covall_err.out

purge: distclean

