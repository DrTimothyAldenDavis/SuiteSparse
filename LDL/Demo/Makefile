#-------------------------------------------------------------------------------
# LDL Demo Makefile
#-------------------------------------------------------------------------------

default: all

include ../../SuiteSparse_config/SuiteSparse_config.mk

I = -I../../SuiteSparse_config -I../Include

C = $(CC) $(CF) $(I)

all: ldlsimple ldllsimple ldlmain ldllmain ldlamd ldllamd

LIB2 = ../../SuiteSparse_config/libsuitesparseconfig.a $(LIB)

library:
	( cd ../../SuiteSparse_config ; $(MAKE) library )
	( cd ../../AMD ; $(MAKE) library )
	( cd ../Lib ; $(MAKE) )

#-------------------------------------------------------------------------------
# stand-alone C programs:
#-------------------------------------------------------------------------------

ldlmain:  ldlmain.c library
	$(C) ldlmain.c ../Lib/libldl.a -o ldlmain $(LIB2)
	- ./ldlmain > my_ldlmain.out
	- diff ldlmain.out my_ldlmain.out

ldllmain:  ldlmain.c library
	$(C) -DLDL_LONG ldlmain.c ../Lib/libldl.a -o ldllmain $(LIB2)
	- ./ldllmain > my_ldllmain.out
	- diff ldlmain.out my_ldllmain.out

ldlsimple:  ldlsimple.c library
	$(C) ldlsimple.c ../Lib/libldl.a -o ldlsimple $(LIB2)
	- ./ldlsimple > my_ldlsimple.out
	- diff ldlsimple.out my_ldlsimple.out

ldllsimple:  ldlsimple.c library
	$(C) $(I) -DLDL_LONG ldlsimple.c ../Lib/libldl.a -o ldllsimple $(LIB2)
	- ./ldllsimple > my_ldllsimple.out
	- diff ldlsimple.out my_ldllsimple.out

ldlamd:  ldlmain.c library
	- $(C) -I../../AMD/Include -DUSE_AMD \
		ldlmain.c ../../AMD/Lib/libamd.a ../Lib/libldl.a -o ldlamd $(LIB2)
	- ./ldlamd > my_ldlamd.out
	- diff ldlamd.out my_ldlamd.out

ldllamd:  ldlmain.c library
	- $(C) -DLDL_LONG $(I) -I../../AMD/Include -DUSE_AMD \
		ldlmain.c ../../AMD/Lib/libamd.a ../Lib/libldl.a -o ldllamd $(LIB2)
	- ./ldllamd > my_ldllamd.out
	- diff ldllamd.out my_ldllamd.out

run:
	- ./ldlsimple > my_ldlsimple.out
	- diff ldlsimple.out my_ldlsimple.out
	- ./ldllsimple > my_ldllsimple.out
	- diff ldlsimple.out my_ldllsimple.out
	- ./ldlmain > my_ldlmain.out
	- diff ldlmain.out my_ldlmain.out
	- ./ldllmain > my_ldllmain.out
	- diff ldlmain.out my_ldllmain.out
	- ./ldlamd > my_ldlamd.out
	- diff ldlamd.out my_ldlamd.out
	- ./ldllamd > my_ldllamd.out
	- diff ldllamd.out my_ldllamd.out


#-------------------------------------------------------------------------------
# clean-up:
#-------------------------------------------------------------------------------

distclean: purge

purge: clean
	- $(RM) ldlmain ldllmain ldlsimple ldllsimple ldlamd ldllamd
	- $(RM) my_ldlmain.out my_ldlamd.out my_ldlsimple.out my_ldllamd.out
	- $(RM) my_ldllsimple.out my_ldllmain.out
	- $(RM) -r *.dSYM

clean:
	- $(RM) $(CLEAN)
