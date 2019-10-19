#-------------------------------------------------------------------------------
# compile the UMFPACK demos (for GNU make and original make)
#-------------------------------------------------------------------------------

# UMFPACK Copyright (c) Timothy A. Davis.
# All Rights Reserved.  See ../Doc/License for License.

default: libs run

all: libs run hb

all32: libs run hb fortran

all64: libs run hb fortran64

include ../../UFconfig/UFconfig.mk

#-------------------------------------------------------------------------------
# the optional Partition module requires METIS, CAMD, and CCOLAMD
LIB_WITH_PARTITION =
CONFIG1 = -DNCHOLMOD
CONFIG2 = -DNPARTITION
LIB_WITH_CHOLMOD =
ifeq (,$(findstring -DNCHOLMOD, $(UMFPACK_CONFIG)))
    # CHOLMOD is requested.  See if it is available
    ifeq (../../CHOLMOD, $(wildcard ../../CHOLMOD))
        ifeq (../../COLAMD, $(wildcard ../../COLAMD))
            # CHOLMOD and COLAMD are available
            CONFIG1 =
            LIB_WITH_CHOLMOD = ../../CHOLMOD/Lib/libcholmod.a \
                ../../COLAMD/Lib/libcolamd.a
            # check if METIS is requested and available
            ifeq (,$(findstring -DNPARTITION, $(CHOLMOD_CONFIG)))
                # METIS is requested.  See if it is available
                ifeq ($(METIS), $(wildcard $(METIS)))
                    ifeq (../../CAMD, $(wildcard ../../CAMD))
                        ifeq (../../CCOLAMD, $(wildcard ../../CCOLAMD))
                            CONFIG2 =
                            LIB_WITH_PARTITION = $(METIS) \
                                ../../CCOLAMD/Lib/libccolamd.a \
                                ../../CAMD/Lib/libcamd.a
                        endif
                    endif
                endif
            endif
        endif
    endif
endif
#-------------------------------------------------------------------------------

C = $(CC) $(CF) $(UMFPACK_CONFIG) $(CONFIG1) $(CONFIG2) \
    -I../Include -I../../AMD/Include -I../../UFconfig

INC = ../Include/umfpack.h ../../AMD/Include/amd.h ../../UFconfig/UFconfig.h

LIBS = $(BLAS) $(XERBLA) $(LIB) $(LIB_WITH_CHOLMOD) $(LIB_WITH_PARTITION)

../Lib/libumfpack.a:
	( cd ../Lib ; $(MAKE) )

../../AMD/Lib/libamd.a:
	( cd ../../AMD ; $(MAKE) library )

../../CHOLMOD/Lib/libcholmod.a:
	- ( cd ../../CHOLMOD && $(MAKE) library )

../../COLAMD/Lib/libcolamd.a:
	- ( cd ../../COLAMD && $(MAKE) library )


../../CCOLAMD/Lib/libccolamd.a:
	- ( cd ../../CCOLAMD && $(MAKE) library )

../../CAMD/Lib/libcamd.a:
	- ( cd ../../CAMD && $(MAKE) library )

$(METIS):
	( cd $(METIS_PATH) && $(MAKE) )

UMFPACK = ../Lib/libumfpack.a ../../AMD/Lib/libamd.a \
    $(WITH_CHOLMOD) $(WITH_PARTITION)

libs: $(UMFPACK)

#-------------------------------------------------------------------------------
# Create the demo programs, run them, and compare the output
#-------------------------------------------------------------------------------

dist: umfpack_di_demo.c umfpack_dl_demo.c umfpack_zi_demo.c umfpack_zl_demo.c

umfpack_simple: umfpack_simple.c $(INC) $(UMFPACK)
	$(C) -o umfpack_simple umfpack_simple.c $(UMFPACK) $(LIBS)

# the GNU rules are simpler:
# umfpack_%_demo.c: umfpack_xx_demo.c umfpack_%_demo.sed
# 	- sed -f umfpack_$*_demo.sed < umfpack_xx_demo.c > umfpack_$*_demo.c
# 
# umfpack_%_demo: umfpack_%_demo.c $(INC) $(UMFPACK)
# 	$(C) -o umfpack_$*_demo umfpack_$*_demo.c $(UMFPACK) $(LIBS)
# 	./umfpack_$*_demo > my_umfpack_$*_demo.out

# but do this via brute-force, so we can use just a single Makefile:

# double-precision, int verion:
umfpack_di_demo.c: umfpack_xx_demo.c umfpack_di_demo.sed
	- sed -f umfpack_di_demo.sed < umfpack_xx_demo.c > umfpack_di_demo.c

umfpack_di_demo: umfpack_di_demo.c $(INC) $(UMFPACK)
	$(C) -o umfpack_di_demo umfpack_di_demo.c $(UMFPACK) $(LIBS)

# double-precision, UF_long verion:
umfpack_dl_demo.c: umfpack_xx_demo.c umfpack_dl_demo.sed
	- sed -f umfpack_dl_demo.sed < umfpack_xx_demo.c > umfpack_dl_demo.c

umfpack_dl_demo: umfpack_dl_demo.c $(INC) $(UMFPACK)
	$(C) -o umfpack_dl_demo umfpack_dl_demo.c $(UMFPACK) $(LIBS)

# complex, int verion:
umfpack_zi_demo.c: umfpack_xx_demo.c umfpack_zi_demo.sed
	- sed -f umfpack_zi_demo.sed < umfpack_xx_demo.c > umfpack_zi_demo.c

umfpack_zi_demo: umfpack_zi_demo.c $(INC) $(UMFPACK)
	$(C) -o umfpack_zi_demo umfpack_zi_demo.c $(UMFPACK) $(LIBS)

# complex, UF_long verion:
umfpack_zl_demo.c: umfpack_xx_demo.c umfpack_zl_demo.sed
	- sed -f umfpack_zl_demo.sed < umfpack_xx_demo.c > umfpack_zl_demo.c

umfpack_zl_demo: umfpack_zl_demo.c $(INC) $(UMFPACK)
	$(C) -o umfpack_zl_demo umfpack_zl_demo.c $(UMFPACK) $(LIBS)

run: umfpack_di_demo umfpack_zi_demo umfpack_dl_demo umfpack_zl_demo umfpack_simple
	./umfpack_simple
	./umfpack_di_demo > my_umfpack_di_demo.out
	- diff umfpack_di_demo.out my_umfpack_di_demo.out
	./umfpack_dl_demo > my_umfpack_dl_demo.out
	- diff umfpack_dl_demo.out my_umfpack_dl_demo.out
	./umfpack_zi_demo > my_umfpack_zi_demo.out
	- diff umfpack_zi_demo.out my_umfpack_zi_demo.out
	./umfpack_zl_demo > my_umfpack_zl_demo.out
	- diff umfpack_zl_demo.out my_umfpack_zl_demo.out

#-------------------------------------------------------------------------------
# create a demo program that reads in Harwell/Boeing matrices, and run it
#-------------------------------------------------------------------------------

# the output of "make hb" is in the file umf4.out
hb: $(UMFPACK) umf4 readhb readhb_nozeros readhb_size
	- ./readhb_nozeros < HB/can_24.psa > tmp/A
	- ./readhb_size    < HB/can_24.psa > tmp/Asize
	- ./umf4
	- ./readhb_nozeros < HB/west0067.rua > tmp/A
	- ./readhb_size    < HB/west0067.rua > tmp/Asize
	- ./umf4
	- ./readhb_nozeros < HB/fs_183_6.rua > tmp/A
	- ./readhb_size    < HB/fs_183_6.rua > tmp/Asize
	- ./umf4
	- ./readhb         < HB/fs_183_6.rua > tmp/A
	- ./readhb_size    < HB/fs_183_6.rua > tmp/Asize
	- ./umf4
	- ./readhb         < HB/arc130.rua > tmp/A
	- ./readhb_size    < HB/arc130.rua > tmp/Asize
	- ./umf4
	- ./readhb_nozeros < HB/arc130.rua > tmp/A
	- ./readhb_size    < HB/arc130.rua > tmp/Asize
	- ./umf4
	- ./readhb_nozeros < HB/arc130.rua > tmp/A
	- ./readhb_size    < HB/arc130.rua > tmp/Asize
	- ./umf4 a 1e-6

umf4: umf4.c $(UMFPACK)
	$(C) -o umf4 umf4.c $(UMFPACK) $(LIBS)

readhb:  readhb.f
	$(F77) $(F77FLAGS) -o readhb readhb.f $(F77LIB)

readhb_size:  readhb_size.f
	$(F77) $(F77FLAGS) -o readhb_size readhb_size.f $(F77LIB)

readhb_nozeros:  readhb_nozeros.f
	$(F77) $(F77FLAGS) -o readhb_nozeros readhb_nozeros.f $(F77LIB)

#-------------------------------------------------------------------------------
# compile the FORTRAN interface and demo
#-------------------------------------------------------------------------------

# This doesn't seem to work on the Mac (Snow Leopard, OS X 10.6.1)
fortran: $(UMFPACK) umf4hb.f umf4_f77wrapper.o umf4zhb.f umf4_f77zwrapper.o $(UMFPACK)
	$(F77) $(F77FLAGS) -o umf4hb umf4hb.f umf4_f77wrapper.o \
	    $(UMFPACK) $(LIBS)
	- ./umf4hb < HB/west0067.rua > my_umf4hb.out
	- diff my_umf4hb.out umf4hb.out
	$(F77) $(F77FLAGS) -o umf4zhb umf4zhb.f umf4_f77zwrapper.o \
	    $(UMFPACK) $(LIBS)
	- ./umf4zhb < HB/qc324.cua > my_umf4zhb.out
	- diff my_umf4zhb.out umf4zhb.out

fortran64: $(UMFPACK) umf4hb64.f umf4_f77wrapper64.o umf4_f77zwrapper64.o $(UMFPACK)
	$(F77) $(F77FLAGS) -o umf4hb64 umf4hb64.f umf4_f77wrapper64.o \
	    $(UMFPACK) $(LIBS)
	- ./umf4hb64 < HB/west0067.rua > my_umf4hb64.out
	- diff my_umf4hb64.out umf4hb64.out

umf4_f77wrapper.o: umf4_f77wrapper.c $(INC)
	$(C) -c umf4_f77wrapper.c -o umf4_f77wrapper.o

umf4_f77zwrapper.o: umf4_f77zwrapper.c $(INC)
	$(C) -c umf4_f77zwrapper.c -o umf4_f77zwrapper.o

umf4_f77wrapper64.o: umf4_f77wrapper.c $(INC)
	$(C) -DDLONG -c umf4_f77wrapper.c -o umf4_f77wrapper64.o

umf4_f77zwrapper64.o: umf4_f77zwrapper.c $(INC)
	$(C) -DDLONG -c umf4_f77zwrapper.c -o umf4_f77zwrapper64.o

#-------------------------------------------------------------------------------
# Remove all but the files in the original distribution
#-------------------------------------------------------------------------------

purge: clean
	- $(RM) umfpack_simple a.out
	- $(RM) umfpack_di_demo my_umfpack_di_demo.out
	- $(RM) umfpack_dl_demo my_umfpack_dl_demo.out
	- $(RM) umfpack_zi_demo my_umfpack_zi_demo.out
	- $(RM) umfpack_zl_demo my_umfpack_zl_demo.out
	- $(RM) umf4hb umf4zhb *.umf my_umf4hb.out
	- $(RM) umf4hb64 my_umf4hb64.out my_umf4zhb.out
	- $(RM) umf4 readhb readhb_nozeros readhb_size tmp/*
	- $(RM) -r *.dSYM

clean:
	- $(RM) $(CLEAN)
