default: all

ccode: all

include ../../UFconfig/UFconfig.mk

# for testing only:
# TEST = -DTESTING

C = $(CC) $(CFLAGS)

INC = ../Include/klu.h ../Include/klu_internal.h ../Include/klu_version.h \
    ../../UFconfig/UFconfig.h Makefile

I = -I../../AMD/Include -I../../COLAMD/Include -I../../BTF/Include \
    -I../Include -I../../UFconfig

all: library

library: libklu.a

KLU_D = klu_d.o klu_d_kernel.o klu_d_dump.o \
    klu_d_factor.o klu_d_free_numeric.o klu_d_solve.o \
    klu_d_scale.o klu_d_refactor.o \
    klu_d_tsolve.o klu_d_diagnostics.o klu_d_sort.o klu_d_extract.o

KLU_Z = klu_z.o klu_z_kernel.o klu_z_dump.o \
    klu_z_factor.o klu_z_free_numeric.o klu_z_solve.o \
    klu_z_scale.o klu_z_refactor.o \
    klu_z_tsolve.o klu_z_diagnostics.o klu_z_sort.o klu_z_extract.o

KLU_L = klu_l.o klu_l_kernel.o klu_l_dump.o \
    klu_l_factor.o klu_l_free_numeric.o klu_l_solve.o \
    klu_l_scale.o klu_l_refactor.o \
    klu_l_tsolve.o klu_l_diagnostics.o klu_l_sort.o klu_l_extract.o

KLU_ZL = klu_zl.o klu_zl_kernel.o klu_zl_dump.o \
    klu_zl_factor.o klu_zl_free_numeric.o klu_zl_solve.o \
    klu_zl_scale.o klu_zl_refactor.o \
    klu_zl_tsolve.o klu_zl_diagnostics.o klu_zl_sort.o klu_zl_extract.o

COMMON = \
    klu_free_symbolic.o klu_defaults.o klu_analyze_given.o \
    klu_analyze.o klu_memory.o \
    klu_l_free_symbolic.o klu_l_defaults.o klu_l_analyze_given.o \
    klu_l_analyze.o klu_l_memory.o

OBJ = $(COMMON) $(KLU_D) $(KLU_Z) $(KLU_L) $(KLU_ZL)

libklu.a: $(OBJ)
	$(AR) libklu.a $(OBJ)
	$(RANLIB) libklu.a

$(OBJ): $(INC)

#-------------------------------------------------------------------------------

klu_d.o: ../Source/klu.c
	$(C) -c $(I) $< -o $@

klu_z.o: ../Source/klu.c
	$(C) -c -DCOMPLEX $(I) $< -o $@

klu_d_kernel.o: ../Source/klu_kernel.c
	$(C) -c $(I) $< -o $@

klu_z_kernel.o: ../Source/klu_kernel.c
	$(C) -c -DCOMPLEX $(I) $< -o $@

klu_d_sort.o: ../Source/klu_sort.c
	$(C) -c $(I) $< -o $@

klu_z_sort.o: ../Source/klu_sort.c
	$(C) -c -DCOMPLEX $(I) $< -o $@

klu_d_diagnostics.o: ../Source/klu_diagnostics.c
	$(C) -c $(I) $< -o $@

klu_z_diagnostics.o: ../Source/klu_diagnostics.c
	$(C) -c -DCOMPLEX $(I) $< -o $@

klu_d_dump.o: ../Source/klu_dump.c
	$(C) -c $(I) $< -o $@

klu_z_dump.o: ../Source/klu_dump.c
	$(C) -c -DCOMPLEX $(I) $< -o $@

klu_d_factor.o: ../Source/klu_factor.c
	$(C) -c $(I) $< -o $@

klu_z_factor.o: ../Source/klu_factor.c
	$(C) -c -DCOMPLEX $(I) $< -o $@

klu_d_free_numeric.o: ../Source/klu_free_numeric.c
	$(C) -c $(I) $< -o $@

klu_z_free_numeric.o: ../Source/klu_free_numeric.c
	$(C) -c -DCOMPLEX $(I) $< -o $@

klu_d_extract.o: ../Source/klu_extract.c
	$(C) -c $(I) $< -o $@

klu_z_extract.o: ../Source/klu_extract.c
	$(C) -c -DCOMPLEX $(I) $< -o $@

klu_d_refactor.o: ../Source/klu_refactor.c
	$(C) -c $(I) $< -o $@

klu_z_refactor.o: ../Source/klu_refactor.c
	$(C) -c -DCOMPLEX $(I) $< -o $@

klu_d_scale.o: ../Source/klu_scale.c
	$(C) -c $(I) $< -o $@

klu_z_scale.o: ../Source/klu_scale.c
	$(C) -c -DCOMPLEX $(I) $< -o $@

klu_d_solve.o: ../Source/klu_solve.c
	$(C) -c $(I) $< -o $@

klu_z_solve.o: ../Source/klu_solve.c
	$(C) -c -DCOMPLEX $(I) $< -o $@

klu_d_tsolve.o: ../Source/klu_tsolve.c
	$(C) -c $(I) $< -o $@

klu_z_tsolve.o: ../Source/klu_tsolve.c
	$(C) -c -DCOMPLEX $(I) $< -o $@

#-------------------------------------------------------------------------------

klu_analyze.o: ../Source/klu_analyze.c
	$(C) -c $(I) $< -o $@

klu_analyze_given.o: ../Source/klu_analyze_given.c
	$(C) -c $(I) $< -o $@

klu_defaults.o: ../Source/klu_defaults.c
	$(C) -c $(I) $< -o $@

klu_free_symbolic.o: ../Source/klu_free_symbolic.c
	$(C) -c $(I) $< -o $@

klu_memory.o: ../Source/klu_memory.c
	$(C) -c $(I) $< -o $@

#-------------------------------------------------------------------------------

purge: distclean

distclean: clean
	- $(RM) libklu.a 

clean:
	- $(RM) $(CLEAN)

#-------------------------------------------------------------------------------

klu_l.o: ../Source/klu.c
	$(C) -c -DDLONG $(I) $< -o $@

klu_zl.o: ../Source/klu.c
	$(C) -c -DCOMPLEX -DDLONG $(I) $< -o $@

klu_l_kernel.o: ../Source/klu_kernel.c
	$(C) -c -DDLONG $(I) $< -o $@

klu_zl_kernel.o: ../Source/klu_kernel.c
	$(C) -c -DCOMPLEX -DDLONG $(I) $< -o $@

klu_l_sort.o: ../Source/klu_sort.c
	$(C) -c -DDLONG $(I) $< -o $@

klu_zl_sort.o: ../Source/klu_sort.c
	$(C) -c -DCOMPLEX -DDLONG $(I) $< -o $@

klu_l_diagnostics.o: ../Source/klu_diagnostics.c
	$(C) -c -DDLONG $(I) $< -o $@

klu_zl_diagnostics.o: ../Source/klu_diagnostics.c
	$(C) -c -DCOMPLEX -DDLONG $(I) $< -o $@

klu_l_dump.o: ../Source/klu_dump.c
	$(C) -c -DDLONG $(I) $< -o $@

klu_zl_dump.o: ../Source/klu_dump.c
	$(C) -c -DCOMPLEX -DDLONG $(I) $< -o $@

klu_l_factor.o: ../Source/klu_factor.c
	$(C) -c -DDLONG $(I) $< -o $@

klu_zl_factor.o: ../Source/klu_factor.c
	$(C) -c -DCOMPLEX -DDLONG $(I) $< -o $@

klu_l_free_numeric.o: ../Source/klu_free_numeric.c
	$(C) -c -DDLONG $(I) $< -o $@

klu_zl_free_numeric.o: ../Source/klu_free_numeric.c
	$(C) -c -DCOMPLEX -DDLONG $(I) $< -o $@

klu_l_extract.o: ../Source/klu_extract.c
	$(C) -c -DDLONG $(I) $< -o $@

klu_zl_extract.o: ../Source/klu_extract.c
	$(C) -c -DCOMPLEX -DDLONG $(I) $< -o $@

klu_l_refactor.o: ../Source/klu_refactor.c
	$(C) -c -DDLONG $(I) $< -o $@

klu_zl_refactor.o: ../Source/klu_refactor.c
	$(C) -c -DCOMPLEX -DDLONG $(I) $< -o $@

klu_l_scale.o: ../Source/klu_scale.c
	$(C) -c -DDLONG $(I) $< -o $@

klu_zl_scale.o: ../Source/klu_scale.c
	$(C) -c -DCOMPLEX -DDLONG $(I) $< -o $@

klu_l_solve.o: ../Source/klu_solve.c
	$(C) -c -DDLONG $(I) $< -o $@

klu_zl_solve.o: ../Source/klu_solve.c
	$(C) -c -DCOMPLEX -DDLONG $(I) $< -o $@

klu_l_tsolve.o: ../Source/klu_tsolve.c
	$(C) -c -DDLONG $(I) $< -o $@

klu_zl_tsolve.o: ../Source/klu_tsolve.c
	$(C) -c -DCOMPLEX -DDLONG $(I) $< -o $@

#-------------------------------------------------------------------------------

klu_l_analyze.o: ../Source/klu_analyze.c
	$(C) -c -DDLONG $(I) $< -o $@

klu_l_analyze_given.o: ../Source/klu_analyze_given.c
	$(C) -c -DDLONG $(I) $< -o $@

klu_l_defaults.o: ../Source/klu_defaults.c
	$(C) -c -DDLONG $(I) $< -o $@

klu_l_free_symbolic.o: ../Source/klu_free_symbolic.c
	$(C) -c -DDLONG $(I) $< -o $@

klu_l_memory.o: ../Source/klu_memory.c
	$(C) -c -DDLONG $(I) $< -o $@

#-------------------------------------------------------------------------------
