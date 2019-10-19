default: all

include ../../UFconfig/UFconfig.mk

# CFLAGS = -Wall -W -Wshadow -Wmissing-prototypes -Wstrict-prototypes \
# 	-Wredundant-decls -Wnested-externs -Wdisabled-optimization \
# 	-pedantic -ansi -O3 -pg

# for statement coverage, picky tests
CFLAGS = -Wall -W -Werror -Wshadow -Wmissing-prototypes -Wstrict-prototypes \
   	-Wredundant-decls -Wnested-externs -Wdisabled-optimization \
 	-ansi -g -ftest-coverage -fprofile-arcs -fexceptions

C = $(CC) $(CFLAGS)

LIB = ../../AMD/Lib/libamd.a ../../COLAMD/libcolamd.a -lm

PRETTY = grep -v "^\#" | indent -bl -nce -ss -bli0 -i4 -sob 

all: ut

TEST = ut.c

ut: cov_klu.a ut.c Makefile
	( cd ../../AMD ; $(MAKE) )
	( cd ../../COLAMD ; $(MAKE) )
	$(C) $(I) ut.c -o ut cov_klu.a $(LIB)

purge: distclean

distclean: clean
	- $(RM) Afile cnum pgrowth options x xt xz xtz
	- $(RM) ut *.c.gcov *.out tmp/*.output *.sort *.a 
	- $(RM) cov_*.c

clean:
	- $(RM) $(CLEAN)

INC =  \
	../Include/klu.h \
	../Include/klu_internal.h \
	../Include/klu_version.h

BTFSRC = \
	../../BTF/Source/btf_order.c \
	../../BTF/Source/maxtrans.c \
	../../BTF/Source/strongcomp.c

BTFOBJ = \
	cov_btf_order.o \
	cov_maxtrans.o \
	cov_strongcomp.o

OBJ = \
	cov_klu_analyze.o \
	cov_klu_analyze_given.o \
	cov_klu_defaults.o \
	cov_klu_free_symbolic.o \
	cov_klu_memory.o \
	cov_klu_d.o \
	cov_klu_d_diagnostics.o \
	cov_klu_d_dump.o \
	cov_klu_d_factor.o \
	cov_klu_d_free_numeric.o \
	cov_klu_d_kernel.o \
	cov_klu_d_extract.o \
	cov_klu_d_refactor.o \
	cov_klu_d_scale.o \
	cov_klu_d_solve.o \
	cov_klu_d_tsolve.o \
	cov_klu_z.o \
	cov_klu_z_diagnostics.o \
	cov_klu_z_dump.o \
	cov_klu_z_factor.o \
	cov_klu_z_free_numeric.o \
	cov_klu_z_kernel.o \
	cov_klu_z_extract.o \
	cov_klu_z_refactor.o \
	cov_klu_z_scale.o \
	cov_klu_z_solve.o \
	cov_klu_z_tsolve.o

LIB2 = cov_klu.a $(LIB)

cov_klu.a: $(OBJ) $(BTFOBJ)
	$(AR) cov_klu.a $(OBJ) $(BTFOBJ)
	$(RANLIB) cov_klu.a

$(OBJ): $(INC)

I = -I../../AMD/Include -I../../AMD/Source -I../../COLAMD -I../../BTF/Include \
	-I../Include

.c.o:
	$(C) -c $(I) $*.c

#-------------------------------------------------------------------------------

cov_klu_d.o: ../Source/klu.c
	$(C) -E $(I) $< | $(PRETTY) > cov_klu_d.c
	$(C) -c $(I) cov_klu_d.c

cov_klu_z.o: ../Source/klu.c
	$(C) -E $(I) -DCOMPLEX $< | $(PRETTY) > cov_klu_z.c
	$(C) -c $(I) cov_klu_z.c

cov_klu_d_kernel.o: ../Source/klu_kernel.c
	$(C) -E $(I) $< | $(PRETTY) > cov_klu_d_kernel.c
	$(C) -c $(I) cov_klu_d_kernel.c

cov_klu_z_kernel.o: ../Source/klu_kernel.c
	$(C) -E $(I) -DCOMPLEX $< | $(PRETTY) > cov_klu_z_kernel.c
	$(C) -c $(I) cov_klu_z_kernel.c

cov_klu_d_diagnostics.o: ../Source/klu_diagnostics.c
	$(C) -E $(I) $< | $(PRETTY) > cov_klu_d_diagnostics.c
	$(C) -c $(I) cov_klu_d_diagnostics.c

cov_klu_z_diagnostics.o: ../Source/klu_diagnostics.c
	$(C) -E $(I) -DCOMPLEX $< | $(PRETTY) > cov_klu_z_diagnostics.c
	$(C) -c $(I) cov_klu_z_diagnostics.c

cov_klu_d_dump.o: ../Source/klu_dump.c
	$(C) -E $(I) $< | $(PRETTY) > cov_klu_d_dump.c
	$(C) -c $(I) cov_klu_d_dump.c

cov_klu_z_dump.o: ../Source/klu_dump.c
	$(C) -E $(I) -DCOMPLEX $< | $(PRETTY) > cov_klu_z_dump.c
	$(C) -c $(I) cov_klu_z_dump.c

cov_klu_d_factor.o: ../Source/klu_factor.c
	$(C) -E $(I) $< | $(PRETTY) > cov_klu_d_factor.c
	$(C) -c $(I) cov_klu_d_factor.c

cov_klu_z_factor.o: ../Source/klu_factor.c
	$(C) -E $(I) -DCOMPLEX $< | $(PRETTY) > cov_klu_z_factor.c
	$(C) -c $(I) cov_klu_z_factor.c

cov_klu_d_free_numeric.o: ../Source/klu_free_numeric.c
	$(C) -E $(I) $< | $(PRETTY) > cov_klu_d_free_numeric.c
	$(C) -c $(I) cov_klu_d_free_numeric.c

cov_klu_z_free_numeric.o: ../Source/klu_free_numeric.c
	$(C) -E $(I) -DCOMPLEX $< | $(PRETTY) > cov_klu_z_free_numeric.c
	$(C) -c $(I) cov_klu_z_free_numeric.c

cov_klu_d_extract.o: ../Source/klu_extract.c
	$(C) -E $(I) $< | $(PRETTY) > cov_klu_d_extract.c
	$(C) -c $(I) cov_klu_d_extract.c

cov_klu_z_extract.o: ../Source/klu_extract.c
	$(C) -E $(I) -DCOMPLEX $< | $(PRETTY) > cov_klu_z_extract.c
	$(C) -c $(I) cov_klu_z_extract.c

cov_klu_d_refactor.o: ../Source/klu_refactor.c
	$(C) -E $(I) $< | $(PRETTY) > cov_klu_d_refactor.c
	$(C) -c $(I) cov_klu_d_refactor.c

cov_klu_z_refactor.o: ../Source/klu_refactor.c
	$(C) -E $(I) -DCOMPLEX $< | $(PRETTY) > cov_klu_z_refactor.c
	$(C) -c $(I) cov_klu_z_refactor.c

cov_klu_d_scale.o: ../Source/klu_scale.c
	$(C) -E $(I) $< | $(PRETTY) > cov_klu_d_scale.c
	$(C) -c $(I) cov_klu_d_scale.c

cov_klu_z_scale.o: ../Source/klu_scale.c
	$(C) -E $(I) -DCOMPLEX $< | $(PRETTY) > cov_klu_z_scale.c
	$(C) -c $(I) cov_klu_z_scale.c

cov_klu_d_solve.o: ../Source/klu_solve.c
	$(C) -E $(I) $< | $(PRETTY) > cov_klu_d_solve.c
	$(C) -c $(I) cov_klu_d_solve.c

cov_klu_z_solve.o: ../Source/klu_solve.c
	$(C) -E $(I) -DCOMPLEX $< | $(PRETTY) > cov_klu_z_solve.c
	$(C) -c $(I) cov_klu_z_solve.c

cov_klu_d_tsolve.o: ../Source/klu_tsolve.c
	$(C) -E $(I) $< | $(PRETTY) > cov_klu_d_tsolve.c
	$(C) -c $(I) cov_klu_d_tsolve.c

cov_klu_z_tsolve.o: ../Source/klu_tsolve.c
	$(C) -E $(I) -DCOMPLEX $< | $(PRETTY) > cov_klu_z_tsolve.c
	$(C) -c $(I) cov_klu_z_tsolve.c

#-------------------------------------------------------------------------------

cov_klu_analyze.o: ../Source/klu_analyze.c
	$(C) -E $(I) $< | $(PRETTY) > cov_klu_analyze.c
	$(C) -c $(I) cov_klu_analyze.c

cov_klu_analyze_given.o: ../Source/klu_analyze_given.c
	$(C) -E $(I) $< | $(PRETTY) > cov_klu_analyze_given.c
	$(C) -c $(I) cov_klu_analyze_given.c

cov_klu_defaults.o: ../Source/klu_defaults.c
	$(C) -E $(I) $< | $(PRETTY) > cov_klu_defaults.c
	$(C) -c $(I) cov_klu_defaults.c

cov_klu_free_symbolic.o: ../Source/klu_free_symbolic.c
	$(C) -E $(I) $< | $(PRETTY) > cov_klu_free_symbolic.c
	$(C) -c $(I) cov_klu_free_symbolic.c

cov_klu_memory.o: ../Source/klu_memory.c
	$(C) -E $(I) $< | $(PRETTY) > cov_klu_memory.c
	$(C) -c $(I) cov_klu_memory.c

#-------------------------------------------------------------------------------

cov_btf_order.o: ../../BTF/Source/btf_order.c
	$(C) -E $(I) $< | $(PRETTY) > cov_btf_order.c
	$(C) -c $(I) cov_btf_order.c

cov_maxtrans.o: ../../BTF/Source/maxtrans.c
	$(C) -E $(I) $< | $(PRETTY) > cov_maxtrans.c
	$(C) -c $(I) cov_maxtrans.c

cov_strongcomp.o: ../../BTF/Source/strongcomp.c
	$(C) -E $(I) $< | $(PRETTY) > cov_strongcomp.c
	$(C) -c $(I) cov_strongcomp.c

#-------------------------------------------------------------------------------
