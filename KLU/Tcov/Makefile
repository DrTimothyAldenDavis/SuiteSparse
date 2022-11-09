# If the libraries (AMD, COLAMD, CAMD, CCOLAMD, metis, and CHOLMOD) are not
# yet built, use "make local ; make install" in SuiteSparse first.  Then "make"
# to compile and run all tests.
#
# This test requires BTF, METIS, CHOLMOD, AMD, COLAMD, CAMD, and CCOLAMD.

default: all

# for statement coverage, picky tests
C = gcc -Wall -W -Wshadow -Wmissing-prototypes -Wstrict-prototypes \
   	-Wredundant-decls -Wnested-externs -Wdisabled-optimization \
	-std=c11 -g --coverage -fprofile-abs-path -fexceptions \
	-fopenmp

LDLIBS = -L../../lib \
    ../../lib/libsuitesparseconfig.so \
    ../../lib/libamd.so \
    ../../lib/libcolamd.so \
    ../../lib/libcamd.so \
    ../../lib/libccolamd.so \
    ../../lib/libcholmod.so \
    -lm -lrt \
    -Wl,-rpath=../../lib -lstdc++

I = -I../../include -I../Include -I../User -I../../BTF/Include -I../Source

all: klutest klultest
	- ./klultests > klultests.out
	- ./klutests > klutests.out
	- ./coverage

valgrind: klutest klultest
	- ./vklutests > klutests.out
	- ./vklultests > klultests.out
	- ./coverage

purge: clean

distclean: clean

clean:
	- $(RM) klutest klultest *.out klu_*.c user_*.c btf_*.c *.o *.gcda \
            klu.c *.gcno *.gcov gmon.out

INC =  \
	../Include/klu.h \
	../Include/klu_internal.h \
	../Include/klu_version.h

BTFOBJ = \
	btf_order.o \
	btf_maxtrans.o \
	btf_strongcomp.o \
	btf_l_order.o \
	btf_l_maxtrans.o \
	btf_l_strongcomp.o

KLUOBJ = \
	klu_analyze.o \
	klu_analyze_given.o \
	klu_defaults.o \
	klu_free_symbolic.o \
	klu_memory.o \
	klu.o \
	klu_diagnostics.o \
	klu_dump.o \
	klu_factor.o \
	klu_free_numeric.o \
	klu_kernel.o \
	klu_extract.o \
	klu_refactor.o \
	klu_scale.o \
	klu_solve.o \
	klu_tsolve.o \
	klu_z.o \
	klu_z_diagnostics.o \
	klu_z_dump.o \
	klu_z_factor.o \
	klu_z_free_numeric.o \
	klu_z_kernel.o \
	klu_z_extract.o \
	klu_z_refactor.o \
	klu_z_scale.o \
	klu_z_solve.o \
	klu_z_tsolve.o \
	klu_l_analyze.o \
	klu_l_analyze_given.o \
	klu_l_defaults.o \
	klu_l_free_symbolic.o \
	klu_l_memory.o \
	klu_l.o \
	klu_l_diagnostics.o \
	klu_l_dump.o \
	klu_l_factor.o \
	klu_l_free_numeric.o \
	klu_l_kernel.o \
	klu_l_extract.o \
	klu_l_refactor.o \
	klu_l_scale.o \
	klu_l_solve.o \
	klu_l_tsolve.o \
	klu_zl.o \
	klu_zl_diagnostics.o \
	klu_zl_dump.o \
	klu_zl_factor.o \
	klu_zl_free_numeric.o \
	klu_zl_kernel.o \
	klu_zl_extract.o \
	klu_zl_refactor.o \
	klu_zl_scale.o \
	klu_zl_solve.o \
	klu_zl_tsolve.o

KLUCHOLMODOBJ = user_klu_cholmod.o user_klu_l_cholmod.o

OBJ = $(BTFOBJ) $(KLUOBJ) $(KLUCHOLMODOBJ)

$(OBJ): $(INC)

klutest: $(OBJ) klutest.c
	$(C) $(I) klutest.c -o klutest $(OBJ) $(LDLIBS)

klultest: $(OBJ) klultest.c klutest.c
	$(C) $(I) klultest.c -o klultest $(OBJ) $(LDLIBS)

klu.c:
	- ln -s ../Source/$@

klu_%.c:
	- ln -s ../Source/$@

user_%.c:
	- ln -s ../User/$*.c user_$*.c

btf_%.c:
	- ln -s ../../BTF/Source/$@

.c.o:
	$(C) -c $(I) $*.c

.PRECIOUS: klu_%.c btf_%.c user_%.c

