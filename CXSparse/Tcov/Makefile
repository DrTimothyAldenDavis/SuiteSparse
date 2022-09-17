# CXSparse/Tcov/Makefile
# CXSparse, Copyright (c) 2006-2022, Timothy A. Davis. All Rights Reserved.
# SPDX-License-Identifier: LGPL-2.1+

# To run with valgrind:
V =
# V = valgrind -q

# Linux test coverage
CC = gcc
CFLAGS = -O -g --coverage -fprofile-abs-path \
    	-Wall -W -Wshadow -Wmissing-prototypes -Wstrict-prototypes \
    	-Wredundant-decls -Wnested-externs -Wdisabled-optimization -std=c11 \
        -Wno-unused-but-set-variable \
	-Wno-unused-parameter -I../Include -I../Demo -I../../SuiteSparse_config

C = $(CC) $(CFLAGS)

run: all run_convert run_di run_dl run_ci run_cl
	./covall

all:	cs_demo1_di cs_demo2_di cs_demo3_di cstcov_test_di \
	cs_demo1_dl cs_demo2_dl cs_demo3_dl cstcov_test_dl \
	cs_demo1_ci cs_demo2_ci cs_demo3_ci cstcov_test_ci \
	cs_demo1_cl cs_demo2_cl cs_demo3_cl cstcov_test_cl \
	cs_idemo cs_ldemo

CS_DI = cs_add.o cs_amd.o cs_chol.o cs_cholsol.o cs_counts.o \
	cs_cumsum.o cs_droptol.o cs_dropzeros.o cs_dupl.o \
	cs_entry.o cs_etree.o cs_fkeep.o cs_gaxpy.o cs_happly.o \
	cs_house.o cs_ipvec.o cs_lsolve.o cs_ltsolve.o cs_lu.o \
	cs_lusol.o cs_util.o cs_multiply.o cs_permute.o \
	cs_pinv.o cs_post.o cs_pvec.o cs_qr.o cs_qrsol.o \
	cs_scatter.o cs_schol.o cs_sqr.o cs_symperm.o cs_tdfs.o \
	cs_transpose.o cs_compress.o cs_usolve.o cs_scc.o \
	cs_maxtrans.o cs_dmperm.o cs_updown.o cs_print.o \
	cs_norm.o cs_load.o cs_dfs.o cstcov_malloc_test.o \
	cs_utsolve.o cs_reach.o cs_spsolve.o \
	cs_leaf.o cs_ereach.o cs_randperm.o

CS_DL = cs_dl_add.o cs_dl_amd.o cs_dl_chol.o cs_dl_cholsol.o cs_dl_compress.o \
        cs_dl_counts.o cs_dl_cumsum.o cs_dl_dfs.o cs_dl_dmperm.o \
        cs_dl_droptol.o cs_dl_dropzeros.o cs_dl_dupl.o cs_dl_entry.o \
        cs_dl_ereach.o cs_dl_etree.o cs_dl_fkeep.o cs_dl_gaxpy.o \
        cs_dl_happly.o cs_dl_house.o cs_dl_ipvec.o cs_dl_leaf.o cs_dl_load.o \
        cs_dl_lsolve.o cs_dl_ltsolve.o cs_dl_lu.o cs_dl_lusol.o \
        cs_dl_maxtrans.o cs_dl_multiply.o cs_dl_norm.o \
        cs_dl_permute.o cs_dl_pinv.o cs_dl_post.o cs_dl_print.o cs_dl_pvec.o \
        cs_dl_qr.o cs_dl_qrsol.o cs_dl_randperm.o cs_dl_reach.o \
        cs_dl_scatter.o cs_dl_scc.o cs_dl_schol.o cs_dl_spsolve.o cs_dl_sqr.o \
        cs_dl_symperm.o cs_dl_tdfs.o cs_dl_transpose.o cs_dl_updown.o \
        cs_dl_usolve.o cs_dl_util.o cs_dl_utsolve.o \
        cstcov_malloc_test_dl.o

CS_CI = cs_ci_add.o cs_ci_amd.o cs_ci_chol.o cs_ci_cholsol.o cs_ci_compress.o \
        cs_ci_counts.o cs_ci_cumsum.o cs_ci_dfs.o cs_ci_dmperm.o \
        cs_ci_droptol.o cs_ci_dropzeros.o cs_ci_dupl.o cs_ci_entry.o \
        cs_ci_ereach.o cs_ci_etree.o cs_ci_fkeep.o cs_ci_gaxpy.o \
        cs_ci_happly.o cs_ci_house.o cs_ci_ipvec.o cs_ci_leaf.o cs_ci_load.o \
        cs_ci_lsolve.o cs_ci_ltsolve.o cs_ci_lu.o cs_ci_lusol.o \
        cs_ci_maxtrans.o cs_ci_multiply.o cs_ci_norm.o \
        cs_ci_permute.o cs_ci_pinv.o cs_ci_post.o cs_ci_print.o cs_ci_pvec.o \
        cs_ci_qr.o cs_ci_qrsol.o cs_ci_randperm.o cs_ci_reach.o \
        cs_ci_scatter.o cs_ci_scc.o cs_ci_schol.o cs_ci_spsolve.o cs_ci_sqr.o \
        cs_ci_symperm.o cs_ci_tdfs.o cs_ci_transpose.o cs_ci_updown.o \
        cs_ci_usolve.o cs_ci_util.o cs_ci_utsolve.o \
        cstcov_malloc_test_ci.o

CS_CL = cs_cl_add.o cs_cl_amd.o cs_cl_chol.o cs_cl_cholsol.o cs_cl_compress.o \
        cs_cl_counts.o cs_cl_cumsum.o cs_cl_dfs.o cs_cl_dmperm.o \
        cs_cl_droptol.o cs_cl_dropzeros.o cs_cl_dupl.o cs_cl_entry.o \
        cs_cl_ereach.o cs_cl_etree.o cs_cl_fkeep.o cs_cl_gaxpy.o \
        cs_cl_happly.o cs_cl_house.o cs_cl_ipvec.o cs_cl_leaf.o cs_cl_load.o \
        cs_cl_lsolve.o cs_cl_ltsolve.o cs_cl_lu.o cs_cl_lusol.o \
        cs_cl_maxtrans.o cs_cl_multiply.o cs_cl_norm.o \
        cs_cl_permute.o cs_cl_pinv.o cs_cl_post.o cs_cl_print.o cs_cl_pvec.o \
        cs_cl_qr.o cs_cl_qrsol.o cs_cl_randperm.o cs_cl_reach.o \
        cs_cl_scatter.o cs_cl_scc.o cs_cl_schol.o cs_cl_spsolve.o cs_cl_sqr.o \
        cs_cl_symperm.o cs_cl_tdfs.o cs_cl_transpose.o cs_cl_updown.o \
        cs_cl_usolve.o cs_cl_util.o cs_cl_utsolve.o \
        cstcov_malloc_test_cl.o

OBJ = $(CS_DI) $(CS_DL) $(CS_CI) $(CS_CL) cs_convert.o

$(OBJ): ../Include/cs.h cstcov_malloc_test.h Makefile

.PRECIOUS: %demo.c %demo1.c %demo2.c %demo3.c cs_%.c cs_%_ci.c cs_%_cl.c cs_%_di.c cs_%_dl.c cstcov_%.c


%demo.c:
	- ln -s ../Demo/$*demo.c

%demo1.c:
	- ln -s ../Demo/$*demo1.c

%demo2.c:
	- ln -s ../Demo/$*demo2.c

%demo3.c:
	- ln -s ../Demo/$*demo3.c

cs_%.c:
	- ln -s ../Source/cs_$*.c

cs_%.o: cs_%.c
	$(C) -c $<

cstcov_%.o: cstcov_%.c
	$(C) -c $<

cs_idemo: $(OBJ) cs_idemo.c
	$(C) -o cs_idemo cs_idemo.c $(OBJ) -lm

cs_ldemo: $(OBJ) cs_ldemo.c
	$(C) -o cs_ldemo cs_ldemo.c $(OBJ) -lm


cs_demo1_di: $(CS_DI) cs_di_demo1.c
	$(C) -o cs_demo1_di cs_di_demo1.c $(CS_DI) -lm

cs_demo2_di: $(CS_DI) cs_di_demo2.c cs_di_demo.c
	$(C) -o cs_demo2_di cs_di_demo2.c cs_di_demo.c $(CS_DI) -lm

cs_demo3_di: $(CS_DI) cs_di_demo3.c cs_di_demo.c
	$(C) -o cs_demo3_di cs_di_demo3.c cs_di_demo.c $(CS_DI) -lm

cstcov_test_di: $(CS_DI) cstcov_test.c cs_di_demo.c
	$(C) -o cstcov_test_di cstcov_test.c cs_di_demo.c $(CS_DI) -lm


cs_demo1_dl: $(CS_DL) cs_dl_demo1.c
	$(C) -o cs_demo1_dl cs_dl_demo1.c $(CS_DL) -lm

cs_demo2_dl: $(CS_DL) cs_dl_demo2.c cs_dl_demo.c
	$(C) -o cs_demo2_dl cs_dl_demo2.c cs_dl_demo.c $(CS_DL) -lm

cs_demo3_dl: $(CS_DL) cs_dl_demo3.c cs_dl_demo.c
	$(C) -o cs_demo3_dl cs_dl_demo3.c cs_dl_demo.c $(CS_DL) -lm

cstcov_test_dl: $(CS_DL) cstcov_test_dl.c cs_dl_demo.c
	$(C) -o cstcov_test_dl cstcov_test_dl.c cs_dl_demo.c $(CS_DL) -lm


cs_demo1_ci: $(CS_CI) cs_ci_demo1.c
	$(C) -o cs_demo1_ci cs_ci_demo1.c $(CS_CI) -lm

cs_demo2_ci: $(CS_CI) cs_ci_demo2.c cs_ci_demo.c
	$(C) -o cs_demo2_ci cs_ci_demo2.c cs_ci_demo.c $(CS_CI) -lm

cs_demo3_ci: $(CS_CI) cs_ci_demo3.c cs_ci_demo.c
	$(C) -o cs_demo3_ci cs_ci_demo3.c cs_ci_demo.c $(CS_CI) -lm

cstcov_test_ci: $(CS_CI) cstcov_test_ci.c cs_ci_demo.c
	$(C) -o cstcov_test_ci cstcov_test_ci.c cs_ci_demo.c $(CS_CI) -lm


cs_demo1_cl: $(CS_CL) cs_cl_demo1.c
	$(C) -o cs_demo1_cl cs_cl_demo1.c $(CS_CL) -lm

cs_demo2_cl: $(CS_CL) cs_cl_demo2.c cs_cl_demo.c
	$(C) -o cs_demo2_cl cs_cl_demo2.c cs_cl_demo.c $(CS_CL) -lm

cs_demo3_cl: $(CS_CL) cs_cl_demo3.c cs_cl_demo.c
	$(C) -o cs_demo3_cl cs_cl_demo3.c cs_cl_demo.c $(CS_CL) -lm

cstcov_test_cl: $(CS_CL) cstcov_test_cl.c cs_cl_demo.c
	$(C) -o cstcov_test_cl cstcov_test_cl.c cs_cl_demo.c $(CS_CL) -lm


run_convert: cs_idemo cs_ldemo
	- $(V) ./cs_idemo < ../Matrix/t2
	- $(V) ./cs_ldemo < ../Matrix/t2

run_di: cs_demo1_di cs_demo2_di cs_demo3_di cstcov_test_di
	- $(V) ./cs_demo1_di < ../Matrix/t1
	- $(V) ./cs_demo1_di < nil
	- $(V) ./cs_demo1_di < zero
	- $(V) ./cs_demo2_di < nil
	- $(V) ./cs_demo2_di < zero
	- $(V) ./cs_demo2_di < ../Matrix/t1
	- $(V) ./cs_demo2_di < ../Matrix/bcsstk01
	- $(V) ./cs_demo2_di < ../Matrix/fs_183_1
	- $(V) ./cs_demo2_di < ../Matrix/west0067
	- $(V) ./cs_demo2_di < ../Matrix/lp_afiro
	- $(V) ./cs_demo2_di < ../Matrix/ash219
	- $(V) ./cs_demo2_di < ../Matrix/mbeacxc
	- $(V) ./cs_demo2_di < ../Matrix/bcsstk16
	- $(V) ./cs_demo2_di < ../Matrix/ibm32a
	- $(V) ./cs_demo2_di < ../Matrix/ibm32b
	- $(V) ./cs_demo3_di < nil
	- $(V) ./cs_demo3_di < ../Matrix/bcsstk01
	- $(V) ./cs_demo3_di < ../Matrix/bcsstk16
	- $(V) ./cstcov_test_di nil > test_di_nil.out
	- $(V) ./cstcov_test_di zero > test_di_zero.out
	- $(V) ./cstcov_test_di ../Matrix/t1 > test_di_t1.out
	- $(V) ./cstcov_test_di ../Matrix/bcsstk01 > test_di_k1.out
	- $(V) ./cstcov_test_di ../Matrix/fs_183_1 > test_di_fs.out
	- $(V) ./cstcov_test_di ../Matrix/west0067 > test_di_we.out
	- $(V) ./cstcov_test_di ../Matrix/ash219 > test_di_ash.out
	- $(V) ./cstcov_test_di ../Matrix/lp_afiro > test_di_afiro.out

run_dl: cs_demo1_dl cs_demo2_dl cs_demo3_dl cstcov_test_dl
	- $(V) ./cs_demo1_dl < ../Matrix/t1
	- $(V) ./cs_demo1_dl < nil
	- $(V) ./cs_demo1_dl < zero
	- $(V) ./cs_demo2_dl < nil
	- $(V) ./cs_demo2_dl < zero
	- $(V) ./cs_demo2_dl < ../Matrix/t1
	- $(V) ./cs_demo2_dl < ../Matrix/bcsstk01
	- $(V) ./cs_demo2_dl < ../Matrix/fs_183_1
	- $(V) ./cs_demo2_dl < ../Matrix/west0067
	- $(V) ./cs_demo2_dl < ../Matrix/lp_afiro
	- $(V) ./cs_demo2_dl < ../Matrix/ash219
	- $(V) ./cs_demo2_dl < ../Matrix/mbeacxc
	- $(V) ./cs_demo2_dl < ../Matrix/bcsstk16
	- $(V) ./cs_demo2_dl < ../Matrix/ibm32a
	- $(V) ./cs_demo2_dl < ../Matrix/ibm32b
	- $(V) ./cs_demo3_dl < nil
	- $(V) ./cs_demo3_dl < ../Matrix/bcsstk01
	- $(V) ./cs_demo3_dl < ../Matrix/bcsstk16
	- $(V) ./cstcov_test_dl nil > test_dl_nil.out
	- $(V) ./cstcov_test_dl zero > test_dl_zero.out
	- $(V) ./cstcov_test_dl ../Matrix/t1 > test_dl_t1.out
	- $(V) ./cstcov_test_dl ../Matrix/bcsstk01 > test_dl_k1.out
	- $(V) ./cstcov_test_dl ../Matrix/fs_183_1 > test_dl_fs.out
	- $(V) ./cstcov_test_dl ../Matrix/west0067 > test_dl_we.out
	- $(V) ./cstcov_test_dl ../Matrix/ash219 > test_dl_ash.out
	- $(V) ./cstcov_test_dl ../Matrix/lp_afiro > test_dl_afiro.out

run_ci: cs_demo1_ci cs_demo2_ci cs_demo3_ci cstcov_test_ci
	- $(V) ./cs_demo1_ci < ../Matrix/t2
	- $(V) ./cs_demo2_ci < ../Matrix/t2
	- $(V) ./cs_demo1_ci < czero
	- $(V) ./cs_demo2_ci < czero
	- $(V) ./cs_demo1_ci < ../Matrix/t3
	- $(V) ./cs_demo2_ci < ../Matrix/t3
	- $(V) ./cs_demo1_ci < ../Matrix/t4
	- $(V) ./cs_demo2_ci < ../Matrix/t4
	- $(V) ./cs_demo2_ci < ../Matrix/c_west0067
	- $(V) ./cs_demo2_ci < ../Matrix/c_mbeacxc
	- $(V) ./cs_demo2_ci < ../Matrix/c_ibm32a
	- $(V) ./cs_demo2_ci < ../Matrix/c_ibm32b
	- $(V) ./cs_demo2_ci < ../Matrix/young1c
	- $(V) ./cs_demo2_ci < ../Matrix/qc324
	- $(V) ./cs_demo2_ci < ../Matrix/neumann
	- $(V) ./cs_demo2_ci < ../Matrix/c4
	- $(V) ./cs_demo3_ci < ../Matrix/c4
	- $(V) ./cs_demo2_ci < ../Matrix/mhd1280b
	- $(V) ./cs_demo3_ci < ../Matrix/mhd1280b
	- $(V) ./cstcov_test_ci ../Matrix/t2 > test_ci_t2.out
	- $(V) ./cstcov_test_ci ../Matrix/young1c > test_ci_young1c.out
	- $(V) ./cstcov_test_ci ../Matrix/qc324 > test_ci_qc324.out
	- $(V) ./cstcov_test_ci ../Matrix/neumann > test_ci_neumann.out
	- $(V) ./cstcov_test_ci ../Matrix/mhd1280b > test_ci_mhd1280b.out

run_cl: cs_demo1_cl cs_demo2_cl cs_demo3_cl cstcov_test_cl
	- $(V) ./cs_demo1_cl < ../Matrix/t2
	- $(V) ./cs_demo2_cl < ../Matrix/t2
	- $(V) ./cs_demo1_cl < czero
	- $(V) ./cs_demo2_cl < czero
	- $(V) ./cs_demo1_cl < ../Matrix/t3
	- $(V) ./cs_demo2_cl < ../Matrix/t3
	- $(V) ./cs_demo1_cl < ../Matrix/t4
	- $(V) ./cs_demo2_cl < ../Matrix/t4
	- $(V) ./cs_demo2_cl < ../Matrix/c_west0067
	- $(V) ./cs_demo2_cl < ../Matrix/c_mbeacxc
	- $(V) ./cs_demo2_cl < ../Matrix/c_ibm32a
	- $(V) ./cs_demo2_cl < ../Matrix/c_ibm32b
	- $(V) ./cs_demo2_cl < ../Matrix/young1c
	- $(V) ./cs_demo2_cl < ../Matrix/qc324
	- $(V) ./cs_demo2_cl < ../Matrix/neumann
	- $(V) ./cs_demo2_cl < ../Matrix/c4
	- $(V) ./cs_demo3_cl < ../Matrix/c4
	- $(V) ./cs_demo2_cl < ../Matrix/mhd1280b
	- $(V) ./cs_demo3_cl < ../Matrix/mhd1280b
	- $(V) ./cstcov_test_cl ../Matrix/t2 > test_cl_t2.out
	- $(V) ./cstcov_test_cl ../Matrix/young1c > test_cl_young1c.out
	- $(V) ./cstcov_test_cl ../Matrix/qc324 > test_cl_qc324.out
	- $(V) ./cstcov_test_cl ../Matrix/neumann > test_cl_neumann.out
	- $(V) ./cstcov_test_cl ../Matrix/mhd1280b > test_cl_mhd1280b.out

clean:
	- $(RM) *.o *.bbg *.da *.gcov *.gcda *.gcno

purge: distclean

distclean: clean
	- $(RM) *.out *.a cov.sort
	- $(RM) cs_demo1_di cs_demo2_di cs_demo3_di cstcov_test_di
	- $(RM) cs_demo1_dl cs_demo2_dl cs_demo3_dl cstcov_test_dl
	- $(RM) cs_demo1_ci cs_demo2_ci cs_demo3_ci cstcov_test_ci
	- $(RM) cs_demo1_cl cs_demo2_cl cs_demo3_cl cstcov_test_cl
	- $(RM) cs_idemo cs_ldemo
	- $(RM) -r cs_*.c *.dSYM
