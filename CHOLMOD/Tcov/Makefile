#===============================================================================
# CHOLMOD/Tcov/Makefile
#===============================================================================

# If you compile CHOLMOD with -DNPARTITION, then you do not need METIS,
# CCOLAMD, or the Partition module.

include ../../UFconfig/UFconfig.mk
include Make.inc

# Tcov requires gcc
CC = gcc

C = $(CC) $(CFLAGS) $(CHOLMOD_CONFIG) $(NANTESTS)

# optimized LAPACK and BLAS
# LIB = $(METIS) -lm $(LAPACK) $(BLAS) $(XERBLA)

# LAPACK and the Fortran reference BLAS, compiled with gfortran -g -O:
# LIB = $(METIS) -lm -llapack_plain -lblas_plain -lg2c
# LIB = $(METIS) -lm -llapack_plain -lblas_plain -lgfortran -lgfortranbegin
# LIB = $(METIS) -lm -llapack -lblas -lgfortran -lgfortranbegin
LIB = $(METIS) -lm -llapack -lblas_plain -lgfortran -lgfortranbegin -lpthread -lg2c

# Solaris
# LIB = $(METIS) -xlic_lib=sunperf

# Use "grep" only, if you do not have "indent"
PRETTY = grep -v "^\#" | indent -bl -nce -ss -bli0 -i4 -sob -l120
# PRETTY = grep -v "^\#" | ssh persimmon indent -bl -nce -ss -bli0 -i4 -sob -l120
# PRETTY = grep -v "^\#"

I = -I../../AMD/Include -I../../COLAMD/Include \
	-I$(METIS_PATH)/Lib -I../../CCOLAMD/Include -I../../CAMD/Include \
	-I../Include -I../../UFconfig

default: cm cl z_demo l_demo cmread clread

TEST = cm.c test_ops.c null.c null2.c lpdemo.c memory.c solve.c aug.c unpack.c \
	raw_factor.c cctest.c ctest.c amdtest.c camdtest.c huge.c

INC =   ../Include/cholmod.h \
	../Include/cholmod_blas.h \
	../Include/cholmod_check.h \
	../Include/cholmod_cholesky.h \
	../Include/cholmod_complexity.h \
	../Include/cholmod_config.h \
	../Include/cholmod_core.h \
	../Include/cholmod_internal.h \
	../Include/cholmod_matrixops.h \
	../Include/cholmod_modify.h \
	../Include/cholmod_partition.h \
	../Include/cholmod_supernodal.h \
	../Include/cholmod_template.h

AMDSRC = ../../AMD/Source/amd_1.c \
	../../AMD/Source/amd_2.c \
	../../AMD/Source/amd_aat.c \
	../../AMD/Source/amd_control.c \
	../../AMD/Source/amd_defaults.c \
	../../AMD/Source/amd_global.c \
	../../AMD/Source/amd_info.c \
	../../AMD/Source/amd_order.c \
	../../AMD/Source/amd_postorder.c \
	../../AMD/Source/amd_post_tree.c \
	../../AMD/Source/amd_preprocess.c \
	../../AMD/Source/amd_valid.c \
	../../AMD/Include/amd.h \
	../../AMD/Include/amd_internal.h

AMDOBJ = \
	zz_amd_1.o \
	zz_amd_2.o \
	zz_amd_aat.o \
	zz_amd_control.o \
	zz_amd_defaults.o \
	zz_amd_global.o \
	zz_amd_info.o \
	zz_amd_dump.o \
	zz_amd_order.o \
	zz_amd_postorder.o \
	zz_amd_post_tree.o \
	zz_amd_preprocess.o \
	zz_amd_valid.o

LAMDOBJ = \
	zl_amd_1.o \
	zl_amd_2.o \
	zl_amd_aat.o \
	zl_amd_control.o \
	zl_amd_defaults.o \
	zl_amd_global.o \
	zl_amd_info.o \
	zl_amd_dump.o \
	zl_amd_order.o \
	zl_amd_postorder.o \
	zl_amd_post_tree.o \
	zl_amd_preprocess.o \
	zl_amd_valid.o

COLAMDSRC = ../../COLAMD/Source/colamd.c ../../COLAMD/Source/colamd_global.c

COLAMDOBJ = zz_colamd.o yz_colamd_global.o

LCOLAMDOBJ = zl_colamd.o yl_colamd_global.o

#-------------------------------------------------------------------------------
# When using the Partition Module:

CCOLAMDSRC = \
	../../CCOLAMD/Source/ccolamd.c \
	../../CCOLAMD/Source/ccolamd_global.c \
	../../CCOLAMD/Include/ccolamd.h

CCOLAMDOBJ = zz_ccolamd.o yz_ccolamd_global.o

LCCOLAMDOBJ = zl_ccolamd.o yl_ccolamd_global.o

$(CCOLAMDOBJ): $(CCOLAMDSRC)

$(LCCOLAMDOBJ): $(CCOLAMDSRC)

IPARTITION_OBJ = \
	z_ccolamd.o \
	z_csymamd.o \
	z_camd.o \
	z_metis.o \
	z_nesdis.o

LPARTITION_OBJ = \
	l_ccolamd.o \
	l_csymamd.o \
	l_camd.o \
	l_metis.o \
	l_nesdis.o

CAMDSRC = ../../CAMD/Source/camd_1.c \
	../../CAMD/Source/camd_2.c \
	../../CAMD/Source/camd_aat.c \
	../../CAMD/Source/camd_control.c \
	../../CAMD/Source/camd_defaults.c \
	../../CAMD/Source/camd_global.c \
	../../CAMD/Source/camd_info.c \
	../../CAMD/Source/camd_order.c \
	../../CAMD/Source/camd_postorder.c \
	../../CAMD/Source/camd_preprocess.c \
	../../CAMD/Source/camd_valid.c \
	../../CAMD/Include/camd.h \
	../../CAMD/Include/camd_internal.h

CAMDOBJ = \
	zz_camd_1.o \
	zz_camd_2.o \
	zz_camd_aat.o \
	zz_camd_control.o \
	zz_camd_defaults.o \
	zz_camd_global.o \
	zz_camd_info.o \
	zz_camd_order.o \
	zz_camd_postorder.o \
	zz_camd_preprocess.o \
	zz_camd_valid.o \
	zz_camd_dump.o

LCAMDOBJ = \
	zl_camd_1.o \
	zl_camd_2.o \
	zl_camd_aat.o \
	zl_camd_control.o \
	zl_camd_defaults.o \
	zl_camd_global.o \
	zl_camd_info.o \
	zl_camd_order.o \
	zl_camd_postorder.o \
	zl_camd_preprocess.o \
	zl_camd_valid.o \
	zl_camd_dump.o

$(CAMDOBJ): $(CAMDSRC)

$(LCAMDOBJ): $(CAMDSRC)

#-------------------------------------------------------------------------------
# If you compile with -DNPARTITION, you may replace the above definitions
# with empty ones (see immediately below), and then you do not need a copy of
# CCOLAMD:
# CCOLAMDSRC =
# CCOLAMDOBJ = 
# LCCOLAMDOBJ =
# IPARTITION_OBJ = 
# LPARTITION_OBJ = 
# CAMDSRC = 
# CAMDOBJ = 
# LCAMDOBJ = 
#-------------------------------------------------------------------------------

IOBJ = \
	z_common.o \
	z_dense.o \
	z_factor.o \
	z_change_factor.o \
	z_memory.o \
	z_sparse.o \
	z_complex.o \
	z_transpose.o \
	z_band.o \
	z_copy.o \
	z_triplet.o \
	z_error.o \
	z_aat.o \
	z_add.o \
	z_check.o \
	z_read.o \
	z_write.o \
	z_amd.o \
	z_analyze.o \
	z_colamd.o \
	z_etree.o \
	z_factorize.o \
	z_postorder.o \
	z_rcond.o \
	z_resymbol.o \
	z_rowcolcounts.o \
	z_rowfac.o \
	z_solve.o \
	z_spsolve.o \
	z_drop.o \
	z_horzcat.o \
	z_norm.o \
	z_scale.o \
	z_sdmult.o \
	z_ssmult.o \
	z_submatrix.o \
	z_vertcat.o \
	z_symmetry.o \
	z_rowadd.o \
	z_rowdel.o \
	z_updown.o \
	z_super_numeric.o \
	z_super_solve.o \
	z_super_symbolic.o \
	$(IPARTITION_OBJ)

LOBJ = \
	l_common.o \
	l_dense.o \
	l_factor.o \
	l_change_factor.o \
	l_memory.o \
	l_sparse.o \
	l_complex.o \
	l_transpose.o \
	l_band.o \
	l_copy.o \
	l_triplet.o \
	l_error.o \
	l_aat.o \
	l_add.o \
	l_check.o \
	l_read.o \
	l_write.o \
	l_amd.o \
	l_analyze.o \
	l_colamd.o \
	l_etree.o \
	l_factorize.o \
	l_postorder.o \
	l_rcond.o \
	l_resymbol.o \
	l_rowcolcounts.o \
	l_rowfac.o \
	l_solve.o \
	l_spsolve.o \
	l_drop.o \
	l_horzcat.o \
	l_norm.o \
	l_scale.o \
	l_sdmult.o \
	l_ssmult.o \
	l_submatrix.o \
	l_vertcat.o \
	l_symmetry.o \
	l_rowadd.o \
	l_rowdel.o \
	l_updown.o \
	l_super_numeric.o \
	l_super_solve.o \
	l_super_symbolic.o \
	$(LPARTITION_OBJ)

IALL = $(IOBJ) $(AMDOBJ)  $(COLAMDOBJ)  $(CCOLAMDOBJ)  $(CAMDOBJ)

LALL = $(LOBJ) $(LAMDOBJ) $(LCOLAMDOBJ) $(LCCOLAMDOBJ) $(LCAMDOBJ)


cm: $(IALL) $(TEST) cm.h Makefile
	$(C) $(I) $(TEST) -o cm $(IALL) $(LIB)

cl: $(LALL) $(TEST) cm.h Makefile
	$(C) -DDLONG $(I) $(TEST) -o cl $(LALL) $(LIB)

cmread: $(IALL) cmread.c Makefile
	$(C) $(I) cmread.c -o cmread $(IALL) $(LIB)

clread: $(LALL) cmread.c Makefile
	$(C) -DDLONG $(I) cmread.c -o clread $(LALL) $(LIB)

z_demo: $(IALL) ../Demo/cholmod_demo.c cm.h Makefile \
	../Demo/cholmod_demo.h
	cat ../Demo/cholmod_demo.c > z_demo.c
	$(C) $(I) -I../Demo z_demo.c -o z_demo $(IALL) $(LIB)

l_demo: $(LALL) ../Demo/cholmod_l_demo.c cm.h Makefile \
	../Demo/cholmod_demo.h
	cat ../Demo/cholmod_l_demo.c > l_demo.c
	$(C) -DDLONG $(I) -I../Demo l_demo.c -o l_demo $(LALL) $(LIB)

go: z_demo l_demo cmread clread cm cl
	$(V) ./z_demo   ../Demo/Matrix/bcsstk01.tri > tmp/demo_k1.out
	$(V) ./z_demo   ../Demo/Matrix/bcsstk02.tri > tmp/demo_k2.out
	$(V) ./z_demo < ../Demo/Matrix/lp_afiro.tri > tmp/demo_afiro.out
	$(V) ./z_demo < ../Demo/Matrix/can___24.mtx > tmp/demo_can24.out
	$(V) ./z_demo < ../Demo/Matrix/c.tri        > tmp/demo_c.out
	$(V) ./z_demo < ../Demo/Matrix/d.tri        > tmp/demo_d.out
	$(V) ./z_demo < ../Demo/Matrix/up.tri       > tmp/demo_up.out
	$(V) ./z_demo < ../Demo/Matrix/c.mtx        > tmp/demo_c_mtx.out
	$(V) ./z_demo < ../Demo/Matrix/0.tri        > tmp/demo_0.out
	$(V) ./z_demo < Matrix/3_2                  > tmp/demo_3_2.out
	$(V) ./z_demo < Matrix/c5lo                 > tmp/demo_c5lo.out
	$(V) ./z_demo < Matrix/c10                  > tmp/demo_c10.out
	$(V) ./z_demo no_such_file                  > tmp/demo_no_such_file.out
	$(V) ./z_demo ../Demo/Matrix/mangle1.mtx    > tmp/demo_mangle1.out
	$(V) ./z_demo ../Demo/Matrix/mangle2.mtx    > tmp/demo_mangle2.out
	$(V) ./z_demo ../Demo/Matrix/mangle3.mtx    > tmp/demo_mangle3.out
	$(V) ./z_demo ../Demo/Matrix/mangle4.mtx    > tmp/demo_mangle4.out
	$(V) ./z_demo ../Demo/Matrix/pts5ldd03.mtx  >  tmp/demo_pts5ldd03.out
	$(V) ./l_demo   ../Demo/Matrix/bcsstk01.tri > tmp/ldemo_k1.out
	$(V) ./l_demo   ../Demo/Matrix/bcsstk02.tri > tmp/ldemo_k2.out
	$(V) ./l_demo < ../Demo/Matrix/lp_afiro.tri > tmp/ldemo_afiro.out
	$(V) ./l_demo < ../Demo/Matrix/can___24.mtx > tmp/ldemo_can24.out
	$(V) ./l_demo < ../Demo/Matrix/c.tri        > tmp/ldemo_c.out
	$(V) ./l_demo < ../Demo/Matrix/d.tri        > tmp/ldemo_d.out
	$(V) ./l_demo < ../Demo/Matrix/up.tri       > tmp/ldemo_up.out
	$(V) ./l_demo < ../Demo/Matrix/c.mtx        > tmp/ldemo_c_mtx.out
	$(V) ./l_demo < ../Demo/Matrix/0.tri        > tmp/ldemo_0.out
	$(V) ./l_demo < Matrix/3_2                  > tmp/ldemo_3_2.out
	$(V) ./l_demo < Matrix/c5lo                 > tmp/ldemo_c5lo.out
	$(V) ./l_demo < Matrix/c10                  > tmp/ldemo_c10.out
	$(V) ./l_demo no_such_file                  > tmp/ldemo_no_such_file.out
	$(V) ./l_demo ../Demo/Matrix/mangle1.mtx    > tmp/ldemo_mangle1.out
	$(V) ./l_demo ../Demo/Matrix/mangle2.mtx    > tmp/ldemo_mangle2.out
	$(V) ./l_demo ../Demo/Matrix/mangle3.mtx    > tmp/ldemo_mangle3.out
	$(V) ./l_demo ../Demo/Matrix/mangle4.mtx    > tmp/ldemo_mangle4.out
	$(V) ./l_demo ../Demo/Matrix/pts5ldd03.mtx  > tmp/ldemo_pts5ldd03.out
	- grep resid tmp/demo*
	$(V) ./cmread no_such_file                  > tmp/no_such_file.out
	$(V) ./cmread Matrix/crud1                  > tmp/crud1.out
	$(V) ./cmread Matrix/crud2                  > tmp/crud2.out
	$(V) ./cmread Matrix/fullcrud.mtx           > tmp/fullcrud.out
	$(V) ./cmread Matrix/fullcrud1.mtx          > tmp/fullcrud1.out
	$(V) ./cmread Matrix/fullcrud2.mtx          > tmp/fullcrud2.out
	$(V) ./cmread Matrix/3by0.mtx               > tmp/3by0.out
	$(V) ./cmread Matrix/fullrza.mtx            > tmp/fullrza.out
	$(V) ./cmread Matrix/fullrsa.mtx            > tmp/fullrsa.out
	$(V) ./cmread Matrix/fullcsa.mtx            > tmp/fullcsa.out
	$(V) ./cmread Matrix/fullcza.mtx            > tmp/fullcza.out
	$(V) ./cmread Matrix/fullcha.mtx            > tmp/fullcha.out
	$(V) ./cmread Matrix/cha.mtx		       > tmp/cha.out
	$(V) ./cmread Matrix/cza.mtx		       > tmp/cza.out
	$(V) ./cmread Matrix/csa.mtx		       > tmp/csa.out
	$(V) ./cmread Matrix/one                    > tmp/one.out
	$(V) ./cmread Matrix/rza.mtx                > tmp/rza.out
	$(V) ./cmread ../Demo/Matrix/mangle5.tri    > tmp/mangle5.out
	$(V) ./cmread ../Demo/Matrix/mangle6.tri    > tmp/mangle6.out
	$(V) ./cmread ../Demo/Matrix/mangle7.tri    > tmp/mangle6.out
	$(V) ./cmread ../Demo/Matrix/mangle8.tri    > tmp/mangle8.out
	$(V) ./cmread ../Demo/Matrix/empty.tri      > tmp/empty.out
	$(V) ./cmread ../Demo/Matrix/one.tri        > tmp/one.out
	$(V) ./cmread Matrix/plskz362.mtx           > tmp/plskz363.out
	$(V) ./cmread Matrix/2diag.tri	       > tmp/2diag.out
	$(V) ./cmread Matrix/r5lo    	       > tmp/r5lo.out
	$(V) ./cmread Matrix/r5lo2    	       > tmp/r5lo2.out
	- diff tmp/r5lo.out tmp/r5lo2.out
	$(V) ./cmread Matrix/cs.mtx	               > tmp/cs.out
	$(V) ./cmread Matrix/2lo.tri	               > tmp/2lo.out
	$(V) ./cmread Matrix/2.tri	               > tmp/2.out
	$(V) ./cmread Matrix/2up.tri	               > tmp/2up.out
	$(V) ./cmread Matrix/huge.tri	       > tmp/huge.out
	$(V) ./cmread Matrix/1e99                   > tmp/1e99.out
	$(V) ./clread no_such_file                  > tmp/l_no_such_file.out
	$(V) ./clread Matrix/crud1                  > tmp/l_crud1.out
	$(V) ./clread Matrix/crud2                  > tmp/l_crud2.out
	$(V) ./clread Matrix/fullcrud.mtx           > tmp/l_fullcrud.out
	$(V) ./clread Matrix/fullcrud1.mtx          > tmp/l_fullcrud1.out
	$(V) ./clread Matrix/fullcrud2.mtx          > tmp/l_fullcrud2.out
	$(V) ./clread Matrix/3by0.mtx               > tmp/l_3by0.out
	$(V) ./clread Matrix/fullrza.mtx            > tmp/l_fullrza.out
	$(V) ./clread Matrix/fullrsa.mtx            > tmp/l_fullrsa.out
	$(V) ./clread Matrix/fullcsa.mtx            > tmp/l_fullcsa.out
	$(V) ./clread Matrix/fullcza.mtx            > tmp/l_fullcza.out
	$(V) ./clread Matrix/fullcha.mtx            > tmp/l_fullcha.out
	$(V) ./clread Matrix/cha.mtx		       > tmp/l_cha.out
	$(V) ./clread Matrix/cza.mtx		       > tmp/l_cza.out
	$(V) ./clread Matrix/csa.mtx		       > tmp/l_csa.out
	$(V) ./clread Matrix/one		       > tmp/l_one.out
	$(V) ./clread Matrix/rza.mtx                > tmp/l_rza.out
	$(V) ./clread ../Demo/Matrix/mangle5.tri    > tmp/l_mangle5.out
	$(V) ./clread ../Demo/Matrix/mangle6.tri    > tmp/l_mangle6.out
	$(V) ./clread ../Demo/Matrix/mangle7.tri    > tmp/l_mangle6.out
	$(V) ./clread ../Demo/Matrix/mangle8.tri    > tmp/l_mangle8.out
	$(V) ./clread ../Demo/Matrix/empty.tri      > tmp/l_empty.out
	$(V) ./clread ../Demo/Matrix/one.tri        > tmp/l_one.out
	$(V) ./clread Matrix/plskz362.mtx           > tmp/l_plskz363.out
	$(V) ./clread Matrix/2diag.tri	       > tmp/l_2diag.out
	$(V) ./clread Matrix/r5lo    	       > tmp/l_r5lo.out
	$(V) ./clread Matrix/r5lo2    	       > tmp/l_r5lo2.out
	- diff tmp/r5lo.out tmp/r5lo2.out
	$(V) ./clread Matrix/cs.mtx	               > tmp/l_cs.out
	$(V) ./clread Matrix/2lo.tri	               > tmp/l_l_2lo.out
	$(V) ./clread Matrix/2.tri	               > tmp/l_2.out
	$(V) ./clread Matrix/2up.tri	               > tmp/l_2up.out
	$(V) ./clread Matrix/huge.tri               > tmp/l_huge.out
	$(V) ./clread Matrix/1e99                   > tmp/l_1e99.out
	$(V) ./cm    < Matrix/galenet > tmp/galenet.out
	$(V) ./cl    < Matrix/galenet > tmp/l_galenet.out
	- $(COVER)
	$(V) ./cm    < Matrix/5by50   > tmp/5by50.out
	$(V) ./cl    < Matrix/5by50   > tmp/l_5by50.out
	- $(COVER)
	$(V) ./cm    < Matrix/r5lo    > tmp/r5lo.out 
	$(V) ./cl    < Matrix/r5lo    > tmp/l_r5lo.out
	$(V) ./cm    < Matrix/r5up    > tmp/r5up.out
	$(V) ./cl    < Matrix/r5up    > tmp/l_r5up.out
	$(V) ./cm    < Matrix/r5up2   > tmp/r5up2.out
	$(V) ./cl    < Matrix/r5up2   > tmp/l_r5up2.out
	$(V) ./cm    < Matrix/c5up2   > tmp/c5up2.out
	$(V) ./cl    < Matrix/c5up2   > tmp/l_c5up2.out
	$(V) ./cm    < Matrix/z5up2   > tmp/z5up2.out
	$(V) ./cl    < Matrix/z5up2   > tmp/l_z5up2.out
	$(V) ./cm -m < Matrix/z5lo    > tmp/z5lo.out
	$(V) ./cl -m < Matrix/z5lo    > tmp/l_z5lo.out
	$(V) ./cm    < Matrix/ibm32   > tmp/ibm.out
	$(V) ./cl    < Matrix/ibm32   > tmp/l_ibm.out
	- $(COVER)
	$(V) ./cm -m < Matrix/c5lo    > tmp/c5lo.out
	$(V) ./cl -m < Matrix/c5lo    > tmp/l_c5lo.out
	$(V) ./cm -m < Matrix/z10     > tmp/z10.out
	$(V) ./cl -m < Matrix/z10     > tmp/l_z10.out
	$(V) ./cm -m < Matrix/z5up    > tmp/z5up.out
	$(V) ./cl -m < Matrix/z5up    > tmp/l_z5up.out
	- $(COVER)
	$(V) ./cm -s < Matrix/3singular  > tmp/3singular.out
	$(V) ./cl -s < Matrix/3singular  > tmp/l_3singular.out
	$(V) ./cm -s < Matrix/z3singular > tmp/z3singular.out
	$(V) ./cl -s < Matrix/z3singular > tmp/l_z3singular.out
	$(V) ./cm -s < Matrix/c3singular > tmp/c3singular.out
	$(V) ./cl -s < Matrix/c3singular > tmp/l_c3singular.out
	$(V) ./cm -m < Matrix/0       > tmp/0.out
	$(V) ./cl -m < Matrix/0       > tmp/l_0.out
	$(V) ./cm -m < Matrix/afiro   > tmp/afiro.out
	$(V) ./cl -m < Matrix/afiro   > tmp/l_afiro.out
	- $(COVER)
	$(V) ./cm -m < Matrix/k01up   > tmp/k01up.out
	$(V) ./cl -m < Matrix/k01up   > tmp/l_k01up.out
	- $(COVER)
	$(V) ./cm    < Matrix/diag    > tmp/diag.out
	$(V) ./cl    < Matrix/diag    > tmp/l_diag.out
	$(V) ./cm -m < Matrix/ex5lo   > tmp/ex5lo.out
	$(V) ./cl -m < Matrix/ex5lo   > tmp/l_ex5lo.out
	- $(COVER)
	$(V) ./cm    < Matrix/20lo    > tmp/20lo.out
	$(V) ./cl    < Matrix/20lo    > tmp/l_20lo.out
	$(V) ./cm    < Matrix/z30lo   > tmp/z30lo.out
	$(V) ./cl    < Matrix/z30lo   > tmp/l_z30lo.out
	- $(COVER)
	$(V) ./cm -m < Matrix/z30up   > tmp/z30up.out
	$(V) ./cl -m < Matrix/z30up   > tmp/l_z30up.out
	$(V) ./cm    < Matrix/c10     > tmp/c10.out
	$(V) ./cl    < Matrix/c10     > tmp/l_c10.out
	$(V) ./cm    < Matrix/c30lo   > tmp/c30lo.out
	$(V) ./cl    < Matrix/c30lo   > tmp/l_c30lo.out
	- $(COVER)
	$(V) ./cm -m < Matrix/c30up   > tmp/c30up.out
	$(V) ./cl -m < Matrix/c30up   > tmp/l_c30up.out
	$(V) ./cm    < Matrix/pi      > tmp/pi.out
	$(V) ./cl    < Matrix/pi      > tmp/l_pi.out
	$(V) ./cm    < Matrix/cpi     > tmp/cpi.out
	$(V) ./cl    < Matrix/cpi     > tmp/l_cpi.out
	$(V) ./cm    < Matrix/1_0     > tmp/1_0.out
	$(V) ./cl    < Matrix/1_0     > tmp/l_1_0.out
	$(V) ./cm -s < Matrix/3b      > tmp/3b.out
	$(V) ./cl -s < Matrix/3b      > tmp/l_3b.out
	$(V) ./cm -s < Matrix/cza     > tmp/cza2.out
	$(V) ./cl -s < Matrix/cza     > tmp/l_cza2.out
	$(V) ./cm    < Matrix/0_1     > tmp/0_1.out
	$(V) ./cl    < Matrix/0_1     > tmp/l_0_1.out
	- $(COVER)
	$(V) ./cm -n < Matrix/galenet > tmp/galenet_nan.out
	- $(COVER)
	$(V) ./cl -n < Matrix/galenet > tmp/l_galenet_nan.out
	- $(COVER)
	$(V) ./cm    < Matrix/a1      > tmp/a1.out
	- $(COVER)
	$(V) ./cl    < Matrix/a1      > tmp/l_a1.out
	- $(COVER)
	$(V) ./cm    < Matrix/zero    > tmp/zero.out
	$(V) ./cl    < Matrix/zero    > tmp/zero.out
	- $(COVER)

cov:
	- $(COVER)

purge: distclean

distclean: clean
	- $(RM) cm cl cmread clread *.c.gcov *.out tmp/*.out z_demo l_demo
	- $(RM) leak zz_*.c z_*.c *.a l_*.c zl_*.c cov.sort yl_*.c yz_*.c
	- $(RM) -r cm.profile cmread.profile z_demo.profile 
	- $(RM) -r cl.profile clread.profile l_demo.profile 
	- $(RM) *.gcda *.gcno
	- $(RM) temp*.mtx

clean:
	- $(RM) $(CLEAN)


$(AMDOBJ): $(AMDSRC)

$(LAMDOBJ): $(AMDSRC)

$(IOBJ): $(INC)

$(LOBJ): $(INC)

.c.o:
	$(C) -c $(I) $*.c

#-------------------------------------------------------------------------------
# AMD
#-------------------------------------------------------------------------------

zz_amd_1.o: ../../AMD/Source/amd_1.c
	$(C) -E $(I) $< | $(PRETTY) > zz_amd_1.c
	$(C) -c $(I) zz_amd_1.c

zz_amd_2.o: ../../AMD/Source/amd_2.c
	$(C) -E $(I) $< | $(PRETTY) > zz_amd_2.c
	$(C) -c $(I) zz_amd_2.c

zz_amd_aat.o: ../../AMD/Source/amd_aat.c
	$(C) -E $(I) $< | $(PRETTY) > zz_amd_aat.c
	$(C) -c $(I) zz_amd_aat.c

zz_amd_control.o: ../../AMD/Source/amd_control.c
	$(C) -E $(I) $< | $(PRETTY) > zz_amd_control.c
	$(C) -c $(I) zz_amd_control.c

zz_amd_defaults.o: ../../AMD/Source/amd_defaults.c
	$(C) -E $(I) $< | $(PRETTY) > zz_amd_defaults.c
	$(C) -c $(I) zz_amd_defaults.c

zz_amd_global.o: ../../AMD/Source/amd_global.c
	$(C) -E $(I) $< | $(PRETTY) > zz_amd_global.c
	$(C) -c $(I) zz_amd_global.c

zz_amd_dump.o: ../../AMD/Source/amd_dump.c
	$(C) -E $(I) $< | $(PRETTY) > zz_amd_dump.c
	$(C) -c $(I) zz_amd_dump.c

zz_amd_info.o: ../../AMD/Source/amd_info.c
	$(C) -E $(I) $< | $(PRETTY) > zz_amd_info.c
	$(C) -c $(I) zz_amd_info.c

zz_amd_order.o: ../../AMD/Source/amd_order.c
	$(C) -E $(I) $< | $(PRETTY) > zz_amd_order.c
	$(C) -c $(I) zz_amd_order.c

zz_amd_postorder.o: ../../AMD/Source/amd_postorder.c
	$(C) -E $(I) $< | $(PRETTY) > zz_amd_postorder.c
	$(C) -c $(I) zz_amd_postorder.c

zz_amd_post_tree.o: ../../AMD/Source/amd_post_tree.c
	$(C) -E $(I) $< | $(PRETTY) > zz_amd_post_tree.c
	$(C) -c $(I) zz_amd_post_tree.c

zz_amd_preprocess.o: ../../AMD/Source/amd_preprocess.c
	$(C) -E $(I) $< | $(PRETTY) > zz_amd_preprocess.c
	$(C) -c $(I) zz_amd_preprocess.c

zz_amd_valid.o: ../../AMD/Source/amd_valid.c
	$(C) -E $(I) $< | $(PRETTY) > zz_amd_valid.c
	$(C) -c $(I) zz_amd_valid.c

#-------------------------------------------------------------------------------

zl_amd_1.o: ../../AMD/Source/amd_1.c
	$(C) -DDLONG -E $(I) $< | $(PRETTY) > zl_amd_1.c
	$(C) -c $(I) zl_amd_1.c

zl_amd_2.o: ../../AMD/Source/amd_2.c
	$(C) -DDLONG -E $(I) $< | $(PRETTY) > zl_amd_2.c
	$(C) -c $(I) zl_amd_2.c

zl_amd_aat.o: ../../AMD/Source/amd_aat.c
	$(C) -DDLONG -E $(I) $< | $(PRETTY) > zl_amd_aat.c
	$(C) -c $(I) zl_amd_aat.c

zl_amd_control.o: ../../AMD/Source/amd_control.c
	$(C) -DDLONG -E $(I) $< | $(PRETTY) > zl_amd_control.c
	$(C) -c $(I) zl_amd_control.c

zl_amd_defaults.o: ../../AMD/Source/amd_defaults.c
	$(C) -DDLONG -E $(I) $< | $(PRETTY) > zl_amd_defaults.c
	$(C) -c $(I) zl_amd_defaults.c

zl_amd_global.o: ../../AMD/Source/amd_global.c
	$(C) -DDLONG -E $(I) $< | $(PRETTY) > zl_amd_global.c
	$(C) -c $(I) zl_amd_global.c

zl_amd_dump.o: ../../AMD/Source/amd_dump.c
	$(C) -DDLONG -E $(I) $< | $(PRETTY) > zl_amd_dump.c
	$(C) -c $(I) zl_amd_dump.c

zl_amd_info.o: ../../AMD/Source/amd_info.c
	$(C) -DDLONG -E $(I) $< | $(PRETTY) > zl_amd_info.c
	$(C) -c $(I) zl_amd_info.c

zl_amd_order.o: ../../AMD/Source/amd_order.c
	$(C) -DDLONG -E $(I) $< | $(PRETTY) > zl_amd_order.c
	$(C) -c $(I) zl_amd_order.c

zl_amd_postorder.o: ../../AMD/Source/amd_postorder.c
	$(C) -DDLONG -E $(I) $< | $(PRETTY) > zl_amd_postorder.c
	$(C) -c $(I) zl_amd_postorder.c

zl_amd_post_tree.o: ../../AMD/Source/amd_post_tree.c
	$(C) -DDLONG -E $(I) $< | $(PRETTY) > zl_amd_post_tree.c
	$(C) -c $(I) zl_amd_post_tree.c

zl_amd_preprocess.o: ../../AMD/Source/amd_preprocess.c
	$(C) -DDLONG -E $(I) $< | $(PRETTY) > zl_amd_preprocess.c
	$(C) -c $(I) zl_amd_preprocess.c

zl_amd_valid.o: ../../AMD/Source/amd_valid.c
	$(C) -DDLONG -E $(I) $< | $(PRETTY) > zl_amd_valid.c
	$(C) -c $(I) zl_amd_valid.c

#-------------------------------------------------------------------------------
# CAMD
#-------------------------------------------------------------------------------

zz_camd_1.o: ../../CAMD/Source/camd_1.c
	$(C) -E $(I) $< | $(PRETTY) > zz_camd_1.c
	$(C) -c $(I) zz_camd_1.c

zz_camd_2.o: ../../CAMD/Source/camd_2.c
	$(C) -E $(I) $< | $(PRETTY) > zz_camd_2.c
	$(C) -c $(I) zz_camd_2.c

zz_camd_aat.o: ../../CAMD/Source/camd_aat.c
	$(C) -E $(I) $< | $(PRETTY) > zz_camd_aat.c
	$(C) -c $(I) zz_camd_aat.c

zz_camd_control.o: ../../CAMD/Source/camd_control.c
	$(C) -E $(I) $< | $(PRETTY) > zz_camd_control.c
	$(C) -c $(I) zz_camd_control.c

zz_camd_defaults.o: ../../CAMD/Source/camd_defaults.c
	$(C) -E $(I) $< | $(PRETTY) > zz_camd_defaults.c
	$(C) -c $(I) zz_camd_defaults.c

zz_camd_global.o: ../../CAMD/Source/camd_global.c
	$(C) -E $(I) $< | $(PRETTY) > zz_camd_global.c
	$(C) -c $(I) zz_camd_global.c

zz_camd_dump.o: ../../CAMD/Source/camd_dump.c
	$(C) -E $(I) $< | $(PRETTY) > zz_camd_dump.c
	$(C) -c $(I) zz_camd_dump.c

zz_camd_info.o: ../../CAMD/Source/camd_info.c
	$(C) -E $(I) $< | $(PRETTY) > zz_camd_info.c
	$(C) -c $(I) zz_camd_info.c

zz_camd_order.o: ../../CAMD/Source/camd_order.c
	$(C) -E $(I) $< | $(PRETTY) > zz_camd_order.c
	$(C) -c $(I) zz_camd_order.c

zz_camd_postorder.o: ../../CAMD/Source/camd_postorder.c
	$(C) -E $(I) $< | $(PRETTY) > zz_camd_postorder.c
	$(C) -c $(I) zz_camd_postorder.c

zz_camd_preprocess.o: ../../CAMD/Source/camd_preprocess.c
	$(C) -E $(I) $< | $(PRETTY) > zz_camd_preprocess.c
	$(C) -c $(I) zz_camd_preprocess.c

zz_camd_valid.o: ../../CAMD/Source/camd_valid.c
	$(C) -E $(I) $< | $(PRETTY) > zz_camd_valid.c
	$(C) -c $(I) zz_camd_valid.c

#-------------------------------------------------------------------------------

zl_camd_1.o: ../../CAMD/Source/camd_1.c
	$(C) -DDLONG -E $(I) $< | $(PRETTY) > zl_camd_1.c
	$(C) -c $(I) zl_camd_1.c

zl_camd_2.o: ../../CAMD/Source/camd_2.c
	$(C) -DDLONG -E $(I) $< | $(PRETTY) > zl_camd_2.c
	$(C) -c $(I) zl_camd_2.c

zl_camd_aat.o: ../../CAMD/Source/camd_aat.c
	$(C) -DDLONG -E $(I) $< | $(PRETTY) > zl_camd_aat.c
	$(C) -c $(I) zl_camd_aat.c

zl_camd_control.o: ../../CAMD/Source/camd_control.c
	$(C) -DDLONG -E $(I) $< | $(PRETTY) > zl_camd_control.c
	$(C) -c $(I) zl_camd_control.c

zl_camd_defaults.o: ../../CAMD/Source/camd_defaults.c
	$(C) -DDLONG -E $(I) $< | $(PRETTY) > zl_camd_defaults.c
	$(C) -c $(I) zl_camd_defaults.c

zl_camd_global.o: ../../CAMD/Source/camd_global.c
	$(C) -DDLONG -E $(I) $< | $(PRETTY) > zl_camd_global.c
	$(C) -c $(I) zl_camd_global.c

zl_camd_dump.o: ../../CAMD/Source/camd_dump.c
	$(C) -DDLONG -E $(I) $< | $(PRETTY) > zl_camd_dump.c
	$(C) -c $(I) zl_camd_dump.c

zl_camd_info.o: ../../CAMD/Source/camd_info.c
	$(C) -DDLONG -E $(I) $< | $(PRETTY) > zl_camd_info.c
	$(C) -c $(I) zl_camd_info.c

zl_camd_order.o: ../../CAMD/Source/camd_order.c
	$(C) -DDLONG -E $(I) $< | $(PRETTY) > zl_camd_order.c
	$(C) -c $(I) zl_camd_order.c

zl_camd_postorder.o: ../../CAMD/Source/camd_postorder.c
	$(C) -DDLONG -E $(I) $< | $(PRETTY) > zl_camd_postorder.c
	$(C) -c $(I) zl_camd_postorder.c

zl_camd_preprocess.o: ../../CAMD/Source/camd_preprocess.c
	$(C) -DDLONG -E $(I) $< | $(PRETTY) > zl_camd_preprocess.c
	$(C) -c $(I) zl_camd_preprocess.c

zl_camd_valid.o: ../../CAMD/Source/camd_valid.c
	$(C) -DDLONG -E $(I) $< | $(PRETTY) > zl_camd_valid.c
	$(C) -c $(I) zl_camd_valid.c

#-------------------------------------------------------------------------------

zz_colamd.o: ../../COLAMD/Source/colamd.c
	$(C) -E $(I) $< | $(PRETTY) > zz_colamd.c
	$(C) -c $(I) zz_colamd.c

yz_colamd_global.o: ../../COLAMD/Source/colamd_global.c
	$(C) -E $(I) $< | $(PRETTY) > yz_colamd_global.c
	$(C) -c $(I) yz_colamd_global.c

zl_colamd.o: ../../COLAMD/Source/colamd.c
	$(C) -DDLONG -E $(I) $< | $(PRETTY) > zl_colamd.c
	$(C) -c $(I) zl_colamd.c

yl_colamd_global.o: ../../COLAMD/Source/colamd_global.c
	$(C) -DDLONG -E $(I) $< | $(PRETTY) > yl_colamd_global.c
	$(C) -c $(I) yl_colamd_global.c

#-------------------------------------------------------------------------------

zz_ccolamd.o: ../../CCOLAMD/Source/ccolamd.c
	$(C) -E $(I) $< | $(PRETTY) > zz_ccolamd.c
	$(C) -c $(I) zz_ccolamd.c

yz_ccolamd_global.o: ../../CCOLAMD/Source/ccolamd_global.c
	$(C) -E $(I) $< | $(PRETTY) > yz_ccolamd_global.c
	$(C) -c $(I) yz_ccolamd_global.c

zl_ccolamd.o: ../../CCOLAMD/Source/ccolamd.c
	$(C) -DDLONG -E $(I) $< | $(PRETTY) > zl_ccolamd.c
	$(C) -c $(I) zl_ccolamd.c

yl_ccolamd_global.o: ../../CCOLAMD/Source/ccolamd_global.c
	$(C) -DDLONG -E $(I) $< | $(PRETTY) > yl_ccolamd_global.c
	$(C) -c $(I) yl_ccolamd_global.c

#-------------------------------------------------------------------------------

z_check.o: ../Check/cholmod_check.c
	$(C) -E $(I) $< | $(PRETTY) > z_check.c
	$(C) -c $(I) z_check.c

z_read.o: ../Check/cholmod_read.c
	$(C) -E $(I) $< | $(PRETTY) > z_read.c
	$(C) -c $(I) z_read.c

z_write.o: ../Check/cholmod_write.c
	$(C) -E $(I) $< | $(PRETTY) > z_write.c
	$(C) -c $(I) z_write.c

#-------------------------------------------------------------------------------

z_common.o: ../Core/cholmod_common.c
	$(C) -E $(I) $< | $(PRETTY) > z_common.c
	$(C) -c $(I) z_common.c

z_dense.o: ../Core/cholmod_dense.c ../Core/t_cholmod_dense.c
	$(C) -E $(I) $< | $(PRETTY) > z_dense.c
	$(C) -c $(I) z_dense.c

z_factor.o: ../Core/cholmod_factor.c
	$(C) -E $(I) $< | $(PRETTY) > z_factor.c
	$(C) -c $(I) z_factor.c

z_change_factor.o: ../Core/cholmod_change_factor.c \
	../Core/t_cholmod_change_factor.c
	$(C) -E $(I) $< | $(PRETTY) > z_change_factor.c
	$(C) -c $(I) z_change_factor.c

z_memory.o: ../Core/cholmod_memory.c
	$(C) -E $(I) $< | $(PRETTY) > z_memory.c
	$(C) -c $(I) z_memory.c

z_sparse.o: ../Core/cholmod_sparse.c
	$(C) -E $(I) $< | $(PRETTY) > z_sparse.c
	$(C) -c $(I) z_sparse.c

z_complex.o: ../Core/cholmod_complex.c
	$(C) -E $(I) $< | $(PRETTY) > z_complex.c
	$(C) -c $(I) z_complex.c

z_transpose.o: ../Core/cholmod_transpose.c ../Core/t_cholmod_transpose.c
	$(C) -E $(I) $< | $(PRETTY) > z_transpose.c
	$(C) -c $(I) z_transpose.c

z_band.o: ../Core/cholmod_band.c
	$(C) -E $(I) $< | $(PRETTY) > z_band.c
	$(C) -c $(I) z_band.c

z_copy.o: ../Core/cholmod_copy.c
	$(C) -E $(I) $< | $(PRETTY) > z_copy.c
	$(C) -c $(I) z_copy.c

z_triplet.o: ../Core/cholmod_triplet.c ../Core/t_cholmod_triplet.c
	$(C) -E $(I) $< | $(PRETTY) > z_triplet.c
	$(C) -c $(I) z_triplet.c

z_error.o: ../Core/cholmod_error.c
	$(C) -E $(I) $< | $(PRETTY) > z_error.c
	$(C) -c $(I) z_error.c

z_aat.o: ../Core/cholmod_aat.c
	$(C) -E $(I) $< | $(PRETTY) > z_aat.c
	$(C) -c $(I) z_aat.c

z_add.o: ../Core/cholmod_add.c
	$(C) -E $(I) $< | $(PRETTY) > z_add.c
	$(C) -c $(I) z_add.c

#-------------------------------------------------------------------------------

z_amd.o: ../Cholesky/cholmod_amd.c
	$(C) -E $(I) $< | $(PRETTY) > z_amd.c
	$(C) -c $(I) z_amd.c

z_analyze.o: ../Cholesky/cholmod_analyze.c
	$(C) -E $(I) $< | $(PRETTY) > z_analyze.c
	$(C) -c $(I) z_analyze.c

z_colamd.o: ../Cholesky/cholmod_colamd.c
	$(C) -E $(I) $< | $(PRETTY) > z_colamd.c
	$(C) -c $(I) z_colamd.c

z_etree.o: ../Cholesky/cholmod_etree.c
	$(C) -E $(I) $< | $(PRETTY) > z_etree.c
	$(C) -c $(I) z_etree.c

z_factorize.o: ../Cholesky/cholmod_factorize.c
	$(C) -E $(I) $< | $(PRETTY) > z_factorize.c
	$(C) -c $(I) z_factorize.c

z_postorder.o: ../Cholesky/cholmod_postorder.c
	$(C) -E $(I) $< | $(PRETTY) > z_postorder.c
	$(C) -c $(I) z_postorder.c

z_rcond.o: ../Cholesky/cholmod_rcond.c 
	$(C) -E $(I) $< | $(PRETTY) > z_rcond.c
	$(C) -c $(I) z_rcond.c

z_resymbol.o: ../Cholesky/cholmod_resymbol.c
	$(C) -E $(I) $< | $(PRETTY) > z_resymbol.c
	$(C) -c $(I) z_resymbol.c

z_rowcolcounts.o: ../Cholesky/cholmod_rowcolcounts.c
	$(C) -E $(I) $< | $(PRETTY) > z_rowcolcounts.c
	$(C) -c $(I) z_rowcolcounts.c

z_solve.o: ../Cholesky/cholmod_solve.c ../Cholesky/t_cholmod_lsolve.c \
	../Cholesky/t_cholmod_ltsolve.c ../Cholesky/t_cholmod_solve.c
	$(C) -E $(I) $< | $(PRETTY) > z_solve.c
	$(C) -c $(I) z_solve.c

z_spsolve.o: ../Cholesky/cholmod_spsolve.c
	$(C) -E $(I) $< | $(PRETTY) > z_spsolve.c
	$(C) -c $(I) z_spsolve.c

z_rowfac.o: ../Cholesky/cholmod_rowfac.c ../Cholesky/t_cholmod_rowfac.c
	$(C) -E $(I) $< | $(PRETTY) > z_rowfac.c
	$(C) -c $(I) z_rowfac.c

#------------------------------------------------------------------------------- 
z_ccolamd.o: ../Partition/cholmod_ccolamd.c
	$(C) -E $(I) $< | $(PRETTY) > z_ccolamd.c
	$(C) -c $(I) z_ccolamd.c

z_csymamd.o: ../Partition/cholmod_csymamd.c
	$(C) -E $(I) $< | $(PRETTY) > z_csymamd.c
	$(C) -c $(I) z_csymamd.c

z_camd.o: ../Partition/cholmod_camd.c
	$(C) -E $(I) $< | $(PRETTY) > z_camd.c
	$(C) -c $(I) z_camd.c

z_metis.o: ../Partition/cholmod_metis.c
	$(C) -E $(I) $< | $(PRETTY) > z_metis.c
	$(C) -c $(I) z_metis.c

z_nesdis.o: ../Partition/cholmod_nesdis.c
	$(C) -E $(I) $< | $(PRETTY) > z_nesdis.c
	$(C) -c $(I) z_nesdis.c

#-------------------------------------------------------------------------------

z_horzcat.o: ../MatrixOps/cholmod_horzcat.c
	$(C) -E $(I) $< | $(PRETTY) > z_horzcat.c
	$(C) -c $(I) z_horzcat.c

z_norm.o: ../MatrixOps/cholmod_norm.c
	$(C) -E $(I) $< | $(PRETTY) > z_norm.c
	$(C) -c $(I) z_norm.c

z_scale.o: ../MatrixOps/cholmod_scale.c
	$(C) -E $(I) $< | $(PRETTY) > z_scale.c
	$(C) -c $(I) z_scale.c

z_drop.o: ../MatrixOps/cholmod_drop.c
	$(C) -E $(I) $< | $(PRETTY) > z_drop.c
	$(C) -c $(I) z_drop.c

z_sdmult.o: ../MatrixOps/cholmod_sdmult.c ../MatrixOps/t_cholmod_sdmult.c
	$(C) -E $(I) $< | $(PRETTY) > z_sdmult.c
	$(C) -c $(I) z_sdmult.c

z_ssmult.o: ../MatrixOps/cholmod_ssmult.c
	$(C) -E $(I) $< | $(PRETTY) > z_ssmult.c
	$(C) -c $(I) z_ssmult.c

z_submatrix.o: ../MatrixOps/cholmod_submatrix.c
	$(C) -E $(I) $< | $(PRETTY) > z_submatrix.c
	$(C) -c $(I) z_submatrix.c

z_vertcat.o: ../MatrixOps/cholmod_vertcat.c
	$(C) -E $(I) $< | $(PRETTY) > z_vertcat.c
	$(C) -c $(I) z_vertcat.c

z_symmetry.o: ../MatrixOps/cholmod_symmetry.c
	$(C) -E $(I) $< | $(PRETTY) > z_symmetry.c
	$(C) -c $(I) z_symmetry.c

#-------------------------------------------------------------------------------

z_rowadd.o: ../Modify/cholmod_rowadd.c
	$(C) -E $(I) $< | $(PRETTY) > z_rowadd.c
	$(C) -c $(I) z_rowadd.c

z_rowdel.o: ../Modify/cholmod_rowdel.c
	$(C) -E $(I) $< | $(PRETTY) > z_rowdel.c
	$(C) -c $(I) z_rowdel.c

z_updown.o: ../Modify/cholmod_updown.c \
	../Modify/t_cholmod_updown.c ../Modify/t_cholmod_updown_numkr.c
	$(C) -E $(I) $< | $(PRETTY) > z_updown.c
	$(C) -c $(I) z_updown.c

#-------------------------------------------------------------------------------

z_super_numeric.o: ../Supernodal/cholmod_super_numeric.c \
		../Supernodal/t_cholmod_super_numeric.c
	$(C) -E $(I) $< | $(PRETTY) > z_super_numeric.c
	$(C) -c $(I) z_super_numeric.c

z_super_symbolic.o: ../Supernodal/cholmod_super_symbolic.c
	$(C) -E $(I) $< | $(PRETTY) > z_super_symbolic.c
	$(C) -c $(I) z_super_symbolic.c

z_super_solve.o: ../Supernodal/cholmod_super_solve.c
	$(C) -E $(I) $< | $(PRETTY) > z_super_solve.c
	$(C) -c $(I) z_super_solve.c

#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------

l_check.o: ../Check/cholmod_check.c
	$(C) -DDLONG -E $(I) $< | $(PRETTY) > l_check.c
	$(C) -c $(I) l_check.c

l_read.o: ../Check/cholmod_read.c
	$(C) -DDLONG -E $(I) $< | $(PRETTY) > l_read.c
	$(C) -c $(I) l_read.c

l_write.o: ../Check/cholmod_write.c
	$(C) -DDLONG -E $(I) $< | $(PRETTY) > l_write.c
	$(C) -c $(I) l_write.c

#-------------------------------------------------------------------------------

l_common.o: ../Core/cholmod_common.c
	$(C) -DDLONG -E $(I) $< | $(PRETTY) > l_common.c
	$(C) -c $(I) l_common.c

l_dense.o: ../Core/cholmod_dense.c ../Core/t_cholmod_dense.c
	$(C) -DDLONG -E $(I) $< | $(PRETTY) > l_dense.c
	$(C) -c $(I) l_dense.c

l_factor.o: ../Core/cholmod_factor.c
	$(C) -DDLONG -E $(I) $< | $(PRETTY) > l_factor.c
	$(C) -c $(I) l_factor.c

l_change_factor.o: ../Core/cholmod_change_factor.c \
	../Core/t_cholmod_change_factor.c
	$(C) -DDLONG -E $(I) $< | $(PRETTY) > l_change_factor.c
	$(C) -c $(I) l_change_factor.c

l_memory.o: ../Core/cholmod_memory.c
	$(C) -DDLONG -E $(I) $< | $(PRETTY) > l_memory.c
	$(C) -c $(I) l_memory.c

l_sparse.o: ../Core/cholmod_sparse.c
	$(C) -DDLONG -E $(I) $< | $(PRETTY) > l_sparse.c
	$(C) -c $(I) l_sparse.c

l_complex.o: ../Core/cholmod_complex.c
	$(C) -DDLONG -E $(I) $< | $(PRETTY) > l_complex.c
	$(C) -c $(I) l_complex.c

l_transpose.o: ../Core/cholmod_transpose.c ../Core/t_cholmod_transpose.c
	$(C) -DDLONG -E $(I) $< | $(PRETTY) > l_transpose.c
	$(C) -c $(I) l_transpose.c

l_band.o: ../Core/cholmod_band.c
	$(C) -DDLONG -E $(I) $< | $(PRETTY) > l_band.c
	$(C) -c $(I) l_band.c

l_copy.o: ../Core/cholmod_copy.c
	$(C) -DDLONG -E $(I) $< | $(PRETTY) > l_copy.c
	$(C) -c $(I) l_copy.c

l_triplet.o: ../Core/cholmod_triplet.c ../Core/t_cholmod_triplet.c
	$(C) -DDLONG -E $(I) $< | $(PRETTY) > l_triplet.c
	$(C) -c $(I) l_triplet.c

l_error.o: ../Core/cholmod_error.c
	$(C) -DDLONG -E $(I) $< | $(PRETTY) > l_error.c
	$(C) -c $(I) l_error.c

l_aat.o: ../Core/cholmod_aat.c
	$(C) -DDLONG -E $(I) $< | $(PRETTY) > l_aat.c
	$(C) -c $(I) l_aat.c

l_add.o: ../Core/cholmod_add.c
	$(C) -DDLONG -E $(I) $< | $(PRETTY) > l_add.c
	$(C) -c $(I) l_add.c

#-------------------------------------------------------------------------------

l_amd.o: ../Cholesky/cholmod_amd.c
	$(C) -DDLONG -E $(I) $< | $(PRETTY) > l_amd.c
	$(C) -c $(I) l_amd.c

l_analyze.o: ../Cholesky/cholmod_analyze.c
	$(C) -DDLONG -E $(I) $< | $(PRETTY) > l_analyze.c
	$(C) -c $(I) l_analyze.c

l_colamd.o: ../Cholesky/cholmod_colamd.c
	$(C) -DDLONG -E $(I) $< | $(PRETTY) > l_colamd.c
	$(C) -c $(I) l_colamd.c

l_etree.o: ../Cholesky/cholmod_etree.c
	$(C) -DDLONG -E $(I) $< | $(PRETTY) > l_etree.c
	$(C) -c $(I) l_etree.c

l_factorize.o: ../Cholesky/cholmod_factorize.c
	$(C) -DDLONG -E $(I) $< | $(PRETTY) > l_factorize.c
	$(C) -c $(I) l_factorize.c

l_postorder.o: ../Cholesky/cholmod_postorder.c
	$(C) -DDLONG -E $(I) $< | $(PRETTY) > l_postorder.c
	$(C) -c $(I) l_postorder.c

l_rcond.o: ../Cholesky/cholmod_rcond.c 
	$(C) -DDLONG -E $(I) $< | $(PRETTY) > l_rcond.c
	$(C) -c $(I) l_rcond.c

l_resymbol.o: ../Cholesky/cholmod_resymbol.c
	$(C) -DDLONG -E $(I) $< | $(PRETTY) > l_resymbol.c
	$(C) -c $(I) l_resymbol.c

l_rowcolcounts.o: ../Cholesky/cholmod_rowcolcounts.c
	$(C) -DDLONG -E $(I) $< | $(PRETTY) > l_rowcolcounts.c
	$(C) -c $(I) l_rowcolcounts.c

l_solve.o: ../Cholesky/cholmod_solve.c ../Cholesky/t_cholmod_lsolve.c \
	../Cholesky/t_cholmod_ltsolve.c ../Cholesky/t_cholmod_solve.c
	$(C) -DDLONG -E $(I) $< | $(PRETTY) > l_solve.c
	$(C) -c $(I) l_solve.c

l_spsolve.o: ../Cholesky/cholmod_spsolve.c
	$(C) -DDLONG -E $(I) $< | $(PRETTY) > l_spsolve.c
	$(C) -c $(I) l_spsolve.c

l_rowfac.o: ../Cholesky/cholmod_rowfac.c ../Cholesky/t_cholmod_rowfac.c
	$(C) -DDLONG -E $(I) $< | $(PRETTY) > l_rowfac.c
	$(C) -c $(I) l_rowfac.c

#------------------------------------------------------------------------------- 
l_ccolamd.o: ../Partition/cholmod_ccolamd.c
	$(C) -DDLONG -E $(I) $< | $(PRETTY) > l_ccolamd.c
	$(C) -c $(I) l_ccolamd.c

l_csymamd.o: ../Partition/cholmod_csymamd.c
	$(C) -DDLONG -E $(I) $< | $(PRETTY) > l_csymamd.c
	$(C) -c $(I) l_csymamd.c

l_camd.o: ../Partition/cholmod_camd.c
	$(C) -DDLONG -E $(I) $< | $(PRETTY) > l_camd.c
	$(C) -c $(I) l_camd.c

l_metis.o: ../Partition/cholmod_metis.c
	$(C) -DDLONG -E $(I) $< | $(PRETTY) > l_metis.c
	$(C) -c $(I) l_metis.c

l_nesdis.o: ../Partition/cholmod_nesdis.c
	$(C) -DDLONG -E $(I) $< | $(PRETTY) > l_nesdis.c
	$(C) -c $(I) l_nesdis.c

#-------------------------------------------------------------------------------

l_horzcat.o: ../MatrixOps/cholmod_horzcat.c
	$(C) -DDLONG -E $(I) $< | $(PRETTY) > l_horzcat.c
	$(C) -c $(I) l_horzcat.c

l_norm.o: ../MatrixOps/cholmod_norm.c
	$(C) -DDLONG -E $(I) $< | $(PRETTY) > l_norm.c
	$(C) -c $(I) l_norm.c

l_scale.o: ../MatrixOps/cholmod_scale.c
	$(C) -DDLONG -E $(I) $< | $(PRETTY) > l_scale.c
	$(C) -c $(I) l_scale.c

l_drop.o: ../MatrixOps/cholmod_drop.c
	$(C) -DDLONG -E $(I) $< | $(PRETTY) > l_drop.c
	$(C) -c $(I) l_drop.c

l_sdmult.o: ../MatrixOps/cholmod_sdmult.c ../MatrixOps/t_cholmod_sdmult.c
	$(C) -DDLONG -E $(I) $< | $(PRETTY) > l_sdmult.c
	$(C) -c $(I) l_sdmult.c

l_ssmult.o: ../MatrixOps/cholmod_ssmult.c
	$(C) -DDLONG -E $(I) $< | $(PRETTY) > l_ssmult.c
	$(C) -c $(I) l_ssmult.c

l_submatrix.o: ../MatrixOps/cholmod_submatrix.c
	$(C) -DDLONG -E $(I) $< | $(PRETTY) > l_submatrix.c
	$(C) -c $(I) l_submatrix.c

l_vertcat.o: ../MatrixOps/cholmod_vertcat.c
	$(C) -DDLONG -E $(I) $< | $(PRETTY) > l_vertcat.c
	$(C) -c $(I) l_vertcat.c

l_symmetry.o: ../MatrixOps/cholmod_symmetry.c
	$(C) -DDLONG -E $(I) $< | $(PRETTY) > l_symmetry.c
	$(C) -c $(I) l_symmetry.c

#-------------------------------------------------------------------------------

l_rowadd.o: ../Modify/cholmod_rowadd.c
	$(C) -DDLONG -E $(I) $< | $(PRETTY) > l_rowadd.c
	$(C) -c $(I) l_rowadd.c

l_rowdel.o: ../Modify/cholmod_rowdel.c
	$(C) -DDLONG -E $(I) $< | $(PRETTY) > l_rowdel.c
	$(C) -c $(I) l_rowdel.c

l_updown.o: ../Modify/cholmod_updown.c \
	../Modify/t_cholmod_updown.c ../Modify/t_cholmod_updown_numkr.c
	$(C) -DDLONG -E $(I) $< | $(PRETTY) > l_updown.c
	$(C) -c $(I) l_updown.c

#-------------------------------------------------------------------------------

l_super_numeric.o: ../Supernodal/cholmod_super_numeric.c \
		../Supernodal/t_cholmod_super_numeric.c
	$(C) -DDLONG -E $(I) $< | $(PRETTY) > l_super_numeric.c
	$(C) -c $(I) l_super_numeric.c

l_super_symbolic.o: ../Supernodal/cholmod_super_symbolic.c
	$(C) -DDLONG -E $(I) $< | $(PRETTY) > l_super_symbolic.c
	$(C) -c $(I) l_super_symbolic.c

l_super_solve.o: ../Supernodal/cholmod_super_solve.c
	$(C) -DDLONG -E $(I) $< | $(PRETTY) > l_super_solve.c
	$(C) -c $(I) l_super_solve.c

#-------------------------------------------------------------------------------
