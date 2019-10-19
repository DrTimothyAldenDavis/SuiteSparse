include ../../../UFconfig/UFconfig.mk

MX = $(MEX) -DCS_LONG
AR = ar cr
RANLIB = ranlib

I = -I../../Include -I../../../UFconfig

all: mexcsparse.a cs_mex.h
	$(MX) cs_thumb_mex.c $(I) mexcsparse.a -output cs_thumb
	$(MX) cs_print_mex.c $(I) mexcsparse.a -output cs_print
	$(MX) cs_updown_mex.c $(I) mexcsparse.a -output cs_updown
	$(MX) cs_gaxpy_mex.c $(I) mexcsparse.a -output cs_gaxpy
	$(MX) cs_transpose_mex.c $(I) mexcsparse.a -output cs_transpose
	$(MX) cs_sparse_mex.c $(I) mexcsparse.a -output cs_sparse
	$(MX) cs_multiply_mex.c $(I) mexcsparse.a -output cs_multiply
	$(MX) cs_add_mex.c $(I) mexcsparse.a -output cs_add
	$(MX) cs_permute_mex.c $(I) mexcsparse.a -output cs_permute
	$(MX) cs_symperm_mex.c $(I) mexcsparse.a -output cs_symperm
	$(MX) cs_lsolve_mex.c $(I) mexcsparse.a -output cs_lsolve
	$(MX) cs_ltsolve_mex.c $(I) mexcsparse.a -output cs_ltsolve
	$(MX) cs_usolve_mex.c $(I) mexcsparse.a -output cs_usolve
	$(MX) cs_utsolve_mex.c $(I) mexcsparse.a -output cs_utsolve
	$(MX) cs_chol_mex.c $(I) mexcsparse.a -output cs_chol
	$(MX) cs_etree_mex.c $(I) mexcsparse.a -output cs_etree
	$(MX) cs_counts_mex.c $(I) mexcsparse.a -output cs_counts
	$(MX) cs_qr_mex.c $(I) mexcsparse.a -output cs_qr
	$(MX) cs_amd_mex.c $(I) mexcsparse.a -output cs_amd
	$(MX) cs_lu_mex.c $(I) mexcsparse.a -output cs_lu
	$(MX) cs_cholsol_mex.c $(I) mexcsparse.a -output cs_cholsol
	$(MX) cs_lusol_mex.c $(I) mexcsparse.a -output cs_lusol
	$(MX) cs_droptol_mex.c $(I) mexcsparse.a -output cs_droptol
	$(MX) cs_qrsol_mex.c $(I) mexcsparse.a -output cs_qrsol
	$(MX) cs_dmperm_mex.c $(I) mexcsparse.a -output cs_dmperm
	$(MX) cs_scc_mex.c $(I) mexcsparse.a -output cs_scc
	$(MX) cs_sqr_mex.c $(I) mexcsparse.a -output cs_sqr
	$(MX) cs_randperm_mex.c $(I) mexcsparse.a -output cs_randperm

CSD = cs_mex.o \
	cs_amd.o \
	cs_chol.o \
	cs_counts.o \
	cs_cumsum.o \
	cs_fkeep.o \
	cs_dfs.o \
	cs_dmperm.o \
	cs_droptol.o \
	cs_dropzeros.o \
	cs_dupl.o \
	cs_entry.o \
	cs_etree.o \
	cs_gaxpy.o \
	cs_ipvec.o \
	cs_lsolve.o \
	cs_ltsolve.o \
	cs_lu.o \
	cs_maxtrans.o \
	cs_util.o \
	cs_malloc.o \
	cs_multiply.o \
	cs_add.o \
	cs_scatter.o \
	cs_permute.o \
	cs_pinv.o \
	cs_post.o \
	cs_tdfs.o \
	cs_pvec.o \
	cs_qr.o \
	cs_happly.o \
	cs_house.o \
	cs_schol.o \
	cs_scc.o \
	cs_sqr.o \
	cs_symperm.o \
	cs_transpose.o \
	cs_compress.o \
	cs_usolve.o \
	cs_utsolve.o \
	cs_cholsol.o \
	cs_lusol.o \
	cs_qrsol.o \
	cs_updown.o \
	cs_norm.o \
	cs_print.o \
	cs_load.o \
	cs_spsolve.o \
	cs_reach.o \
	cs_ereach.o \
	cs_leaf.o \
	cs_randperm.o

CSC = \
	cs_cl_amd.o \
	cs_cl_chol.o \
	cs_cl_counts.o \
	cs_cl_cumsum.o \
	cs_cl_fkeep.o \
	cs_cl_dfs.o \
	cs_cl_dmperm.o \
	cs_cl_droptol.o \
	cs_cl_dropzeros.o \
	cs_cl_dupl.o \
	cs_cl_entry.o \
	cs_cl_etree.o \
	cs_cl_gaxpy.o \
	cs_cl_ipvec.o \
	cs_cl_lsolve.o \
	cs_cl_ltsolve.o \
	cs_cl_lu.o \
	cs_cl_maxtrans.o \
	cs_cl_util.o \
	cs_cl_malloc.o \
	cs_cl_multiply.o \
	cs_cl_add.o \
	cs_cl_scatter.o \
	cs_cl_permute.o \
	cs_cl_pinv.o \
	cs_cl_post.o \
	cs_cl_tdfs.o \
	cs_cl_pvec.o \
	cs_cl_qr.o \
	cs_cl_happly.o \
	cs_cl_house.o \
	cs_cl_schol.o \
	cs_cl_scc.o \
	cs_cl_sqr.o \
	cs_cl_symperm.o \
	cs_cl_transpose.o \
	cs_cl_compress.o \
	cs_cl_usolve.o \
	cs_cl_utsolve.o \
	cs_cl_cholsol.o \
	cs_cl_lusol.o \
	cs_cl_qrsol.o \
	cs_cl_updown.o \
	cs_cl_norm.o \
	cs_cl_print.o \
	cs_cl_load.o \
	cs_cl_spsolve.o \
	cs_cl_reach.o \
	cs_cl_ereach.o \
	cs_cl_leaf.o \
	cs_cl_randperm.o

CS = $(CSD) $(CSC)

mexcsparse.a: $(CS)
	$(AR) mexcsparse.a $(CS)
	$(RANLIB) mexcsparse.a

$(CS): ../../Include/cs.h


cs_mex.o: cs_mex.c cs_mex.h
	$(MX) -c $(I) $<

cs_cl_%.o: ../../Source/cs_%.c
	cp -f $< cs_cl_$*.c
	$(MX) -DCS_COMPLEX -c $(I) cs_cl_$*.c

cs_%.o: ../../Source/cs_%.c
	$(MX) -c $(I) $<


clean:
	- rm -f *.o

distclean: clean
	- rm -f *.mex* *.dll *.a cs_cl_*.c

purge: distclean
