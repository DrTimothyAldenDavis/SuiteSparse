MEX = mex -O
AR = ar cr
RANLIB = ranlib

I = -I../../Include

all: mexcsparse.a cs_mex.h
	$(MEX) cs_thumb_mex.c $(I) mexcsparse.a -output cs_thumb
	$(MEX) cs_print_mex.c $(I) mexcsparse.a -output cs_print
	$(MEX) cs_updown_mex.c $(I) mexcsparse.a -output cs_updown
	$(MEX) cs_gaxpy_mex.c $(I) mexcsparse.a -output cs_gaxpy
	$(MEX) cs_transpose_mex.c $(I) mexcsparse.a -output cs_transpose
	$(MEX) cs_sparse_mex.c $(I) mexcsparse.a -output cs_sparse
	$(MEX) cs_multiply_mex.c $(I) mexcsparse.a -output cs_multiply
	$(MEX) cs_add_mex.c $(I) mexcsparse.a -output cs_add
	$(MEX) cs_permute_mex.c $(I) mexcsparse.a -output cs_permute
	$(MEX) cs_symperm_mex.c $(I) mexcsparse.a -output cs_symperm
	$(MEX) cs_lsolve_mex.c $(I) mexcsparse.a -output cs_lsolve
	$(MEX) cs_ltsolve_mex.c $(I) mexcsparse.a -output cs_ltsolve
	$(MEX) cs_usolve_mex.c $(I) mexcsparse.a -output cs_usolve
	$(MEX) cs_utsolve_mex.c $(I) mexcsparse.a -output cs_utsolve
	$(MEX) cs_chol_mex.c $(I) mexcsparse.a -output cs_chol
	$(MEX) cs_etree_mex.c $(I) mexcsparse.a -output cs_etree
	$(MEX) cs_counts_mex.c $(I) mexcsparse.a -output cs_counts
	$(MEX) cs_qr_mex.c $(I) mexcsparse.a -output cs_qr
	$(MEX) cs_amd_mex.c $(I) mexcsparse.a -output cs_amd
	$(MEX) cs_lu_mex.c $(I) mexcsparse.a -output cs_lu
	$(MEX) cs_cholsol_mex.c $(I) mexcsparse.a -output cs_cholsol
	$(MEX) cs_lusol_mex.c $(I) mexcsparse.a -output cs_lusol
	$(MEX) cs_droptol_mex.c $(I) mexcsparse.a -output cs_droptol
	$(MEX) cs_qrsol_mex.c $(I) mexcsparse.a -output cs_qrsol
	$(MEX) cs_dmperm_mex.c $(I) mexcsparse.a -output cs_dmperm
	$(MEX) cs_scc_mex.c $(I) mexcsparse.a -output cs_scc
	$(MEX) cs_sqr_mex.c $(I) mexcsparse.a -output cs_sqr
	$(MEX) cs_randperm_mex.c $(I) mexcsparse.a -output cs_randperm

CS = cs_mex.o \
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

mexcsparse.a: $(CS)
	$(AR) mexcsparse.a $(CS)
	$(RANLIB) mexcsparse.a

$(CS): ../../Include/cs.h


cs_mex.o: cs_mex.c cs_mex.h
	$(MEX) -c $(I) $<

cs_amd.o: ../../Source/cs_amd.c
	$(MEX) -c $(I) $<

cs_chol.o: ../../Source/cs_chol.c
	$(MEX) -c $(I) $<

cs_ereach.o: ../../Source/cs_ereach.c
	$(MEX) -c $(I) $<

cs_cholsol.o: ../../Source/cs_cholsol.c
	$(MEX) -c $(I) $<

cs_lusol.o: ../../Source/cs_lusol.c
	$(MEX) -c $(I) $<

cs_qrsol.o: ../../Source/cs_qrsol.c
	$(MEX) -c $(I) $<

cs_counts.o: ../../Source/cs_counts.c
	$(MEX) -c $(I) $<

cs_leaf.o: ../../Source/cs_leaf.c
	$(MEX) -c $(I) $<

cs_cumsum.o: ../../Source/cs_cumsum.c
	$(MEX) -c $(I) $<

cs_fkeep.o: ../../Source/cs_fkeep.c
	$(MEX) -c $(I) $<

cs_dfs.o: ../../Source/cs_dfs.c
	$(MEX) -c $(I) $<

cs_droptol.o: ../../Source/cs_droptol.c
	$(MEX) -c $(I) $<

cs_dropzeros.o: ../../Source/cs_dropzeros.c
	$(MEX) -c $(I) $<

cs_dupl.o: ../../Source/cs_dupl.c
	$(MEX) -c $(I) $<

cs_entry.o: ../../Source/cs_entry.c
	$(MEX) -c $(I) $<

cs_etree.o: ../../Source/cs_etree.c
	$(MEX) -c $(I) $<

cs_gaxpy.o: ../../Source/cs_gaxpy.c
	$(MEX) -c $(I) $<

cs_ipvec.o: ../../Source/cs_ipvec.c
	$(MEX) -c $(I) $<

cs_lsolve.o: ../../Source/cs_lsolve.c
	$(MEX) -c $(I) $<

cs_ltsolve.o: ../../Source/cs_ltsolve.c
	$(MEX) -c $(I) $<

cs_lu.o: ../../Source/cs_lu.c
	$(MEX) -c $(I) $<

cs_util.o: ../../Source/cs_util.c
	$(MEX) -c $(I) $<

cs_malloc.o: ../../Source/cs_malloc.c
	$(MEX) -c $(I) $<

cs_multiply.o: ../../Source/cs_multiply.c
	$(MEX) -c $(I) $<

cs_add.o: ../../Source/cs_add.c
	$(MEX) -c $(I) $<

cs_scatter.o: ../../Source/cs_scatter.c
	$(MEX) -c $(I) $<

cs_permute.o: ../../Source/cs_permute.c
	$(MEX) -c $(I) $<

cs_pinv.o: ../../Source/cs_pinv.c
	$(MEX) -c $(I) $<

cs_post.o: ../../Source/cs_post.c
	$(MEX) -c $(I) $<

cs_tdfs.o: ../../Source/cs_tdfs.c
	$(MEX) -c $(I) $<

cs_pvec.o: ../../Source/cs_pvec.c
	$(MEX) -c $(I) $<

cs_qr.o: ../../Source/cs_qr.c
	$(MEX) -c $(I) $<

cs_happly.o: ../../Source/cs_happly.c
	$(MEX) -c $(I) $<

cs_house.o: ../../Source/cs_house.c
	$(MEX) -c $(I) $<

cs_schol.o: ../../Source/cs_schol.c
	$(MEX) -c $(I) $<

cs_spsolve.o: ../../Source/cs_spsolve.c
	$(MEX) -c $(I) $<

cs_reach.o: ../../Source/cs_reach.c
	$(MEX) -c $(I) $<

cs_sqr.o: ../../Source/cs_sqr.c
	$(MEX) -c $(I) $<

cs_symperm.o: ../../Source/cs_symperm.c
	$(MEX) -c $(I) $<

cs_transpose.o: ../../Source/cs_transpose.c
	$(MEX) -c $(I) $<

cs_compress.o: ../../Source/cs_compress.c
	$(MEX) -c $(I) $<

cs_usolve.o: ../../Source/cs_usolve.c
	$(MEX) -c $(I) $<

cs_utsolve.o: ../../Source/cs_utsolve.c
	$(MEX) -c $(I) $<

cs_dmperm.o: ../../Source/cs_dmperm.c
	$(MEX) -c $(I) $<

cs_randperm.o: ../../Source/cs_randperm.c
	$(MEX) -c $(I) $<

cs_maxtrans.o: ../../Source/cs_maxtrans.c
	$(MEX) -c $(I) $<

cs_scc.o: ../../Source/cs_scc.c
	$(MEX) -c $(I) $<

cs_updown.o: ../../Source/cs_updown.c
	$(MEX) -c $(I) $<

cs_print.o: ../../Source/cs_print.c
	$(MEX) -c $(I) $<

cs_norm.o: ../../Source/cs_norm.c
	$(MEX) -c $(I) $<

cs_load.o: ../../Source/cs_load.c
	$(MEX) -c $(I) $<

clean:
	rm -f *.o

distclean: clean
	rm -f *.mex* *.dll *.a

purge: distclean
