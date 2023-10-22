
all: CHOLMOD_UserGuide.pdf

I = ../Include/cholmod.h

C = ../Demo/cholmod_simple.c

M = \
	../MATLAB/analyze.m \
	../MATLAB/bisect.m \
	../MATLAB/chol2.m \
	../MATLAB/cholmod2.m \
	../MATLAB/cholmod_demo.m \
	../MATLAB/cholmod_make.m \
	../MATLAB/etree2.m \
	../MATLAB/graph_demo.m \
	../MATLAB/lchol.m \
	../MATLAB/ldlchol.m \
	../MATLAB/ldl_normest.m \
	../MATLAB/ldlsolve.m \
	../MATLAB/ldlsplit.m \
	../MATLAB/ldlupdate.m \
	../MATLAB/ldlrowmod.m \
	../MATLAB/metis.m \
	../MATLAB/nesdis.m \
	../MATLAB/resymbol.m \
	../MATLAB/sdmult.m \
	../MATLAB/sparse2.m \
	../MATLAB/spsym.m \
	../MATLAB/mread.m \
	../MATLAB/mwrite.m \
	../MATLAB/symbfact2.m

CHOLMOD_UserGuide.pdf: CHOLMOD_UserGuide.tex UserGuide.bib $(I) $(C) $(M) Makefile getproto rule.awk header.tex footer.tex getmproto mfooter.tex mheader.tex mfile.awk cholmod_version.tex
	./getmproto ../MATLAB/analyze.m > _analyze_m.tex
	./getmproto ../MATLAB/bisect.m > _bisect_m.tex
	./getmproto ../MATLAB/chol2.m > _chol2_m.tex
	./getmproto ../MATLAB/cholmod2.m > _cholmod2_m.tex
	./getmproto ../MATLAB/cholmod_demo.m > _cholmod_demo_m.tex
	./getmproto ../MATLAB/cholmod_make.m > _cholmod_make_m.tex
	./getmproto ../MATLAB/etree2.m > _etree2_m.tex
	./getmproto ../MATLAB/graph_demo.m > _graph_demo_m.tex
	./getmproto ../MATLAB/lchol.m > _lchol_m.tex
	./getmproto ../MATLAB/ldlchol.m > _ldlchol_m.tex
	./getmproto ../MATLAB/ldl_normest.m > _ldl_normest_m.tex
	./getmproto ../MATLAB/ldlsolve.m > _ldlsolve_m.tex
	./getmproto ../MATLAB/ldlsplit.m > _ldlsplit_m.tex
	./getmproto ../MATLAB/ldlupdate.m > _ldlupdate_m.tex
	./getmproto ../MATLAB/ldlrowmod.m > _ldlrowmod_m.tex
	./getmproto ../MATLAB/metis.m > _metis_m.tex
	./getmproto ../MATLAB/mread.m > _mread_m.tex
	./getmproto ../MATLAB/spsym.m > _spsym_m.tex
	./getmproto ../MATLAB/mwrite.m > _mwrite_m.tex
	./getmproto ../MATLAB/nesdis.m > _nesdis_m.tex
	./getmproto ../MATLAB/resymbol.m > _resymbol_m.tex
	./getmproto ../MATLAB/sdmult.m > _sdmult_m.tex
	./getmproto ../MATLAB/sparse2.m > _sparse2_m.tex
	./getmproto ../MATLAB/symbfact2.m > _symbfact2_m.tex
	./getproto '/include/, /^}/' ../Demo/cholmod_simple.c > _simple.tex
	./getproto '/typedef struct cholmod_common/, /^}/' ../Include/cholmod.h > _common.tex
	./getproto '/int cholmod_start/, /\*\) ;/' ../Include/cholmod.h > _start.tex
	./getproto '/int cholmod_finish/, /\*\) ;/' ../Include/cholmod.h > _finish.tex
	./getproto '/int cholmod_defaults/, /\*\) ;/' ../Include/cholmod.h > _defaults.tex
	./getproto '/size_t cholmod_maxrank/, /\*\) ;/' ../Include/cholmod.h > _maxrank.tex
	./getproto '/int cholmod_allocate_work/, /\*\) ;/' ../Include/cholmod.h > _allocate_work.tex
	./getproto '/int cholmod_free_work/, /\*\) ;/' ../Include/cholmod.h > _free_work.tex
	./getproto '/int64_t cholmod_clear_flag/, /\*\) ;/' ../Include/cholmod.h > _clear_flag.tex
	./getproto '/int cholmod_error/, /\*\) ;/' ../Include/cholmod.h > _error.tex
	./getproto '/double cholmod_dbound/, /l_sbound/' ../Include/cholmod.h > _dbound.tex
	./getproto '/double cholmod_hypot/, /double\) ;/' ../Include/cholmod.h > _hypot.tex
	./getproto '/int cholmod_divcomplex/, /\*\) ;/' ../Include/cholmod.h > _divcomplex.tex
	./getproto '/typedef struct cholmod_sparse/, /^}/' ../Include/cholmod.h > _sparse.tex
	./getproto '/cholmod_sparse \*cholmod_allocate_sparse/, /\*\) ;/' ../Include/cholmod.h > _allocate_sparse.tex
	./getproto '/int cholmod_free_sparse/, /\*\) ;/' ../Include/cholmod.h > _free_sparse.tex
	./getproto '/int cholmod_reallocate_sparse/, /\*\) ;/' ../Include/cholmod.h > _reallocate_sparse.tex
	./getproto '/int64_t cholmod_nnz/, /\*\) ;/' ../Include/cholmod.h > _nnz.tex
	./getproto '/int64_t cholmod_dense_nnz/, /\*\) ;/' ../Include/cholmod.h > _dense_nnz.tex
	./getproto '/cholmod_sparse \*cholmod_speye/, /\*\) ;/' ../Include/cholmod.h > _speye.tex
	./getproto '/cholmod_sparse \*cholmod_spzeros/, /\*\) ;/' ../Include/cholmod.h > _spzeros.tex
	./getproto '/cholmod_sparse \*cholmod_transpose/, /\*\) ;/' ../Include/cholmod.h > _transpose.tex
	./getproto '/int cholmod_transpose_unsym/, /\*\) ;/' ../Include/cholmod.h > _transpose_unsym.tex
	./getproto '/int cholmod_transpose_sym/, /\*\) ;/' ../Include/cholmod.h > _transpose_sym.tex
	./getproto '/cholmod_sparse \*cholmod_ptranspose/, /\*\) ;/' ../Include/cholmod.h > _ptranspose.tex
	./getproto '/int cholmod_sort/, /\*\) ;/' ../Include/cholmod.h > _sort.tex
	./getproto '/cholmod_sparse \*cholmod_band/, /\*\) ;/' ../Include/cholmod.h > _band.tex
	./getproto '/int cholmod_band_inplace/, /\*\) ;/' ../Include/cholmod.h > _band_inplace.tex
	./getproto '/cholmod_sparse \*cholmod_aat/, /\*\) ;/' ../Include/cholmod.h > _aat.tex
	./getproto '/cholmod_sparse \*cholmod_copy_sparse/, /\*\) ;/' ../Include/cholmod.h > _copy_sparse.tex
	./getproto '/cholmod_sparse \*cholmod_copy/, /\*\) ;/' ../Include/cholmod.h > _copy.tex
	./getproto '/cholmod_sparse \*cholmod_add/, /\*\) ;/' ../Include/cholmod.h > _add.tex
	./getproto '/int cholmod_sparse_xtype/, /\*\) ;/' ../Include/cholmod.h > _sparse_xtype.tex 
	./getproto '/typedef struct cholmod_factor/, /^}/' ../Include/cholmod.h > _factor.tex 
	./getproto '/cholmod_factor \*cholmod_allocate_factor/, /\*\) ;/' ../Include/cholmod.h > _allocate_factor.tex 
	./getproto '/int cholmod_free_factor/, /\*\) ;/' ../Include/cholmod.h > _free_factor.tex 
	./getproto '/int cholmod_reallocate_factor/, /\*\) ;/' ../Include/cholmod.h > _reallocate_factor.tex 
	./getproto '/int cholmod_change_factor/, /\*\) ;/' ../Include/cholmod.h > _change_factor.tex 
	./getproto '/int cholmod_pack_factor/, /\*\) ;/' ../Include/cholmod.h > _pack_factor.tex 
	./getproto '/int cholmod_reallocate_column/, /\*\) ;/' ../Include/cholmod.h > _reallocate_column.tex 
	./getproto '/cholmod_sparse \*cholmod_factor_to_sparse/, /\*\) ;/' ../Include/cholmod.h > _factor_to_sparse.tex 
	./getproto '/cholmod_factor \*cholmod_copy_factor/, /\*\) ;/' ../Include/cholmod.h > _copy_factor.tex 
	./getproto '/int cholmod_factor_xtype/, /\*\) ;/' ../Include/cholmod.h > _factor_xtype.tex 
	./getproto '/typedef struct cholmod_dense/, /^}/' ../Include/cholmod.h > _dense.tex 
	./getproto '/cholmod_dense \*cholmod_allocate_dense/, /\*\) ;/' ../Include/cholmod.h > _allocate_dense.tex 
	./getproto '/cholmod_dense \*cholmod_ensure_dense/, /\*\) ;/' ../Include/cholmod.h > _ensure_dense.tex 
	./getproto '/cholmod_dense \*cholmod_zeros/, /\*\) ;/' ../Include/cholmod.h > _zeros.tex 
	./getproto '/cholmod_dense \*cholmod_ones/, /\*\) ;/' ../Include/cholmod.h > _ones.tex 
	./getproto '/cholmod_dense \*cholmod_eye/, /\*\) ;/' ../Include/cholmod.h > _eye.tex 
	./getproto '/int cholmod_free_dense/, /\*\) ;/' ../Include/cholmod.h > _free_dense.tex 
	./getproto '/cholmod_dense \*cholmod_sparse_to_dense/, /\*\) ;/' ../Include/cholmod.h > _sparse_to_dense.tex 
	./getproto '/cholmod_sparse \*cholmod_dense_to_sparse/, /\*\) ;/' ../Include/cholmod.h > _dense_to_sparse.tex 
	./getproto '/cholmod_dense \*cholmod_copy_dense/, /\*\) ;/' ../Include/cholmod.h > _copy_dense.tex 
	./getproto '/int cholmod_copy_dense2/, /\*\) ;/' ../Include/cholmod.h > _copy_dense2.tex 
	./getproto '/int cholmod_dense_xtype/, /\*\) ;/' ../Include/cholmod.h > _dense_xtype.tex 
	./getproto '/typedef struct cholmod_triplet/, /^}/' ../Include/cholmod.h > _triplet.tex 
	./getproto '/cholmod_triplet \*cholmod_allocate_triplet/, /\*\) ;/' ../Include/cholmod.h > _allocate_triplet.tex 
	./getproto '/int cholmod_free_triplet/, /\*\) ;/' ../Include/cholmod.h > _free_triplet.tex 
	./getproto '/int cholmod_reallocate_triplet/, /\*\) ;/' ../Include/cholmod.h > _reallocate_triplet.tex 
	./getproto '/cholmod_triplet \*cholmod_sparse_to_triplet/, /\*\) ;/' ../Include/cholmod.h > _sparse_to_triplet.tex 
	./getproto '/cholmod_sparse \*cholmod_triplet_to_sparse/, /\*\) ;/' ../Include/cholmod.h > _triplet_to_sparse.tex 
	./getproto '/cholmod_triplet \*cholmod_copy_triplet/, /\*\) ;/' ../Include/cholmod.h > _copy_triplet.tex 
	./getproto '/int cholmod_triplet_xtype/, /\*\) ;/' ../Include/cholmod.h > _triplet_xtype.tex 
	./getproto '/void \*cholmod_malloc/, /\*\) ;/' ../Include/cholmod.h > _malloc.tex 
	./getproto '/void \*cholmod_calloc/, /\*\) ;/' ../Include/cholmod.h > _calloc.tex 
	./getproto '/void \*cholmod_free/, /\*\) ;/' ../Include/cholmod.h > _free.tex 
	./getproto '/void \*cholmod_realloc/, /\*\) ;/' ../Include/cholmod.h > _realloc.tex 
	./getproto '/int cholmod_realloc_multiple/, /\*\) ;/' ../Include/cholmod.h > _realloc_multiple.tex 
	./getproto '/int cholmod_version/, /l_version/' ../Include/cholmod.h > _version.tex 
	./getproto '/itype: integer sizes/, /define CHOLMOD_SUPERNODAL/' ../Include/cholmod.h > _defn.tex 
	./getproto '/int cholmod_check_common/, /\*\) ;/' ../Include/cholmod.h > _check_common.tex 
	./getproto '/int cholmod_print_common/, /\*\) ;/' ../Include/cholmod.h > _print_common.tex 
	./getproto '/int cholmod_check_sparse/, /\*\) ;/' ../Include/cholmod.h > _check_sparse.tex 
	./getproto '/int cholmod_print_sparse/, /\*\) ;/' ../Include/cholmod.h > _print_sparse.tex 
	./getproto '/int cholmod_check_dense/, /\*\) ;/' ../Include/cholmod.h > _check_dense.tex 
	./getproto '/int cholmod_print_dense/, /\*\) ;/' ../Include/cholmod.h > _print_dense.tex 
	./getproto '/int cholmod_check_factor/, /\*\) ;/' ../Include/cholmod.h > _check_factor.tex 
	./getproto '/int cholmod_print_factor/, /\*\) ;/' ../Include/cholmod.h > _print_factor.tex 
	./getproto '/int cholmod_check_triplet/, /\*\) ;/' ../Include/cholmod.h > _check_triplet.tex 
	./getproto '/int cholmod_print_triplet/, /\*\) ;/' ../Include/cholmod.h > _print_triplet.tex 
	./getproto '/int cholmod_check_subset/, /\*\) ;/' ../Include/cholmod.h > _check_subset.tex 
	./getproto '/int cholmod_print_subset/, /\*\) ;/' ../Include/cholmod.h > _print_subset.tex 
	./getproto '/int cholmod_check_perm/, /\*\) ;/' ../Include/cholmod.h > _check_perm.tex
	./getproto '/int cholmod_print_perm/, /\*\) ;/' ../Include/cholmod.h > _print_perm.tex
	./getproto '/int cholmod_check_parent/, /\*\) ;/' ../Include/cholmod.h > _check_parent.tex
	./getproto '/int cholmod_print_parent/, /\*\) ;/' ../Include/cholmod.h > _print_parent.tex
	./getproto '/cholmod_triplet \*cholmod_read_triplet/, /\*\) ;/' ../Include/cholmod.h > _read_triplet.tex
	./getproto '/cholmod_sparse \*cholmod_read_sparse/, /\*\) ;/' ../Include/cholmod.h > _read_sparse.tex
	./getproto '/cholmod_dense \*cholmod_read_dense/, /\*\) ;/' ../Include/cholmod.h > _read_dense.tex
	./getproto '/void \*cholmod_read_matrix/, /\*\) ;/' ../Include/cholmod.h > _read_matrix.tex
	./getproto '/int cholmod_write_sparse/, /\*\) ;/' ../Include/cholmod.h > _write_sparse.tex
	./getproto '/int cholmod_write_dense/, /\*\) ;/' ../Include/cholmod.h > _write_dense.tex
	./getproto '/cholmod_factor \*cholmod_analyze /, /\*\) ;/' ../Include/cholmod.h > _analyze.tex
	./getproto '/cholmod_factor \*cholmod_analyze_p/, /\*\) ;/' ../Include/cholmod.h > _analyze_p.tex
	./getproto '/int cholmod_factorize /, /\*\) ;/' ../Include/cholmod.h > _factorize.tex
	./getproto '/int cholmod_factorize_p/, /\*\) ;/' ../Include/cholmod.h > _factorize_p.tex
	./getproto '/cholmod_dense \*cholmod_solve/, /\*\) ;/' ../Include/cholmod.h > _solve.tex 
	./getproto '/int cholmod_solve2/, /\*\) ;/' ../Include/cholmod.h > _solve2.tex 
	./getproto '/cholmod_sparse \*cholmod_spsolve/, /\*\) ;/' ../Include/cholmod.h > _spsolve.tex 
	./getproto '/int cholmod_etree/, /\*\) ;/' ../Include/cholmod.h > _etree.tex 
	./getproto '/int cholmod_rowcolcounts/, /\*\) ;/' ../Include/cholmod.h > _rowcolcounts.tex 
	./getproto '/int cholmod_analyze_ordering/, /\*\) ;/' ../Include/cholmod.h > _analyze_ordering.tex 
	./getproto '/int cholmod_amd/, /\*\) ;/' ../Include/cholmod.h > _amd.tex 
	./getproto '/int cholmod_colamd/, /\*\) ;/' ../Include/cholmod.h > _colamd.tex 
	./getproto '/int cholmod_rowfac/, /\*\) ;/' ../Include/cholmod.h > _rowfac.tex 
	./getproto '/int cholmod_rowfac_mask/, /\*\) ;/' ../Include/cholmod.h > _rowfac_mask.tex 
	./getproto '/int cholmod_row_subtree/, /\*\) ;/' ../Include/cholmod.h > _row_subtree.tex 
	./getproto '/int cholmod_row_lsubtree/, /\*\) ;/' ../Include/cholmod.h > _row_lsubtree.tex 
	./getproto '/int cholmod_lsolve_pattern/, /\*\) ;/' ../Include/cholmod.h > _lsolve_pattern.tex 
	./getproto '/int cholmod_resymbol /, /\*\) ;/' ../Include/cholmod.h > _resymbol.tex
	./getproto '/int cholmod_resymbol_noperm/, /\*\) ;/' ../Include/cholmod.h > _resymbol_noperm.tex 
	./getproto '/double cholmod_rcond/, /\*\) ;/' ../Include/cholmod.h > _rcond.tex
	./getproto '/int32_t cholmod_postorder/, /\*\) ;/' ../Include/cholmod.h > _postorder.tex
	./getproto '/int cholmod_updown /, /\*\) ;/' ../Include/cholmod.h > _updown.tex
	./getproto '/int cholmod_updown_solve/, /\*\) ;/' ../Include/cholmod.h > _updown_solve.tex
	./getproto '/int cholmod_updown_mark/, /\*\) ;/' ../Include/cholmod.h > _updown_mark.tex
	./getproto '/int cholmod_updown_mask/, /\*\) ;/' ../Include/cholmod.h > _updown_mask.tex
	./getproto '/int cholmod_rowadd /, /\*\) ;/' ../Include/cholmod.h > _rowadd.tex
	./getproto '/int cholmod_rowadd_solve/, /\*\) ;/' ../Include/cholmod.h > _rowadd_solve.tex
	./getproto '/int cholmod_rowadd_mark/, /\*\) ;/' ../Include/cholmod.h > _rowadd_mark.tex
	./getproto '/int cholmod_rowdel /, /\*\) ;/' ../Include/cholmod.h > _rowdel.tex
	./getproto '/int cholmod_rowdel_solve/, /\*\) ;/' ../Include/cholmod.h > _rowdel_solve.tex
	./getproto '/int cholmod_rowdel_mark/, /\*\) ;/' ../Include/cholmod.h > _rowdel_mark.tex
	./getproto '/int cholmod_drop/, /\*\) ;/' ../Include/cholmod.h > _drop.tex
	./getproto '/double cholmod_norm_dense/, /\*\) ;/' ../Include/cholmod.h > _norm_dense.tex
	./getproto '/double cholmod_norm_sparse/, /\*\) ;/' ../Include/cholmod.h > _norm_sparse.tex
	./getproto '/cholmod_sparse \*cholmod_horzcat/, /\*\) ;/' ../Include/cholmod.h > _horzcat.tex
	./getproto '/define CHOLMOD_SCALAR/, /\*\) ;/' ../Include/cholmod.h > _scale.tex
	./getproto '/int cholmod_sdmult/, /\*\) ;/' ../Include/cholmod.h > _sdmult.tex
	./getproto '/cholmod_sparse \*cholmod_ssmult/, /\*\) ;/' ../Include/cholmod.h > _ssmult.tex
	./getproto '/cholmod_sparse \*cholmod_submatrix/, /\*\) ;/' ../Include/cholmod.h > _submatrix.tex
	./getproto '/cholmod_sparse \*cholmod_vertcat/, /\*\) ;/' ../Include/cholmod.h > _vertcat.tex
	./getproto '/int cholmod_symmetry/, /\*\) ;/' ../Include/cholmod.h > _symmetry.tex
	./getproto '/int cholmod_super_symbolic/, /\*\) ;/' ../Include/cholmod.h > _super_symbolic.tex
	./getproto '/int cholmod_super_numeric/, /\*\) ;/' ../Include/cholmod.h > _super_numeric.tex
	./getproto '/int cholmod_super_lsolve/, /\*\) ;/' ../Include/cholmod.h > _super_lsolve.tex
	./getproto '/int cholmod_super_ltsolve/, /\*\) ;/' ../Include/cholmod.h > _super_ltsolve.tex
	./getproto '/int64_t cholmod_nested_dissection/, /\*\) ;/' ../Include/cholmod.h > _nested_dissection.tex
	./getproto '/int cholmod_metis/, /\*\) ;/' ../Include/cholmod.h > _metis.tex
	./getproto '/int cholmod_ccolamd/, /\*\) ;/' ../Include/cholmod.h > _ccolamd.tex
	./getproto '/int cholmod_camd/, /\*\) ;/' ../Include/cholmod.h > _camd.tex
	./getproto '/int cholmod_csymamd/, /\*\) ;/' ../Include/cholmod.h > _csymamd.tex
	./getproto '/int cholmod_csymamd/, /\*\) ;/' ../Include/cholmod.h > _csymamd.tex
	./getproto '/int64_t cholmod_bisect/, /\*\) ;/' ../Include/cholmod.h > _bisect.tex
	./getproto '/int64_t cholmod_metis_bisector/, /\*\) ;/' ../Include/cholmod.h > _metis_bisector.tex
	./getproto '/int64_t cholmod_collapse_septree/, /\*\) ;/' ../Include/cholmod.h > _collapse_septree.tex
	pdflatex CHOLMOD_UserGuide
	bibtex CHOLMOD_UserGuide
	pdflatex CHOLMOD_UserGuide
	pdflatex CHOLMOD_UserGuide
	- $(RM) _temp.awk
	- $(RM) -r *.aux *.bbl *.blg *.log *.toc *.dvi *.lof *.lot

distclean: purge

purge: clean

clean:
	- $(RM) _*.tex
	- $(RM) _temp.awk
	- $(RM) -r *.aux *.bbl *.blg *.log *.toc *.dvi *.lof *.lot

