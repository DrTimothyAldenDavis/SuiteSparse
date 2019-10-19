#-------------------------------------------------------------------------------
# UMFPACK Makefile for compiling on Unix systems (for original make only)
#-------------------------------------------------------------------------------

# This is a very ugly Makefile, and is only provided for those who do not
# have GNU make.  Note that it is not used if you have GNU make.  It ignores
# dependency checking and just compiles everything.

default: everything

include ../../UFconfig/UFconfig.mk

C = $(CC) $(CF) $(UMFPACK_CONFIG) \
    -I../Include -I../Source -I../../AMD/Include -I../../UFconfig \
    -I../../CHOLMOD/Include

everything:
	$(C) -c ../Source/umfpack_global.c -o umfpack_gn_global.o
	$(C) -DDINT -c ../Source/umf_analyze.c -o umf_i_analyze.o
	$(C) -DDINT -c ../Source/umf_apply_order.c -o umf_i_apply_order.o
	$(C) -DDINT -c ../Source/umf_colamd.c -o umf_i_colamd.o
	$(C) -DDINT -c ../Source/umf_cholmod.c -o umf_i_cholmod.o
	$(C) -DDINT -c ../Source/umf_free.c -o umf_i_free.o
	$(C) -DDINT -c ../Source/umf_fsize.c -o umf_i_fsize.o
	$(C) -DDINT -c ../Source/umf_is_permutation.c -o umf_i_is_permutation.o
	$(C) -DDINT -c ../Source/umf_malloc.c -o umf_i_malloc.o
	$(C) -DDINT -c ../Source/umf_realloc.c -o umf_i_realloc.o
	$(C) -DDINT -c ../Source/umf_report_perm.c -o umf_i_report_perm.o
	$(C) -DDINT -c ../Source/umf_singletons.c -o umf_i_singletons.o
	$(C) -DDLONG -c ../Source/umf_analyze.c -o umf_l_analyze.o
	$(C) -DDLONG -c ../Source/umf_apply_order.c -o umf_l_apply_order.o
	$(C) -DDLONG -c ../Source/umf_colamd.c -o umf_l_colamd.o
	$(C) -DDLONG -c ../Source/umf_cholmod.c -o umf_l_cholmod.o
	$(C) -DDLONG -c ../Source/umf_free.c -o umf_l_free.o
	$(C) -DDLONG -c ../Source/umf_fsize.c -o umf_l_fsize.o
	$(C) -DDLONG -c ../Source/umf_is_permutation.c -o umf_l_is_permutation.o
	$(C) -DDLONG -c ../Source/umf_malloc.c -o umf_l_malloc.o
	$(C) -DDLONG -c ../Source/umf_realloc.c -o umf_l_realloc.o
	$(C) -DDLONG -c ../Source/umf_report_perm.c -o umf_l_report_perm.o
	$(C) -DDLONG -c ../Source/umf_singletons.c -o umf_l_singletons.o
	$(C) -c ../Source/umfpack_timer.c -o umfpack_gn_timer.o
	$(C) -c ../Source/umfpack_tictoc.c -o umfpack_gn_tictoc.o
	$(C) -DDINT -DCONJUGATE_SOLVE -c ../Source/umf_ltsolve.c -o umf_di_lhsolve.o
	$(C) -DDINT -DCONJUGATE_SOLVE -c ../Source/umf_utsolve.c -o umf_di_uhsolve.o
	$(C) -DDINT -DDO_MAP -c ../Source/umf_triplet.c -o umf_di_triplet_map_nox.o
	$(C) -DDINT -DDO_VALUES -c ../Source/umf_triplet.c -o umf_di_triplet_nomap_x.o
	$(C) -DDINT -c ../Source/umf_triplet.c -o umf_di_triplet_nomap_nox.o
	$(C) -DDINT -DDO_MAP -DDO_VALUES -c ../Source/umf_triplet.c -o umf_di_triplet_map_x.o
	$(C) -DDINT -DFIXQ -c ../Source/umf_assemble.c -o umf_di_assemble_fixq.o
	$(C) -DDINT -DDROP -c ../Source/umf_store_lu.c -o umf_di_store_lu_drop.o
	$(C) -DDINT -c ../Source/umf_assemble.c -o umf_di_assemble.o
	$(C) -DDINT -c ../Source/umf_blas3_update.c -o umf_di_blas3_update.o
	$(C) -DDINT -c ../Source/umf_build_tuples.c -o umf_di_build_tuples.o
	$(C) -DDINT -c ../Source/umf_create_element.c -o umf_di_create_element.o
	$(C) -DDINT -c ../Source/umf_dump.c -o umf_di_dump.o
	$(C) -DDINT -c ../Source/umf_extend_front.c -o umf_di_extend_front.o
	$(C) -DDINT -c ../Source/umf_garbage_collection.c -o umf_di_garbage_collection.o
	$(C) -DDINT -c ../Source/umf_get_memory.c -o umf_di_get_memory.o
	$(C) -DDINT -c ../Source/umf_init_front.c -o umf_di_init_front.o
	$(C) -DDINT -c ../Source/umf_kernel.c -o umf_di_kernel.o
	$(C) -DDINT -c ../Source/umf_kernel_init.c -o umf_di_kernel_init.o
	$(C) -DDINT -c ../Source/umf_kernel_wrapup.c -o umf_di_kernel_wrapup.o
	$(C) -DDINT -c ../Source/umf_local_search.c -o umf_di_local_search.o
	$(C) -DDINT -c ../Source/umf_lsolve.c -o umf_di_lsolve.o
	$(C) -DDINT -c ../Source/umf_ltsolve.c -o umf_di_ltsolve.o
	$(C) -DDINT -c ../Source/umf_mem_alloc_element.c -o umf_di_mem_alloc_element.o
	$(C) -DDINT -c ../Source/umf_mem_alloc_head_block.c -o umf_di_mem_alloc_head_block.o
	$(C) -DDINT -c ../Source/umf_mem_alloc_tail_block.c -o umf_di_mem_alloc_tail_block.o
	$(C) -DDINT -c ../Source/umf_mem_free_tail_block.c -o umf_di_mem_free_tail_block.o
	$(C) -DDINT -c ../Source/umf_mem_init_memoryspace.c -o umf_di_mem_init_memoryspace.o
	$(C) -DDINT -c ../Source/umf_report_vector.c -o umf_di_report_vector.o
	$(C) -DDINT -c ../Source/umf_row_search.c -o umf_di_row_search.o
	$(C) -DDINT -c ../Source/umf_scale_column.c -o umf_di_scale_column.o
	$(C) -DDINT -c ../Source/umf_set_stats.c -o umf_di_set_stats.o
	$(C) -DDINT -c ../Source/umf_solve.c -o umf_di_solve.o
	$(C) -DDINT -c ../Source/umf_symbolic_usage.c -o umf_di_symbolic_usage.o
	$(C) -DDINT -c ../Source/umf_transpose.c -o umf_di_transpose.o
	$(C) -DDINT -c ../Source/umf_tuple_lengths.c -o umf_di_tuple_lengths.o
	$(C) -DDINT -c ../Source/umf_usolve.c -o umf_di_usolve.o
	$(C) -DDINT -c ../Source/umf_utsolve.c -o umf_di_utsolve.o
	$(C) -DDINT -c ../Source/umf_valid_numeric.c -o umf_di_valid_numeric.o
	$(C) -DDINT -c ../Source/umf_valid_symbolic.c -o umf_di_valid_symbolic.o
	$(C) -DDINT -c ../Source/umf_grow_front.c -o umf_di_grow_front.o
	$(C) -DDINT -c ../Source/umf_start_front.c -o umf_di_start_front.o
	$(C) -DDINT -c ../Source/umf_store_lu.c -o umf_di_store_lu.o
	$(C) -DDINT -c ../Source/umf_scale.c -o umf_di_scale.o
	$(C) -DDINT -DWSOLVE -c ../Source/umfpack_solve.c -o umfpack_di_wsolve.o
	$(C) -DDINT -c ../Source/umfpack_col_to_triplet.c -o umfpack_di_col_to_triplet.o
	$(C) -DDINT -c ../Source/umfpack_defaults.c -o umfpack_di_defaults.o
	$(C) -DDINT -c ../Source/umfpack_free_numeric.c -o umfpack_di_free_numeric.o
	$(C) -DDINT -c ../Source/umfpack_free_symbolic.c -o umfpack_di_free_symbolic.o
	$(C) -DDINT -c ../Source/umfpack_get_numeric.c -o umfpack_di_get_numeric.o
	$(C) -DDINT -c ../Source/umfpack_get_lunz.c -o umfpack_di_get_lunz.o
	$(C) -DDINT -c ../Source/umfpack_get_symbolic.c -o umfpack_di_get_symbolic.o
	$(C) -DDINT -c ../Source/umfpack_get_determinant.c -o umfpack_di_get_determinant.o
	$(C) -DDINT -c ../Source/umfpack_numeric.c -o umfpack_di_numeric.o
	$(C) -DDINT -c ../Source/umfpack_qsymbolic.c -o umfpack_di_qsymbolic.o
	$(C) -DDINT -c ../Source/umfpack_report_control.c -o umfpack_di_report_control.o
	$(C) -DDINT -c ../Source/umfpack_report_info.c -o umfpack_di_report_info.o
	$(C) -DDINT -c ../Source/umfpack_report_matrix.c -o umfpack_di_report_matrix.o
	$(C) -DDINT -c ../Source/umfpack_report_numeric.c -o umfpack_di_report_numeric.o
	$(C) -DDINT -c ../Source/umfpack_report_perm.c -o umfpack_di_report_perm.o
	$(C) -DDINT -c ../Source/umfpack_report_status.c -o umfpack_di_report_status.o
	$(C) -DDINT -c ../Source/umfpack_report_symbolic.c -o umfpack_di_report_symbolic.o
	$(C) -DDINT -c ../Source/umfpack_report_triplet.c -o umfpack_di_report_triplet.o
	$(C) -DDINT -c ../Source/umfpack_report_vector.c -o umfpack_di_report_vector.o
	$(C) -DDINT -c ../Source/umfpack_solve.c -o umfpack_di_solve.o
	$(C) -DDINT -c ../Source/umfpack_symbolic.c -o umfpack_di_symbolic.o
	$(C) -DDINT -c ../Source/umfpack_transpose.c -o umfpack_di_transpose.o
	$(C) -DDINT -c ../Source/umfpack_triplet_to_col.c -o umfpack_di_triplet_to_col.o
	$(C) -DDINT -c ../Source/umfpack_scale.c -o umfpack_di_scale.o
	$(C) -DDINT -c ../Source/umfpack_load_numeric.c -o umfpack_di_load_numeric.o
	$(C) -DDINT -c ../Source/umfpack_save_numeric.c -o umfpack_di_save_numeric.o
	$(C) -DDINT -c ../Source/umfpack_load_symbolic.c -o umfpack_di_load_symbolic.o
	$(C) -DDINT -c ../Source/umfpack_save_symbolic.c -o umfpack_di_save_symbolic.o
	$(C) -DDLONG -DCONJUGATE_SOLVE -c ../Source/umf_ltsolve.c -o umf_dl_lhsolve.o
	$(C) -DDLONG -DCONJUGATE_SOLVE -c ../Source/umf_utsolve.c -o umf_dl_uhsolve.o
	$(C) -DDLONG -DDO_MAP -c ../Source/umf_triplet.c -o umf_dl_triplet_map_nox.o
	$(C) -DDLONG -DDO_VALUES -c ../Source/umf_triplet.c -o umf_dl_triplet_nomap_x.o
	$(C) -DDLONG -c ../Source/umf_triplet.c -o umf_dl_triplet_nomap_nox.o
	$(C) -DDLONG -DDO_MAP -DDO_VALUES -c ../Source/umf_triplet.c -o umf_dl_triplet_map_x.o
	$(C) -DDLONG -DFIXQ -c ../Source/umf_assemble.c -o umf_dl_assemble_fixq.o
	$(C) -DDLONG -DDROP -c ../Source/umf_store_lu.c -o umf_dl_store_lu_drop.o
	$(C) -DDLONG -c ../Source/umf_assemble.c -o umf_dl_assemble.o
	$(C) -DDLONG -c ../Source/umf_blas3_update.c -o umf_dl_blas3_update.o
	$(C) -DDLONG -c ../Source/umf_build_tuples.c -o umf_dl_build_tuples.o
	$(C) -DDLONG -c ../Source/umf_create_element.c -o umf_dl_create_element.o
	$(C) -DDLONG -c ../Source/umf_dump.c -o umf_dl_dump.o
	$(C) -DDLONG -c ../Source/umf_extend_front.c -o umf_dl_extend_front.o
	$(C) -DDLONG -c ../Source/umf_garbage_collection.c -o umf_dl_garbage_collection.o
	$(C) -DDLONG -c ../Source/umf_get_memory.c -o umf_dl_get_memory.o
	$(C) -DDLONG -c ../Source/umf_init_front.c -o umf_dl_init_front.o
	$(C) -DDLONG -c ../Source/umf_kernel.c -o umf_dl_kernel.o
	$(C) -DDLONG -c ../Source/umf_kernel_init.c -o umf_dl_kernel_init.o
	$(C) -DDLONG -c ../Source/umf_kernel_wrapup.c -o umf_dl_kernel_wrapup.o
	$(C) -DDLONG -c ../Source/umf_local_search.c -o umf_dl_local_search.o
	$(C) -DDLONG -c ../Source/umf_lsolve.c -o umf_dl_lsolve.o
	$(C) -DDLONG -c ../Source/umf_ltsolve.c -o umf_dl_ltsolve.o
	$(C) -DDLONG -c ../Source/umf_mem_alloc_element.c -o umf_dl_mem_alloc_element.o
	$(C) -DDLONG -c ../Source/umf_mem_alloc_head_block.c -o umf_dl_mem_alloc_head_block.o
	$(C) -DDLONG -c ../Source/umf_mem_alloc_tail_block.c -o umf_dl_mem_alloc_tail_block.o
	$(C) -DDLONG -c ../Source/umf_mem_free_tail_block.c -o umf_dl_mem_free_tail_block.o
	$(C) -DDLONG -c ../Source/umf_mem_init_memoryspace.c -o umf_dl_mem_init_memoryspace.o
	$(C) -DDLONG -c ../Source/umf_report_vector.c -o umf_dl_report_vector.o
	$(C) -DDLONG -c ../Source/umf_row_search.c -o umf_dl_row_search.o
	$(C) -DDLONG -c ../Source/umf_scale_column.c -o umf_dl_scale_column.o
	$(C) -DDLONG -c ../Source/umf_set_stats.c -o umf_dl_set_stats.o
	$(C) -DDLONG -c ../Source/umf_solve.c -o umf_dl_solve.o
	$(C) -DDLONG -c ../Source/umf_symbolic_usage.c -o umf_dl_symbolic_usage.o
	$(C) -DDLONG -c ../Source/umf_transpose.c -o umf_dl_transpose.o
	$(C) -DDLONG -c ../Source/umf_tuple_lengths.c -o umf_dl_tuple_lengths.o
	$(C) -DDLONG -c ../Source/umf_usolve.c -o umf_dl_usolve.o
	$(C) -DDLONG -c ../Source/umf_utsolve.c -o umf_dl_utsolve.o
	$(C) -DDLONG -c ../Source/umf_valid_numeric.c -o umf_dl_valid_numeric.o
	$(C) -DDLONG -c ../Source/umf_valid_symbolic.c -o umf_dl_valid_symbolic.o
	$(C) -DDLONG -c ../Source/umf_grow_front.c -o umf_dl_grow_front.o
	$(C) -DDLONG -c ../Source/umf_start_front.c -o umf_dl_start_front.o
	$(C) -DDLONG -c ../Source/umf_store_lu.c -o umf_dl_store_lu.o
	$(C) -DDLONG -c ../Source/umf_scale.c -o umf_dl_scale.o
	$(C) -DDLONG -DWSOLVE -c ../Source/umfpack_solve.c -o umfpack_dl_wsolve.o
	$(C) -DDLONG -c ../Source/umfpack_col_to_triplet.c -o umfpack_dl_col_to_triplet.o
	$(C) -DDLONG -c ../Source/umfpack_defaults.c -o umfpack_dl_defaults.o
	$(C) -DDLONG -c ../Source/umfpack_free_numeric.c -o umfpack_dl_free_numeric.o
	$(C) -DDLONG -c ../Source/umfpack_free_symbolic.c -o umfpack_dl_free_symbolic.o
	$(C) -DDLONG -c ../Source/umfpack_get_numeric.c -o umfpack_dl_get_numeric.o
	$(C) -DDLONG -c ../Source/umfpack_get_lunz.c -o umfpack_dl_get_lunz.o
	$(C) -DDLONG -c ../Source/umfpack_get_symbolic.c -o umfpack_dl_get_symbolic.o
	$(C) -DDLONG -c ../Source/umfpack_get_determinant.c -o umfpack_dl_get_determinant.o
	$(C) -DDLONG -c ../Source/umfpack_numeric.c -o umfpack_dl_numeric.o
	$(C) -DDLONG -c ../Source/umfpack_qsymbolic.c -o umfpack_dl_qsymbolic.o
	$(C) -DDLONG -c ../Source/umfpack_report_control.c -o umfpack_dl_report_control.o
	$(C) -DDLONG -c ../Source/umfpack_report_info.c -o umfpack_dl_report_info.o
	$(C) -DDLONG -c ../Source/umfpack_report_matrix.c -o umfpack_dl_report_matrix.o
	$(C) -DDLONG -c ../Source/umfpack_report_numeric.c -o umfpack_dl_report_numeric.o
	$(C) -DDLONG -c ../Source/umfpack_report_perm.c -o umfpack_dl_report_perm.o
	$(C) -DDLONG -c ../Source/umfpack_report_status.c -o umfpack_dl_report_status.o
	$(C) -DDLONG -c ../Source/umfpack_report_symbolic.c -o umfpack_dl_report_symbolic.o
	$(C) -DDLONG -c ../Source/umfpack_report_triplet.c -o umfpack_dl_report_triplet.o
	$(C) -DDLONG -c ../Source/umfpack_report_vector.c -o umfpack_dl_report_vector.o
	$(C) -DDLONG -c ../Source/umfpack_solve.c -o umfpack_dl_solve.o
	$(C) -DDLONG -c ../Source/umfpack_symbolic.c -o umfpack_dl_symbolic.o
	$(C) -DDLONG -c ../Source/umfpack_transpose.c -o umfpack_dl_transpose.o
	$(C) -DDLONG -c ../Source/umfpack_triplet_to_col.c -o umfpack_dl_triplet_to_col.o
	$(C) -DDLONG -c ../Source/umfpack_scale.c -o umfpack_dl_scale.o
	$(C) -DDLONG -c ../Source/umfpack_load_numeric.c -o umfpack_dl_load_numeric.o
	$(C) -DDLONG -c ../Source/umfpack_save_numeric.c -o umfpack_dl_save_numeric.o
	$(C) -DDLONG -c ../Source/umfpack_load_symbolic.c -o umfpack_dl_load_symbolic.o
	$(C) -DDLONG -c ../Source/umfpack_save_symbolic.c -o umfpack_dl_save_symbolic.o
	$(C) -DZINT -DCONJUGATE_SOLVE -c ../Source/umf_ltsolve.c -o umf_zi_lhsolve.o
	$(C) -DZINT -DCONJUGATE_SOLVE -c ../Source/umf_utsolve.c -o umf_zi_uhsolve.o
	$(C) -DZINT -DDO_MAP -c ../Source/umf_triplet.c -o umf_zi_triplet_map_nox.o
	$(C) -DZINT -DDO_VALUES -c ../Source/umf_triplet.c -o umf_zi_triplet_nomap_x.o
	$(C) -DZINT -c ../Source/umf_triplet.c -o umf_zi_triplet_nomap_nox.o
	$(C) -DZINT -DDO_MAP -DDO_VALUES -c ../Source/umf_triplet.c -o umf_zi_triplet_map_x.o
	$(C) -DZINT -DFIXQ -c ../Source/umf_assemble.c -o umf_zi_assemble_fixq.o
	$(C) -DZINT -DDROP -c ../Source/umf_store_lu.c -o umf_zi_store_lu_drop.o
	$(C) -DZINT -c ../Source/umf_assemble.c -o umf_zi_assemble.o
	$(C) -DZINT -c ../Source/umf_blas3_update.c -o umf_zi_blas3_update.o
	$(C) -DZINT -c ../Source/umf_build_tuples.c -o umf_zi_build_tuples.o
	$(C) -DZINT -c ../Source/umf_create_element.c -o umf_zi_create_element.o
	$(C) -DZINT -c ../Source/umf_dump.c -o umf_zi_dump.o
	$(C) -DZINT -c ../Source/umf_extend_front.c -o umf_zi_extend_front.o
	$(C) -DZINT -c ../Source/umf_garbage_collection.c -o umf_zi_garbage_collection.o
	$(C) -DZINT -c ../Source/umf_get_memory.c -o umf_zi_get_memory.o
	$(C) -DZINT -c ../Source/umf_init_front.c -o umf_zi_init_front.o
	$(C) -DZINT -c ../Source/umf_kernel.c -o umf_zi_kernel.o
	$(C) -DZINT -c ../Source/umf_kernel_init.c -o umf_zi_kernel_init.o
	$(C) -DZINT -c ../Source/umf_kernel_wrapup.c -o umf_zi_kernel_wrapup.o
	$(C) -DZINT -c ../Source/umf_local_search.c -o umf_zi_local_search.o
	$(C) -DZINT -c ../Source/umf_lsolve.c -o umf_zi_lsolve.o
	$(C) -DZINT -c ../Source/umf_ltsolve.c -o umf_zi_ltsolve.o
	$(C) -DZINT -c ../Source/umf_mem_alloc_element.c -o umf_zi_mem_alloc_element.o
	$(C) -DZINT -c ../Source/umf_mem_alloc_head_block.c -o umf_zi_mem_alloc_head_block.o
	$(C) -DZINT -c ../Source/umf_mem_alloc_tail_block.c -o umf_zi_mem_alloc_tail_block.o
	$(C) -DZINT -c ../Source/umf_mem_free_tail_block.c -o umf_zi_mem_free_tail_block.o
	$(C) -DZINT -c ../Source/umf_mem_init_memoryspace.c -o umf_zi_mem_init_memoryspace.o
	$(C) -DZINT -c ../Source/umf_report_vector.c -o umf_zi_report_vector.o
	$(C) -DZINT -c ../Source/umf_row_search.c -o umf_zi_row_search.o
	$(C) -DZINT -c ../Source/umf_scale_column.c -o umf_zi_scale_column.o
	$(C) -DZINT -c ../Source/umf_set_stats.c -o umf_zi_set_stats.o
	$(C) -DZINT -c ../Source/umf_solve.c -o umf_zi_solve.o
	$(C) -DZINT -c ../Source/umf_symbolic_usage.c -o umf_zi_symbolic_usage.o
	$(C) -DZINT -c ../Source/umf_transpose.c -o umf_zi_transpose.o
	$(C) -DZINT -c ../Source/umf_tuple_lengths.c -o umf_zi_tuple_lengths.o
	$(C) -DZINT -c ../Source/umf_usolve.c -o umf_zi_usolve.o
	$(C) -DZINT -c ../Source/umf_utsolve.c -o umf_zi_utsolve.o
	$(C) -DZINT -c ../Source/umf_valid_numeric.c -o umf_zi_valid_numeric.o
	$(C) -DZINT -c ../Source/umf_valid_symbolic.c -o umf_zi_valid_symbolic.o
	$(C) -DZINT -c ../Source/umf_grow_front.c -o umf_zi_grow_front.o
	$(C) -DZINT -c ../Source/umf_start_front.c -o umf_zi_start_front.o
	$(C) -DZINT -c ../Source/umf_store_lu.c -o umf_zi_store_lu.o
	$(C) -DZINT -c ../Source/umf_scale.c -o umf_zi_scale.o
	$(C) -DZINT -DWSOLVE -c ../Source/umfpack_solve.c -o umfpack_zi_wsolve.o
	$(C) -DZINT -c ../Source/umfpack_col_to_triplet.c -o umfpack_zi_col_to_triplet.o
	$(C) -DZINT -c ../Source/umfpack_defaults.c -o umfpack_zi_defaults.o
	$(C) -DZINT -c ../Source/umfpack_free_numeric.c -o umfpack_zi_free_numeric.o
	$(C) -DZINT -c ../Source/umfpack_free_symbolic.c -o umfpack_zi_free_symbolic.o
	$(C) -DZINT -c ../Source/umfpack_get_numeric.c -o umfpack_zi_get_numeric.o
	$(C) -DZINT -c ../Source/umfpack_get_lunz.c -o umfpack_zi_get_lunz.o
	$(C) -DZINT -c ../Source/umfpack_get_symbolic.c -o umfpack_zi_get_symbolic.o
	$(C) -DZINT -c ../Source/umfpack_get_determinant.c -o umfpack_zi_get_determinant.o
	$(C) -DZINT -c ../Source/umfpack_numeric.c -o umfpack_zi_numeric.o
	$(C) -DZINT -c ../Source/umfpack_qsymbolic.c -o umfpack_zi_qsymbolic.o
	$(C) -DZINT -c ../Source/umfpack_report_control.c -o umfpack_zi_report_control.o
	$(C) -DZINT -c ../Source/umfpack_report_info.c -o umfpack_zi_report_info.o
	$(C) -DZINT -c ../Source/umfpack_report_matrix.c -o umfpack_zi_report_matrix.o
	$(C) -DZINT -c ../Source/umfpack_report_numeric.c -o umfpack_zi_report_numeric.o
	$(C) -DZINT -c ../Source/umfpack_report_perm.c -o umfpack_zi_report_perm.o
	$(C) -DZINT -c ../Source/umfpack_report_status.c -o umfpack_zi_report_status.o
	$(C) -DZINT -c ../Source/umfpack_report_symbolic.c -o umfpack_zi_report_symbolic.o
	$(C) -DZINT -c ../Source/umfpack_report_triplet.c -o umfpack_zi_report_triplet.o
	$(C) -DZINT -c ../Source/umfpack_report_vector.c -o umfpack_zi_report_vector.o
	$(C) -DZINT -c ../Source/umfpack_solve.c -o umfpack_zi_solve.o
	$(C) -DZINT -c ../Source/umfpack_symbolic.c -o umfpack_zi_symbolic.o
	$(C) -DZINT -c ../Source/umfpack_transpose.c -o umfpack_zi_transpose.o
	$(C) -DZINT -c ../Source/umfpack_triplet_to_col.c -o umfpack_zi_triplet_to_col.o
	$(C) -DZINT -c ../Source/umfpack_scale.c -o umfpack_zi_scale.o
	$(C) -DZINT -c ../Source/umfpack_load_numeric.c -o umfpack_zi_load_numeric.o
	$(C) -DZINT -c ../Source/umfpack_save_numeric.c -o umfpack_zi_save_numeric.o
	$(C) -DZINT -c ../Source/umfpack_load_symbolic.c -o umfpack_zi_load_symbolic.o
	$(C) -DZINT -c ../Source/umfpack_save_symbolic.c -o umfpack_zi_save_symbolic.o
	$(C) -DZLONG -DCONJUGATE_SOLVE -c ../Source/umf_ltsolve.c -o umf_zl_lhsolve.o
	$(C) -DZLONG -DCONJUGATE_SOLVE -c ../Source/umf_utsolve.c -o umf_zl_uhsolve.o
	$(C) -DZLONG -DDO_MAP -c ../Source/umf_triplet.c -o umf_zl_triplet_map_nox.o
	$(C) -DZLONG -DDO_VALUES -c ../Source/umf_triplet.c -o umf_zl_triplet_nomap_x.o
	$(C) -DZLONG -c ../Source/umf_triplet.c -o umf_zl_triplet_nomap_nox.o
	$(C) -DZLONG -DDO_MAP -DDO_VALUES -c ../Source/umf_triplet.c -o umf_zl_triplet_map_x.o
	$(C) -DZLONG -DFIXQ -c ../Source/umf_assemble.c -o umf_zl_assemble_fixq.o
	$(C) -DZLONG -DDROP -c ../Source/umf_store_lu.c -o umf_zl_store_lu_drop.o
	$(C) -DZLONG -c ../Source/umf_assemble.c -o umf_zl_assemble.o
	$(C) -DZLONG -c ../Source/umf_blas3_update.c -o umf_zl_blas3_update.o
	$(C) -DZLONG -c ../Source/umf_build_tuples.c -o umf_zl_build_tuples.o
	$(C) -DZLONG -c ../Source/umf_create_element.c -o umf_zl_create_element.o
	$(C) -DZLONG -c ../Source/umf_dump.c -o umf_zl_dump.o
	$(C) -DZLONG -c ../Source/umf_extend_front.c -o umf_zl_extend_front.o
	$(C) -DZLONG -c ../Source/umf_garbage_collection.c -o umf_zl_garbage_collection.o
	$(C) -DZLONG -c ../Source/umf_get_memory.c -o umf_zl_get_memory.o
	$(C) -DZLONG -c ../Source/umf_init_front.c -o umf_zl_init_front.o
	$(C) -DZLONG -c ../Source/umf_kernel.c -o umf_zl_kernel.o
	$(C) -DZLONG -c ../Source/umf_kernel_init.c -o umf_zl_kernel_init.o
	$(C) -DZLONG -c ../Source/umf_kernel_wrapup.c -o umf_zl_kernel_wrapup.o
	$(C) -DZLONG -c ../Source/umf_local_search.c -o umf_zl_local_search.o
	$(C) -DZLONG -c ../Source/umf_lsolve.c -o umf_zl_lsolve.o
	$(C) -DZLONG -c ../Source/umf_ltsolve.c -o umf_zl_ltsolve.o
	$(C) -DZLONG -c ../Source/umf_mem_alloc_element.c -o umf_zl_mem_alloc_element.o
	$(C) -DZLONG -c ../Source/umf_mem_alloc_head_block.c -o umf_zl_mem_alloc_head_block.o
	$(C) -DZLONG -c ../Source/umf_mem_alloc_tail_block.c -o umf_zl_mem_alloc_tail_block.o
	$(C) -DZLONG -c ../Source/umf_mem_free_tail_block.c -o umf_zl_mem_free_tail_block.o
	$(C) -DZLONG -c ../Source/umf_mem_init_memoryspace.c -o umf_zl_mem_init_memoryspace.o
	$(C) -DZLONG -c ../Source/umf_report_vector.c -o umf_zl_report_vector.o
	$(C) -DZLONG -c ../Source/umf_row_search.c -o umf_zl_row_search.o
	$(C) -DZLONG -c ../Source/umf_scale_column.c -o umf_zl_scale_column.o
	$(C) -DZLONG -c ../Source/umf_set_stats.c -o umf_zl_set_stats.o
	$(C) -DZLONG -c ../Source/umf_solve.c -o umf_zl_solve.o
	$(C) -DZLONG -c ../Source/umf_symbolic_usage.c -o umf_zl_symbolic_usage.o
	$(C) -DZLONG -c ../Source/umf_transpose.c -o umf_zl_transpose.o
	$(C) -DZLONG -c ../Source/umf_tuple_lengths.c -o umf_zl_tuple_lengths.o
	$(C) -DZLONG -c ../Source/umf_usolve.c -o umf_zl_usolve.o
	$(C) -DZLONG -c ../Source/umf_utsolve.c -o umf_zl_utsolve.o
	$(C) -DZLONG -c ../Source/umf_valid_numeric.c -o umf_zl_valid_numeric.o
	$(C) -DZLONG -c ../Source/umf_valid_symbolic.c -o umf_zl_valid_symbolic.o
	$(C) -DZLONG -c ../Source/umf_grow_front.c -o umf_zl_grow_front.o
	$(C) -DZLONG -c ../Source/umf_start_front.c -o umf_zl_start_front.o
	$(C) -DZLONG -c ../Source/umf_store_lu.c -o umf_zl_store_lu.o
	$(C) -DZLONG -c ../Source/umf_scale.c -o umf_zl_scale.o
	$(C) -DZLONG -DWSOLVE -c ../Source/umfpack_solve.c -o umfpack_zl_wsolve.o
	$(C) -DZLONG -c ../Source/umfpack_col_to_triplet.c -o umfpack_zl_col_to_triplet.o
	$(C) -DZLONG -c ../Source/umfpack_defaults.c -o umfpack_zl_defaults.o
	$(C) -DZLONG -c ../Source/umfpack_free_numeric.c -o umfpack_zl_free_numeric.o
	$(C) -DZLONG -c ../Source/umfpack_free_symbolic.c -o umfpack_zl_free_symbolic.o
	$(C) -DZLONG -c ../Source/umfpack_get_numeric.c -o umfpack_zl_get_numeric.o
	$(C) -DZLONG -c ../Source/umfpack_get_lunz.c -o umfpack_zl_get_lunz.o
	$(C) -DZLONG -c ../Source/umfpack_get_symbolic.c -o umfpack_zl_get_symbolic.o
	$(C) -DZLONG -c ../Source/umfpack_get_determinant.c -o umfpack_zl_get_determinant.o
	$(C) -DZLONG -c ../Source/umfpack_numeric.c -o umfpack_zl_numeric.o
	$(C) -DZLONG -c ../Source/umfpack_qsymbolic.c -o umfpack_zl_qsymbolic.o
	$(C) -DZLONG -c ../Source/umfpack_report_control.c -o umfpack_zl_report_control.o
	$(C) -DZLONG -c ../Source/umfpack_report_info.c -o umfpack_zl_report_info.o
	$(C) -DZLONG -c ../Source/umfpack_report_matrix.c -o umfpack_zl_report_matrix.o
	$(C) -DZLONG -c ../Source/umfpack_report_numeric.c -o umfpack_zl_report_numeric.o
	$(C) -DZLONG -c ../Source/umfpack_report_perm.c -o umfpack_zl_report_perm.o
	$(C) -DZLONG -c ../Source/umfpack_report_status.c -o umfpack_zl_report_status.o
	$(C) -DZLONG -c ../Source/umfpack_report_symbolic.c -o umfpack_zl_report_symbolic.o
	$(C) -DZLONG -c ../Source/umfpack_report_triplet.c -o umfpack_zl_report_triplet.o
	$(C) -DZLONG -c ../Source/umfpack_report_vector.c -o umfpack_zl_report_vector.o
	$(C) -DZLONG -c ../Source/umfpack_solve.c -o umfpack_zl_solve.o
	$(C) -DZLONG -c ../Source/umfpack_symbolic.c -o umfpack_zl_symbolic.o
	$(C) -DZLONG -c ../Source/umfpack_transpose.c -o umfpack_zl_transpose.o
	$(C) -DZLONG -c ../Source/umfpack_triplet_to_col.c -o umfpack_zl_triplet_to_col.o
	$(C) -DZLONG -c ../Source/umfpack_scale.c -o umfpack_zl_scale.o
	$(C) -DZLONG -c ../Source/umfpack_load_numeric.c -o umfpack_zl_load_numeric.o
	$(C) -DZLONG -c ../Source/umfpack_save_numeric.c -o umfpack_zl_save_numeric.o
	$(C) -DZLONG -c ../Source/umfpack_load_symbolic.c -o umfpack_zl_load_symbolic.o
	$(C) -DZLONG -c ../Source/umfpack_save_symbolic.c -o umfpack_zl_save_symbolic.o
	$(ARCHIVE)  ../Lib/libumfpack.a \
	    umfpack_gn_global.o \
	    umf_i_analyze.o umf_i_apply_order.o umf_i_colamd.o umf_i_free.o \
            umf_i_cholmod.o umf_i_fsize.o umf_i_is_permutation.o \
            umf_i_malloc.o umf_i_realloc.o \
	    umf_i_report_perm.o umf_i_singletons.o \
	    umf_l_analyze.o umf_l_apply_order.o umf_l_colamd.o umf_l_free.o \
            umf_l_cholmod.o umf_l_fsize.o umf_l_is_permutation.o \
            umf_l_malloc.o umf_l_realloc.o \
	    umf_l_report_perm.o umf_l_singletons.o \
	    umfpack_gn_timer.o umfpack_gn_tictoc.o \
	    umf_di_lhsolve.o \
	    umf_di_uhsolve.o umf_di_triplet_map_nox.o umf_di_triplet_nomap_x.o \
	    umf_di_triplet_nomap_nox.o umf_di_triplet_map_x.o \
	    umf_di_assemble_fixq.o umf_di_store_lu_drop.o umf_di_assemble.o \
	    umf_di_blas3_update.o umf_di_build_tuples.o \
	    umf_di_create_element.o umf_di_dump.o umf_di_extend_front.o \
	    umf_di_garbage_collection.o umf_di_get_memory.o \
	    umf_di_init_front.o umf_di_kernel.o umf_di_kernel_init.o \
	    umf_di_kernel_wrapup.o umf_di_local_search.o umf_di_lsolve.o \
	    umf_di_ltsolve.o umf_di_mem_alloc_element.o \
	    umf_di_mem_alloc_head_block.o umf_di_mem_alloc_tail_block.o \
	    umf_di_mem_free_tail_block.o umf_di_mem_init_memoryspace.o \
	    umf_di_report_vector.o umf_di_row_search.o umf_di_scale_column.o \
	    umf_di_set_stats.o umf_di_solve.o umf_di_symbolic_usage.o \
	    umf_di_transpose.o umf_di_tuple_lengths.o umf_di_usolve.o \
	    umf_di_utsolve.o umf_di_valid_numeric.o umf_di_valid_symbolic.o \
	    umf_di_grow_front.o umf_di_start_front.o \
	    umf_di_store_lu.o umf_di_scale.o umfpack_di_wsolve.o \
	    umfpack_di_col_to_triplet.o umfpack_di_defaults.o \
	    umfpack_di_free_numeric.o umfpack_di_free_symbolic.o \
	    umfpack_di_get_numeric.o umfpack_di_get_lunz.o \
	    umfpack_di_get_symbolic.o umfpack_di_get_determinant.o \
	    umfpack_di_numeric.o \
	    umfpack_di_qsymbolic.o umfpack_di_report_control.o \
	    umfpack_di_report_info.o umfpack_di_report_matrix.o \
	    umfpack_di_report_numeric.o umfpack_di_report_perm.o \
	    umfpack_di_report_status.o umfpack_di_report_symbolic.o \
	    umfpack_di_report_triplet.o umfpack_di_report_vector.o \
	    umfpack_di_solve.o umfpack_di_symbolic.o umfpack_di_transpose.o \
	    umfpack_di_triplet_to_col.o umfpack_di_scale.o \
	    umfpack_di_load_numeric.o umfpack_di_save_numeric.o \
	    umfpack_di_load_symbolic.o umfpack_di_save_symbolic.o \
	    umf_dl_lhsolve.o \
	    umf_dl_uhsolve.o umf_dl_triplet_map_nox.o umf_dl_triplet_nomap_x.o \
	    umf_dl_triplet_nomap_nox.o umf_dl_triplet_map_x.o \
	    umf_dl_assemble_fixq.o umf_dl_store_lu_drop.o umf_dl_assemble.o \
	    umf_dl_blas3_update.o umf_dl_build_tuples.o \
	    umf_dl_create_element.o umf_dl_dump.o umf_dl_extend_front.o \
	    umf_dl_garbage_collection.o umf_dl_get_memory.o \
	    umf_dl_init_front.o umf_dl_kernel.o umf_dl_kernel_init.o \
	    umf_dl_kernel_wrapup.o umf_dl_local_search.o umf_dl_lsolve.o \
	    umf_dl_ltsolve.o umf_dl_mem_alloc_element.o \
	    umf_dl_mem_alloc_head_block.o umf_dl_mem_alloc_tail_block.o \
	    umf_dl_mem_free_tail_block.o umf_dl_mem_init_memoryspace.o \
	    umf_dl_report_vector.o umf_dl_row_search.o umf_dl_scale_column.o \
	    umf_dl_set_stats.o umf_dl_solve.o umf_dl_symbolic_usage.o \
	    umf_dl_transpose.o umf_dl_tuple_lengths.o umf_dl_usolve.o \
	    umf_dl_utsolve.o umf_dl_valid_numeric.o umf_dl_valid_symbolic.o \
	    umf_dl_grow_front.o umf_dl_start_front.o \
	    umf_dl_store_lu.o umf_dl_scale.o umfpack_dl_wsolve.o \
	    umfpack_dl_col_to_triplet.o umfpack_dl_defaults.o \
	    umfpack_dl_free_numeric.o umfpack_dl_free_symbolic.o \
	    umfpack_dl_get_numeric.o umfpack_dl_get_lunz.o \
	    umfpack_dl_get_symbolic.o umfpack_dl_get_determinant.o \
	    umfpack_dl_numeric.o \
	    umfpack_dl_qsymbolic.o umfpack_dl_report_control.o \
	    umfpack_dl_report_info.o umfpack_dl_report_matrix.o \
	    umfpack_dl_report_numeric.o umfpack_dl_report_perm.o \
	    umfpack_dl_report_status.o umfpack_dl_report_symbolic.o \
	    umfpack_dl_report_triplet.o umfpack_dl_report_vector.o \
	    umfpack_dl_solve.o umfpack_dl_symbolic.o umfpack_dl_transpose.o \
	    umfpack_dl_triplet_to_col.o umfpack_dl_scale.o \
	    umfpack_dl_load_numeric.o umfpack_dl_save_numeric.o \
	    umfpack_dl_load_symbolic.o umfpack_dl_save_symbolic.o \
	    umf_zi_lhsolve.o \
	    umf_zi_uhsolve.o umf_zi_triplet_map_nox.o umf_zi_triplet_nomap_x.o \
	    umf_zi_triplet_nomap_nox.o umf_zi_triplet_map_x.o \
	    umf_zi_assemble_fixq.o umf_zi_store_lu_drop.o umf_zi_assemble.o \
	    umf_zi_blas3_update.o umf_zi_build_tuples.o \
	    umf_zi_create_element.o umf_zi_dump.o umf_zi_extend_front.o \
	    umf_zi_garbage_collection.o umf_zi_get_memory.o \
	    umf_zi_init_front.o umf_zi_kernel.o umf_zi_kernel_init.o \
	    umf_zi_kernel_wrapup.o umf_zi_local_search.o umf_zi_lsolve.o \
	    umf_zi_ltsolve.o umf_zi_mem_alloc_element.o \
	    umf_zi_mem_alloc_head_block.o umf_zi_mem_alloc_tail_block.o \
	    umf_zi_mem_free_tail_block.o umf_zi_mem_init_memoryspace.o \
	    umf_zi_report_vector.o umf_zi_row_search.o umf_zi_scale_column.o \
	    umf_zi_set_stats.o umf_zi_solve.o umf_zi_symbolic_usage.o \
	    umf_zi_transpose.o umf_zi_tuple_lengths.o umf_zi_usolve.o \
	    umf_zi_utsolve.o umf_zi_valid_numeric.o umf_zi_valid_symbolic.o \
	    umf_zi_grow_front.o umf_zi_start_front.o \
	    umf_zi_store_lu.o umf_zi_scale.o umfpack_zi_wsolve.o \
	    umfpack_zi_col_to_triplet.o umfpack_zi_defaults.o \
	    umfpack_zi_free_numeric.o umfpack_zi_free_symbolic.o \
	    umfpack_zi_get_numeric.o umfpack_zi_get_lunz.o \
	    umfpack_zi_get_symbolic.o umfpack_zi_get_determinant.o \
	    umfpack_zi_numeric.o \
	    umfpack_zi_qsymbolic.o umfpack_zi_report_control.o \
	    umfpack_zi_report_info.o umfpack_zi_report_matrix.o \
	    umfpack_zi_report_numeric.o umfpack_zi_report_perm.o \
	    umfpack_zi_report_status.o umfpack_zi_report_symbolic.o \
	    umfpack_zi_report_triplet.o umfpack_zi_report_vector.o \
	    umfpack_zi_solve.o umfpack_zi_symbolic.o umfpack_zi_transpose.o \
	    umfpack_zi_triplet_to_col.o umfpack_zi_scale.o \
	    umfpack_zi_load_numeric.o umfpack_zi_save_numeric.o \
	    umfpack_zi_load_symbolic.o umfpack_zi_save_symbolic.o \
	    umf_zl_lhsolve.o \
	    umf_zl_uhsolve.o umf_zl_triplet_map_nox.o umf_zl_triplet_nomap_x.o \
	    umf_zl_triplet_nomap_nox.o umf_zl_triplet_map_x.o \
	    umf_zl_assemble_fixq.o umf_zl_store_lu_drop.o umf_zl_assemble.o \
	    umf_zl_blas3_update.o umf_zl_build_tuples.o \
	    umf_zl_create_element.o umf_zl_dump.o umf_zl_extend_front.o \
	    umf_zl_garbage_collection.o umf_zl_get_memory.o \
	    umf_zl_init_front.o umf_zl_kernel.o umf_zl_kernel_init.o \
	    umf_zl_kernel_wrapup.o umf_zl_local_search.o umf_zl_lsolve.o \
	    umf_zl_ltsolve.o umf_zl_mem_alloc_element.o \
	    umf_zl_mem_alloc_head_block.o umf_zl_mem_alloc_tail_block.o \
	    umf_zl_mem_free_tail_block.o umf_zl_mem_init_memoryspace.o \
	    umf_zl_report_vector.o umf_zl_row_search.o umf_zl_scale_column.o \
	    umf_zl_set_stats.o umf_zl_solve.o umf_zl_symbolic_usage.o \
	    umf_zl_transpose.o umf_zl_tuple_lengths.o umf_zl_usolve.o \
	    umf_zl_utsolve.o umf_zl_valid_numeric.o umf_zl_valid_symbolic.o \
	    umf_zl_grow_front.o umf_zl_start_front.o \
	    umf_zl_store_lu.o umf_zl_scale.o umfpack_zl_wsolve.o \
	    umfpack_zl_col_to_triplet.o umfpack_zl_defaults.o \
	    umfpack_zl_free_numeric.o umfpack_zl_free_symbolic.o \
	    umfpack_zl_get_numeric.o umfpack_zl_get_lunz.o \
	    umfpack_zl_get_symbolic.o umfpack_zl_get_determinant.o \
	    umfpack_zl_numeric.o \
	    umfpack_zl_qsymbolic.o umfpack_zl_report_control.o \
	    umfpack_zl_report_info.o umfpack_zl_report_matrix.o \
	    umfpack_zl_report_numeric.o umfpack_zl_report_perm.o \
	    umfpack_zl_report_status.o umfpack_zl_report_symbolic.o \
	    umfpack_zl_report_triplet.o umfpack_zl_report_vector.o \
	    umfpack_zl_solve.o umfpack_zl_symbolic.o umfpack_zl_transpose.o \
	    umfpack_zl_triplet_to_col.o umfpack_zl_scale.o \
	    umfpack_zl_load_numeric.o umfpack_zl_save_numeric.o \
	    umfpack_zl_load_symbolic.o umfpack_zl_save_symbolic.o
	- $(RANLIB) ../Lib/libumfpack.a

#-------------------------------------------------------------------------------
# Remove all but the files in the original distribution
#-------------------------------------------------------------------------------

purge: clean
	- $(RM) ../Lib/libumfpack.a

clean:
	- $(RM) $(CLEAN)
