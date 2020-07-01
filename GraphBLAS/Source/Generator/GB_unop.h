if_operator_is_enabled

GrB_Info GB_unop_apply
(
    GB_ctype *Cx,
    const GB_atype *Ax,
    int64_t anz,
    int nthreads
) ;

endif_operator_is_enabled

GrB_Info GB_unop_tran
(
    GrB_Matrix C,
    const GrB_Matrix A,
    int64_t *GB_RESTRICT *Rowcounts,
    GBI_single_iterator Iter,
    const int64_t *GB_RESTRICT A_slice,
    int naslice
) ;

