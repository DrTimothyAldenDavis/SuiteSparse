#include "spqr.hpp"

template <typename Int> int spqr_start
(
    cholmod_common *Common
) ;

template <> int spqr_start <int32_t>
(
    cholmod_common *Common
) ;

template <> int spqr_start <int64_t>
(
    cholmod_common *Common
) ;

template <typename Int> int spqr_finish
(
    cholmod_common *Common
) ;

template <> int spqr_finish <int32_t>
(
    cholmod_common *Common
) ;

template <> int spqr_finish <int64_t>
(
    cholmod_common *Common
) ;

template <typename Int> void *spqr_malloc (size_t n, size_t size, cholmod_common *Common) ;
template <> void *spqr_malloc <int32_t> (size_t n, size_t size, cholmod_common *Common) ;
template <> void *spqr_malloc <int64_t> (size_t n, size_t size, cholmod_common *Common) ;

template <typename Int> void *spqr_calloc (size_t n, size_t size, cholmod_common *Common) ;
template <> void *spqr_calloc <int32_t> (size_t n, size_t size, cholmod_common *Common) ;
template <> void *spqr_calloc <int64_t> (size_t n, size_t size, cholmod_common *Common) ;

template <typename Int> void *spqr_free (size_t n, size_t size, void *p, cholmod_common *Common) ;
template <> void *spqr_free <int32_t> (size_t n, size_t size, void *p, cholmod_common *Common) ;
template <> void *spqr_free <int64_t> (size_t n, size_t size, void *p, cholmod_common *Common) ;

template <typename Int> void *spqr_realloc	/* returns pointer to reallocated block */
(
    /* ---- input ---- */
    size_t nnew,	/* requested # of items in reallocated block */
    size_t size,	/* size of each item */
    /* ---- in/out --- */
    void *p,		/* block of memory to realloc */
    size_t *n,		/* current size on input, nnew on output if successful*/
    /* --------------- */
    cholmod_common *Common
) ;
template <> void *spqr_realloc <int32_t>	/* returns pointer to reallocated block */
(
    /* ---- input ---- */
    size_t nnew,	/* requested # of items in reallocated block */
    size_t size,	/* size of each item */
    /* ---- in/out --- */
    void *p,		/* block of memory to realloc */
    size_t *n,		/* current size on input, nnew on output if successful*/
    /* --------------- */
    cholmod_common *Common
) ;
template <> void *spqr_realloc <int64_t>	/* returns pointer to reallocated block */
(
    /* ---- input ---- */
    size_t nnew,	/* requested # of items in reallocated block */
    size_t size,	/* size of each item */
    /* ---- in/out --- */
    void *p,		/* block of memory to realloc */
    size_t *n,		/* current size on input, nnew on output if successful*/
    /* --------------- */
    cholmod_common *Common
) ;

template <typename Int> cholmod_sparse *spqr_allocate_sparse
(
    /* ---- input ---- */
    size_t nrow,	/* # of rows of A */
    size_t ncol,	/* # of columns of A */
    size_t nzmax,	/* max # of nonzeros of A */
    int sorted,		/* TRUE if columns of A sorted, FALSE otherwise */
    int packed,		/* TRUE if A will be packed, FALSE otherwise */
    int stype,		/* stype of A */
    int xtype,		/* CHOLMOD_PATTERN, _REAL, _COMPLEX, or _ZOMPLEX */
    /* --------------- */
    cholmod_common *Common
) ;
template <> cholmod_sparse *spqr_allocate_sparse <int32_t>
(
    /* ---- input ---- */
    size_t nrow,	/* # of rows of A */
    size_t ncol,	/* # of columns of A */
    size_t nzmax,	/* max # of nonzeros of A */
    int sorted,		/* TRUE if columns of A sorted, FALSE otherwise */
    int packed,		/* TRUE if A will be packed, FALSE otherwise */
    int stype,		/* stype of A */
    int xtype,		/* CHOLMOD_PATTERN, _REAL, _COMPLEX, or _ZOMPLEX */
    /* --------------- */
    cholmod_common *Common
) ;
template <> cholmod_sparse *spqr_allocate_sparse <int64_t>
(
    /* ---- input ---- */
    size_t nrow,	/* # of rows of A */
    size_t ncol,	/* # of columns of A */
    size_t nzmax,	/* max # of nonzeros of A */
    int sorted,		/* TRUE if columns of A sorted, FALSE otherwise */
    int packed,		/* TRUE if A will be packed, FALSE otherwise */
    int stype,		/* stype of A */
    int xtype,		/* CHOLMOD_PATTERN, _REAL, _COMPLEX, or _ZOMPLEX */
    /* --------------- */
    cholmod_common *Common
) ;

template <typename Int> int spqr_free_sparse
(
    /* ---- in/out --- */
    cholmod_sparse **A,	/* matrix to deallocate, NULL on output */
    /* --------------- */
    cholmod_common *Common
) ;
template <> int spqr_free_sparse <int32_t>
(
    /* ---- in/out --- */
    cholmod_sparse **A,	/* matrix to deallocate, NULL on output */
    /* --------------- */
    cholmod_common *Common
) ;
template <> int spqr_free_sparse <int64_t>
(
    /* ---- in/out --- */
    cholmod_sparse **A,	/* matrix to deallocate, NULL on output */
    /* --------------- */
    cholmod_common *Common
) ;

template <typename Int> int spqr_reallocate_sparse
(
    /* ---- input ---- */
    size_t nznew,	/* new # of entries in A */
    /* ---- in/out --- */
    cholmod_sparse *A,	/* matrix to reallocate */
    /* --------------- */
    cholmod_common *Common
) ;
template <> int spqr_reallocate_sparse <int32_t>
(
    /* ---- input ---- */
    size_t nznew,	/* new # of entries in A */
    /* ---- in/out --- */
    cholmod_sparse *A,	/* matrix to reallocate */
    /* --------------- */
    cholmod_common *Common
) ;
template <> int spqr_reallocate_sparse <int64_t>
(
    /* ---- input ---- */
    size_t nznew,	/* new # of entries in A */
    /* ---- in/out --- */
    cholmod_sparse *A,	/* matrix to reallocate */
    /* --------------- */
    cholmod_common *Common
) ;

template <typename Int> cholmod_dense *spqr_allocate_dense
(
    /* ---- input ---- */
    size_t nrow,	/* # of rows of matrix */
    size_t ncol,	/* # of columns of matrix */
    size_t d,		/* leading dimension */
    int xtype,		/* CHOLMOD_REAL, _COMPLEX, or _ZOMPLEX */
    /* --------------- */
    cholmod_common *Common
) ;
template <> cholmod_dense *spqr_allocate_dense <int32_t>
(
    /* ---- input ---- */
    size_t nrow,	/* # of rows of matrix */
    size_t ncol,	/* # of columns of matrix */
    size_t d,		/* leading dimension */
    int xtype,		/* CHOLMOD_REAL, _COMPLEX, or _ZOMPLEX */
    /* --------------- */
    cholmod_common *Common
) ;

template <> cholmod_dense *spqr_allocate_dense <int64_t>
(
    /* ---- input ---- */
    size_t nrow,	/* # of rows of matrix */
    size_t ncol,	/* # of columns of matrix */
    size_t d,		/* leading dimension */
    int xtype,		/* CHOLMOD_REAL, _COMPLEX, or _ZOMPLEX */
    /* --------------- */
    cholmod_common *Common
) ;

template <typename Int> int spqr_free_dense
(
    /* ---- in/out --- */
    cholmod_dense **X,	/* dense matrix to deallocate, NULL on output */
    /* --------------- */
    cholmod_common *Common
) ;
template <> int spqr_free_dense <int32_t>
(
    /* ---- in/out --- */
    cholmod_dense **X,	/* dense matrix to deallocate, NULL on output */
    /* --------------- */
    cholmod_common *Common
) ;
template <> int spqr_free_dense <int64_t>
(
    /* ---- in/out --- */
    cholmod_dense **X,	/* dense matrix to deallocate, NULL on output */
    /* --------------- */
    cholmod_common *Common
) ;

template <typename Int> cholmod_factor *spqr_allocate_factor
(
    /* ---- input ---- */
    size_t n,		/* L is n-by-n */
    /* --------------- */
    cholmod_common *Common
) ;
template <> cholmod_factor *spqr_allocate_factor <int32_t>
(
    /* ---- input ---- */
    size_t n,		/* L is n-by-n */
    /* --------------- */
    cholmod_common *Common
) ;
template <> cholmod_factor *spqr_allocate_factor <int64_t>
(
    /* ---- input ---- */
    size_t n,		/* L is n-by-n */
    /* --------------- */
    cholmod_common *Common
) ;

template <typename Int> int spqr_free_factor
(
    /* ---- in/out --- */
    cholmod_factor **L,	/* factor to free, NULL on output */
    /* --------------- */
    cholmod_common *Common
) ;
template <> int spqr_free_factor <int32_t>
(
    /* ---- in/out --- */
    cholmod_factor **L,	/* factor to free, NULL on output */
    /* --------------- */
    cholmod_common *Common
) ;
template <> int spqr_free_factor <int64_t>
(
    /* ---- in/out --- */
    cholmod_factor **L,	/* factor to free, NULL on output */
    /* --------------- */
    cholmod_common *Common
) ;

template <typename Int> int spqr_allocate_work
(
    /* ---- input ---- */
    size_t nrow,	/* size: Common->Flag (nrow), Common->Head (nrow+1) */
    size_t iworksize,	/* size of Common->Iwork */
    size_t xworksize,	/* size of Common->Xwork */
    /* --------------- */
    cholmod_common *Common
) ;
template <> int spqr_allocate_work <int32_t>
(
    /* ---- input ---- */
    size_t nrow,	/* size: Common->Flag (nrow), Common->Head (nrow+1) */
    size_t iworksize,	/* size of Common->Iwork */
    size_t xworksize,	/* size of Common->Xwork */
    /* --------------- */
    cholmod_common *Common
) ;
template <> int spqr_allocate_work <int64_t>
(
    /* ---- input ---- */
    size_t nrow,	/* size: Common->Flag (nrow), Common->Head (nrow+1) */
    size_t iworksize,	/* size of Common->Iwork */
    size_t xworksize,	/* size of Common->Xwork */
    /* --------------- */
    cholmod_common *Common
) ;

template <typename Int> int spqr_amd
(
    cholmod_sparse *A, Int *fset, size_t fsize, Int *Perm, cholmod_common *Common
) ;
template <> int spqr_amd <int64_t> 
(
    cholmod_sparse *A, int64_t *fset, size_t fsize, int64_t *Perm, cholmod_common *Common
) ;
template <> int spqr_amd <int32_t> 
(
    cholmod_sparse *A, int32_t *fset, size_t fsize, int32_t *Perm, cholmod_common *Common
) ;

template <typename Int> int spqr_metis 
(
    cholmod_sparse *A, Int *fset, size_t fsize, int postorder, Int *Perm, cholmod_common *Common
) ;
template <> int spqr_metis <int64_t> 
(
    cholmod_sparse *A, int64_t *fset, size_t fsize, int postorder, int64_t *Perm, cholmod_common *Common
);

template <> int spqr_metis <int32_t> 
(
    cholmod_sparse *A, int32_t *fset, size_t fsize, int postorder, int32_t *Perm, cholmod_common *Common
);

template <typename Int> cholmod_sparse *spqr_transpose 
(
    cholmod_sparse *A, int values, cholmod_common *Common
) ;
template <> cholmod_sparse *spqr_transpose <int64_t> 
(
    cholmod_sparse *A, int values, cholmod_common *Common
) ;
template <> cholmod_sparse *spqr_transpose <int32_t> 
(
    cholmod_sparse *A, int values, cholmod_common *Common
) ;

template <typename Int>
cholmod_factor *spqr_analyze_p2
(
    /* ---- input ---- */
    int for_whom,       /* FOR_SPQR     (0): for SPQR but not GPU-accelerated
                           FOR_CHOLESKY (1): for Cholesky (GPU or not)
                           FOR_SPQRGPU  (2): for SPQR with GPU acceleration */
    cholmod_sparse *A,	/* matrix to order and analyze */
    Int *UserPerm,	/* user-provided permutation, size A->nrow */
    Int *fset,	/* subset of 0:(A->ncol)-1 */
    size_t fsize,	/* size of fset */
    /* --------------- */
    cholmod_common *Common
) ;
template <>
cholmod_factor *spqr_analyze_p2 <int32_t>
(
    /* ---- input ---- */
    int for_whom,       /* FOR_SPQR     (0): for SPQR but not GPU-accelerated
                           FOR_CHOLESKY (1): for Cholesky (GPU or not)
                           FOR_SPQRGPU  (2): for SPQR with GPU acceleration */
    cholmod_sparse *A,	/* matrix to order and analyze */
    int32_t *UserPerm,	/* user-provided permutation, size A->nrow */
    int32_t *fset,	/* subset of 0:(A->ncol)-1 */
    size_t fsize,	/* size of fset */
    /* --------------- */
    cholmod_common *Common
) ;
template <>
cholmod_factor *spqr_analyze_p2 <int64_t>
(
    /* ---- input ---- */
    int for_whom,       /* FOR_SPQR     (0): for SPQR but not GPU-accelerated
                           FOR_CHOLESKY (1): for Cholesky (GPU or not)
                           FOR_SPQRGPU  (2): for SPQR with GPU acceleration */
    cholmod_sparse *A,	/* matrix to order and analyze */
    int64_t *UserPerm,	/* user-provided permutation, size A->nrow */
    int64_t *fset,	/* subset of 0:(A->ncol)-1 */
    size_t fsize,	/* size of fset */
    /* --------------- */
    cholmod_common *Common
) ;

template <typename Int> int spqr_colamd
(
    /* ---- input ---- */
    cholmod_sparse *A,	/* matrix to order */
    Int *fset,	/* subset of 0:(A->ncol)-1 */
    size_t fsize,	/* size of fset */
    int postorder,	/* if TRUE, follow with a coletree postorder */
    /* ---- output --- */
    Int *Perm,	/* size A->nrow, output permutation */
    /* --------------- */
    cholmod_common *Common
) ;
template <> int spqr_colamd <int32_t>
(
    /* ---- input ---- */
    cholmod_sparse *A,	/* matrix to order */
    int32_t *fset,	/* subset of 0:(A->ncol)-1 */
    size_t fsize,	/* size of fset */
    int postorder,	/* if TRUE, follow with a coletree postorder */
    /* ---- output --- */
    int32_t *Perm,	/* size A->nrow, output permutation */
    /* --------------- */
    cholmod_common *Common
) ;
template <> int spqr_colamd <int64_t>
(
    /* ---- input ---- */
    cholmod_sparse *A,	/* matrix to order */
    int64_t *fset,	/* subset of 0:(A->ncol)-1 */
    size_t fsize,	/* size of fset */
    int postorder,	/* if TRUE, follow with a coletree postorder */
    /* ---- output --- */
    int64_t *Perm,	/* size A->nrow, output permutation */
    /* --------------- */
    cholmod_common *Common
) ;

template <typename Int> Int spqr_postorder	/* return # of nodes postordered */
(
    /* ---- input ---- */
    Int *Parent,	/* size n. Parent [j] = p if p is the parent of j */
    size_t n,
    Int *Weight_p,	/* size n, optional. Weight [j] is weight of node j */
    /* ---- output --- */
    Int *Post,	/* size n. Post [k] = j is kth in postordered tree */
    /* --------------- */
    cholmod_common *Common
) ;
template <> int32_t spqr_postorder <int32_t>	/* return # of nodes postordered */
(
    /* ---- input ---- */
    int32_t *Parent,	/* size n. Parent [j] = p if p is the parent of j */
    size_t n,
    int32_t *Weight_p,	/* size n, optional. Weight [j] is weight of node j */
    /* ---- output --- */
    int32_t *Post,	/* size n. Post [k] = j is kth in postordered tree */
    /* --------------- */
    cholmod_common *Common
) ;
template <> int64_t spqr_postorder <int64_t>	/* return # of nodes postordered */
(
    /* ---- input ---- */
    int64_t *Parent,	/* size n. Parent [j] = p if p is the parent of j */
    size_t n,
    int64_t *Weight_p,	/* size n, optional. Weight [j] is weight of node j */
    /* ---- output --- */
    int64_t *Post,	/* size n. Post [k] = j is kth in postordered tree */
    /* --------------- */
    cholmod_common *Common
) ;

template <typename Int> int64_t spqr_nnz
(
    cholmod_sparse *A,
    cholmod_common *Common
) ;
template <> int64_t spqr_nnz <int64_t>
(
    cholmod_sparse *A,
    cholmod_common *Common
) ;
template <> int64_t spqr_nnz <int32_t>
(
    cholmod_sparse *A,
    cholmod_common *Common
) ;

template <typename Int> cholmod_sparse *spqr_dense_to_sparse
(
    /* ---- input ---- */
    cholmod_dense *X,	/* matrix to copy */
    int values,		/* TRUE if values to be copied, FALSE otherwise */
    /* --------------- */
    cholmod_common *Common
) ;
template <> cholmod_sparse *spqr_dense_to_sparse <int32_t>
(
    /* ---- input ---- */
    cholmod_dense *X,	/* matrix to copy */
    int values,		/* TRUE if values to be copied, FALSE otherwise */
    /* --------------- */
    cholmod_common *Common
) ;
template <> cholmod_sparse *spqr_dense_to_sparse <int64_t>
(
    /* ---- input ---- */
    cholmod_dense *X,	/* matrix to copy */
    int values,		/* TRUE if values to be copied, FALSE otherwise */
    /* --------------- */
    cholmod_common *Common
) ;

template <typename Int> cholmod_dense *spqr_sparse_to_dense
(
    /* ---- input ---- */
    cholmod_sparse *A,	/* matrix to copy */
    /* --------------- */
    cholmod_common *Common
) ;
template <> cholmod_dense *spqr_sparse_to_dense <int32_t>
(
    /* ---- input ---- */
    cholmod_sparse *A,	/* matrix to copy */
    /* --------------- */
    cholmod_common *Common
) ;
template <> cholmod_dense *spqr_sparse_to_dense <int64_t>
(
    /* ---- input ---- */
    cholmod_sparse *A,	/* matrix to copy */
    /* --------------- */
    cholmod_common *Common
) ;

template <typename Int> cholmod_sparse *spqr_speye
(
    /* ---- input ---- */
    size_t nrow,	/* # of rows of A */
    size_t ncol,	/* # of columns of A */
    int xtype,		/* CHOLMOD_PATTERN, _REAL, _COMPLEX, or _ZOMPLEX */
    /* --------------- */
    cholmod_common *Common
) ;
template <> cholmod_sparse *spqr_speye <int64_t>
(
    /* ---- input ---- */
    size_t nrow,	/* # of rows of A */
    size_t ncol,	/* # of columns of A */
    int xtype,		/* CHOLMOD_PATTERN, _REAL, _COMPLEX, or _ZOMPLEX */
    /* --------------- */
    cholmod_common *Common
) ;
template <> cholmod_sparse *spqr_speye <int32_t>
(
    /* ---- input ---- */
    size_t nrow,	/* # of rows of A */
    size_t ncol,	/* # of columns of A */
    int xtype,		/* CHOLMOD_PATTERN, _REAL, _COMPLEX, or _ZOMPLEX */
    /* --------------- */
    cholmod_common *Common
) ;


template <typename Int> int spqr_free_work
(
    cholmod_common *Common
) ;

template <> int spqr_free_work <int32_t>
(
    cholmod_common *Common
) ;

template <> int spqr_free_work <int64_t>
(
    cholmod_common *Common
) ;

template <typename Int> cholmod_dense *spqr_zeros
(
    /* ---- input ---- */
    size_t nrow,	/* # of rows of matrix */
    size_t ncol,	/* # of columns of matrix */
    int xtype,		/* CHOLMOD_REAL, _COMPLEX, or _ZOMPLEX */
    /* --------------- */
    cholmod_common *Common
) ;

template <> cholmod_dense *spqr_zeros <int32_t>
(
    /* ---- input ---- */
    size_t nrow,	/* # of rows of matrix */
    size_t ncol,	/* # of columns of matrix */
    int xtype,		/* CHOLMOD_REAL, _COMPLEX, or _ZOMPLEX */
    /* --------------- */
    cholmod_common *Common
) ;

template <> cholmod_dense *spqr_zeros <int64_t>
(
    /* ---- input ---- */
    size_t nrow,	/* # of rows of matrix */
    size_t ncol,	/* # of columns of matrix */
    int xtype,		/* CHOLMOD_REAL, _COMPLEX, or _ZOMPLEX */
    /* --------------- */
    cholmod_common *Common
) ;

template <typename Int> cholmod_dense *spqr_ones
(
    /* ---- input ---- */
    size_t nrow,	/* # of rows of matrix */
    size_t ncol,	/* # of columns of matrix */
    int xtype,		/* CHOLMOD_REAL, _COMPLEX, or _ZOMPLEX */
    /* --------------- */
    cholmod_common *Common
) ;

template <> cholmod_dense *spqr_ones <int32_t>
(
    /* ---- input ---- */
    size_t nrow,	/* # of rows of matrix */
    size_t ncol,	/* # of columns of matrix */
    int xtype,		/* CHOLMOD_REAL, _COMPLEX, or _ZOMPLEX */
    /* --------------- */
    cholmod_common *Common
) ;

template <> cholmod_dense *spqr_ones <int64_t>
(
    /* ---- input ---- */
    size_t nrow,	/* # of rows of matrix */
    size_t ncol,	/* # of columns of matrix */
    int xtype,		/* CHOLMOD_REAL, _COMPLEX, or _ZOMPLEX */
    /* --------------- */
    cholmod_common *Common
) ;

template <typename Int> cholmod_sparse *spqr_ssmult
(
    /* ---- input ---- */
    cholmod_sparse *A,	/* left matrix to multiply */
    cholmod_sparse *B,	/* right matrix to multiply */
    int stype,		/* requested stype of C */
    int values,		/* TRUE: do numerical values, FALSE: pattern only */
    int sorted,		/* if TRUE then return C with sorted columns */
    /* --------------- */
    cholmod_common *Common
) ;

template <> cholmod_sparse *spqr_ssmult <int32_t>
(
    /* ---- input ---- */
    cholmod_sparse *A,	/* left matrix to multiply */
    cholmod_sparse *B,	/* right matrix to multiply */
    int stype,		/* requested stype of C */
    int values,		/* TRUE: do numerical values, FALSE: pattern only */
    int sorted,		/* if TRUE then return C with sorted columns */
    /* --------------- */
    cholmod_common *Common
) ;

template <> cholmod_sparse *spqr_ssmult <int64_t>
(
    /* ---- input ---- */
    cholmod_sparse *A,	/* left matrix to multiply */
    cholmod_sparse *B,	/* right matrix to multiply */
    int stype,		/* requested stype of C */
    int values,		/* TRUE: do numerical values, FALSE: pattern only */
    int sorted,		/* if TRUE then return C with sorted columns */
    /* --------------- */
    cholmod_common *Common
) ;


template <typename Int> cholmod_sparse *spqr_ssadd
(
    /* ---- input ---- */
    cholmod_sparse *A,	    /* matrix to add */
    cholmod_sparse *B,	    /* matrix to add */
    double alpha [2],	    /* scale factor for A */
    double beta [2],	    /* scale factor for B */
    int values,		    /* if TRUE compute the numerical values of C */
    int sorted,		    /* if TRUE, sort columns of C */
    /* --------------- */
    cholmod_common *Common
) ;

template <> cholmod_sparse *spqr_ssadd <int32_t>
(
    /* ---- input ---- */
    cholmod_sparse *A,	    /* matrix to add */
    cholmod_sparse *B,	    /* matrix to add */
    double alpha [2],	    /* scale factor for A */
    double beta [2],	    /* scale factor for B */
    int values,		    /* if TRUE compute the numerical values of C */
    int sorted,		    /* if TRUE, sort columns of C */
    /* --------------- */
    cholmod_common *Common
) ;

template <> cholmod_sparse *spqr_ssadd <int64_t>
(
    /* ---- input ---- */
    cholmod_sparse *A,	    /* matrix to add */
    cholmod_sparse *B,	    /* matrix to add */
    double alpha [2],	    /* scale factor for A */
    double beta [2],	    /* scale factor for B */
    int values,		    /* if TRUE compute the numerical values of C */
    int sorted,		    /* if TRUE, sort columns of C */
    /* --------------- */
    cholmod_common *Common
) ;


template <typename Int> int spqr_sdmult
(
    /* ---- input ---- */
    cholmod_sparse *A,	/* sparse matrix to multiply */
    int transpose,	/* use A if 0, or A' otherwise */
    double alpha [2],   /* scale factor for A */
    double beta [2],    /* scale factor for Y */
    cholmod_dense *X,	/* dense matrix to multiply */
    /* ---- in/out --- */
    cholmod_dense *Y,	/* resulting dense matrix */
    /* --------------- */
    cholmod_common *Common
) ;

template <> int spqr_sdmult <int32_t>
(
    /* ---- input ---- */
    cholmod_sparse *A,	/* sparse matrix to multiply */
    int transpose,	/* use A if 0, or A' otherwise */
    double alpha [2],   /* scale factor for A */
    double beta [2],    /* scale factor for Y */
    cholmod_dense *X,	/* dense matrix to multiply */
    /* ---- in/out --- */
    cholmod_dense *Y,	/* resulting dense matrix */
    /* --------------- */
    cholmod_common *Common
) ;

template <> int spqr_sdmult <int64_t>
(
    /* ---- input ---- */
    cholmod_sparse *A,	/* sparse matrix to multiply */
    int transpose,	/* use A if 0, or A' otherwise */
    double alpha [2],   /* scale factor for A */
    double beta [2],    /* scale factor for Y */
    cholmod_dense *X,	/* dense matrix to multiply */
    /* ---- in/out --- */
    cholmod_dense *Y,	/* resulting dense matrix */
    /* --------------- */
    cholmod_common *Common
) ;



template <typename Int> double spqr_norm_sparse
(
    /* ---- input ---- */
    cholmod_sparse *A,	/* matrix to compute the norm of */
    int norm,		/* type of norm: 0: inf. norm, 1: 1-norm */
    /* --------------- */
    cholmod_common *Common
) ;

template <> double spqr_norm_sparse <int32_t>
(
    /* ---- input ---- */
    cholmod_sparse *A,	/* matrix to compute the norm of */
    int norm,		/* type of norm: 0: inf. norm, 1: 1-norm */
    /* --------------- */
    cholmod_common *Common
) ;

template <> double spqr_norm_sparse <int64_t>
(
    /* ---- input ---- */
    cholmod_sparse *A,	/* matrix to compute the norm of */
    int norm,		/* type of norm: 0: inf. norm, 1: 1-norm */
    /* --------------- */
    cholmod_common *Common
) ;


template <typename Int> double spqr_norm_dense
(
    /* ---- input ---- */
    cholmod_dense *X,	/* matrix to compute the norm of */
    int norm,		/* type of norm: 0: inf. norm, 1: 1-norm, 2: 2-norm */
    /* --------------- */
    cholmod_common *Common
) ;

template <> double spqr_norm_dense <int32_t>
(
    /* ---- input ---- */
    cholmod_dense *X,	/* matrix to compute the norm of */
    int norm,		/* type of norm: 0: inf. norm, 1: 1-norm, 2: 2-norm */
    /* --------------- */
    cholmod_common *Common
) ;

template <> double spqr_norm_dense <int64_t>
(
    /* ---- input ---- */
    cholmod_dense *X,	/* matrix to compute the norm of */
    int norm,		/* type of norm: 0: inf. norm, 1: 1-norm, 2: 2-norm */
    /* --------------- */
    cholmod_common *Common
) ;


template <typename Int> cholmod_dense *spqr_copy_dense
(
    /* ---- input ---- */
    cholmod_dense *X,	/* matrix to copy */
    /* --------------- */
    cholmod_common *Common
) ;

template <> cholmod_dense *spqr_copy_dense <int32_t>
(
    /* ---- input ---- */
    cholmod_dense *X,	/* matrix to copy */
    /* --------------- */
    cholmod_common *Common
) ;

template <> cholmod_dense *spqr_copy_dense <int64_t>
(
    /* ---- input ---- */
    cholmod_dense *X,	/* matrix to copy */
    /* --------------- */
    cholmod_common *Common
) ;

template <typename Int> cholmod_sparse *spqr_copy
(
    /* ---- input ---- */
    cholmod_sparse *A,	/* matrix to copy */
    int stype,		/* requested stype of C */
    int mode,		/* >0: numerical, 0: pattern, <0: pattern (no diag) */
    /* --------------- */
    cholmod_common *Common
) ;

template <> cholmod_sparse *spqr_copy <int32_t>
(
    /* ---- input ---- */
    cholmod_sparse *A,	/* matrix to copy */
    int stype,		/* requested stype of C */
    int mode,		/* >0: numerical, 0: pattern, <0: pattern (no diag) */
    /* --------------- */
    cholmod_common *Common
) ;

template <> cholmod_sparse *spqr_copy <int64_t>
(
    /* ---- input ---- */
    cholmod_sparse *A,	/* matrix to copy */
    int stype,		/* requested stype of C */
    int mode,		/* >0: numerical, 0: pattern, <0: pattern (no diag) */
    /* --------------- */
    cholmod_common *Common
) ;

template <typename Int> int spqr_sparse_xtype
(
    /* ---- input ---- */
    int to_xtype,	/* requested xtype (pattern, real, complex, zomplex) */
    /* ---- in/out --- */
    cholmod_sparse *A,	/* sparse matrix to change */
    /* --------------- */
    cholmod_common *Common
) ;

template <> int spqr_sparse_xtype <int32_t>
(
    /* ---- input ---- */
    int to_xtype,	/* requested xtype (pattern, real, complex, zomplex) */
    /* ---- in/out --- */
    cholmod_sparse *A,	/* sparse matrix to change */
    /* --------------- */
    cholmod_common *Common
) ;

template <> int spqr_sparse_xtype <int64_t>
(
    /* ---- input ---- */
    int to_xtype,	/* requested xtype (pattern, real, complex, zomplex) */
    /* ---- in/out --- */
    cholmod_sparse *A,	/* sparse matrix to change */
    /* --------------- */
    cholmod_common *Common
) ;

template <typename Int> int spqr_gpu_memorysize
(
    size_t         *total_mem,
    size_t         *available_mem,
    cholmod_common *Common
) ;

template <> int spqr_gpu_memorysize <int32_t>
(
    size_t         *total_mem,
    size_t         *available_mem,
    cholmod_common *Common
) ;

template <> int spqr_gpu_memorysize <int64_t>
(
    size_t         *total_mem,
    size_t         *available_mem,
    cholmod_common *Common
) ;


template <typename Int> cholmod_sparse *spqr_read_sparse
(
    /* ---- input ---- */
    FILE *f,		/* file to read from, must already be open */
    /* --------------- */
    cholmod_common *Common
) ;

template <> cholmod_sparse *spqr_read_sparse <int32_t>
(
    /* ---- input ---- */
    FILE *f,		/* file to read from, must already be open */
    /* --------------- */
    cholmod_common *Common
) ;

template <> cholmod_sparse *spqr_read_sparse <int64_t>
(
    /* ---- input ---- */
    FILE *f,		/* file to read from, must already be open */
    /* --------------- */
    cholmod_common *Common
) ;

