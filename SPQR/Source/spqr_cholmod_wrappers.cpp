#include "spqr.hpp"

template <> void *spqr_malloc <int64_t> (size_t n, size_t size, cholmod_common *Common)
{
    return cholmod_l_malloc (n, size, Common) ;
}
template <> void *spqr_malloc <int32_t> (size_t n, size_t size, cholmod_common *Common)
{
    return cholmod_malloc (n, size, Common) ;
}

template <> void *spqr_calloc <int64_t> (size_t n, size_t size, cholmod_common *Common)
{
    return cholmod_l_calloc (n, size, Common) ;
}
template <> void *spqr_calloc <int32_t> (size_t n, size_t size, cholmod_common *Common)
{
    return cholmod_calloc (n, size, Common) ;
}

template <> void *spqr_free <int64_t> (size_t n, size_t size, void *p, cholmod_common *Common)
{
    return cholmod_l_free (n, size, p, Common) ;
}
template <> void *spqr_free <int32_t> (size_t n, size_t size, void *p, cholmod_common *Common)
{
    return cholmod_free (n, size, p, Common) ;
}

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
)
{
    return cholmod_realloc (nnew, size, p, n, Common) ;
}
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
)
{
    return cholmod_l_realloc (nnew, size, p, n, Common) ;
}

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
)
{
    return cholmod_allocate_sparse (nrow, ncol, nzmax, sorted, packed, stype, xtype, Common) ;
}
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
)
{
    return cholmod_l_allocate_sparse (nrow, ncol, nzmax, sorted, packed, stype, xtype, Common) ;
}

template <> int spqr_free_sparse <int32_t>
(
    /* ---- in/out --- */
    cholmod_sparse **A,	/* matrix to deallocate, NULL on output */
    /* --------------- */
    cholmod_common *Common
)
{
    return cholmod_free_sparse (A, Common) ;
}
template <> int spqr_free_sparse <int64_t>
(
    /* ---- in/out --- */
    cholmod_sparse **A,	/* matrix to deallocate, NULL on output */
    /* --------------- */
    cholmod_common *Common
)
{
    return cholmod_l_free_sparse (A, Common) ;
}

template <> int spqr_reallocate_sparse <int32_t>
(
    /* ---- input ---- */
    size_t nznew,	/* new # of entries in A */
    /* ---- in/out --- */
    cholmod_sparse *A,	/* matrix to reallocate */
    /* --------------- */
    cholmod_common *Common
)
{
    return cholmod_reallocate_sparse (nznew, A, Common) ;
}
template <> int spqr_reallocate_sparse <int64_t>
(
    /* ---- input ---- */
    size_t nznew,	/* new # of entries in A */
    /* ---- in/out --- */
    cholmod_sparse *A,	/* matrix to reallocate */
    /* --------------- */
    cholmod_common *Common
)
{
    return cholmod_l_reallocate_sparse (nznew, A, Common) ;
}

template <> cholmod_dense *spqr_allocate_dense <int32_t>
(
    /* ---- input ---- */
    size_t nrow,	/* # of rows of matrix */
    size_t ncol,	/* # of columns of matrix */
    size_t d,		/* leading dimension */
    int xtype,		/* CHOLMOD_REAL, _COMPLEX, or _ZOMPLEX */
    /* --------------- */
    cholmod_common *Common
)
{
    return cholmod_allocate_dense (nrow, ncol, d, xtype, Common) ;
}

template <> cholmod_dense *spqr_allocate_dense <int64_t>
(
    /* ---- input ---- */
    size_t nrow,	/* # of rows of matrix */
    size_t ncol,	/* # of columns of matrix */
    size_t d,		/* leading dimension */
    int xtype,		/* CHOLMOD_REAL, _COMPLEX, or _ZOMPLEX */
    /* --------------- */
    cholmod_common *Common
)
{
    return cholmod_l_allocate_dense (nrow, ncol, d, xtype, Common) ;
}

template <> int spqr_free_dense <int32_t>
(
    /* ---- in/out --- */
    cholmod_dense **X,	/* dense matrix to deallocate, NULL on output */
    /* --------------- */
    cholmod_common *Common
)
{
    return cholmod_free_dense (X, Common) ;
}
template <> int spqr_free_dense <int64_t>
(
    /* ---- in/out --- */
    cholmod_dense **X,	/* dense matrix to deallocate, NULL on output */
    /* --------------- */
    cholmod_common *Common
)
{
    return cholmod_l_free_dense (X, Common) ;
}

template <> cholmod_factor *spqr_allocate_factor <int32_t>
(
    /* ---- input ---- */
    size_t n,		/* L is n-by-n */
    /* --------------- */
    cholmod_common *Common
)
{
    return cholmod_allocate_factor (n, Common) ;
}
template <> cholmod_factor *spqr_allocate_factor <int64_t>
(
    /* ---- input ---- */
    size_t n,		/* L is n-by-n */
    /* --------------- */
    cholmod_common *Common
)
{
    return cholmod_l_allocate_factor (n, Common) ;
}


template <> int spqr_free_factor <int32_t>
(
    /* ---- in/out --- */
    cholmod_factor **L,	/* factor to free, NULL on output */
    /* --------------- */
    cholmod_common *Common
)
{
    return cholmod_free_factor (L, Common) ;
}
template <> int spqr_free_factor <int64_t>
(
    /* ---- in/out --- */
    cholmod_factor **L,	/* factor to free, NULL on output */
    /* --------------- */
    cholmod_common *Common
)
{
    return cholmod_l_free_factor (L, Common) ;
}

template <> int spqr_allocate_work <int32_t>
(
    /* ---- input ---- */
    size_t nrow,	/* size: Common->Flag (nrow), Common->Head (nrow+1) */
    size_t iworksize,	/* size of Common->Iwork */
    size_t xworksize,	/* size of Common->Xwork */
    /* --------------- */
    cholmod_common *Common
)
{
    return cholmod_allocate_work (nrow, iworksize, xworksize, Common) ;
}
template <> int spqr_allocate_work <int64_t>
(
    /* ---- input ---- */
    size_t nrow,	/* size: Common->Flag (nrow), Common->Head (nrow+1) */
    size_t iworksize,	/* size of Common->Iwork */
    size_t xworksize,	/* size of Common->Xwork */
    /* --------------- */
    cholmod_common *Common
)
{
    return cholmod_l_allocate_work (nrow, iworksize, xworksize, Common) ;
}


template <> int spqr_amd <int64_t> 
(
    cholmod_sparse *A, int64_t *fset, size_t fsize, int64_t *Perm, cholmod_common *Common
)
{
    return cholmod_l_amd (A, fset, fsize, Perm, Common) ;
}

template <> int spqr_amd <int32_t> 
(
    cholmod_sparse *A, int32_t *fset, size_t fsize, int32_t *Perm, cholmod_common *Common
)
{
    return cholmod_amd (A, fset, fsize, Perm, Common) ;
}

template <> int spqr_metis <int64_t> 
(
    cholmod_sparse *A, int64_t *fset, size_t fsize, int postorder, int64_t *Perm, cholmod_common *Common
)
{
    return cholmod_l_metis (A, fset, fsize, postorder, Perm, Common) ;
}

template <> int spqr_metis <int32_t> 
(
    cholmod_sparse *A, int32_t *fset, size_t fsize, int postorder, int32_t *Perm, cholmod_common *Common
)
{
    return cholmod_metis (A, fset, fsize, postorder, Perm, Common) ;
}

template <> cholmod_sparse *spqr_transpose <int64_t> 
(
    cholmod_sparse *A, int values, cholmod_common *Common
)
{
    return cholmod_l_transpose (A, values, Common) ;
}
template <> cholmod_sparse *spqr_transpose <int32_t> 
(
    cholmod_sparse *A, int values, cholmod_common *Common
)
{
    return cholmod_transpose (A, values, Common) ;
}

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
)
{
    return cholmod_analyze_p2 (for_whom, A, UserPerm, fset, fsize, Common) ;
}
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
)
{
    return cholmod_l_analyze_p2 (for_whom, A, UserPerm, fset, fsize, Common) ;
}

template <>
int spqr_colamd <int32_t>
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
)
{
    return cholmod_colamd (A, fset, fsize, postorder, Perm, Common) ;
}
template <>
int spqr_colamd <int64_t>
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
)
{
    return cholmod_l_colamd (A, fset, fsize, postorder, Perm, Common) ;
}

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
)
{
    return cholmod_postorder (Parent, n, Weight_p, Post, Common) ;
}

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
)
{
    return cholmod_l_postorder (Parent, n, Weight_p, Post, Common) ;
}

template <> int64_t spqr_nnz <int64_t>
(
    cholmod_sparse *A,
    cholmod_common *Common
)
{
    return cholmod_l_nnz(A, Common) ;
}

template <> int64_t spqr_nnz <int32_t>
(
    cholmod_sparse *A,
    cholmod_common *Common
)
{
    return cholmod_nnz(A, Common) ;
}

template <> cholmod_sparse *spqr_dense_to_sparse <int32_t>
(
    /* ---- input ---- */
    cholmod_dense *X,	/* matrix to copy */
    int values,		/* TRUE if values to be copied, FALSE otherwise */
    /* --------------- */
    cholmod_common *Common
)
{
    return cholmod_dense_to_sparse(X, values, Common) ;
}
template <> cholmod_sparse *spqr_dense_to_sparse <int64_t>
(
    /* ---- input ---- */
    cholmod_dense *X,	/* matrix to copy */
    int values,		/* TRUE if values to be copied, FALSE otherwise */
    /* --------------- */
    cholmod_common *Common
)
{
    return cholmod_l_dense_to_sparse(X, values, Common) ;
}

template <> cholmod_dense *spqr_sparse_to_dense <int32_t>
(
    /* ---- input ---- */
    cholmod_sparse *A,	/* matrix to copy */
    /* --------------- */
    cholmod_common *Common
)
{
    return cholmod_sparse_to_dense (A, Common) ;
}
template <> cholmod_dense *spqr_sparse_to_dense <int64_t>
(
    /* ---- input ---- */
    cholmod_sparse *A,	/* matrix to copy */
    /* --------------- */
    cholmod_common *Common
)
{
    return cholmod_l_sparse_to_dense (A, Common) ;
}

template <> cholmod_sparse *spqr_speye <int64_t>
(
    /* ---- input ---- */
    size_t nrow,	/* # of rows of A */
    size_t ncol,	/* # of columns of A */
    int xtype,		/* CHOLMOD_PATTERN, _REAL, _COMPLEX, or _ZOMPLEX */
    /* --------------- */
    cholmod_common *Common
)
{
    return cholmod_l_speye(nrow, ncol, xtype, Common) ;
}
template <> cholmod_sparse *spqr_speye <int32_t>
(
    /* ---- input ---- */
    size_t nrow,	/* # of rows of A */
    size_t ncol,	/* # of columns of A */
    int xtype,		/* CHOLMOD_PATTERN, _REAL, _COMPLEX, or _ZOMPLEX */
    /* --------------- */
    cholmod_common *Common
)
{
    return cholmod_speye(nrow, ncol, xtype, Common) ;
}