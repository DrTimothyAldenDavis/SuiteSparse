////////////////////////////////////////////////////////////////////////////////
///////////////////////////////// ParU_Analyze /////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

// ParU, Copyright (c) 2022, Mohsen Aznaveh and Timothy A. Davis,
// All Rights Reserved.
// SPDX-License-Identifier: GNU GPL 3.0

/*! @brief Computing etree and do the symbolic analysis. In this file I am going
 * to use umfpack symbolic analysis instaed of spqrsym. However, I will keep the
 * style of spqr mostly.
 *
 *************************  Relaxed amalgamation:
 *
 *                           Example: ./Matrix/b1_ss.mtx
 *      original post ordered etree:
 *
 *        5(2) <-- front(number of pivotal cols)
 *        |   \
 *        0(1) 4(1)__
 *             |     \
 *             1(1)   3(1)
 *                     \

 *                     2(1)
 *
 **********  Relaxed tree:  threshold=3  front(number of pivotal cols)##oldfront
 *          3(2)##5
 *          |       \
 *          0(1)##0 2(3)##2,3,4
 *                   |
 *                   1(1)##1
 *
 *                   0 1 2 3 4 5 -1
 *             fmap: 0 1 2 2 2 3 -1  last one is necessary for my check
 *             fmap[oldf] == fmap [oldf+1]  amalgamated
 *             fmap[oldf-1] == fmap [oldf]  amalgamated  and root of subtree
 *
 ****************  Augmented tree creation:
 *
 *                      Example: ./Matrix/problem.mtx
 *  original post ordered etree:          augmented tree: (Row elements)
 *
 *        4___                                     16_____
 *        |   \                                    |   \  (14-15)
 *        0    3__                                 3    13_____
 *             |  \                              (0-2)  |  \   \
 *             1   2                                    7   10 (11-12)
 *                                                      |    \
 *                                                    (4-6) (8-9)
 *
 *      Note: the original etree use a dummy parent for all the tree(forest)
 *              the augmented tree does not
 * @author Aznaveh
 * */
// =============================================================================

#define FREE_WORK                                   \
    paru_free((n + 1), sizeof(int64_t), fmap);          \
    paru_free((n + 1), sizeof(int64_t), newParent);     \
    paru_free(n, sizeof(int64_t), inv_Diag_map);        \
    paru_free((n + 1), sizeof(int64_t), Front_parent);  \
    paru_free((n + 1), sizeof(int64_t), Front_npivcol); \
    paru_free((m - n1), sizeof(int64_t), Ps);           \
    paru_free((std::max(m, n) + 2), sizeof(int64_t), Work);  \
    paru_free((cs1 + 1), sizeof(int64_t), cSup);        \
    paru_free((rs1 + 1), sizeof(int64_t), cSlp);        \
    umfpack_dl_free_symbolic(&Symbolic);            \
    umfpack_dl_paru_free_sw(&SW);

// =============================================================================

#include <algorithm>

#include "paru_internal.hpp"

ParU_Ret ParU_Analyze(cholmod_sparse *A, ParU_Symbolic **S_handle,
                      ParU_Control *user_Control)
{
    DEBUGLEVEL(0);
    PARU_DEFINE_PRLEVEL;
#ifndef NTIME
    double start_time = PARU_OPENMP_GET_WTIME;
#endif
    ParU_Symbolic *Sym;
    Sym = static_cast<ParU_Symbolic*>(paru_alloc(1, sizeof(ParU_Symbolic)));
    if (!Sym)
    {
        return PARU_OUT_OF_MEMORY;
    }
    *S_handle = Sym;

    int64_t m = A->nrow;
    int64_t n = A->ncol;
    if (m != n)
    {
        PRLEVEL(1, ("ParU: Input matrix is not square!\n"));
        paru_free(1, sizeof(ParU_Symbolic), Sym);
        *S_handle = NULL;
        return PARU_INVALID;
    }

    if (A->xtype != CHOLMOD_REAL)
    {
        PRLEVEL(1, ("ParU: input matrix must be real\n"));
        paru_free(1, sizeof(ParU_Symbolic), Sym);
        *S_handle = NULL;
        return PARU_INVALID;
    }

    int64_t *Ap = static_cast<int64_t*>(A->p);
    int64_t *Ai = static_cast<int64_t*>(A->i);
    double *Ax = static_cast<double*>(A->x);

    // Initializaing pointers with NULL; just in case for an early exit
    // not to free an uninitialized space
    // Sym->Chain_start = Sym->Chain_maxrows = Sym->Chain_maxcols = NULL;
    Sym->Qfill = Sym->Diag_map = NULL;
    Sym->ustons.Sup = Sym->lstons.Slp = NULL;
    Sym->ustons.Suj = Sym->lstons.Sli = NULL;
    Sym->Super = Sym->Depth = NULL;
    Sym->Pinit = NULL;
    Sym->n1 = -1;
    int64_t cs1 = -1;
    int64_t rs1 = -1;

    int64_t *Cm = Sym->Cm = NULL;
    int64_t *Fm = Sym->Fm = NULL;
    int64_t *Parent = Sym->Parent = NULL;
    int64_t *Child = Sym->Child = NULL;
    int64_t *Childp = Sym->Childp = NULL;
    int64_t *Sp = Sym->Sp = NULL;
    int64_t *Sj = Sym->Sj = NULL;
    int64_t *Sleft = Sym->Sleft = NULL;
    double *front_flop_bound = Sym->front_flop_bound = NULL;
    double *stree_flop_bound = Sym->stree_flop_bound = NULL;
    int64_t *task_map = Sym->task_map = NULL;
    int64_t *task_parent = Sym->task_parent = NULL;
    int64_t *task_num_child = Sym->task_num_child = NULL;
    int64_t *task_depth = Sym->task_depth = NULL;
    int64_t *Pinv = Sym->Pinv = NULL;

    // for computing augmented tree
    int64_t *aParent = Sym->aParent = NULL;  // augmented tree size m+nf
    int64_t *aChildp = Sym->aChildp = NULL;  // size m+nf+2
    int64_t *aChild = Sym->aChild = NULL;    // size m+nf+1
    int64_t *rM = Sym->row2atree = NULL;     // row map
    int64_t *snM = Sym->super2atree = NULL;  // and supernode map
    int64_t *first = Sym->first = NULL;      // first descendent in the tree
    // augmented tree size nf+1

    int64_t *fmap = NULL;  // temp amalgamation data structure
    int64_t *newParent = NULL;
    int64_t *Super = NULL;
    int64_t *Depth = NULL;
    int64_t *Work = NULL;
    int64_t *Ps = NULL;    // new row permutation for just the Submatrix part
    int64_t *Sup = NULL;   // Singlton u p
    int64_t *cSup = NULL;  // copy of Singlton u p
    int64_t *Slp = NULL;   // Singlton l p
    int64_t *cSlp = NULL;  // copy Singlton l p
    int64_t *Suj = NULL;
    int64_t *Sli = NULL;

    //~~~~~~~~~~~~  Calling UMFPACK and retrieving data structure ~~~~~~~~~~~~~~

    /* ---------------------------------------------------------------------- */
    /*    The varialbes are needed for the UMFPACK symbolic analysis phase    */
    /* ---------------------------------------------------------------------- */

    int64_t n1 = 0,  // The number of pivots with zero Markowitz cost.
                 // Info[UMFPACK_COL_SINGLETONS]+Info[UMFPACK_ROW_SINGLETONS]
                 // They apper first in the output permutations P and Q
                 //
                 //
                 //  --- Copied from UMFPACK
                 //  ---   x x x x x x x x x
                 //  ---   . x x x x x x x x
                 //  ---   . . x x x x x x x
                 //  ---   . . . x . . . . .
                 //  ---   . . . x x . . . .
                 //  ---   . . . x x s s s s
                 //  ---   . . . x x s s s s
                 //  ---   . . . x x s s s s
                 //  ---   . . . x x s s s s
                 //
                 //  ---   The above example has 3 column singletons (the first
                 //  ---  three   columns and their corresponding pivot rows) 
                 //  ---  and 2 row   singletons.  The singletons are ordered 
                 //  ---  first, because they  have zero Markowitz cost. The LU
                 //  ---  factorization for these first five rows and columns is
                 //  ---  free - there is no work to do (except to scale the 
                 //  ---  pivot columns for the 2 row singletons), and no 
                 //  ---  fill-in occurs.  The remaining submatrix (4-by-4 in 
                 //  ---  the above example) has no rows or columns with degree 
                 //  ---  one.  It may have empty rows or columns.
                 //
                 //
                 //        _______________
                 //       |\**************r
                 //       |  \************r -> UMFPACK_COL_SINGLETONS
                 //       |   \***********r
                 //       |    *\         |
                 //       |    ***\       |
                 //       |    ***xx\xxxxx|            |
                 //       |    ***xxxx\xxx|            +   = n1
                 //       -----ccc--------            /
                 //             |
                 //            UMFPACK_ROW_SINGLETONS

        nfr,  // The number of frontam matrices; nf in SPQR analysis

            // nchains,  // The frontal matrices are related to one another by
            // the supernodal column elimination tree. Each nod in this tree is
            // one frontal matrix. The tree is partitioned into a set of
            // disjoint paths, and a frontal matrix chaing is one path in this
            // tree.  UMFPACK uses unifrontal technique to factroize chains,
            // with a single working array that holds each frontal matrix in the
            // chain, one at a time. nchains is in the range 0 to nfr

        *Pinit = NULL,
        // The inital row permutation. If P [k] = i, then this means
        // that row i is the kth row in the pre-ordered matrix.
        // For the unsymmetric strategy, P defines the row-merge
        // order. Let j be the column index of the leftmost nonzero
        // entry in row i of A*Q. The P defines a sort of the rows
        // according to this value. A row can appear earlier in this
        // ordering if it is aggressively absorbed before it can
        // become a pivot row. If P [k]= i, row i typically will not
        // be the kth pivot row.
        // For the symmetric strategy, P = Q. If no pivoting occurs
        // during numerical factorization, P[k] = i also defines the
        // fianl permutation of umfpack_*_numeric, for the symmetric
        // strategy.
        // NOTE: SPQR uses Pinv for stairecase structure and that is
        // an invert permutation that I have to compute the direct
        // permutation in paru_write.

        *Qinit = NULL,
        // The inital column permutation. If Q [k] = j, then this
        // means that column j is the kth pivot column in pre-ordered
        // matrix. Q is not necessearily the same as final column
        // permutation in UMFPACK. In UMFPACK if the matrix is
        // structurally singular, and if the symmetric strategy is
        // used (or if Control [UMFPACK_FIXQ] > 0), then this Q will
        // be the same as the final column permutaion.
        // NOTE: there is no column permutation in paru. So the
        // initial permutaion would stay. SPQR uses Qfill for
        // staircase structure and that is the column permutation for
        // paru also.

        *Diag_map = NULL,
        // Diag_map[newcol] = newrow; It is how UMFPACK see it
        // it should be updated during makding staircase structure
        // my Diag_map would be Diag_map[col_s] = row_s
        // col_s = newcol - n1, row_s comes from staircase structure
        // I have to make initial Diag_map inverse to be able to compute mine
        *inv_Diag_map = NULL,
        // it will be freed in symbolic anyway but I will make another copy
        // in paru_init_rowFronts; that copy can be updated

        *Front_npivcol = NULL,
        // size = n_col +1;  actual size = nfr+1
        // NOTE: This is not the case for SPQR
        // I think SPQR is easier:
        // Front_npivcol [f] = Super [f+1] - Super [f]
        // In the case of amalgamation of a child to the parent
        // then the number of pivotal column of the amalgamted node is:
        //  Super [Parent[f]+1] - Super [f]
        // Front_parent [nfr+1] is a place holder for columns
        // with no entries

        *Front_parent = NULL;
    // size = n_col +1;  actual size = nfr+1
    // NOTE: This is not the case for SPQR
    // Parent is the one I should use instead.

    // *Front_1strow,  // size = n_col +1;  actual size = nfr+1
    // Front_1strow [k] is the row index of the first row in
    // A (P,Q) whose leftmost entry is in pivot column for
    // kth front.
    // This is necessary only to properly factorize singular
    // matrices. Rows in the range Front_1strow [k] to
    // Front_1strow [k+1]-1 first become pivot row candidate
    // at the kth front. Any rows not eliminated in the kth
    // front maybe selected as pivot rows in the parent of k
    // (Front_1strow [k]) and so on up the tree.
    // Aznaveh: I am now using it at least for the rowMarks.

    // *Front_leftmostdesc,  // size = n_col +1;  actual size = nfr+1
    // Aznaveh: I have a module computing leftmostdesc
    // for my augmented tree; so maybe do not need it

    //*Chain_start,  // size = n_col +1;  actual size = nfr+1
    // The kth frontal matrix chain consists of frontal
    // matrices Chain_start [k] through Chain_start [k+1]-1.
    // Thus, Chain_start [0] is always 0 and
    // Chain_start[nchains] is the total number of frontal
    // matrices, nfr. For two adjacent fornts f and f+1
    // within a single chian, f+1 is always the parent of f
    // (that is, Front_parent [f] = f+1).
    //
    // *Chain_maxrows,  // size = n_col +1;  actual size = nfr+1
    // *Chain_maxcols;  // The kth frontal matrix chain requires a single
    // working array of dimension Chain_maxrows [k] by
    // Chain_maxcols [k], for the unifrontal technique that
    // factorizes the frontal matrix chain. Since the
    // symbolic factorization only provides

    void *Symbolic;  // Output argument in umf_dl_symbolc;
    // holds a pointer to the Symbolic object  if succesful
    // and NULL otherwise

    double status,           // Info [UMFPACK_STATUS]
        Info[UMFPACK_INFO],  // Contains statistics about the symbolic analysis

        umf_Control[UMFPACK_CONTROL];  // it is set in umfpack_dl_defaults and
    // is used in umfpack_dl_symbolic; if
    // passed NULL it will use the defaults

    /* ---------------------------------------------------------------------- */
    /*    Setting up umfpakc symbolic analysis and do the  analysis phase     */
    /* ---------------------------------------------------------------------- */

    /* get the default control parameters */

    // Here is where the pre-ordering strategy is being chosen. I need a nested
    // dissection method here like metis:
    //      umf_Control [UMFPACK_ORDERING] = UMFPACK_ORDERING_METIS;
    //      However I am using the default for now; Page 40 UMFPACK_UserGuide
    //      Page 22 UserGuide
    umfpack_dl_defaults(umf_Control);

    // before using the Control checking user input
    ParU_Control my_Control = *user_Control;
    {
        int64_t umfpack_ordering = my_Control.umfpack_ordering;
        // I dont support UMFPACK_ORDERING_GIVEN or UMFPACK_ORDERING_USER now
        if (umfpack_ordering != UMFPACK_ORDERING_METIS &&
            umfpack_ordering != UMFPACK_ORDERING_METIS_GUARD &&
            umfpack_ordering != UMFPACK_ORDERING_AMD &&
            umfpack_ordering != UMFPACK_ORDERING_CHOLMOD &&
            umfpack_ordering != UMFPACK_ORDERING_BEST &&
            umfpack_ordering != UMFPACK_ORDERING_NONE)
            //my_Control.umfpack_ordering = UMFPACK_ORDERING_METIS;
            my_Control.umfpack_ordering = UMFPACK_ORDERING_AMD;
        int64_t umfpack_strategy = my_Control.umfpack_strategy;
        // I dont support UMFPACK_ORDERING_GIVEN or UMFPACK_ORDERING_USER now
        if (umfpack_strategy != UMFPACK_STRATEGY_AUTO &&
            umfpack_strategy != UMFPACK_STRATEGY_SYMMETRIC &&
            umfpack_strategy != UMFPACK_STRATEGY_UNSYMMETRIC)
            my_Control.umfpack_strategy = UMFPACK_STRATEGY_AUTO;

        int64_t threshold = my_Control.relaxed_amalgamation_threshold;
        if (threshold < 0 || threshold > 512)
            my_Control.relaxed_amalgamation_threshold = 32;
        int64_t paru_strategy = my_Control.paru_strategy;
        if (paru_strategy != UMFPACK_STRATEGY_AUTO &&
            paru_strategy != UMFPACK_STRATEGY_SYMMETRIC &&
            paru_strategy != UMFPACK_STRATEGY_UNSYMMETRIC)
            my_Control.paru_strategy = PARU_STRATEGY_AUTO;
        int64_t umfpack_default_singleton = my_Control.umfpack_default_singleton;
        if (umfpack_default_singleton != 0 && umfpack_default_singleton != 1)
            my_Control.umfpack_default_singleton = 1;
    }
    ParU_Control *Control = &my_Control;

    umf_Control[UMFPACK_SINGLETONS] = Control->umfpack_default_singleton;
    umf_Control[UMFPACK_ORDERING] = Control->umfpack_ordering;
    // umf_Control[UMFPACK_ORDERING] = UMFPACK_ORDERING_METIS;
    umf_Control[UMFPACK_FIXQ] = -1;
    umf_Control[UMFPACK_STRATEGY] = Control->umfpack_strategy;
    // umf_Control[UMFPACK_STRATEGY] = UMFPACK_STRATEGY_AUTO;
    // umf_Control[UMFPACK_STRATEGY] = UMFPACK_STRATEGY_UNSYMMETRIC;
    // umf_Control[UMFPACK_STRATEGY] = UMFPACK_STRATEGY_SYMMETRIC;
    umf_Control[UMFPACK_STRATEGY_THRESH_SYM] = 0.3;

#ifndef NDEBUG
    /* print the control parameters */
    if (PR <= 0) umfpack_dl_report_control(umf_Control);
    PRLEVEL (1, ("max_threads =%d\n", control_nthreads (Control)));
#endif

    /* performing the symbolic analysis */

    void *SW = NULL;
    status = umfpack_dl_paru_symbolic(m, n, Ap, Ai, Ax,
                                      NULL,   // user provided ordering
                                      FALSE,  // No user ordering
                                      NULL,   // user params
                                      &Symbolic,
                                      &SW,  // new in/out
                                      umf_Control, Info);

    if (status < 0)
    {
#ifndef NDEBUG
        umfpack_dl_report_info(umf_Control, Info);
        umfpack_dl_report_status(umf_Control, status);
#endif
        PRLEVEL(1, ("ParU: umfpack_dl_symbolic failed\n"));
        umfpack_dl_free_symbolic(&Symbolic);
        FREE_WORK;
        paru_free(1, sizeof(ParU_Symbolic), Sym);
        *S_handle = NULL;
        return PARU_INVALID;
    }
    /* ---------------------------------------------------------------------- */
    /* startegy UMFPACK used*/
    /* ---------------------------------------------------------------------- */
    int64_t strategy = Info[UMFPACK_STRATEGY_USED];
    if (Control->paru_strategy == PARU_STRATEGY_AUTO)
    {  // if user didn't choose the strategy I will pick the same strategy
        // as umfpack.        However I cannot save it in current control
        Sym->strategy = strategy;
    }
    else
        Sym->strategy = Control->paru_strategy;

#ifndef NDEBUG
    PR = 0;
    if (strategy == UMFPACK_STRATEGY_SYMMETRIC)
    {
        PRLEVEL(PR, ("\n%% strategy used:  symmetric\n"));
        if (Info[UMFPACK_ORDERING_USED] == UMFPACK_ORDERING_AMD)
        {
            PRLEVEL(PR, ("%% ordering used:  amd on A+A'\n"));
        }
        else if (Info[UMFPACK_ORDERING_USED] == UMFPACK_ORDERING_GIVEN)
        {
            PRLEVEL(PR, ("%% ordering used: user perm.\n"));
        }
        else if (Info[UMFPACK_ORDERING_USED] == UMFPACK_ORDERING_USER)
        {
            PRLEVEL(PR, ("%% ordering used:  user function\n"));
        }
        else if (Info[UMFPACK_ORDERING_USED] == UMFPACK_ORDERING_NONE)
        {
            PRLEVEL(PR, ("%% ordering used: none\n"));
        }
        else if (Info[UMFPACK_ORDERING_USED] == UMFPACK_ORDERING_METIS)
        {
            PRLEVEL(PR, ("%% ordering used: metis on A+A'\n"));
        }
        else
        {
            PRLEVEL(PR, ("%% ordering used: not computed\n"));
        }
    }
    else
    {
        PRLEVEL(PR, ("\n%% strategy used:unsymmetric\n"));
        if (Info[UMFPACK_ORDERING_USED] == UMFPACK_ORDERING_AMD)
        {
            PRLEVEL(PR, ("%% ordering used: colamd on A\n"));
        }
        else if (Info[UMFPACK_ORDERING_USED] == UMFPACK_ORDERING_GIVEN)
        {
            PRLEVEL(PR, ("%% ordering used: user perm.\n"));
        }
        else if (Info[UMFPACK_ORDERING_USED] == UMFPACK_ORDERING_USER)
        {
            PRLEVEL(PR, ("%% ordering used: user function\n"));
        }
        else if (Info[UMFPACK_ORDERING_USED] == UMFPACK_ORDERING_NONE)
        {
            PRLEVEL(PR, ("%% ordering used: none\n"));
        }
        else if (Info[UMFPACK_ORDERING_USED] == UMFPACK_ORDERING_METIS)
        {
            PRLEVEL(PR, ("%% ordering used: metis on A'A\n"));
        }
        else
        {
            PRLEVEL(PR, ("%% ordering used: not computed\n"));
        }
    }

    PR = -1;
    PRLEVEL(PR, ("\n%% Symbolic factorization of A: "));
    if (PR <= 0) (void)umfpack_dl_report_symbolic(Symbolic, umf_Control);
    PR = 1;
#endif

    /* ---------------------------------------------------------------------- */
    /*    Copy the contents of Symbolic in my data structure                  */
    /* ---------------------------------------------------------------------- */
    SymbolicType *Sym_umf = static_cast<SymbolicType*>(Symbolic);  // just an alias
    // temp amalgamation data structure
    fmap = static_cast<int64_t*>(paru_alloc((n + 1), sizeof(int64_t)));
    newParent = static_cast<int64_t*>(paru_alloc((n + 1), sizeof(int64_t)));
    if (Sym->strategy == PARU_STRATEGY_SYMMETRIC)
        Diag_map = Sym_umf->Diagonal_map;
    else
        Diag_map = NULL;

    if (Diag_map)
        inv_Diag_map = static_cast<int64_t*>(paru_alloc(n, sizeof(int64_t)));
    else
        inv_Diag_map = NULL;
    if (!fmap || !newParent || (!Diag_map != !inv_Diag_map))
    {
        PRLEVEL(1, ("ParU: out of memory\n"));
        ParU_Freesym(S_handle, Control);
        umfpack_dl_free_symbolic(&Symbolic);
        umfpack_dl_paru_free_sw(&SW);
        FREE_WORK;
        return PARU_OUT_OF_MEMORY;
    }

    n1 = Sym_umf->n1;
    int64_t anz = Sym_umf->nz;
    nfr = Sym_umf->nfr;
#ifndef NDEBUG
    int64_t nr = Sym_umf->n_row;
    int64_t nc = Sym_umf->n_col;
    PRLEVEL(1, ("In: " LD "x" LD " nnz = " LD " \n", nr, nc, anz));
#endif

    // nchains = Sym_umf->nchains;
    cs1 = Sym_umf->n1c;
    rs1 = Sym_umf->n1r;

    // brain transplant

    Pinit = Sym_umf->Rperm_init;
    Sym->Pinit = Pinit;

    Qinit = Sym_umf->Cperm_init;
    Front_npivcol = Sym_umf->Front_npivcol;
    Front_parent = Sym_umf->Front_parent;

    // Chain_start = Sym_umf->Chain_start;
    // Chain_maxrows = Sym_umf->Chain_maxrows;
    // Chain_maxcols = Sym_umf->Chain_maxcols;

    if (Diag_map) Sym_umf->Diagonal_map = NULL;
    Sym_umf->Rperm_init = NULL;
    Sym_umf->Cperm_init = NULL;
    Sym_umf->Front_npivcol = NULL;
    Sym_umf->Front_parent = NULL;
    // Sym_umf->Chain_start = NULL;
    // Sym_umf->Chain_maxrows = NULL;
    // Sym_umf->Chain_maxcols = NULL;

#ifndef NDEBUG
    PR = 1;
    PRLEVEL(-1, ("\n%%\tcs1 = " LD ", rs1 = " LD " n1 = " LD "\n", cs1, rs1, n1));
    PRLEVEL(PR, ("From the Symbolic object,\
                C is of dimension " LD "-by-" LD "\n",
                 nr, nc));
    PRLEVEL(PR, ("   with nz = " LD ", number of fronts = " LD ",\n", anz, nfr));
    PR = 1;
    // PRLEVEL(PR, ("   number of frontal matrix chains = " LD "\n", nchains));

    PRLEVEL(1, ("\nPivot columns in each front, and parent of each front:\n"));
    int64_t k = 0;

    PR = 1;
    for (int64_t i = 0; i < nfr; i++)
    {
        int64_t fnpiv = Front_npivcol[i];
        PRLEVEL(PR, ("Front " LD ": parent front: " LD " number of pivot cols: " LD "\n",
                     i, Front_parent[i], fnpiv));
        // PRLEVEL(PR, ("%% first row is " LD "\n", Front_1strow[i]));

        for (int64_t j = 0; j < fnpiv; j++)
        {
            int64_t col = Qinit[k];
            PRLEVEL(PR, ("" LD "-th pivot column is column " LD ""
                         " in original matrix\n",
                         k, col));
            k++;
        }
    }
    PR = -1;

    PRLEVEL(PR, ("\nTotal number of pivot columns "
                 "in frontal matrices: " LD "\n",
                 k));

    // PRLEVEL(PR, ("\nFrontal matrix chains:\n"));
    //    for (int64_t j = 0; j < nchains; j++)
    //    {
    //        PRLEVEL(PR, ("Frontal matrices " LD " to " LD " in chain\n",
    //        Chain_start[j],
    //                     Chain_start[j + 1] - 1));
    //        PRLEVEL(PR, ("\tworking array of size " LD "-by-" LD "\n",
    //        Chain_maxrows[j],
    //                     Chain_maxcols[j]));
    //    }

    PRLEVEL(PR, ("Forthwith Pinit =\n"));
    for (int64_t i = 0; i < std::min(77, m); i++) PRLEVEL(PR, ("" LD " ", Pinit[i]));
    PRLEVEL(PR, ("\n"));
    PRLEVEL(PR, ("Forthwith Qinit =\n"));
    for (int64_t i = 0; i < std::min(77, m); i++) PRLEVEL(PR, ("" LD " ", Qinit[i]));
    PRLEVEL(PR, ("\n"));
    PR = -1;
    if (Diag_map)
    {
        PRLEVEL(PR, ("Forthwith Diag_map =\n"));
        for (int64_t i = 0; i < std::min(77, n); i++) PRLEVEL(PR, ("" LD " ", Diag_map[i]));
        PRLEVEL(PR, ("\n"));
    }
    PR = 1;

#endif

    umfpack_dl_free_symbolic(&Symbolic);

    //@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

    ASSERT(m == nr);
    ASSERT(n == nc);

    Sym->m = m;
    Sym->n = n;
    Sym->n1 = n1;
    Sym->rs1 = rs1;
    Sym->cs1 = cs1;
    Sym->anz = anz;
    int64_t nf = Sym->nf = nfr;

    // Sym->Chain_start = Chain_start;
    // Sym->Chain_maxrows = Chain_maxrows;
    // Sym->Chain_maxcols = Chain_maxcols;
    Sym->Qfill = Qinit;
    Sym->Diag_map = Diag_map;

    PRLEVEL(0, ("%% A  is  " LD " x " LD " \n", m, n));
    PRLEVEL(1, ("LU = zeros(" LD "," LD ");\n", m, n));
    PRLEVEL(1, ("npivots =[]; \n"));
    PRLEVEL(1, ("S = zeros(" LD "," LD "); %% n1 = " LD "\n", m, n, n1));
    PRLEVEL(1, ("%% nf=" LD "\n", nf));
    //
    /* ---------------------------------------------------------------------- */
    /*           Fixing Parent and computing Children datat structure         */
    /* ---------------------------------------------------------------------- */

    // Parent size is nf+1 potentially smaller than what UMFPACK allocate
    size_t size = n + 1;
    Parent = static_cast<int64_t*>(paru_realloc(nf + 1, sizeof(int64_t), Front_parent, &size));
    Sym->Parent = Parent;
    ASSERT(size <= static_cast<size_t>(n) + 1);
    if (size != static_cast<size_t>(nf) + 1)
    {  // should not happen anyway it is always shrinking
        PRLEVEL(1, ("ParU: out of memory\n"));
        Sym->Parent = NULL;
        ParU_Freesym(S_handle, Control);
        umfpack_dl_paru_free_sw(&SW);
        FREE_WORK;
        return PARU_OUT_OF_MEMORY;
    }
    Front_parent = NULL;

    // Making Super data structure
    // like SPQR: Super[f]<= pivotal columns of (f) < Super[f+1]
    if (nf > 0)
    {
        Super = Sym->Super = static_cast<int64_t*>(paru_alloc((nf + 1), sizeof(int64_t)));
        Depth = Sym->Depth = static_cast<int64_t*>(paru_calloc(nf, sizeof(int64_t)));
        if (Super == NULL || Depth == NULL)
        {
            PRLEVEL(1, ("ParU: out of memory\n"));
            ParU_Freesym(S_handle, Control);
            umfpack_dl_paru_free_sw(&SW);
            FREE_WORK;
            return PARU_OUT_OF_MEMORY;
        }

        Super[0] = 0;
        int64_t sum = 0;
        for (int64_t k = 1; k <= nf; k++)
        {
            sum += Front_npivcol[k - 1];
            Super[k] = sum;
            // Super[k] = Front_npivcol[k - 1];
        }
        // paru_cumsum(nf + 1, Super);
    }

    /* ---------------------------------------------------------------------- */
    /*                          Relaxed amalgamation                          */
    /* ---------------------------------------------------------------------- */
#ifndef NDEBUG
    PR = 1;
    PRLEVEL(PR, ("%%%% Before relaxed amalgmation\n"));

    if (nf > 0)
    {
        PRLEVEL(PR, ("%%%% Super:\n"));
        for (int64_t k = 0; k <= nf; k++) PRLEVEL(PR, ("  " LD "", Super[k]));
        PRLEVEL(PR, ("\n"));

        PR = 1;
        PRLEVEL(PR, ("%%%% Parent:\n"));
        for (int64_t k = 0; k <= nf; k++) PRLEVEL(PR, ("  " LD "", Parent[k]));
        PRLEVEL(PR, ("\n"));
    }
    PR = 1;
#endif

    // Relax amalgamation before making the list of children
    // The gist is to make a cumsum over Pivotal columns
    // then start from children and see if I merge it to the father how many
    // pivotal columns will it have;
    // if it has less than threshold merge the child to the father
    //
    // Super and Parent  and upperbounds
    // Parent needs an extra work space
    // Super can be changed in space upperbound can be changed in space
    // Upperbound how to do: maximum of pervious upperbounds
    // Number of the columns of the root of each subtree
    //
    int64_t threshold = Control->relaxed_amalgamation_threshold;
    PRLEVEL(1, ("Relaxed amalgamation threshold = " LD "\n", threshold));
    int64_t newF = 0;

    // int64_t num_roots = 0; //deprecated I dont use it anymore

    for (int64_t f = 0; f < nf; f++)
    {  // finding representative for each front
        int64_t repr = f;
        // amalgamate till number of pivot columns is small
        PRLEVEL(PR, ("%% repr = " LD " Parent =" LD "\n", repr, Parent[repr]));
        PRLEVEL(PR, ("%%size of Potential pivot= " LD "\n",
                     Super[Parent[repr] + 1] - Super[f]));
        while (Super[Parent[repr] + 1] - Super[f] < threshold &&
               Parent[repr] != -1)
        {
            repr = Parent[repr];
            PRLEVEL(PR, ("%%Middle stage f= " LD " repr = " LD "\n", f, repr));
            PRLEVEL(PR, ("%%number of pivot cols= " LD "\n",
                         Super[repr + 1] - Super[f]));
            PRLEVEL(PR, ("%%number of pivot cols if Parent collapsed= " LD "\n",
                         Super[Parent[repr] + 1] - Super[f]));
        }
        // if (Parent[repr] == -1) num_roots++;

        PRLEVEL(PR, ("%% newF = " LD " for:\n", newF));
        for (int64_t k = f; k <= repr; k++)
        {
            PRLEVEL(PR, ("%%  " LD " ", k));
            fmap[k] = newF;
        }
        PRLEVEL(PR, ("%%repr = " LD "\n", repr));
        newF++;
        f = repr;
    }
    // Sym->num_roots = num_roots;

#ifndef NDEBUG
    PR = 1;
#endif

    int64_t newNf = newF;  // new size of number of fronts
    fmap[nf] = -1;

    // newParent size is newF+1 potentially smaller than nf
    if (newF > 0)
        newParent =
            static_cast<int64_t*>(paru_realloc(newF + 1, sizeof(int64_t), newParent, &size));
    if (size != (size_t)newF + 1)
    {
        PRLEVEL(1, ("ParU: out of memory\n"));
        // newParent = tmp_newParent;
        ParU_Freesym(S_handle, Control);
        umfpack_dl_paru_free_sw(&SW);
        FREE_WORK;
        return PARU_OUT_OF_MEMORY;
    }
    ASSERT(newF <= nf);

    for (int64_t oldf = 0; oldf < nf; oldf++)
    {  // maping old to new
        int64_t newf = fmap[oldf];
        int64_t oldParent = Parent[oldf];
        newParent[newf] = oldParent >= 0 ? fmap[oldParent] : -1;
    }
    PRLEVEL(-1, ("%% newF = " LD " and nf=" LD "\n", newNf, nf));

    /* ---------------------------------------------------------------------- */
    /*         Finding the Upper bound of rows and cols                       */
    /* ---------------------------------------------------------------------- */
    SWType *mySW = static_cast<SWType*>(SW);
    int64_t *Front_nrows = static_cast<int64_t*>(mySW->Front_nrows);
    int64_t *Front_ncols = static_cast<int64_t*>(mySW->Front_ncols);

    if (newNf > 0)
    {
        Fm = static_cast<int64_t*>(paru_calloc((newNf + 1), sizeof(int64_t)));
        Cm = static_cast<int64_t*>(paru_alloc((newNf + 1), sizeof(int64_t)));
        // int64_t *roots = (int64_t *)paru_alloc((num_roots), sizeof(int64_t));
        Sym->Fm = Fm;
        Sym->Cm = Cm;
        // Sym->roots = roots;

        // if (Fm == NULL || Cm == NULL || roots == NULL)
        if (Fm == NULL || Cm == NULL)
        {
            PRLEVEL(1, ("ParU: out of memory\n"));
            // paru_free(n, sizeof(int64_t), inv_Diag_map);
            ParU_Freesym(S_handle, Control);
            umfpack_dl_paru_free_sw(&SW);
            FREE_WORK;
            return PARU_OUT_OF_MEMORY;
        }
    }

    // after relaxed amalgamation
    // Copying first nf+1 elements of Front_nrows(UMFPACK) into Fm(SPQR like)
    for (int64_t oldf = 0; oldf < nf; oldf++)
    {
        PRLEVEL(PR, ("oldf=" LD "\n", oldf));
        int64_t newf = fmap[oldf];
        PRLEVEL(PR, ("newf=" LD "\n", newf));
        PRLEVEL(PR, ("next=" LD "\n", fmap[oldf + 1]));

        if (newf != fmap[oldf + 1])
        {                                   // either root or not amalgamated
            Fm[newf] += Front_nrows[oldf];  // + Front_npivcol[oldf];
            Cm[newf] = Front_ncols[oldf] - Front_npivcol[oldf];
            // newSuper[newf+1] = Super[oldf+1] ;
            Super[newf + 1] = Super[oldf + 1];
            PRLEVEL(PR, ("Fm[newf]=" LD "\n", Fm[newf]));
            PRLEVEL(PR, ("Cm[newf]=" LD "\n", Cm[newf]));
        }
        else
        {
            Fm[newf] += Front_npivcol[oldf];
            // Cm[newf] += Front_npivcol[oldf];
            PRLEVEL(PR, ("Fm[newf]=" LD "\n", Fm[newf]));
            PRLEVEL(PR, ("Cm[newf]=" LD "\n", Cm[newf]));
        }
    }
    if (Super) Super[newNf] = Super[nf];

#ifndef NDEBUG
    PR = 1;
    PRLEVEL(PR, ("%%%% After relaxed amalgmation\n"));

    if (Cm)
    {
        PRLEVEL(PR, ("Cm =\n"));
        for (int64_t i = 0; i < nf + 1; i++) PRLEVEL(PR, ("" LD " ", Cm[i]));
        PRLEVEL(PR, ("\n"));
    }

    if (Fm)
    {
        PRLEVEL(PR, ("Fm =\n"));
        for (int64_t i = 0; i < nf + 1; i++) PRLEVEL(PR, ("" LD " ", Fm[i]));
        PRLEVEL(PR, ("\n"));
    }

    PRLEVEL(PR, ("Pivot cols=\n"));
    for (int64_t i = 0; i < nf + 1; i++) PRLEVEL(PR, ("" LD " ", Front_npivcol[i]));
    PRLEVEL(PR, ("\n"));

    PRLEVEL(PR, ("Upper bound on Rows =\n"));
    for (int64_t i = 0; i < nf + 1; i++) PRLEVEL(PR, ("" LD " ", Front_nrows[i]));
    PRLEVEL(PR, ("\n"));

    PRLEVEL(PR, ("Upper bound on Cols=\n"));
    for (int64_t i = 0; i < nf + 1; i++) PRLEVEL(PR, ("" LD " ", Front_ncols[i]));
    PRLEVEL(PR, ("\n"));

    PRLEVEL(PR, ("%%%% Super:\n"));
    if (nf > 0)
        for (int64_t k = 0; k <= nf; k++) PRLEVEL(PR, ("  " LD "", Super[k]));
    PRLEVEL(PR, ("\n"));

    PRLEVEL(PR, ("%%%% fmap:\n"));
    for (int64_t k = 0; k <= nf; k++) PRLEVEL(PR, ("  " LD "", fmap[k]));
    PRLEVEL(PR, ("\n"));

    PR = -1;
    PRLEVEL(PR, ("%%%% newParent(" LD "):\n", newNf));
    for (int64_t k = 0; k < newNf; k++) PRLEVEL(PR, ("" LD " ", newParent[k]));
    PRLEVEL(PR, ("\n"));
#endif
    // This prints my etree; I keep it in comments to find it easily
    // printf ("%%%% newParent:\n");
    // for (int64_t k = 0; k < newNf; k++) printf("" LD " ", newParent[k]);
    // printf("\n");

    paru_free(nf + 1, sizeof(int64_t), Sym->Parent);
    Sym->Parent = Parent = newParent;
    newParent = NULL;
    nf = Sym->nf = newNf;

    umfpack_dl_paru_free_sw(&SW);
    SW = NULL;
    //////////////////////end of relaxed amalgamation//////////////////////////

    //////////////////////computing the depth of each front ///////////////////
#ifndef NDEBUG
    PR = 1;
#endif

    for (int64_t f = 0; f < nf; f++)
    {
        if (Parent[f] != -1 && Depth[f] == 0)  // not computed so far
        {
            PRLEVEL(PR, ("%% Precompute Depth[" LD "]=" LD "\n", f, Depth[f]));
            int64_t d = 1;
            int64_t p = Parent[f];
            while (Parent[p] != -1 && Depth[p] == 0)
            {
                p = Parent[p];
                d++;
                PRLEVEL(PR, ("%% Up Depth[" LD "]=" LD " d=" LD "\n", p, Depth[p], d));
            }
            Depth[f] = d + Depth[p];  // depth is computed
            PRLEVEL(PR, ("%% Depth[" LD "]=" LD " Depth[" LD "]=" LD "\n", p, Depth[p], f,
                         Depth[f]));
            // updating ancestors
            d = Depth[f];
            p = Parent[f];
            while (Parent[p] != -1 && Depth[p] == 0)
            {
                Depth[p] = --d;
                p = Parent[p];
                PRLEVEL(PR, ("%% down Depth[" LD "]=" LD " d=" LD "\n", p, Depth[p], d));
            }
        }
        PRLEVEL(PR, ("%% Postcompute Depth[" LD "]=" LD "\n", f, Depth[f]));
    }

    // depth of each non leave node is the max of depth of its children
    // for (int64_t f = 0; f < nf; f++)
    //{
    //    int64_t p = f;
    //    while (Parent[p] != -1 && Depth[Parent[p]] < Depth[p])
    //    {
    //        Depth[Parent[p]] = Depth[p];
    //        p = Parent[p];
    //    }
    //}
#ifndef NDEBUG
    PR = 1;
#endif

    //////////////////////end of computing the depth of each front ////////////

    // Making Children list and computing the bound sizes
    if (nf > 0)
    {
        Childp = (int64_t *)paru_calloc((nf + 2), sizeof(int64_t));
        Sym->Childp = Childp;
        if (Childp == NULL)
        {
            PRLEVEL(1, ("ParU: out of memory\n"));
            ParU_Freesym(S_handle, Control);
            FREE_WORK;
            return PARU_OUT_OF_MEMORY;
        }
    }

#ifndef NDEBUG
    int64_t Us_bound_size = 0;
    int64_t LUs_bound_size = 0;
    int64_t row_Int_bound = 0;
    int64_t col_Int_bound = 0;
#endif
    for (int64_t f = 0; f < nf; f++)
    {
#ifndef NDEBUG
        int64_t fp = Super[f + 1] - Super[f];
        int64_t fm = Sym->Fm[f];
        int64_t fn = Sym->Cm[f]; /* Upper bound number of cols of F */
        Us_bound_size += fp * fn;
        LUs_bound_size += fp * fm;
        row_Int_bound += fm;
        col_Int_bound += fn;
#endif
        if (Parent[f] > 0) Childp[Parent[f] + 1]++;
    }
    // see GraphBLAS/Source/GB_cumsum.c
    paru_cumsum(nf + 2, Childp, Control);
#ifndef NDEBUG
    Sym->Us_bound_size = Us_bound_size;
    Sym->LUs_bound_size = LUs_bound_size;
    Sym->row_Int_bound = row_Int_bound;
    Sym->col_Int_bound = col_Int_bound;
    PR = 1;
    PRLEVEL(PR, ("%%row_Int_bound=" LD ", col_Int_bound=" LD "", row_Int_bound,
                 col_Int_bound));
    PRLEVEL(PR,
            ("%%-Us_bound_size = " LD " LUs_bound_size = " LD " sum = " LD "\n",
             Us_bound_size, LUs_bound_size,
             row_Int_bound + col_Int_bound + Us_bound_size + LUs_bound_size));
    PR = 1;
    PRLEVEL(PR, ("%%%%-Chidlp-----"));
    if (Childp)
    {
        for (int64_t f = 0; f < nf + 2; f++) PRLEVEL(PR, ("" LD " ", Childp[f]));
        PRLEVEL(PR, ("\n"));
    }
#endif
    if (nf > 0)
    {
        Child = (int64_t *)paru_calloc((nf + 1), sizeof(int64_t));
        Sym->Child = Child;
        if (Child == NULL)
        {
            PRLEVEL(1, ("ParU: out of memory\n"));
            ParU_Freesym(S_handle, Control);
            FREE_WORK;
            return PARU_OUT_OF_MEMORY;
        }
    }
    // copy of Childp using Work for other places also
    Work = (int64_t *)paru_alloc(std::max(m, n) + 2, sizeof(int64_t));
    int64_t *cChildp = Work;
    if (cChildp == NULL)
    {
        PRLEVEL(1, ("ParU: out of memory\n"));
        ParU_Freesym(S_handle, Control);
        FREE_WORK;
        return PARU_OUT_OF_MEMORY;
    }

    if (nf > 0)
        paru_memcpy(cChildp, Childp, (nf + 2) * sizeof(int64_t), Control);
    else
        cChildp = NULL;

    for (int64_t f = 0; f < nf; f++)
    {
        if (Parent[f] > 0) Child[cChildp[Parent[f]]++] = f;
    }
#ifndef NDEBUG
    if (cChildp)
    {
        PR = 1;
        PRLEVEL(PR, ("%%%%_cChidlp_____"));
        for (int64_t f = 0; f < nf + 2; f++) PRLEVEL(PR, ("" LD " ", cChildp[f]));
        PRLEVEL(PR, ("\n"));
    }
#endif

    /* ---------------------------------------------------------------------- */
    /*                   computing the Staircase structures                   */
    /* ---------------------------------------------------------------------- */

    Sp = Sym->Sp = (int64_t *)paru_calloc(m + 1 - n1, sizeof(int64_t));
    Sleft = Sym->Sleft = (int64_t *)paru_alloc(n + 2 - n1, sizeof(int64_t));
    Pinv = (int64_t *)paru_alloc(m + 1, sizeof(int64_t));

    if (Sp == NULL || Sleft == NULL || Pinv == NULL)
    {
        PRLEVEL(1, ("ParU: out of memory\n"));
        ParU_Freesym(S_handle, Control);
        FREE_WORK;
        return PARU_OUT_OF_MEMORY;
    }
    Sym->Pinv = Pinv;

    //-------- computing the inverse permutation for P and Diag_map
#pragma omp taskloop default(none) shared(m, Pinv, Pinit) grainsize(512)
    for (int64_t i = 0; i < m; i++)
    {
        Pinv[Pinit[i]] = i;
    }

    if (Diag_map)
    {
#pragma omp taskloop default(none) shared(m, Diag_map, inv_Diag_map) \
    grainsize(512)
        for (int64_t i = 0; i < m; i++)
        {
            int64_t newrow = Diag_map[i];  // Diag_map[newcol] = newrow
                                       // it can be confusing while it is a
                                       // mapping from col to row and vice verse
            inv_Diag_map[newrow] = i;
        }
    }

#ifndef NDEBUG
    PRLEVEL(PR, ("Qinit =\n"));
    for (int64_t j = 0; j < std::min(64, n); j++) PRLEVEL(PR, ("" LD " ", Qinit[j]));
    PRLEVEL(PR, ("\n"));

    PRLEVEL(PR, ("Pinit =\n"));
    for (int64_t i = 0; i < std::min(64, m); i++) PRLEVEL(PR, ("" LD " ", Pinit[i]));
    PRLEVEL(PR, ("\n"));

    PR = 1;
    PRLEVEL(PR, ("Pinv =\n"));
    for (int64_t i = 0; i < m; i++) PRLEVEL(PR, ("" LD " ", Pinv[i]));
    PRLEVEL(PR, ("\n"));

    PR = 1;
    if (inv_Diag_map)
    {
        PRLEVEL(PR, ("inv_Diag_map =\n"));
        for (int64_t i = 0; i < std::min(64, n); i++)
            PRLEVEL(PR, ("" LD " ", inv_Diag_map[i]));
        PRLEVEL(PR, ("\n"));
    }
    PR = 1;

#endif

    Ps = static_cast<int64_t*>(paru_calloc(m - n1, sizeof(int64_t)));
    if (cs1 > 0)
    {
        Sup = Sym->ustons.Sup = static_cast<int64_t*>(paru_calloc(cs1 + 1, sizeof(int64_t)));
        cSup = static_cast<int64_t*>(paru_alloc(cs1 + 1, sizeof(int64_t)));
    }
    if (rs1 > 0)
    {
        Slp = Sym->lstons.Slp = static_cast<int64_t*>(paru_calloc(rs1 + 1, sizeof(int64_t)));
        cSlp = static_cast<int64_t*>(paru_alloc(rs1 + 1, sizeof(int64_t)));
    }

    if (((Slp == NULL || cSlp == NULL) && rs1 != 0) ||
        ((Sup == NULL || cSup == NULL) && cs1 != 0) || Ps == NULL)
    {
        PRLEVEL(1, ("ParU: rs1=" LD " cs1=" LD " memory problem\n", rs1, cs1));
        ParU_Freesym(S_handle, Control);
        FREE_WORK;
        return PARU_OUT_OF_MEMORY;
    }
    int64_t sunz = 0;  // U nnz: singlteton nnzero of s
    int64_t slnz = 0;  // L nnz: singlteton nnzero of s
    int64_t snz = 0;   // s nonzero: nnz in submatrix excluding singletons
    int64_t rowcount = 0;
    Sleft[0] = 0;
    // counting number of entries in each row of submatrix Sp and also making
    // the singelton matrices
#ifndef NDEBUG
    PR = 1;
    PRLEVEL(PR, ("Computing Staircase Structure and singleton structure\n"));
    PRLEVEL(PR, ("rs1= " LD " cs1=" LD "\n", rs1, cs1));
    PR = 1;
#endif
    for (int64_t newcol = 0; newcol < n1; newcol++)
    {  // The columns that are just in singleton
        int64_t oldcol = Qinit[newcol];
        PRLEVEL(PR, ("newcol = " LD " oldcol=" LD "\n", newcol, oldcol));
        for (int64_t p = Ap[oldcol]; p < Ap[oldcol + 1]; p++)
        {
            int64_t oldrow = Ai[p];
            int64_t newrow = Pinv[oldrow];
            PRLEVEL(PR, ("newrow=" LD " oldrow=" LD "\n", newrow, oldrow));
            if (newrow < cs1)
            {  // inside U singletons
                PRLEVEL(PR, ("Counting phase,Inside U singletons\n"));
                sunz++;
                Sup[newrow + 1]++;
            }
            else
            {  // inside L singletons
#ifndef NDEBUG
                PR = 1;
#endif
                PRLEVEL(PR, ("Counting phase,Inside L singletons "));
                PRLEVEL(PR, ("newrow=" LD " oldrow=" LD "\n", newrow, oldrow));
                slnz++;
                // if (newcol-cs1 +1 == 0)
                if (newcol < cs1)
                {
                    PRLEVEL(PR, ("newrow=" LD " oldrow=" LD "\n", newrow, oldrow));
                    PRLEVEL(PR, ("!!!! newcol=" LD " cs1=" LD "\n", newcol, cs1));
                }
                ASSERT(newcol - cs1 + 1 != 0);
                ASSERT(newcol >= cs1);
                Slp[newcol - cs1 + 1]++;
            }
        }
    }

#ifndef NDEBUG
    PR = 1;
    PRLEVEL(PR, ("Analyze: Sup and Slp in the middle\n"));
    if (cs1 > 0)
    {
        PRLEVEL(PR, ("(" LD ") Sup =", sunz));
        for (int64_t k = 0; k <= cs1; k++) PRLEVEL(PR, ("" LD " ", Sup[k]));
        PRLEVEL(PR, ("\n"));
    }
    if (rs1 > 0)
    {
        PRLEVEL(PR, ("(" LD ") Slp =", slnz));
        for (int64_t k = cs1; k <= n1; k++) PRLEVEL(PR, ("" LD " ", Slp[k - cs1]));
        PRLEVEL(PR, ("\n"));
    }
#endif
    for (int64_t newcol = n1; newcol < n; newcol++)
    {
        int64_t oldcol = Qinit[newcol];
        PRLEVEL(1, ("newcol = " LD " oldcol=" LD "\n", newcol, oldcol));
        for (int64_t p = Ap[oldcol]; p < Ap[oldcol + 1]; p++)
        {
            int64_t oldrow = Ai[p];
            int64_t newrow = Pinv[oldrow];
            int64_t srow = newrow - n1;
            PRLEVEL(1, ("\tnewrow=" LD " oldrow=" LD " srow=" LD "\n", newrow, oldrow,
                        srow));
            if (srow >= 0)
            {  // it is insdie S otherwise it is part of singleton
                // and update Diag_map
                if (Sp[srow + 1] == 0)
                {  // first time seen
                    PRLEVEL(1, ("\tPs[" LD "]= " LD "\n", rowcount, srow));
                    Ps[rowcount] = srow;
                    Pinit[n1 + rowcount] = oldrow;
                    if (Diag_map)
                    {
                        int64_t diag_col = inv_Diag_map[newrow];
                        if (diag_col < n1)
                        {
                            PRLEVEL(0, ("diag_col= " LD "\n", diag_col));
                        }
                        // This assertions is not correct because
                        // sometimes singltones can steal diagonals
                        // ASSERT(diag_col >= n1);
                        // row of s ~~~~~~> col Qfill confusing be aware
                        Diag_map[diag_col] = rowcount + n1;  // updating
                        // diag_map
                    }

                    rowcount++;
                }
                snz++;
                Sp[srow + 1]++;
            }
            else
            {  // inside the U singletons
                Sup[newrow + 1]++;
                sunz++;
            }
        }
        Sleft[newcol - n1 + 1] = rowcount;
    }
    Sleft[n - n1 + 1] = m - n1 - rowcount;  // empty rows of S if any
    Sym->snz = snz;

#ifndef NDEBUG
    PR = 1;
    PRLEVEL(PR, ("Sup and Slp finished (before cumsum)U-sing =" LD " L-sing=" LD "\n",
                 sunz, slnz));
    if (cs1 > 0)
    {
        PRLEVEL(PR, ("(" LD ") Sup =", sunz));
        for (int64_t k = 0; k <= cs1; k++) PRLEVEL(PR, ("" LD " ", Sup[k]));
        PRLEVEL(PR, ("\n"));
    }
    if (rs1 > 0)
    {
        PRLEVEL(PR, ("(" LD ") Slp =", slnz));
        for (int64_t k = cs1; k <= n1; k++) PRLEVEL(PR, ("" LD " ", Slp[k - cs1]));
        PRLEVEL(PR, ("\n"));
    }
    PR = 1;
    PRLEVEL(PR, ("Ps =\n"));
    for (int64_t k = 0; k < rowcount; k++) PRLEVEL(PR, ("" LD " ", Ps[k]));
    PRLEVEL(PR, ("\n"));
    if (Diag_map)
    {
        PRLEVEL(PR, ("Symbolic Diag_map (" LD ") =\n", n));
        for (int64_t i = 0; i < std::min(64, n); i++) PRLEVEL(PR, ("" LD " ", Diag_map[i]));
        PRLEVEL(PR, ("\n"));
    }
    PR = 1;

#endif

    if (rowcount < m - n1)
    {
        // That must not happen anyway if umfpack finds it
        PRLEVEL(1, ("ParU: Empty rows in submatrix\n"));
#ifndef NDEBUG
        PRLEVEL(1, ("m = " LD ", n1 = " LD ", rowcount = " LD ", snz = " LD "\n", m, n1,
                    rowcount, snz));
        for (int64_t srow = 0; srow < m - n1; srow++)
        {
            if (Sp[srow] == 0)
            {
                PRLEVEL(1, ("Row " LD " is empty\n", srow));
            }
        }
#endif
        ParU_Freesym(S_handle, Control);
        FREE_WORK;
        return PARU_SINGULAR;
    }
    ASSERT(rowcount == m - n1);

    // Update Sp based on new permutation for Sp[1...m+1]
#ifndef NDEBUG
    PR = 1;
    PRLEVEL(PR, ("Before permutation Sp =\n"));
    for (int64_t i = n1; i <= m; i++) PRLEVEL(PR, ("" LD " ", Sp[i - n1]));
    PRLEVEL(PR, ("\n"));
#endif

    // update Pinv
#pragma omp taskloop default(none) shared(m, n1, Pinv, Pinit) grainsize(512)
    for (int64_t i = n1; i < m; i++)
    {
        Pinv[Pinit[i]] = i;
    }

    ///////////////////////////////////////////////////////////////
    int64_t *cSp = Work;
    if (n1 == m)
    {  // no fronts
        cSp[0] = 0;
    }
    else
    {
#pragma omp taskloop default(none) shared(m, n1, cSp, Sp, Ps) grainsize(512)
        for (int64_t i = n1; i < m; i++)
        {
            int64_t row = i - n1;
            cSp[row + 1] = Sp[Ps[row] + 1];
        }
    }

#ifndef NDEBUG
    PRLEVEL(PR, ("After permutation Sp =\n"));
    for (int64_t i = n1; i <= m; i++) PRLEVEL(PR, ("" LD " ", cSp[i - n1]));
    PRLEVEL(PR, ("\n"));
    PR = 1;
    if (rs1 > 0)
    {
        PRLEVEL(PR, ("(" LD ") Slp =", slnz));
        for (int64_t k = 0; k <= rs1; k++) PRLEVEL(PR, ("" LD " ", Slp[k]));
    }
    PRLEVEL(PR, ("\n"));
#endif

    paru_cumsum(m + 1 - n1, cSp, Control);
    paru_memcpy(Sp, cSp, (m + 1 - n1) * sizeof(int64_t), Control);

    if (cs1 > 0)
    {
        paru_cumsum(cs1 + 1, Sup, Control);
        paru_memcpy(cSup, Sup, (cs1 + 1) * sizeof(int64_t), Control);
    }
    if (rs1 > 0)
    {
        paru_cumsum(rs1 + 1, Slp, Control);
        paru_memcpy(cSlp, Slp, (rs1 + 1) * sizeof(int64_t), Control);
    }

#ifndef NDEBUG
    PR = 1;
    PRLEVEL(PR, ("Analyze: Sup and Slp in the after cumsum\n"));
    if (cs1 > 0)
    {
        PRLEVEL(PR, ("(" LD ") Sup =", sunz));
        for (int64_t k = 0; k <= cs1; k++)
        {
            PRLEVEL(PR, ("" LD " ", Sup[k]));
            PRLEVEL(PR + 2, ("c" LD " ", cSup[k]));
            if (Sup[k] != cSup[k])
                PRLEVEL(PR, ("Sup[" LD "] =" LD ", cSup=" LD "", k, Sup[k], cSup[k]));
            ASSERT(Sup[k] == cSup[k]);
        }
        PRLEVEL(PR, ("\n"));
    }
    if (rs1 > 0)
    {
        PRLEVEL(PR, ("(" LD ") Slp =", slnz));
        for (int64_t k = 0; k <= rs1; k++)
        {
            PRLEVEL(PR, ("" LD " ", Slp[k]));
            PRLEVEL(PR + 2, ("o" LD " ", cSlp[k]));
            if (Slp[k] != cSlp[k])
                PRLEVEL(PR,
                        ("\nSup[" LD "] =" LD ", cSup=" LD "\n", k, Slp[k], cSlp[k]));
            ASSERT(Slp[k] == cSlp[k]);
        }
        PRLEVEL(PR, ("\n"));
    }
    PR = 1;
    PRLEVEL(PR, ("After cumsum  Sp =\n"));
    for (int64_t i = n1; i <= m; i++) PRLEVEL(PR, ("" LD " ", Sp[i - n1]));
    PRLEVEL(PR, ("\n"));
#endif

#ifndef NDEBUG
    PR = 1;
    PRLEVEL(PR, ("After Stair case Pinv =\n"));
    for (int64_t i = 0; i < m; i++) PRLEVEL(PR, ("" LD " ", Pinv[i]));
    PRLEVEL(PR, ("\n"));
#endif
    // PofA
    Sym->Pinit = Pinit;
    Sym->Pinv = Pinv;
    ///////////////////////////////////////////////////////////////
#ifndef NDEBUG
    PR = 1;
    PRLEVEL(PR, ("Before Sj Sp =\n"));
    for (int64_t i = n1; i <= m; i++) PRLEVEL(PR, ("" LD " ", cSp[i - n1]));
    PRLEVEL(PR, ("\n"));
#endif

    Sym->ustons.nnz = sunz;
    Sym->lstons.nnz = slnz;
    if (cs1 > 0)
    {
        Suj = static_cast<int64_t*>(paru_alloc(sunz, sizeof(int64_t)));
    }
    Sym->ustons.Suj = Suj;

    if (rs1 > 0)
    {
        Sli = static_cast<int64_t*>(paru_alloc(slnz, sizeof(int64_t)));
    }
    Sym->lstons.Sli = Sli;

    // Updating Sj using copy of Sp
    Sj = static_cast<int64_t*>(paru_alloc(snz, sizeof(int64_t)));
    Sym->Sj = Sj;

    if (Sj == NULL || (cs1 > 0 && Suj == NULL) || (rs1 > 0 && Sli == NULL))
    {
        PRLEVEL(1, ("ParU: out of memory\n"));
        ParU_Freesym(S_handle, Control);
        FREE_WORK;
        return PARU_OUT_OF_MEMORY;
    }

    // construct Sj and singltons
#ifndef NDEBUG
    PR = 1;
    PRLEVEL(PR, ("Constructing Sj and singletons\n"));
#endif
    for (int64_t newcol = 0; newcol < n1; newcol++)
    {  // The columns that are just in singleton
        int64_t oldcol = Qinit[newcol];
        PRLEVEL(PR, ("newcol = " LD " oldcol=" LD "\n", newcol, oldcol));
        for (int64_t p = Ap[oldcol]; p < Ap[oldcol + 1]; p++)
        {
            int64_t oldrow = Ai[p];
            int64_t newrow = Pinv[oldrow];
            PRLEVEL(PR, ("newrow=" LD " oldrow=" LD "\n", newrow, oldrow));
            if (newrow < cs1)
            {  // inside U singletons CSR
                PRLEVEL(PR, ("Inside U singletons\n"));
                if (newcol == newrow)
                {  // diagonal entry
                    Suj[Sup[newrow]] = newcol;
                }
                else
                {
                    Suj[++cSup[newrow]] = newcol;
                }
            }
            else
            {  // inside L singletons CSC
                PRLEVEL(PR, ("Inside L singletons\n"));
                if (newcol == newrow)
                {  // diagonal entry
                    Sli[Slp[newcol - cs1]] = newrow;
                }
                else
                {
                    Sli[++cSlp[newcol - cs1]] = newrow;
                }
            }
        }
    }

    PRLEVEL(PR, ("The rest of matrix after singletons\n"));
    for (int64_t newcol = n1; newcol < n; newcol++)
    {
        int64_t oldcol = Qinit[newcol];
        for (int64_t p = Ap[oldcol]; p < Ap[oldcol + 1]; p++)
        {
            int64_t oldrow = Ai[p];
            int64_t newrow = Pinv[oldrow];
            int64_t srow = newrow - n1;
            int64_t scol = newcol - n1;
            if (srow >= 0)
            {  // it is insdie S otherwise it is part of singleton
                Sj[cSp[srow]++] = scol;
            }
            else
            {  // inside the U singletons
                PRLEVEL(PR, ("Usingleton rest newcol = " LD " newrow=" LD "\n",
                             newcol, newrow));
                ASSERT(newrow != newcol);  // not a diagonal entry
                Suj[++cSup[newrow]] = newcol;
            }
        }
    }
    PRLEVEL(PR, ("Constructing Sj and singletons finished here\n"));
#ifndef NDEBUG
    PR = 1;
    PRLEVEL(PR, ("Sup and Slp after making \n"));
    if (cs1 > 0)
    {
        PRLEVEL(PR, ("U singltons(CSR) (nnz=" LD ") Sup =", sunz));
        for (int64_t k = 0; k <= cs1; k++) PRLEVEL(PR, ("" LD " ", Sup[k]));
        PRLEVEL(PR, ("\n"));

        for (int64_t newrow = 0; newrow < cs1; newrow++)
        {
            PRLEVEL(PR, ("row = " LD "\n", newrow));
            for (int64_t p = Sup[newrow]; p < Sup[newrow + 1]; p++)
            {
                PRLEVEL(PR, (" (" LD ")", Suj[p]));
            }
            PRLEVEL(PR, ("\n"));
        }
    }
    if (rs1 > 0)
    {
        PRLEVEL(PR, ("L singleons(CSC) (nnz=" LD ") Slp =", slnz));
        for (int64_t k = cs1; k <= n1; k++) PRLEVEL(PR, ("" LD " ", Slp[k - cs1]));
        PRLEVEL(PR, ("\n"));

        for (int64_t newcol = cs1; newcol < n1; newcol++)
        {
            PRLEVEL(PR, ("col = " LD "\n", newcol));
            for (int64_t p = Slp[newcol - cs1]; p < Slp[newcol - cs1 + 1]; p++)
            {
                PRLEVEL(PR, (" (" LD ")", Sli[p]));
            }
            PRLEVEL(PR, ("\n"));
        }
    }
    PR = 1;
    PRLEVEL(PR, ("Sp =\n"));
    for (int64_t i = n1; i <= m; i++) PRLEVEL(PR, ("" LD " ", Sp[i - n1]));
    PRLEVEL(PR, ("\n"));

    PRLEVEL(PR, ("Sj =\n"));
    for (int64_t k = 0; k < snz; k++) PRLEVEL(PR, ("" LD " ", Sj[k]));
    PRLEVEL(PR, ("\n"));
    PR = 1;
    for (int64_t i = 0; i < m - n1; i++)
    {
        PRLEVEL(PR, (" i=" LD " cSp=" LD " Sp=" LD "\n", i, cSp[i], Sp[i + 1]));
        ASSERT(cSp[i] == Sp[i + 1]);
    }
#endif
    /* ---------------------------------------------------------------------- */
    /*    computing the augmented tree and the data structures related        */
    /* ---------------------------------------------------------------------- */

#ifndef NDEBUG
    PR = 1;
    // print fronts
    for (int64_t f = 0; f < nf; f++)
    {
        int64_t col1 = Super[f];
        int64_t col2 = Super[f + 1];
        int64_t fp = Super[f + 1] - Super[f];
        PRLEVEL(PR, ("%% Front=" LD " Parent=" LD "\n", f, Parent[f]));
        PRLEVEL(PR, ("%% " LD "..." LD " npivotal=" LD "\n", col1, col2, fp));
        PRLEVEL(PR, ("%% list of " LD " children: ", Childp[f + 1] - Childp[f]));
        for (int64_t i = Childp[f]; i <= Childp[f + 1] - 1; i++)
            PRLEVEL(PR, ("" LD " ", Child[i]));
        PRLEVEL(PR, ("\n"));
    }
    PR = 1;
#endif
    // submatrix is msxns
    int64_t ms = m - n1;
#ifndef NDEBUG
    int64_t ns = n - n1;
#endif

    if (nf > 0)
    {
        Sym->aParent = aParent = static_cast<int64_t*>(paru_alloc(ms + nf, sizeof(int64_t)));
        Sym->aChild = aChild = static_cast<int64_t*>(paru_alloc(ms + nf + 1, sizeof(int64_t)));
        Sym->aChildp = aChildp = static_cast<int64_t*>(paru_alloc(ms + nf + 2, sizeof(int64_t)));
        Sym->first = first = static_cast<int64_t*>(paru_alloc(nf + 1, sizeof(int64_t)));
        Sym->row2atree = rM = static_cast<int64_t*>(paru_alloc(ms, sizeof(int64_t)));
        Sym->super2atree = snM = static_cast<int64_t*>(paru_alloc(nf, sizeof(int64_t)));

        Sym->front_flop_bound = front_flop_bound =
            static_cast<double*>(paru_alloc(nf + 1, sizeof(double)));
        Sym->stree_flop_bound = stree_flop_bound =
            static_cast<double*>(paru_calloc(nf + 1, sizeof(double)));

        if (aParent == NULL || aChild == NULL || aChildp == NULL ||
            rM == NULL || snM == NULL || first == NULL ||
            front_flop_bound == NULL || stree_flop_bound == NULL)
        {
            PRLEVEL(1, ("ParU: Out of memory in symbolic phase"));
            ParU_Freesym(S_handle, Control);
            FREE_WORK;
            return PARU_OUT_OF_MEMORY;
        }
        // initialization
        paru_memset(aParent, -1, (ms + nf) * sizeof(int64_t), Control);
#ifndef NDEBUG
        paru_memset(aChild, -1, (ms + nf + 1) * sizeof(int64_t), Control);
        PR = 1;
#endif
        paru_memset(aChildp, -1, (ms + nf + 2) * sizeof(int64_t), Control);
        paru_memset(first, -1, (nf + 1) * sizeof(int64_t), Control);

        aChildp[0] = 0;
    }
    int64_t offset = 0;  // number of rows visited in each iteration orig front+
                     // rows
    int64_t lastChildFlag = 0;
    int64_t childpointer = 0;
    // int64_t root_count = 0;

    for (int64_t f = 0; f < nf; f++)
    {
#ifndef NDEBUG
        PRLEVEL(PR, ("%% Front " LD "\n", f));
        PRLEVEL(PR, ("%% pivot columns [ " LD " to " LD " ] n: " LD " \n", Super[f],
                     Super[f + 1] - 1, ns));
#endif

        // computing works in each front
        int64_t fp = Super[f + 1] - Super[f];  // k
        int64_t fm = Sym->Fm[f];               // m
        int64_t fn = Sym->Cm[f];               // n Upper bound number of cols of f
                              // if (Parent[f] == -1) roots[root_count++] = f;
        front_flop_bound[f] = (double)(fp * fm * fn + fp * fm + fp * fn);
        stree_flop_bound[f] += front_flop_bound[f];
        for (int64_t i = Childp[f]; i <= Childp[f + 1] - 1; i++)
        {
            stree_flop_bound[f] += stree_flop_bound[Child[i]];
            PRLEVEL(PR, ("%% child=" LD " fl=%lf ", Child[i],
                         front_flop_bound[Child[i]]));
        }
        PRLEVEL(PR, ("%% flops bound= %lf\n ", front_flop_bound[f]));
        PRLEVEL(PR, ("%% " LD " " LD " " LD "\n ", fp, fm, fn));
        PRLEVEL(PR, ("%% stree bound= %lf\n ", stree_flop_bound[f]));
        ASSERT(Super[f + 1] <= ns);
        int64_t numRow = Sleft[Super[f + 1]] - Sleft[Super[f]];

        int64_t numoforiginalChild = 0;
        if (lastChildFlag)
        {  // the current node is the parent
            PRLEVEL(PR, ("%% Childs of " LD ": ", f));
            numoforiginalChild = Childp[f + 1] - Childp[f];

            for (int64_t i = Childp[f]; i <= Childp[f + 1] - 1; i++)
            {
                PRLEVEL(PR, ("" LD " ", Child[i]));
                int64_t c = Child[i];
                ASSERT(snM[c] < ms + nf + 1);
                aParent[snM[c]] = offset + numRow;
                PRLEVEL(PR, ("%% aParent[" LD "] =" LD "\n", aParent[snM[c]],
                             offset + numRow));
                ASSERT(childpointer < ms + nf + 1);
                aChild[childpointer++] = snM[c];
            }
        }

        PRLEVEL(1, ("%% numRow=" LD " ", numRow));
        PRLEVEL(1, ("#offset=" LD "\n", offset));
        for (int64_t i = offset; i < offset + numRow; i++)
        {
            ASSERT(aChildp[i + 1] == -1);
            aChildp[i + 1] = aChildp[i];
            PRLEVEL(1, ("%% @i=" LD " aCp[" LD "]=" LD " aCp[" LD "]=" LD "", i, i + 1,
                        aChildp[i + 1], i, aChildp[i]));
        }

        for (int64_t i = Sleft[Super[f]]; i < Sleft[Super[f + 1]]; i++)
        {
            // number of rows
            ASSERT(i < ms);

            rM[i] = i + f;
            ASSERT(i + f < ms + nf + 1);
            aParent[i + f] = offset + numRow;
            ASSERT(childpointer < ms + nf + 1);
            aChild[childpointer++] = i + f;
        }

        offset += numRow;
        snM[f] = offset++;
        ASSERT(offset < ms + nf + 1);
        ASSERT(aChildp[offset] == -1);
        aChildp[offset] = aChildp[offset - 1] + numRow + numoforiginalChild;
        PRLEVEL(
            1, ("\n %% f=" LD " numoforiginalChild=" LD "\n", f, numoforiginalChild));

        if (Parent[f] == f + 1)
        {  // last child due to staircase
            PRLEVEL(1, ("%% last Child =" LD "\n", f));
            lastChildFlag = 1;
        }
        else
            lastChildFlag = 0;
    }
    //    PRLEVEL(1, ("%% root_count=" LD ", num_roots=" LD "\n ", root_count,
    //    num_roots)); ASSERT(root_count == num_roots);

    // Initialize first descendent of the etree
    PRLEVEL(PR, ("%% computing first of\n "));
    for (int64_t i = 0; i < nf; i++)
    {
        for (int64_t r = i; r != -1 && first[r] == -1; r = Parent[r])
        {
            PRLEVEL(PR, ("%% first of " LD " is " LD "\n", r, i));
            first[r] = i;
        }
    }

    ////////////////    computing task tree and such ///////////////////////////
    // std::vector<int64_t> task_helper(nf); //use Work instead of a new vector
    int64_t *task_helper = Work;  // just use first nf of the Work
    const double flop_thresh = 1024;
    int64_t ntasks = 0;

    for (int64_t f = 0; f < nf; f++)
    {
        int64_t par = Parent[f];
        if (par == -1)
        {
            task_helper[f] = ntasks++;
        }
        else
        {
            int64_t num_sibl = Childp[par + 1] - Childp[par] - 1;
            if (num_sibl == 0)  // getting rid of chains
            {
                task_helper[f] = -1;
            }
            else if (stree_flop_bound[f] < flop_thresh)
            {  // getting rid of  small tasks
                task_helper[f] = -1;
            }
            else
            {
                task_helper[f] = ntasks++;
            }
        }
    }

    PRLEVEL(1, ("%% ntasks = " LD "\n", ntasks));
    Sym->ntasks = ntasks;
    if (nf > 0)
    {
        Sym->task_map = task_map = static_cast<int64_t*>(paru_alloc(ntasks + 1, sizeof(int64_t)));
        Sym->task_parent = task_parent = static_cast<int64_t*>(paru_alloc(ntasks, sizeof(int64_t)));
        Sym->task_num_child = task_num_child =
            static_cast<int64_t*>(paru_calloc(ntasks, sizeof(int64_t)));
        Sym->task_depth = task_depth = static_cast<int64_t*>(paru_calloc(ntasks, sizeof(int64_t)));

        if (task_map == NULL || task_parent == NULL || task_num_child == NULL ||
            task_depth == NULL)
        {
            PRLEVEL(1, ("ParU: Out of memory in symbolic phase"));
            ParU_Freesym(S_handle, Control);
            FREE_WORK;
            return PARU_OUT_OF_MEMORY;
        }
        task_map[0] = -1;
    }
    int64_t i = 0;
    for (int64_t f = 0; f < nf; f++)
    {
        if (task_helper[f] != -1) task_map[++i] = f;
    }

    for (int64_t t = 0; t < ntasks; t++)
    {
        int64_t node = task_map[t + 1];
        int64_t rep = Parent[node];
        PRLEVEL(1, ("t=" LD " node=" LD " rep =" LD "\n", t, node, rep));
        if (rep == -1)
        {
            task_parent[t] = -1;
            if (t == 0 && nf > 1)
            {
                task_depth[t] = Depth[0];
            }
        }
        else
        {
            task_depth[t] = std::max(Depth[node], task_depth[t]);
            while (task_helper[rep] < 0) rep = Parent[rep];
            PRLEVEL(1,
                    ("After a while t=" LD " node=" LD " rep =" LD "\n", t, node, rep));
            task_parent[t] = task_helper[rep];
        }
    }

    for (int64_t t = 0; t < ntasks; t++)
    {
        int64_t par = task_parent[t];
        if (par != -1) task_num_child[par]++;
    }

    // finding max size of chains
    // int64_t max_chain = 0;
    // int64_t ii = 0;
    // while (ii < nf)
    //{
    //    int64_t chain_size = 0;
    //    PRLEVEL(1, ("ii = " LD "\n", ii));
    //    while (Parent[ii] != -1 && ii < nf &&
    //      Childp[Parent[ii] + 1] - Childp[Parent[ii]] == 1)
    //    {
    //        chain_size++;
    //        ii++;
    //    }
    //    PRLEVEL(1, ("after ii = " LD "\n", ii));
    //    max_chain = std::max(chain_size, max_chain);
    //    PRLEVEL(1, ("max_chain = " LD "\n", max_chain));
    //    ii++;
    //}
    // Sym->max_chain = max_chain;
    // PRLEVEL(1, ("max_chain = " LD "\n", max_chain));

#ifndef NDEBUG
    if (ntasks > 0)
    {
        PR = 1;
        PRLEVEL(PR, ("%% Task tree helper:\n"));
        for (int64_t i = 0; i < nf; i++)
            PRLEVEL(PR, ("" LD ")" LD " ", i, task_helper[i]));
        PRLEVEL(PR, ("\n%% tasknodes map (" LD "):\n", ntasks));
        for (int64_t i = 0; i <= ntasks; i++) PRLEVEL(PR, ("" LD " ", task_map[i]));
        PR = -1;
        PRLEVEL(PR, ("%% tasktree :\n"));
        for (int64_t i = 0; i < ntasks; i++) PRLEVEL(PR, ("" LD " ", task_parent[i]));
        PRLEVEL(PR, ("\n"));
        PR = 1;
        PRLEVEL(PR, ("\n%% task_num_child:\n"));
        for (int64_t i = 0; i < ntasks; i++)
            PRLEVEL(PR, ("" LD " ", task_num_child[i]));
        PRLEVEL(PR, ("\n"));
        PRLEVEL(PR, ("\n%% " LD " tasks:\n", ntasks));
        for (int64_t t = 0; t < ntasks; t++)
        {
            PRLEVEL(PR, ("" LD "[" LD "-" LD "] ", t, task_map[t] + 1, task_map[t + 1]));
        }
        PRLEVEL(PR, ("\n%% task depth:\n"));
        for (int64_t t = 0; t < ntasks; t++)
        {
            PRLEVEL(PR, ("" LD " ", task_depth[t]));
        }
        PRLEVEL(PR, ("\n"));
    }

    PR = 1;
    PRLEVEL(PR, ("%% super node mapping ,snM (and rows): "));
    for (int64_t f = 0; f < nf; f++)
    {
        ASSERT(snM[f] != -1);
        PRLEVEL(PR, ("" LD " (" LD ") ", snM[f], Sleft[Super[f]]));
    }
    if (nf > 0)
    {
        PRLEVEL(PR, ("- (" LD ") ", Sleft[Super[nf]]));
        PRLEVEL(PR, ("\n"));
    }

    PR = 1;
    if (rM)
    {
        PRLEVEL(PR, ("%% row mapping (rM): "));
        for (int64_t i = 0; i < ms; i++)
        {
            ASSERT(rM[i] != -1);
            PRLEVEL(PR, ("" LD " ", rM[i]));
        }
        PRLEVEL(PR, ("\n"));
    }

    if (aParent)
    {
        PRLEVEL(PR, ("%% aParent: "));
        for (int64_t i = 0; i < ms + nf; i++) PRLEVEL(PR, ("" LD " ", aParent[i]));
        PRLEVEL(PR, ("%% \n"));
    }

    if (aChildp)
    {
        PRLEVEL(PR, ("%% aChildp: "));
        for (int64_t i = 0; i < ms + nf + 2; i++) PRLEVEL(PR, ("" LD " ", aChildp[i]));
        PRLEVEL(PR, ("\n"));
    }

    if (aChild)
    {
        PRLEVEL(PR, ("%% aChild: "));
        for (int64_t i = 0; i < ms + nf + 1; i++) PRLEVEL(PR, ("" LD " ", aChild[i]));
        PRLEVEL(PR, ("\n"));
    }

    PR = 1;
    if (aChild && aChildp)
    {
        for (int64_t i = 0; i < ms + nf; i++)
        {
            if (aChildp[i] == aChildp[i + 1]) continue;
            PRLEVEL(PR, ("%% anode:" LD "", i));
            for (int64_t c = aChildp[i]; c < aChildp[i + 1]; c++)
                PRLEVEL(PR, (" " LD ",", aChild[c]));
            PRLEVEL(PR, ("\n"));
        }
    }
    PR = 1;

    if (first)
    {
        PRLEVEL(PR, ("%% first: "));
        for (int64_t i = 0; i < nf + 1; i++)
            PRLEVEL(PR, ("first[" LD "]=" LD " ", i, first[i]));
        PRLEVEL(PR, ("\n"));
    }

    PR = 1;
    //    PRLEVEL(PR, ("%%%% roots:\n"));
    //    for (int64_t k = 0; k < num_roots; k++) PRLEVEL(PR, ("  " LD "", roots[k]));
    //    PRLEVEL(PR, ("\n"));

    PRLEVEL(PR, ("%%%% Depth:\n"));
    for (int64_t k = 0; k < nf; k++) PRLEVEL(PR, ("  " LD "", Depth[k]));
    PRLEVEL(PR, ("\n"));

#endif

    // This prints the task tree; I keep it in comments to find it easily
    // printf ("%% tasktree :\n");
    // for (int64_t i = 0; i < ntasks; i++) printf("" LD " ", task_parent[i]);
    // printf ("\n");

#ifndef NTIME
    double time = PARU_OPENMP_GET_WTIME;
    time -= start_time;
    PRLEVEL(1, ("%% mRHS paru_apply_inv_perm %lf seconds\n", time));
#endif
    FREE_WORK;
    return PARU_SUCCESS;
}
