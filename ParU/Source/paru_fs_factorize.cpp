////////////////////////////////////////////////////////////////////////////////
//////////////////////////  paru_fs_factorize  /////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

// ParU, Copyright (c) 2022-2024, Mohsen Aznaveh and Timothy A. Davis,
// All Rights Reserved.
// SPDX-License-Identifier: GPL-3.0-or-later

/*! @brief Doing the BLAS factorization in different panels and call degree
 * update when it is necessary.
 *
 * @author Aznaveh
 */

#include "paru_internal.hpp"

//------------------------------------------------------------------------------
// paru_swap_rows: swap pivot rows
//------------------------------------------------------------------------------

void paru_swap_rows
(
    double *F,              // swap rows r1 and r2 of front f
    int64_t *frowList,      // global id's of the rows of this fron
    int64_t m,              // F is m-by-n
    int64_t n,
    int64_t r1,             // r1, r2: rows to swap
    int64_t r2
)
{
    // This function also swap rows r1 and r2 wholly and indices
    if (r1 == r2) return;
    std::swap(frowList[r1], frowList[r2]);
    for (int64_t j = 0; j < n; j++)
    {
        // each column
        std::swap(F[j * m + r1], F[j * m + r2]);
    }
}

//------------------------------------------------------------------------------
// paru_panel_factorize: factorize a single panel
//------------------------------------------------------------------------------

bool paru_panel_factorize
(
    int64_t f,
    int64_t m,
    int64_t n,
    int64_t panel_num,
    int64_t row_end,
    paru_work *Work,
    const ParU_Symbolic Sym,
    ParU_Numeric Num
)
{

    // get Control
    int64_t panel_width = Work->panel_width ;
    double piv_toler    = Work->piv_toler ;
    double diag_toler   = Work->diag_toler ;

    // works like dgetf2f.f in netlib v3.0  here is a link:
    // https://github.com/xianyi/OpenBLAS/blob/develop/reference/dgetf2f.f
    DEBUGLEVEL(0);
    PARU_DEFINE_PRLEVEL;
    PRLEVEL(1, ("%% Inside panel factorization " LD " \n", panel_num));

    int64_t *row_degree_bound = Work->row_degree_bound;

    int64_t j1 = panel_num * panel_width;  // panel starting column

    //  j1 <= panel columns < j2
    //     last panel might be smaller
    int64_t j2 = (j1 + panel_width < n) ? j1 + panel_width : n;

    PRLEVEL(1, ("%% j1= " LD " j2 =" LD " \n", j1, j2));
    PRLEVEL(1, ("%% row_end= " LD "\n", row_end));

    // ASSERT(row_end >= j2);

    int64_t *frowList = Num->frowList[f];
    ParU_Factors *LUs = Num->partial_LUs;
    double *F = LUs[f].p;

#ifndef NDEBUG  // Printing the panel
    int64_t num_col_panel = j2 - j1;
    PRLEVEL(PR, ("%% Starting the factorization\n"));
    PRLEVEL(PR, ("%% This Panel:\n"));
    for (int64_t r = j1; r < row_end; r++)
    {
        PRLEVEL(PR, ("%% " LD "\t", frowList[r]));
        for (int64_t c = j1; c < j2; c++) PRLEVEL(PR, (" %2.5lf\t", F[c * m + r]));
        PRLEVEL(PR, ("\n"));
    }
#endif

    const int64_t *Super = Sym->Super;
    int64_t col1 = Super[f]; /* fornt F has columns col1:col2-1 */
    int64_t *Diag_map = Work->Diag_map;
    int64_t n1 = Sym->n1;

    // column jth of the panel
    for (int64_t j = j1; j < j2; j++)
    {
        // for fat fronts
        if (j >= row_end) break;

        PRLEVEL(1, ("%% j = " LD "\n", j));

        // Initializing maximum element in the column
        int64_t row_max = j;

        double maxval = F[j * m + row_max];

#ifndef NDEBUG
        int64_t row_deg_max = row_degree_bound[frowList[row_max]];
        PRLEVEL(1, ("%% before search max value= %2.4lf row_deg = " LD "\n",
                    maxval, row_deg_max));
#endif

        int64_t row_diag = (Diag_map) ? Diag_map[col1 + j + n1] - n1 : -1;
        double diag_val = maxval;  // initialization
        int64_t diag_found = (frowList[j] == row_diag ? j : -1) ;
        PRLEVEL(1, ("%%curCol=" LD " row_diag=" LD "\n", j + col1 + n1, row_diag));
        PRLEVEL(1, ("%%##j=" LD " value= %2.4lf\n", j, F[j * m + j]));

        for (int64_t i = j + 1; i < row_end; i++)
        {
            // find max
            PRLEVEL(1, ("%%i=" LD " value= %2.4lf", i, F[j * m + i]));
            PRLEVEL(1, (" deg = " LD " \n", row_degree_bound[frowList[i]]));
            if (fabs(maxval) < fabs(F[j * m + i]))
            {
                row_max = i;
                maxval = F[j * m + i];
            }
            if (frowList[i] == row_diag)  // find diag
            {
                PRLEVEL(1, ("%%Found it %2.4lf\n", F[j * m + i]));
                // row_diag = i;
                diag_found = i;
                diag_val = F[j * m + i];
            }
        }

#ifndef NDEBUG
        row_deg_max = row_degree_bound[frowList[row_max]];
#endif

        PRLEVEL(1, ("%% max value= %2.4lf\n", maxval));

        if (maxval == 0)
        {
            PRLEVEL(1, ("%% NO pivot found in " LD "\n", n1 + col1 + j));
#pragma omp atomic write
            Num->res = PARU_SINGULAR;
            continue;
        }
        // initialzing pivot as max numeric value
        double piv = maxval;
        int64_t row_piv = row_max;
        int64_t chose_diag = 0;

        if (Sym->strategy_used == PARU_STRATEGY_SYMMETRIC)
        {
            if (diag_found != -1)
            {
                if (fabs(diag_toler * maxval) < fabs(diag_val))
                {
                    piv = diag_val;
                    row_piv = diag_found;
                    PRLEVEL(1, ("%% symmetric pivot piv value= %2.4lf"
                                " row_piv=" LD "\n",
                                piv, row_piv));
                    chose_diag = 1;
                }
#ifndef NDEBUG
                else
                {
                    PRLEVEL(1, ("%% diag found but too small " LD ""
                                " maxval=%2.4lf diag_val=%e \n",
                                row_piv, maxval, diag_val));
                }
#endif
            }
#ifndef NDEBUG
            else
            {
                PRLEVEL(1, ("%% diag not found " LD "\n", row_piv));
            }
#endif
        }

        // find sparsest between accepteble ones
        // if not symmetric or the diagonal is not good enough
        int64_t row_deg_sp = row_degree_bound[frowList[row_max]];
        if (chose_diag == 0)
        {
            int64_t row_sp = row_max;
            // pragma omp taskloop  default(none) shared(maxval, F, row_sp, j,
            // row_end, m, piv, frowList, row_degree_bound, row_deg_sp)
            // grainsize(512)

            for (int64_t i = j; i < row_end; i++)
            {
                double value = F[j * m + i];
                if (fabs(piv_toler * maxval) < fabs(value) &&
                    row_degree_bound[frowList[i]] < row_deg_sp)
                {
                    // numerically acceptalbe and sparser
                    // pragma omp critical
                    {
                        piv = value;
                        row_deg_sp = row_degree_bound[frowList[i]];
                        row_sp = i;
                    }
                }
            }
            row_piv = row_sp;
        }

        if (Sym->strategy_used == PARU_STRATEGY_SYMMETRIC && chose_diag == 0)
        {
            int64_t pivcol = col1 + j + n1;      // S col index + n1
            int64_t pivrow = frowList[row_piv];  // S row index
            paru_diag_update(pivcol, pivrow, Work);
            PRLEVEL(1, ("%% symmetric matrix but the diag didn't picked for "
                        "row_piv=" LD "\n",
                        row_piv));
        }
        PRLEVEL(1, ("%% piv value= %2.4lf row_deg=" LD "\n", piv, row_deg_sp));
        PRLEVEL(1, ("%% piv value= %e \n", piv));
        // swap rows
        PRLEVEL(1, ("%% Swaping rows j=" LD ", row_piv=" LD "\n", j, row_piv));
        paru_swap_rows(F, frowList, m, n, j, row_piv);

#ifndef NDEBUG  // Printing the pivotal front
        PR = 1;
        if (row_piv != row_max) PRLEVEL(PR, ("%% \n"));
        PRLEVEL(PR, ("%% After Swaping\n"));
        PRLEVEL(PR, (" \n"));
        for (int64_t r = 0; r < row_end; r++)
        {
            PRLEVEL(PR, ("%% " LD "\t", frowList[r]));
            for (int64_t c = 0; c < num_col_panel; c++)
                PRLEVEL(PR, (" %2.5lf\t", F[c * m + r]));
            PRLEVEL(PR, ("\n"));
        }
#endif

        // dscal loop unroll is also possible

        if (j < row_end - 1)
        {
            PRLEVEL(1, ("%% dscal\n"));
// pragma omp taskloop simd default(none)
// shared(j, row_end, F, m, piv) if(row_end-j > 1024)
#pragma omp simd
            for (int64_t i = j + 1; i < row_end; i++)
            {
                F[j * m + i] /= piv;
            }
        }

        // dger
        /*               dgemm   A := alpha *x*y**T + A
         *
         *
         *                <----------fp------------------->
         *                        j1 current j2
         *                         ^  panel  ^
         *                         |         |
         *                         |   j     |
         *             F           |   ^     |
         *              \  ____..._|___|_____|__________...___
         * ^              |\      |         |                 |
         * |              | \     |<--panel | rest of         |
         * |              |  \    | width-> |  piv front      |
         * |              |___\...|_______ _|_________ ... ___|
         * |   ^    j1--> |       |\* *|    |                 |
         * | panel        |       |**\*|    |                 |
         * | width        |       |___|Pyyyy|                 |
         * |   v          |____...|___|xAAAA__________...____ |
         * |        j2--> |       |   |xAAAA|                 |
         * rowCount       |       |   |xAAAA|                 |
         * |              |       |   |xAAAA|                 |
         * |              |       |   |xAAAA|                 |
         * |              |       |row_end  |                 |
         * |              .       .         .                 .
         * |              .       .         .                 .
         * |              .       .         .                 .
         * v              |___....____________________..._____|
         *
         */

        if (j < j2 - 1)
        {
            double *X = F + j * m + j + 1;
            double alpha = -1.0;
            double *Y = F + j * m + j + m;
            double *A = F + j * m + j + m + 1;

#ifndef NDEBUG  // Printing dger input
            int64_t M = (int64_t)row_end - 1 - j;
            int64_t N = (int64_t)j2 - 1 - j;
            int64_t Incx = (int64_t)1;
            int64_t Incy = (int64_t)m;
            int64_t lda = (int64_t)m;
            int64_t PR = 1;
            PRLEVEL(PR, ("%% lda = " LD " ", lda));
            PRLEVEL(PR, ("%% M = " LD " ", M));
            PRLEVEL(PR, ("N = " LD " \n %%", N));
            PRLEVEL(PR, ("%% x= ( " LD ")", N));
            for (int64_t i = 0; i < M; i++) PRLEVEL(PR, (" %lf ", X[i]));
            PRLEVEL(PR, ("\n %% y= ( " LD ")", N));
            for (int64_t j = 0; j < N; j++) PRLEVEL(PR, (" %lf ", Y[j * m]));
            PRLEVEL(PR, ("\n"));

#endif
            bool blas_ok = true ;
            SUITESPARSE_BLAS_dger(row_end - 1 - j, j2 - 1 - j, &alpha, X, 1, Y,
                                  m, A, m, blas_ok);
            if (!blas_ok) return (false);
#ifdef COUNT_FLOPS
#pragma omp atomic update
            Num->flp_cnt_dger += (double)2 * M * N;
#ifndef NDEBUG
            PRLEVEL(PR, ("\n%% FlopCount Dger fac  " LD "  " LD " ", M, N));
            PRLEVEL(PR, ("cnt = %lf\n ", Num->flp_cnt_dger));
#endif
#endif
        }

#ifndef NDEBUG  // Printing the pivotal front
        int64_t PR = 1;
        PRLEVEL(PR, ("%% After dger\n"));
        for (int64_t r = j; r < row_end; r++)
        {
            PRLEVEL(PR, ("%% " LD "\t", frowList[r]));
            for (int64_t c = j; c < j2; c++)
                PRLEVEL(PR, (" %2.5lf\t", F[c * m + r]));
            PRLEVEL(PR, ("\n"));
        }
#endif
    }
    return (true) ;
}

//------------------------------------------------------------------------------
// paru_factorize_full_summed: factorize a frontal matrix
//------------------------------------------------------------------------------

ParU_Info paru_factorize_full_summed
(
    int64_t f,
    int64_t start_fac,
    std::vector<int64_t> &panel_row,
    std::set<int64_t> &stl_colSet,
    std::vector<int64_t> &pivotal_elements,
    paru_work *Work,
    const ParU_Symbolic Sym,
    ParU_Numeric Num
)
{

    // get Control
    int64_t panel_width = Work->panel_width ;

    DEBUGLEVEL(0);
    PARU_DEFINE_PRLEVEL;

    const int64_t *Super = Sym->Super;
    int64_t col1 = Super[f]; /* fornt F has columns col1:col2-1 */
    int64_t col2 = Super[f + 1];
    int64_t fp = col2 - col1; /* first fp columns are pivotal */

    ParU_Factors *LUs = Num->partial_LUs;
    int64_t rowCount = Num->frowCount[f];
    double *F = LUs[f].p;

    int64_t num_panels =
        (fp % panel_width == 0) ? fp / panel_width : fp / panel_width + 1;
    for (int64_t panel_num = 0; panel_num < num_panels; panel_num++)
    {
#ifndef NDEBUG  // Printing the pivotal front
        int64_t *frowList = Num->frowList[f];
        PRLEVEL(PR, ("%%Pivotal Front Before " LD "\n", panel_num));

        for (int64_t r = 0; r < rowCount; r++)
        {
            PRLEVEL(PR, ("%% " LD "\t", frowList[r]));
            for (int64_t c = 0; c < fp; c++)
            {
                PRLEVEL(PR, (" %2.5lf\t", F[c * rowCount + r]));
            }
            PRLEVEL(PR, ("\n"));
        }
#endif

        int64_t row_end = panel_row[panel_num];
        int64_t j1 = panel_num * panel_width;
        int64_t j2 = (panel_num + 1) * panel_width;
        // factorize current panel
        bool blas_ok = paru_panel_factorize(f, rowCount, fp,
            panel_num, row_end, Work, Sym, Num);
        if (!blas_ok) return (PARU_TOO_LARGE);
        // int64_t naft; //number of active frontal tasks
        // pragma omp atomic read
        // naft = Work->naft;
        // pragma omp parallel  proc_bind(close) if(naft == 1)
        // pragma omp single
        {
            // update row degree and dgeem can be done in parallel
            // pragma omp task default(none) mergeable
            // shared(Num, pivotal_elements, stl_colSet)
            // shared(panel_num, row_end, f, start_fac)

            if (Sym->Cm[f] != 0)
            {
                // if there is potential column left
                paru_update_rowDeg(panel_num, row_end, f, start_fac, stl_colSet,
                                   pivotal_elements, Work, Sym, Num);
            }

            /*               trsm
             *
             *        F = fully summed part of the pivotal front
             *           op( A ) * B = alpha*B
             *
             *                <----------fp------------------->
             *                        j1 current j2
             *    F                    ^  panel  ^
             *     \           ____..._|_________|__________...___
             * ^              |\      |         |                 |
             * |              | \     |<--panel | rest of         |
             * |              |  \    | width-> |  piv front      |
             * |              |___\...|_______ _|_________ ... ___|
             * |   ^    j1--> |       |\        |                 |
             * | panel        |       |**\ A    |   B(In out)     |
             * | width        |       |*L**\    |                 |
             * |   v          |____...|******\ _|_________...____ |
             * |        j2--> |       |         |                 |
             * rowCount       |       |         |                 |
             * |              .       .         .                 .
             * |              .       .         .                 .
             * |              .       .         .                 .
             * v              |___....____________________..._____|
             *
             */
            // pragma omp task  shared(F)
            // shared(panel_width, j1, j2, fp, f, rowCount)
            if (j2 < fp)  // if it is not the last
            {
                int64_t M = (int64_t)panel_width;
                int64_t N = (int64_t)fp - j2;
                double alpha = 1.0;
                double *A = F + j1 * rowCount + j1;
                int64_t lda = (int64_t)rowCount;
                double *B = F + j2 * rowCount + j1;
                int64_t ldb = (int64_t)rowCount;
#ifndef NDEBUG
                int64_t PR = 1;
                PRLEVEL(PR, ("%% M = " LD " N =  " LD " alpha = %f \n", M, N, alpha));
                PRLEVEL(PR, ("%% lda = " LD " ldb = " LD "\n", lda, ldb));
                PRLEVEL(PR, ("%% Pivotal Front Before Trsm: " LD " x " LD "\n", fp,
                             rowCount));
                for (int64_t r = 0; r < rowCount; r++)
                {
                    PRLEVEL(PR, ("%% " LD "\t", frowList[r]));
                    for (int64_t c = 0; c < fp; c++)
                        PRLEVEL(PR, (" %2.5lf\t", F[c * rowCount + r]));
                    PRLEVEL(PR, ("\n"));
                }

#endif
                blas_ok = paru_tasked_dtrsm(f, M, N, alpha, A, lda, B, ldb,
                    Work, Num);
                if (!blas_ok) return (PARU_TOO_LARGE);
#ifndef NDEBUG
                PRLEVEL(PR, ("%% Pivotal Front After Trsm: " LD " x " LD "\n %%", fp,
                             rowCount));
                for (int64_t r = 0; r < rowCount; r++)
                {
                    PRLEVEL(PR, ("%% " LD "\t", frowList[r]));
                    for (int64_t c = 0; c < fp; c++)
                        PRLEVEL(PR, (" %2.5lf\t", F[c * rowCount + r]));
                    PRLEVEL(PR, ("\n"));
                }
#endif
            }
        }  // end of parallel region; it doesn't show good performance

        /*               dgemm   C := alpha*op(A)*op(B) + beta*C
         *
         *        F = fully summed part of the pivotal front
         *
         *                <----------fp------------------->
         *                        j1 current j2
         *    F                    ^  panel  ^
         *     \           ____..._|_________|__________...___
         * ^              |\      |         |                 |
         * |              | \     |<--panel | rest of         |
         * |              |  \    | width-> |  piv front      |
         * |              |___\...|_______ _|_________ ... ___|
         * |   ^    j1--> |       |\        |**************** |
         * | panel        |       |  \      |****In***B****** |
         * | width        |       |    \    |**************** |
         * |   v          |____...|_______\___________...____ |
         * |        j2--> |       |******** |ccccccccccccccccc|
         * rowCount       |       |******** |ccccccccccccccccc|
         * |              .       .******** .ccccccCcccccccccc.
         * |              .       .***In*** .ccccOutcccccccccc.
         * |              .       .***A**** .ccccccccccccccccc.
         * |              |       |******** |ccccccccccccccccc|
         * |              |       |******** |ccccccccccccccccc|
         * |              |       |row_end  |                 |
         * |              |       |         |                 |
         * v              |___....|_______ _|__________...____|
         *
         */

        if (j2 < fp)
        {
            int64_t M = (int64_t)(row_end - j2);

            int64_t N = (int64_t)fp - j2;
            int64_t K = (int64_t)panel_width;
            // alpha = -1;
            double *A = F + j1 * rowCount + j2;
            int64_t lda = (int64_t)rowCount;
            double *B = F + j2 * rowCount + j1;
            int64_t ldb = (int64_t)rowCount;
            // double beta = 1;  // keep current values
            double *C = F + j2 * rowCount + j2;
            int64_t ldc = (int64_t)rowCount;
#ifndef NDEBUG
            int64_t PR = 1;
            PRLEVEL(PR, ("%% DGEMM "));
            PRLEVEL(PR, ("%% M = " LD " K =  " LD " N =  " LD " \n", M, K, N));
            PRLEVEL(PR, ("%% lda = " LD " ldb = " LD "\n", lda, ldb));
            PRLEVEL(PR, ("%% j2 =" LD " j1=" LD "\n", j2, j1));
            PRLEVEL(PR, ("\n %%"));
#endif
            blas_ok = paru_tasked_dgemm(f, M, N, K, A, lda, B, ldb, 1, C, ldc,
                                        Work, Num);
            if (!blas_ok) return (PARU_TOO_LARGE);
#ifdef COUNT_FLOPS
                //#pragma omp atomic
                // Num->flp_cnt_real_dgemm += (double)2 * M * N * K;
#ifndef NDEBUG
            PRLEVEL(PR, ("\n%% FlopCount Dgemm factorize  " LD "  " LD "  " LD " ", M, N, K));
            PRLEVEL(PR, (" " LD "  " LD "  " LD " \n", M, N, K));
#endif
#endif
        }

#ifndef NDEBUG
        if (j2 < fp)
        {
            PRLEVEL(PR, ("%% Pivotal Front After Dgemm: " LD " x " LD "\n %%", fp,
                         rowCount));
            for (int64_t r = 0; r < rowCount; r++)
            {
                PRLEVEL(PR, ("%% " LD "\t", frowList[r]));
                for (int64_t c = 0; c < fp; c++)
                    PRLEVEL(PR, (" %2.5lf\t", F[c * rowCount + r]));
                PRLEVEL(PR, ("\n"));
            }
        }
#endif
    }
    return PARU_SUCCESS;
}
