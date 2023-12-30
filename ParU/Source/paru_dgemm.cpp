////////////////////////////////////////////////////////////////////////////////
//////////////////////////  paru_dgemm /////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////

// ParU, Copyright (c) 2022, Mohsen Aznaveh and Timothy A. Davis,
// All Rights Reserved.
// SPDX-License-Identifier: GNU GPL 3.0

/*! @brief      A wraper around dgemm for outer product.
 *
 *          This does the outer product
 *              C := alpha*op( A )*op( B ) + beta*C,
 *              el:= -1*part(pF) * uPart + zero
 *              TRANSA = 'N'   op(A) = A
 *              TRANSB = 'N'   op(B) = B
 *
 * @author Aznaveh
 */
#include "paru_internal.hpp"

int64_t paru_dgemm(int64_t f, double *pF, double *uPart, double *el, int64_t fp,
               int64_t rowCount, int64_t colCount, paru_work *Work, ParU_Numeric *Num)
{
    DEBUGLEVEL(0);
    PRLEVEL(1, ("%% rowCount =" LD "  ", rowCount));
    PRLEVEL(1, ("%% colCount =" LD "  ", colCount));
    PRLEVEL(1, ("%% fp =" LD "\n", fp));

    int64_t mA = (int64_t)(rowCount - fp);
    int64_t nB = (int64_t)colCount;
    int64_t nA = (int64_t)fp;

    PRLEVEL(1, ("%% mA = " LD " ", mA));
    PRLEVEL(1, ("%% nB = " LD " " , nB));
    PRLEVEL(1, ("%% nA = " LD "\n", nA));

#ifndef NDEBUG
    double *Ap = pF + fp;
    PRLEVEL(1, ("%% A =\n"));
    for (int64_t i = 0; i < mA; i++)
    {
        PRLEVEL(1, ("%% "));
        for (int64_t j = 0; j < nA; j++)
            PRLEVEL(1, ("%2.4lf\t", Ap[j * rowCount + i]));
        PRLEVEL(1, ("\n"));
    }

    int64_t mB = nA;
    double *Bp = uPart;
    PRLEVEL(1, ("%% B =\n"));
    for (int64_t i = 0; i < mB; i++)
    {
        PRLEVEL(1, ("%% "));
        for (int64_t j = 0; j < nB; j++) PRLEVEL(1, ("%2.4lf\t", Bp[j * mB + i]));
        PRLEVEL(1, ("\n"));
    }

    double *Cp = el;
    PRLEVEL(1, ("%%Before DGEMM C =\n"));
    for (int64_t i = 0; i < mA; i++)
    {
        PRLEVEL(1, ("%% "));
        for (int64_t j = 0; j < nB; j++)
        {
            PRLEVEL(1, ("%2.4lf\t", Cp[j * mA + i]));
        }
        PRLEVEL(1, ("\n"));
    }
#endif

    // alpha = -1;
    int64_t lda = (int64_t)rowCount;
    int64_t ldb = (int64_t)fp;
    int64_t ldc = (int64_t)(rowCount - fp);

    PRLEVEL(1, ("%% lda = " LD " ", lda));
    PRLEVEL(1, ("%% ldb = " LD " ", ldb));
    PRLEVEL(1, ("%% ldc = " LD "\n", ldc));

    // double beta = 0;  // U part is not initialized

    int64_t blas_ok = paru_tasked_dgemm(f, mA, nB, nA, pF + fp, 
            lda, uPart, ldb, 0, el, ldc, Work, Num);

#ifndef NDEBUG
    int64_t PR = 1;
    PRLEVEL(PR, ("%%After DGEMM C =\n"));
    for (int64_t i = 0; i < mA; i++)
    {
        PRLEVEL(PR, ("%% "));
        for (int64_t j = 0; j < nB; j++) PRLEVEL(PR, ("%2.4lf\t", Cp[j * mA + i]));
        PRLEVEL(PR, ("\n"));
    }
#endif

    return blas_ok;
}
