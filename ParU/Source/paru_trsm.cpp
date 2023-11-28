////////////////////////////////////////////////////////////////////////////////
//////////////////////////  paru_trsm //////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

// ParU, Copyright (c) 2022, Mohsen Aznaveh and Timothy A. Davis,
// All Rights Reserved.
// SPDX-License-Identifier: GNU GPL 3.0

/*! @brief trsm wrapper
 *
 *       l11*u12=a12 and u12 is unkonwn  so we need this:
 *                op( A ) * X = alpha*B
 *               part(pF) * X = 1 * upart
 *               part(pF) * upart = 1 * upart
 *        SIDE = 'L' or 'l'
 *        UPLO = 'L' or 'l'; A is a lower triangular matrix.
 *        TRANSA = 'N' or 'n'   op( A ) = A.
 *        DIAG = 'U' or 'u'   A is assumed to be unit triangular.
 *        M     M specifies the number of rows of B.    (fp)
 *        N     N specifies the number of columns of B. (colCount)
 *        ALPHA, (alpha = 1.0)
 *        A (pF)
 *        LDA  leading dimension of A. (rowCount)
 *        B (upart)
 *        LDB  leading dimension of B.  (fp)
 *
 * @author Aznaveh
 */
#include "paru_internal.hpp"

int64_t paru_trsm(int64_t f, double *pF, double *uPart, int64_t fp, int64_t rowCount,
              int64_t colCount, paru_work *Work, ParU_Numeric *Num)
{
    DEBUGLEVEL(0);
    int64_t mB = (int64_t)fp;
    int64_t nB = (int64_t)colCount;
    double alpha = 1.0;
    int64_t lda = (int64_t)rowCount;
    int64_t ldb = (int64_t)fp;

#ifndef NDEBUG  // Printing the  U part
    PRLEVEL(1, ("TRSM (" LD "x" LD ") (" LD "x" LD ") \n", mB, mB, mB, nB));
    int64_t p = 1;
    PRLEVEL(p, ("mB=" LD " nB = " LD " alpha = %f \n", mB, nB, alpha));
    PRLEVEL(p, ("lda =" LD " ldb =" LD "\n", lda, ldb));
    PRLEVEL(p, ("(I)U Before Trsm: " LD " x " LD "\n", fp, colCount));
    for (int64_t i = 0; i < fp; i++)
    {
        for (int64_t j = 0; j < colCount; j++)
            PRLEVEL(p, (" %2.5lf\t", uPart[j * fp + i]));
        PRLEVEL(p, ("\n"));
    }
#endif

    int64_t blas_ok = paru_tasked_trsm(f, mB, nB, alpha, pF, 
            lda, uPart, ldb, Work, Num);

#ifndef NDEBUG  // Printing the  U part
    PRLEVEL(p, ("(I)U After Trsm: " LD " x " LD "\n", fp, colCount));
    for (int64_t i = 0; i < fp; i++)
    {
        for (int64_t j = 0; j < colCount; j++)
            PRLEVEL(p, (" %2.5lf\t", uPart[j * fp + i]));
        PRLEVEL(p, ("\n"));
    }
#endif

    return blas_ok;
}
