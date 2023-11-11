////////////////////////////////////////////////////////////////////////////////
//////////////////////////  paru_write /////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

// ParU, Copyright (c) 2022, Mohsen Aznaveh and Timothy A. Davis,
// All Rights Reserved.
// SPDX-License-Identifier: GNU GPL 3.0

/*! @brief Writing the results into a file
 *    it must be called after the results are computed
 *  @author Aznaveh
 */

#include "paru_internal.hpp"
void paru_write(int scale, char *id, paru_work *Work, ParU_Numeric *Num)
{
    DEBUGLEVEL(0);
    PRLEVEL(1, ("%% Start Writing\n"));
    ParU_Symbolic *Sym = Work->Sym;
    int64_t nf = Sym->nf;

    int64_t m = Sym->m;
    int64_t n = Sym->n;
    int64_t n1 = Sym->n1;  // row+col singletons

    int64_t *Qfill = Sym->Qfill;

    ParU_Factors *LUs = Num->partial_LUs;
    ParU_Factors *Us = Num->partial_Us;
    int64_t *Super = Sym->Super;

    char default_name[] = "0";
    char *name;
    if (id)
        name = id;
    else
        name = default_name;

    char dpath[] = "../Demo/Res/";

    //-------------------- writing column permutation to a file
    {
        FILE *colfptr;

        char fname[100] = "";
        strcat(fname, dpath);
        strcat(fname, name);
        strcat(fname, "_col.txt");
        colfptr = (fopen(fname, "w"));

        if (colfptr == NULL)
        {
            printf("Error in making %s to write the results!\n", fname);
            return;
        }
        fprintf(colfptr, "%%cols\n");

        for (int64_t col = 0; col < n; col++)
        {                                      // for each column of A(:,Qfill)
            int64_t j = Qfill ? Qfill[col] : col;  // col of S is column j of A
            fprintf(colfptr, LD "\n", j);
        }

        fclose(colfptr);
        PRLEVEL(1, ("%% column permutaion DONE\n"));
    }
    //--------------------

    //--------------------computing and  writing row permutation to a file

    // some working memory that is freed in this function
    int64_t *oldRofS = NULL;
    int64_t *newRofS = NULL;
    int64_t *Pinit = Sym->Pinit;

    oldRofS = static_cast<int64_t*>(paru_alloc(m, sizeof(int64_t)));  // S -> LU P
    newRofS = static_cast<int64_t*>(paru_alloc(m, sizeof(int64_t)));  // Pinv of S

    if (oldRofS == NULL || newRofS == NULL)
    {
        printf("memory problem for writing into files\n");
        paru_free(m, sizeof(int64_t), oldRofS);
        paru_free(m, sizeof(int64_t), newRofS);
        return;
    }

    //   I have this transition   A ---> S ---> LU
    //   Mostly in the code I have rows of S.
    //   It is why I compute oldRofS and newRofS
    //              oldRofS
    //             -------->
    //           S           LU
    //             <-------
    //              newRofS
    //

    {
        FILE *rowfptr;
        char fname[100] = "";
        strcat(fname, dpath);
        strcat(fname, name);
        strcat(fname, "_row.txt");
        rowfptr = (fopen(fname, "w"));

        if (rowfptr == NULL)
        {
            printf("Error in opening a file");
            return;
        }
        fprintf(rowfptr, "%%rows\n");

        int64_t ip = 0;  // number of rows seen so far
        for (int64_t k = 0; k < n1; k++)
            // first singletons
            fprintf(rowfptr, LD "\n", Pinit[k]);

        for (int64_t f = 0; f < nf; f++)
        {  // rows for each front
            int64_t col1 = Super[f];
            int64_t col2 = Super[f + 1];
            int64_t fp = col2 - col1;
            int64_t *frowList = Num->frowList[f];

            for (int64_t k = 0; k < fp; k++)
            {
                oldRofS[ip++] = frowList[k];  // computing permutation for S
                // P[k] = i
                fprintf(rowfptr, LD "\n", Pinit[frowList[k]]);
            }
        }
        fclose(rowfptr);
        PRLEVEL(1, ("%% row permutaion DONE\n"));
    }
    //--------------------

    //-------- computing the direct permutation of S
    for (int64_t k = 0; k < m - n1; k++)
        // Inv permutation for S Pinv[i] = k;
        newRofS[oldRofS[k]] = k;

    //--------------------

    //-------------------- writing row scales to a file
    if (scale)
    {
        double *Rs = Num->Rs;
        FILE *scalefptr;
        char fname[100] = "";
        strcat(fname, dpath);
        strcat(fname, name);
        strcat(fname, "_scale.txt");
        scalefptr = (fopen(fname, "w"));

        if (scalefptr == NULL)
        {
            printf("Error in opening a file");
            return;
        }
        for (int64_t row = 0; row < m; row++)
            fprintf(scalefptr, "%.17g\n", Rs[row]);
        fclose(scalefptr);
    }
    //--------------------

    //-------------------- writing info to a file
    {
        FILE *infofptr;
        char fname[100] = "";
        strcat(fname, dpath);
        strcat(fname, name);
        strcat(fname, "_info.txt");
        infofptr = (fopen(fname, "w"));

        if (infofptr == NULL)
        {
            printf("Error in opening a file");
            return;
        }
        // I don't use umf_time inside a user visible DS anymore
        // maybe I can use another DS for internal use if I need it in future
        //fprintf(infofptr, "%.17g\n", Num->my_time + Sym->my_time);
        // fprintf(infofptr, "%.17g\n", Num->umf_time);

#ifdef COUNT_FLOPS
        fprintf(infofptr, "%.17g\n", Work->flp_cnt_dgemm);
        fprintf(infofptr, "%.17g\n", Work->flp_cnt_trsm);
        fprintf(infofptr, "%.17g\n", Work->flp_cnt_dger);
        fprintf(infofptr, "%.17g\n", Work->flp_cnt_real_dgemm);
#endif
        fclose(infofptr);
    }
    //--------------------

    //-------------------- writing results to a file
    FILE *LUfptr;
    char fname[100] = "";
    strcat(fname, dpath);
    strcat(fname, name);
    strcat(fname, "_LU.txt");
    LUfptr = (fopen(fname, "w"));

    if (LUfptr == NULL)
    {
        printf("Error in opening a file");
        return;
    }

    // computing nnz of factorized S
    int64_t nnz = 0;
    for (int64_t f = 0; f < nf; f++)
    {
        int64_t colCount = Num->fcolCount[f];
        int64_t rowCount = Num->frowCount[f];
        int64_t col1 = Super[f];
        int64_t col2 = Super[f + 1];
        int64_t fp = col2 - col1;
        // nnz += fp * (rowCount + colCount);

        double *pivotalFront = LUs[f].p;
        double *uPart = Us[f].p;
        for (int64_t j = col1; j < col2; j++)
        {
            for (int64_t i = 0; i < rowCount; i++)
            {
                if (pivotalFront[(j - col1) * rowCount + i] != 0.0) nnz++;
            }
        }

        for (int64_t j = 0; j < colCount; j++)
            for (int64_t i = 0; i < fp; i++)
            {
                {
                    if (uPart[fp * j + i] != 0.0) nnz++;
                }
            }
    }
    nnz += Sym->anz - Sym->snz;  // adding singletons

    fprintf(LUfptr, "%%%%MatrixMarket matrix coordinate real general\n");
    fprintf(LUfptr, "%%-----------produced by ParU ---------------\n");
    fprintf(LUfptr, LD "  " LD " " LD "\n", m, n, nnz);

    // writing the singletons
    // (I haven't add that part yet)

    // writing the L and U factors
    for (int64_t f = 0; f < nf; f++)
    {
        int64_t colCount = Num->fcolCount[f];
        int64_t *fcolList = Num->fcolList[f];
        int64_t rowCount = Num->frowCount[f];
        int64_t *frowList = Num->frowList[f];
        int64_t col1 = Super[f];
        int64_t col2 = Super[f + 1];
        int64_t fp = col2 - col1;

        // Printing LU part
        double *pivotalFront = LUs[f].p;
        PRLEVEL(1, ("%% pivotalFront =%p \n", pivotalFront));
        for (int64_t j = col1; j < col2; j++)
            for (int64_t i = 0; i < rowCount; i++)
            {
                if (pivotalFront[(j - col1) * rowCount + i] != 0.0)
                    fprintf(LUfptr, LD "  " LD " %.17g\n",
                            newRofS[frowList[i]] + n1 + 1, j + n1 + 1,
                            pivotalFront[(j - col1) * rowCount + i]);
            }

#ifndef NDEBUG  // Printing the pivotal front
        int64_t p = 1;
        PRLEVEL(p, ("\n%%Inside paru_write Luf{" LD "}= [", f + 1));
        for (int64_t r = 0; r < rowCount; r++)
        {
            PRLEVEL(p, (" "));
            for (int64_t c = col1; c < col2; c++)
                PRLEVEL(p,
                        (" %.17g ", pivotalFront[(c - col1) * rowCount + r]));
            PRLEVEL(p, (";\n%% "));
        }
        PRLEVEL(p, (";]\n"));
#endif

        // Printing U part
        double *uPart = Us[f].p;
        for (int64_t j = 0; j < colCount; j++)
            for (int64_t i = 0; i < fp; i++)
            {
                if (uPart[fp * j + i] != 0.0)
                    fprintf(LUfptr, LD "  " LD " %.17g\n",
                            newRofS[frowList[i]] + n1 + 1, fcolList[j] + n1 + 1,
                            uPart[fp * j + i]);
            }
#ifndef NDEBUG  // Printing the  U part
        p = 1;
        PRLEVEL(p, ("\n"));
        PRLEVEL(p, ("%% fp = " LD ", colCount = " LD "\n", fp, colCount));
        if (colCount != 0)
        {
            for (int64_t i = 0; i < fp; i++)
            {
                for (int64_t j = 0; j < colCount; j++)
                    PRLEVEL(p, (" %2.5lf\t", uPart[j * fp + i]));
                PRLEVEL(p, (";\n  %% "));
            }

            PRLEVEL(p, ("\n"));
        }
#endif
    }

    fclose(LUfptr);

    paru_free(m, sizeof(int64_t), oldRofS);
    paru_free(m, sizeof(int64_t), newRofS);
}
