/* klu_simple: a simple KLU demo; solution is x = (1,2,3,4,5) */

#include <stdio.h>
#include "klu.h"

int    n = 5 ;
int    Ap [ ] = {0, 2, 5, 9, 10, 12} ;
int    Ai [ ] = { 0,  1,  0,   2,  4,  1,  2,  3,   4,  2,  1,  4} ;
double Ax [ ] = {2., 3., 3., -1., 4., 4., -3., 1., 2., 2., 6., 1.} ;
double b [ ] = {8., 45., -3., 3., 19.} ;

// dump int array
void dump_array(int* p, int sz, const char* name)
{
    printf("%s=[", name);
    for (int i = 0; i < sz; ++ i) printf("%d ", p[i]);
    printf("]\n");
}

// dump size_t array
void dump_array_s(size_t* p, int sz, const char* name)
{
    printf("%s=[", name);
    for (int i = 0; i < sz; ++ i) printf("%ld ", p[i]);
    printf("]\n");
}

// dump double array
void dump_array_d(double* p, int sz, const char* name)
{
    printf("%s=[", name);
    for (int i = 0; i < sz; ++ i) printf("%g ", p[i]);
    printf("]\n");
}

int dump_symbolic(klu_symbolic* sym)
{
    printf("\nsymbolic dump:\n");

    printf("n=%d, nz=%d, nzoff=%d, nblocks=%d, maxblock=%d, ordering=%d, do_btf=%d\n",
            sym->n, sym->nz, sym->nzoff, sym->nblocks, sym->maxblock, sym->ordering, sym->do_btf);
    if (sym->do_btf)
        printf("structural_rank=%d: %s\n",
                sym->structural_rank, (sym->structural_rank==sym->n)? "full": "singular");

    dump_array(sym->P, sym->n, "P");
    dump_array(sym->Q, sym->n, "Q");
    dump_array(sym->R, sym->nblocks+1, "R");
    dump_array_d(sym->Lnz, sym->nblocks, "Lnz");

    return 0;
}

/*
 * LUbx[i] is orgainized as below (take `double` as the data type for each entry) :
 *
 * void** LUbx -->+---------+          double[]
 *                | block 0 |-----> +============+ <--- Xip, Xlen
 *                +---------+       |   chunk 0  |
 *                | block 1 |       |   indices  |
 *                +---------+       |~~~~~~~~~~~~|
 *                | ...     |       |   chunk 0  |
 *                +---------+       |   entries  |
 *                                  |            |
 *                                  |            |
 *                                  +============+ <--- Xip, Xlen
 *                                  |   chunck 1 |
 *                                  |   indices  |
 *                                  |~~~~~~~~~~~~|
 *                                  |   chunck 1 |
 *                                  |   entries  |
 *                                  |            |
 *                                  |            |
 *                                  +============+ <-- Xip, Xlen
 *                                  |            |
 *                                  z            z
 *                                  |  ...       |
 *
 * Basically, each diagonal block (after LU decomp) contains multiple columns,
 * the indices and values(entries) of L and U (L/U is called X as the difference
 * between L and U is not visible at `LUbx` level), excluding the diagonal of U,
 * of all columns in the block are stored in the single double array `LUbx[blk_idx]`.
 *
 * LUbx[i] is piece-wisely combined by all L/U for the block, where the memory *chunk*
 * of each X (be it L or U) is continous. The smallest allocation unit for each chunk is
 * double, so as for the the whole LUbx[i].
 *
 * For each chunk, the indices is stored first, followed by the values. As the
 * index type (int) is of smaller size of value type (double), the area for storing
 * indices may be padded to be multiple of double size (cf `UNITS` macro in `klu_version.h`).
 *
 * The index values are local to the block, started from 0.
 *
 * The diagonals of L and U are not stored in `LUbx[]`:
 *   - L is unit triangular, so the 1s are implicitly known;
 *   - The diagonal of U for the *whole* matrix is stored in `Udiag`.
 */
int dump_lubx(double* LU, // LU array for a block
              int Xip,    // offset for the chunk
              int Xlen,   // # of entries for the chunk
              const char* name)
{
    int* idx = (int*)(LU + Xip);
    int units = (Xlen * sizeof(int) + sizeof(double) - 1) / sizeof(double);
    double* value = LU + Xip + units;

    //printf("%s: ", name);
    for (int i = 0; i < Xlen; ++ i) {
        printf("(%d, %g), ", idx[i], value[i]);
    }
    printf("\n");
    return 0;
}

int dump_numeric(klu_symbolic* sym, klu_numeric* num)
{
    printf("\nnumeric dump:\n");
    printf("lnz=%d, unz=%d, max_lnz_block=%d, max_unz_block=%d, worksize=%ldB, nzoff=%d\n",
            num->lnz, num->unz, num->max_lnz_block, num->max_unz_block, num->worksize, num->nzoff);

    dump_array(num->Pnum, num->n, "Pnum");
    dump_array(num->Pinv, num->n, "Pinv");
    dump_array(num->Lip, num->n, "Lip");
    dump_array(num->Uip, num->n, "Uip");
    dump_array(num->Llen, num->n, "Llen");
    dump_array(num->Ulen, num->n, "Ulen");
    if (num->Rs)
        dump_array_d(num->Rs, num->n, "Rs");
    else
        printf("no row-scaling\n");
    dump_array_s(num->LUsize, num->nblocks, "LUsize");
    dump_array_d(num->Udiag, num->n, "Udiag");

    // LU for each block
    for (int i = 0; i < num->nblocks; ++ i) {
        int k1 = sym->R[i];   // start row/col idx
        int k2 = sym->R[i+1]; // start row/col idx of the next block
        int nk = k2 - k1 ;    // dim of the current block

        printf("\nblock %d (dim=%d, LUsize=%ld):\n", i, nk, num->LUsize[i]);
        if (num->LUsize[i] > 0) {
            double* LU = (double *) num->LUbx[i];
            for (int k = k1; k < k2; ++ k) {
                int Lip = num->Lip[k];
                int Uip = num->Uip[k];
                int Llen = num->Llen[k];
                int Ulen = num->Ulen[k];
                printf("  col[%d]: \n", k);
                if (Llen > 0) {
                    printf("    Llen=%d, Lip=%d: ", Llen, Lip);
                    dump_lubx(LU, Lip, Llen, "L");
                }
                if (Ulen > 0) {
                    printf("    Ulen=%d, Uip=%d: ", Ulen, Uip);
                    dump_lubx(LU, Uip, Ulen, "U");
                }
            }
        }
    }

    // off-diagonal: note that within a column, the row indices are not sorted
    dump_array(num->Offp, num->n+1, "Offp");
    dump_array(num->Offi, num->nzoff, "Offi");
    dump_array_d(num->Offx, num->nzoff, "Offx");

    return 0;
}


int main (void)
{
    klu_symbolic *Symbolic ;
    klu_numeric *Numeric ;
    klu_common Common ;
    int i ;

    klu_defaults (&Common) ;
    Common.scale = 0; // no row scaling

    Symbolic = klu_analyze (n, Ap, Ai, &Common) ;
    dump_symbolic(Symbolic);

    Numeric = klu_factor (Ap, Ai, Ax, Symbolic, &Common) ;
    dump_numeric(Symbolic, Numeric);

    klu_solve (Symbolic, Numeric, 5, 1, b, &Common) ;

    printf("\nsolution dump:\n");
    for (i = 0 ; i < n ; i++) printf ("x[%d]=%g\n", i, b [i]) ;


    klu_free_symbolic (&Symbolic, &Common) ;
    klu_free_numeric (&Numeric, &Common) ;

    return (0) ;
}

