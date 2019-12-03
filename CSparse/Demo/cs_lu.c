#include "cs.h"
#include "mmio.h"

// a test 3x3 matrix: A = LU
// | 2  0  3 |   | 1               || 2  0    3   |
// | 3  6  0 | = | 1.5   1         ||    6   -4.5 |
// | 0  1  4 |   | 0     1.667   1 ||         4.75|

csi    g_p[] = {0,  0,  0,  1,  1,  1,  2,  2,  2};   // col idx
csi    g_i[] = {0,  1,  2,  0,  1,  2,  0,  1,  2};   // row idx
double g_x[] = {2,  3,  0,  0,  6,  1,  3,  0,  4};

//const char g_path[] = "../../KLU/Matrix/impcol_a.mtx"; // pivoting failed
//const char g_path[] = "./fpga_dcop_01.mtx";
//const char g_path[] = "./rajat19.mtx"; // https://www.cise.ufl.edu/research/sparse/matrices/Rajat/rajat19.html
const char g_path[] = "../../SPQR/Matrix/lfat5b.mtx";

// read mtx format, uncompressed
cs* read_mm(const char* path);
int dump_cs(const cs* M, const char* text);

int main (void)
{
#if (1)
    // triple format
    cs* T = read_mm(g_path);
    if (!T) {
        printf("read_mm() failed\n");
        exit(1);
    }
#else
    cs* T = (cs*)malloc(sizeof(cs));
    T->nzmax = 9;
    T->nz = 9;
    T->m = 3;
    T->n = 3;
    T->p = (csi*)malloc(sizeof(csi) * 9);
    T->i = (csi*)malloc(sizeof(csi) * 9);
    T->x = (double*)malloc(sizeof(double) * 9);
    for (int i = 0; i < 9; ++i) {
        T->p[i] = g_p[i];
        T->i[i] = g_i[i];
        T->x[i] = g_x[i];
    }
#endif

    // compressed-column format
    cs* A = cs_compress (T) ;

    // symbolic LU ordering and analysis
    int order = 0; // 0: natural ordering, 1: A+A'
    int is_qr = 0; // lu decomp
    css* sym = cs_sqr(order, A, is_qr);

    if (!sym) {
        printf("ERROR: cs_sqr() failed\n");
        exit(1);
    }

    if (sym->pinv) {
        printf("sym->pinv != NULL\n");
    } else {
        printf("sym->pinv = NULL\n");
    }

    if (sym->q) {
        printf("sym->q[] = [");
        for (int i = 0; i < A->n; ++i) {
            printf("%ld ", sym->q[i]);
        }
        printf("]\n");
    } else {
        printf("sym->q = NULL\n");
    }

    if (sym->parent) {
        printf("sym->parent != NULL\n");
    } else {
        printf("sym->parent = NULL\n");
    }

    if (sym->cp) {
        printf("sym->cp != NULL\n");
    } else {
        printf("sym->cp = NULL\n");
    }

    if (sym->leftmost) {
        printf("sym->leftmost != NULL\n");
    } else {
        printf("sym->leftmost = NULL\n");
    }

    printf("sym->m2=%ld\n", sym->m2);
    printf("sym->lnz=%lf\nsym->unz=%lf\n", sym->lnz, sym->unz);

    // LU factor
    double tol = 0.;
    csn* num = cs_lu(A, sym, tol); // sym can't be NULL

    if (!num) {
        printf("ERROR: cs_lu() failed\n");
        exit(1);
    }

    // L/U
    int brief = 0;
    dump_cs(num->L, "L");
    cs_print(num->L, brief);
    dump_cs(num->U, "U");
    cs_print(num->U, brief);

    if (num->pinv) {
        printf("num->pinv[]=[");
        for (int i = 0; i < A->m; ++ i) printf("%ld, ", num->pinv[i]);
        printf("]\n");
    } else {
        printf("num->pinv = NULL\n");
    }

    // tear down
    cs_nfree(num);
    cs_sfree(sym);
    cs_spfree (A) ;
    cs_spfree (T) ;

    return 0;
}

// read mtx format, uncompressed
// cf: https://math.nist.gov/MatrixMarket/mmio/c/example_read.c
cs* read_mm(const char* path)
{
    MM_typecode matcode;
    FILE *f;
    int M, N, nz;
    csi i, *I, *J;
    double *val;
    int tmp;

    if ((f = fopen(path, "r")) == NULL) {
        printf("file %s not founc!", path);
        return NULL;
    }

    if (mm_read_banner(f, &matcode) != 0) {
        printf("Could not process Matrix Market banner.\n");
        return NULL;
    }


    /*  This is how one can screen matrix types if their application */
    /*  only supports a subset of the Matrix Market data types.      */

    if (mm_is_complex(matcode) && mm_is_matrix(matcode) && mm_is_sparse(matcode)) {
        printf("Sorry, this application does not support ");
        printf("Market Market type: [%s]\n", mm_typecode_to_str(matcode));
        return NULL;
    }

    /* find out size of sparse matrix .... */

    if (0 != mm_read_mtx_crd_size(f, &M, &N, &nz)) {
        printf("Sorry, mm_read_mtx_crd_size() failed!");
        return NULL;
    }

    printf("(M, N, nz)=(%d, %d, %d)\n", M, N, nz);

    /* reseve memory for matrices */
    I = (csi*) malloc(nz * sizeof(csi));
    J = (csi*) malloc(nz * sizeof(csi));
    val = (double *) malloc(nz * sizeof(double));


    /* NOTE: when reading in doubles, ANSI C requires the use of the "l"  */
    /*   specifier as in "%lg", "%lf", "%le", otherwise errors will occur */
    /*  (ANSI C X3.159-1989, Sec. 4.9.6.2, p. 136 lines 13-15)            */

    for (i=0; i<nz; i++)
    {
        tmp = fscanf(f, "%ld %ld %lg\n", &I[i], &J[i], &val[i]);
        I[i]--;  /* adjust from 1-based to 0-based */
        J[i]--;
    }

    fclose(f);

    cs* A = (cs*)malloc(sizeof(cs));
    A->m = M;
    A->n = N;
    A->nzmax = nz;
    A->nz = nz;  // not compressed
    A->p = J;
    A->i = I;
    A->x = val;


    return A;
}

int dump_cs(const cs* M, const char* text)
{
    printf("\n%s: (m, n, nzmax, nz) = (%ld, %ld, %ld, %ld)\n", text, M->m, M->n, M->nzmax, M->nz);
    if (M->nz != - 1) {
        printf("p[]=[");
        for (int i = 0; i < M->n; ++ i) printf("%ld, ", M->p[i]);
        printf("]  <- col start indices\n");

        printf("i[]=[");
        for (int i = 0; i < M->nz; ++ i) printf("%ld, ", M->i[i]);
        printf("] <- row indices\n");

        printf("x[]=[");
        for (int i = 0; i < M->nz; ++ i) printf("%lf, ", M->x[i]);
        printf("]\n");
    } else {
        printf("p[]=[");
        for (int i = 0; i < M->n; ++ i) printf("%ld, ", M->p[i]);
        printf("] <- col start indices\n");

        printf("i[]=[");
        for (int i = 0; i < M->nzmax; ++ i) printf("%ld, ", M->i[i]);
        printf("] <- row indices\n");

        printf("x[]=[");
        for (int i = 0; i < M->nzmax; ++ i) printf("%lf, ", M->x[i]);
        printf("]\n");
    }

    return 0;
}
