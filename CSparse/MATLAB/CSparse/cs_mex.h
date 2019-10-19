#include "cs.h"
#include "mex.h"
cs *cs_mex_get_sparse (cs *A, int square, int values, const mxArray *Amatlab) ;
mxArray *cs_mex_put_sparse (cs **A) ;
void cs_mex_check (int nel, int m, int n, int square, int sparse, int values,
    const mxArray *A) ;
int *cs_mex_get_int (int n, const mxArray *Imatlab, int *imax, int lo) ;
mxArray *cs_mex_put_int (int *p, int n, int offset, int do_free) ;
double *cs_mex_get_double (int n, const mxArray *X) ;
double *cs_mex_put_double (int n, const double *b, mxArray **X) ;
