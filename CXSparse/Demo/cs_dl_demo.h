#include "cs.h"
typedef struct problem_struct
{
    cs_dl *A ;
    cs_dl *C ;
    UF_long sym ;
    double *x ;
    double *b ;
    double *resid ;
} problem ;

problem *get_problem (FILE *f, double tol) ;
UF_long demo2 (problem *Prob) ;
UF_long demo3 (problem *Prob) ;
problem *free_problem (problem *Prob) ;
