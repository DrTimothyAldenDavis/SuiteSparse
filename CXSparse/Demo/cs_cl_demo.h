#include "cs.h"
typedef struct problem_struct
{
    cs_cl *A ;
    cs_cl *C ;
    UF_long sym ;
    double _Complex *x ;
    double _Complex *b ;
    double _Complex *resid ;
} problem ;

problem *get_problem (FILE *f, double tol) ;
UF_long demo2 (problem *Prob) ;
UF_long demo3 (problem *Prob) ;
problem *free_problem (problem *Prob) ;
