#include "cs.h"
typedef struct problem_struct
{
    cs_cl *A ;
    cs_cl *C ;
    UF_long sym ;
    cs_complex_t *x ;
    cs_complex_t *b ;
    cs_complex_t *resid ;
} problem ;

problem *get_problem (FILE *f, double tol) ;
UF_long demo2 (problem *Prob) ;
UF_long demo3 (problem *Prob) ;
problem *free_problem (problem *Prob) ;
