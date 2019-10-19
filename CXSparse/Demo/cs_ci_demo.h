#include "cs.h"
typedef struct problem_struct
{
    cs_ci *A ;
    cs_ci *C ;
    int sym ;
    double _Complex *x ;
    double _Complex *b ;
    double _Complex *resid ;
} problem ;

problem *get_problem (FILE *f, double tol) ;
int demo2 (problem *Prob) ;
int demo3 (problem *Prob) ;
problem *free_problem (problem *Prob) ;
