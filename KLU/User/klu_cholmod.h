#include "klu.h"
#include "UFconfig.h"

int klu_cholmod (int n, int Ap [ ], int Ai [ ], int Perm [ ], klu_common *) ;

UF_long klu_l_cholmod (UF_long n, UF_long Ap [ ], UF_long Ai [ ],
    UF_long Perm [ ], klu_l_common *) ;

