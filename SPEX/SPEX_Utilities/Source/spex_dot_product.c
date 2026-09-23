//------------------------------------------------------------------------------
// SPEX_Utilities/spex_dot_product: Get dot product of two column vectors
//------------------------------------------------------------------------------

// SPEX_QR: (c) 2020-2023, Lorena Mejia Domenzain, Christopher Lourenco,
// Timothy A. Davis, and Erick Moreno-Centeno.
// All Rights Reserved.
// SPDX-License-Identifier: GPL-2.0-or-later or LGPL-3.0-or-later

//------------------------------------------------------------------------------

#include "spex_util_internal.h"

SPEX_info spex_dot_product
(
    //Output
    mpz_t prod,
    //Input
    const SPEX_matrix A,
    const int64_t k,             //column number for A A(:,k)
    const SPEX_matrix B,
    const int64_t j,             //column number for B B(:,j)
    const SPEX_options option  
)
{
    SPEX_info info;
    //int64_t pA, pB, iA, iB;
    int sgnA, sgnB;
    int64_t p,q;
    SPEX_MPZ_INIT(prod);
    SPEX_MPZ_SET_UI(prod,0);

    ASSERT(A->m==B->m);

    /*for(pA=A->p[k]; pA < A->p[k+1]; pA++) //goes through every element of col j of A 
    {
        iA=A->i[pA]; //A(Ai,j) is non zero
        for(pB=B->p[j]; pB < B->p[j+1]; pB++ )//goes through every element of col j of B
        {
            iB=B->i[pB];//B(Bi,j) is non zero
            if(iA==iB)
            {
                //prod=prod+A->x.mpz[p]*B->x.mpz[p]
                SPEX_MPZ_ADDMUL(prod,A->x.mpz[pA],B->x.mpz[pB]);
            }
        }
    }*/
    
    p=A->p[k];
    q=B->p[j];
    
    while(p < A->p[k+1] && q < B->p[j+1])
    {
        if(A->i[p] < B->i[q])
        {
            p++;
        }
        else if(A->i[p] > B->i[q])
        {
            q++;
        }
        else
        {
            SPEX_MPZ_SGN(&sgnA, A->x.mpz[p]);
            SPEX_MPZ_SGN(&sgnB, B->x.mpz[q]);
            if(sgnA!=0 && sgnB!=0)
            {
                //prod=prod+A->x.mpz[p]*B->x.mpz[p]
                SPEX_MPZ_ADDMUL(prod,A->x.mpz[p],B->x.mpz[q]);
            }
            p++;
            q++;
        }
        
    }

    return SPEX_OK;
}
