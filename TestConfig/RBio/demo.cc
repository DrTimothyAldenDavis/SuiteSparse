//------------------------------------------------------------------------------
// SuiteSparse/TestConfig/RBio/demo.cc
//------------------------------------------------------------------------------

// Copyright (c) 2024, Timothy A. Davis, All Rights Reserved.
// SPDX-License-Identifier: BSD-3-clause

//------------------------------------------------------------------------------

// TestConfig program

#include <iostream>

#include "RBio.h"

int main (void)
{
    #define N 2
    #define NNZ 4
    int64_t n = N;
    int64_t nzmax = NNZ;
    int64_t Ap[N+1];
    int64_t Ai[NNZ];
    double Ax[NNZ];
    Ap [0] = 0;
    Ap [1] = 2;
    Ap [2] = NNZ;
    Ai [0] = 0;
    Ai [1] = 1;
    Ai [2] = 0;
    Ai [3] = 1;
    Ax [0] = 11.0;
    Ax [1] = 21.0;
    Ax [2] = 12.0;
    Ax [3] = 22.0;

    char mtype [4];
    std::string key {"simple"};
    std::string title {"2-by-2 matrix"};
    mtype [0] = '\0';
    int64_t njumbled, nzeros;
    int result = RBok (n, n, NNZ, Ap, Ai, Ax, nullptr, nullptr, 0, &njumbled, &nzeros);
    std::cout << "njumbled " << njumbled << ", nzeros " << nzeros << std::endl;
    std::string filename {"temp.rb"};
    result = RBwrite (filename.c_str (), title.c_str (), key.c_str (), n, n,
        Ap, Ai, Ax, nullptr, nullptr, nullptr, 0, mtype);
    std::cout << "result " << result << std::endl;
    std::cout << "mtype: " << mtype << std::endl;

    // dump out the file
    FILE *f = fopen ("temp.rb", "r");
    int c;
    while (1)
    {
        c = fgetc (f);
        if (c == EOF)
            break;
        fputc (c, stdout);
    }
    fclose (f) ;

    return 0;
}
