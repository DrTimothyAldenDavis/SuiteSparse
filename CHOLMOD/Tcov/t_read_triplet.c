//------------------------------------------------------------------------------
// CHOLMOD/Tcov/t_read_triplet: read a triplet matrix
//------------------------------------------------------------------------------

// CHOLMOD/Tcov Module.  Copyright (C) 2005-2023, Timothy A. Davis.
// All Rights Reserved.
// SPDX-License-Identifier: GPL-2.0+

//------------------------------------------------------------------------------

// Read a triplet matrix from a file.

cholmod_triplet *read_triplet
(
    FILE *f
)
{
    cholmod_triplet *T ;
    Real *Tx, *Tz ;
    long long x1, x2, x3, x4, x5 ;
    Int *Ti, *Tj ;
    Int n, j, k, nrow, ncol, nz, stype, arrowhead, tridiag_plus_denserow,
        xtype, is_complex ;
    char s [MAXLINE] ;

    //--------------------------------------------------------------------------
    // read in a triplet matrix from a file
    //--------------------------------------------------------------------------

    dot = 0 ;
    xtype = 0 ;
    if (fgets (s, MAXLINE, f) == NULL)
    {
        return (NULL) ;
    }

    // header line: nrow ncol nz stype (xtype-1)

    x1 = 0 ;
    x2 = 0 ;
    x3 = 0 ;
    x4 = 0 ;
    x5 = 0 ;
    k = sscanf (s, "%lld %lld %lld %lld %lld\n", &x1, &x2, &x3, &x4, &x5) ;
    nrow = x1 ;
    ncol = x2 ;
    nz = x3 ;
    stype = x4 ;
    xtype = x5 ;

    xtype++ ;
    is_complex = (xtype != CHOLMOD_REAL) ;

    printf ("read_triplet: nrow "ID" ncol "ID" nz "ID" stype "ID" xtype "ID"\n",
            nrow, ncol, nz, stype, xtype) ;

    arrowhead = FALSE ;
    tridiag_plus_denserow = FALSE ;

    n = MAX (nrow, ncol) ;
    if (stype == 2)
    {
        // ignore nz and the rest of the file, and create an arrowhead matrix
        arrowhead = TRUE ;
        nz = nrow + ncol + 1 ;
        stype = (nrow == ncol) ? (1) : (0) ;
    }
    else if (stype == 3)
    {
        tridiag_plus_denserow = TRUE ;
        nrow = n ;
        ncol = n ;
        nz = 4*n + 4 ;
        stype = 0 ;
    }

    T = CHOLMOD(allocate_triplet) (nrow, ncol, nz, stype,
            (is_complex ? CHOLMOD_ZOMPLEX : CHOLMOD_REAL) + DTYPE, cm) ;
    if (T == NULL)
    {
        ERROR (CHOLMOD_INVALID, "cannot create triplet matrix") ;
        return (NULL) ;
    }
    Ti = T->i ;
    Tj = T->j ;
    Tx = T->x ;
    Tz = T->z ;

    if (arrowhead)
    {
        for (k = 0 ; k < MIN (nrow,ncol) ; k++)
        {
            Ti [k] = k ;
            Tj [k] = k ;
            Tx [k] = nrow + xrand (1) ;                         // RAND
            if (is_complex)
            {
                Tz [k] = nrow + xrand (1) ;                     // RAND
            }
        }
        for (j = 0 ; j < ncol ; j++)
        {
            Ti [k] = 0 ;
            Tj [k] = j ;
            Tx [k] = - xrand (1) ;                              // RAND
            if (is_complex)
            {
                Tz [k] = - xrand (1) ;                          // RAND
            }
            k++ ;
        }
        T->nnz = k ;
    }
    else if (tridiag_plus_denserow)
    {
        // dense row, except for the last column
        for (k = 0 ; k < n-1 ; k++)
        {
            Ti [k] = 0 ;
            Tj [k] = k ;
            Tx [k] = xrand (1) ;                                // RAND
            if (is_complex)
            {
                Tz [k] = xrand (1) ;                            // RAND
            }
        }

        // diagonal
        for (j = 0 ; j < n ; j++)
        {
            Ti [k] = j ;
            Tj [k] = j ;
            Tx [k] = nrow + xrand (1) ;                         // RAND
            if (is_complex)
            {
                Tz [k] = nrow + xrand (1) ;                     // RAND
            }
            k++ ;
        }

        // superdiagonal
        for (j = 1 ; j < n ; j++)
        {
            Ti [k] = j-1 ;
            Tj [k] = j ;
            Tx [k] = xrand (1) ;                                // RAND
            if (is_complex)
            {
                Tz [k] = xrand (1) ;                            // RAND
            }
            k++ ;
        }

        // subdiagonal
        for (j = 0 ; j < n-1 ; j++)
        {
            Ti [k] = j+1 ;
            Tj [k] = j ;
            Tx [k] = xrand (1) ;                                // RAND
            if (is_complex)
            {
                Tz [k] = xrand (1) ;                            // RAND
            }
            k++ ;
        }

        // a few extra terms in the last column
        Ti [k] = MAX (0, n-3) ;
        Tj [k] = n-1 ;
        Tx [k] = xrand (1) ;                                    // RAND
        if (is_complex)
        {
            Tz [k] = xrand (1) ;                                // RAND
        }
        k++ ;

        Ti [k] = MAX (0, n-4) ;
        Tj [k] = n-1 ;
        Tx [k] = xrand (1) ;                                    // RAND
        if (is_complex)
        {
            Tz [k] = xrand (1) ;                                // RAND
        }
        k++ ;

        Ti [k] = MAX (0, n-5) ;
        Tj [k] = n-1 ;
        Tx [k] = xrand (1) ;                                    // RAND
        if (is_complex)
        {
            Tz [k] = xrand (1) ;                                // RAND
        }
        k++ ;

        T->nnz = k ;
    }
    else
    {
        if (is_complex)
        {
            for (k = 0 ; k < nz ; k++)
            {
                int64_t i, j ;
                double x, z ;
                if (fscanf (f, "%ld %ld %lg %lg\n", &i, &j, &x, &z)
                    == EOF)
                {
                    ERROR (CHOLMOD_INVALID, "Error reading triplet matrix\n") ;
                }
                Ti [k] = i ;
                Tj [k] = j ;
                Tx [k] = x ;
                Tz [k] = z ;
            }
        }
        else
        {
            for (k = 0 ; k < nz ; k++)
            {
                int64_t i, j ;
                double x ;
                if (fscanf (f, "%ld %ld %lg\n", &i, &j, &x) == EOF)
                {
                    ERROR (CHOLMOD_INVALID, "Error reading triplet matrix\n") ;
                }
                Ti [k] = i ;
                Tj [k] = j ;
                Tx [k] = x ;
            }
        }
        T->nnz = nz ;
    }

    CHOLMOD(triplet_xtype) (xtype + DTYPE, T, cm) ;

    //--------------------------------------------------------------------------
    // print the triplet matrix
    //--------------------------------------------------------------------------

    const int psave = cm->print ;
    cm->print = 4 ;
    CHOLMOD(print_triplet) (T, "T input", cm) ;
    cm->print = psave ;
    printf ("\n\n======================================================\n"
            "Test matrix: "ID"-by-"ID" with "ID" entries, stype: "ID"\n",
            (Int) T->nrow, (Int) T->ncol, (Int) T->nnz, (Int) T->stype) ;
    return (T) ;
}

