#include <stdio.h>
#include "klu.h"
#include "assert.h"
#include <string.h>

#undef NPRINT
#define NPRINT
#ifdef NPRINT
#define PRINTF(x) 
#else
#define PRINTF(x) printf x
#endif

static void read_options (klu_common *Common)
{
    char line [80] ;
    FILE *input ;
    
    if (!(input = fopen ("options", "r")))
	fprintf (stderr, "cannot open file %s\n", "options") ;
    fgets (line, 80, input) ;
    assert (strstr (line, "options")) ;
    fgets (line, 80, input) ;
    klu_defaults (Common) ;
    if (strstr (line, "default") != 0)
    {
	;
    }
    else
    {
	Common->tol = atof (line) ;
	fgets (line, 80, input) ;
	Common->growth = atof (line) ;
	fgets (line, 80, input) ;
	Common->initmem_amd = atof (line) ;
	fgets (line, 80, input) ;
	Common->initmem = atof (line) ;
	fgets (line, 80, input) ;
	Common->btf = atoi (line) ;
	fgets (line, 80, input) ;
	Common->ordering = atoi (line) ;
	fgets (line, 80, input) ;
	Common->scale = atoi (line) ;
	fgets (line, 80, input) ;
	Common->halt_if_singular = atoi (line) ;
    }
    fclose (input) ;
}

static void open (FILE **input)
{
    if (!(*input = fopen ("Afile", "r")))
	fprintf (stderr, "cannot open file %s\n", "Afile") ;
}

static void close (FILE *input)
{
    fclose (input) ;
}

static void read_ints (FILE *input, int *A, int n, int convert_zero_based)
{
    assert (A) ;
    char line [80] ;
    int i ;
    if (convert_zero_based)
    {
	for (i = 0 ; i < n ; i++)
	{
	    fgets (line, 80, input) ;
	    *A++ = atoi (line)  - 1 ;
	}
    }
    else
    {
	for (i = 0 ; i < n ; i++)
	{
	    fgets (line, 80, input) ;
	    *A++ = atoi (line) ;
	}
    }    
}

static void read_doubles (FILE *input, double *Ax, int n)
{
    double x ;
    int i ;
    assert (Ax) ;
    for (i = 0 ; i < n ; i++)
    {
	fscanf (input, "%lg\n", &x) ;
	*Ax++ = x ;
    }	
}

static void read_rhs (FILE *input, double *Ax, double *At, int n)
{
    double x ;
    int i ;
    assert (Ax) ;
    for (i = 0 ; i < n ; i++)
    {
	fscanf (input, "%lg\n", &x) ;
	*Ax++ = x ;
	*At++ = x ;
    }	
}

static void read_complex_rhs (FILE *input, double *Ax, double *At, int n, 
			     int isRHSreal)
{
    double x ;
    int i ;
    assert (Ax) ;
    if (isRHSreal)
    {
	for (i = 0 ; i < n ; i++)
	{
	    fscanf (input, "%lg\n", &x) ;
	    *Ax++ = x ;
	    *Ax++ = 0.0 ;
	    *At++ = x ;
	    *At++ = 0.0 ;
	}	
    }
    else
    {
	for (i = 0 ; i < 2*n ; i++)
	{
	    fscanf (input, "%lg\n", &x) ;
	    *Ax++ = x ;
	    *At++ = x ;
	}	
    }
}

static void writeResult (double *Bx, int n, const char *name, const char * iname,
		 int isreal)
{
    FILE *output ;
    FILE *ioutput ;
    int i ;

    if (!(output = fopen (name, "w")))
	fprintf (stderr, "cannot open file %s\n", "X") ;

    if (!isreal)
    {
	if (!(ioutput = fopen (iname, "w")))
	    fprintf (stderr, "cannot open file %s\n", "Xz") ;
	for (i = 0 ; i < 2*n ; i += 2)
	{
	    fprintf (output, "%30.20e\n", Bx [i]) ;
	    fprintf (ioutput, "%30.20e\n", Bx [i + 1]) ;
	}
	fclose (ioutput) ;
    }
    else
    {
	for (i = 0 ; i < n ; i++)
	    fprintf (output, "%30.20e\n", Bx [i]) ;
    }

    fclose (output) ;
    
}

static void writeDiagnostics (double growth, double condest, double rcond)
{
    FILE *output ;

    if (!(output = fopen ("cnum", "w")))
	fprintf (stderr, "cannot open file %s\n", "cnum") ;

    fprintf (output, "%10.20e\n", condest) ;
    fclose (output) ;

    if (!(output = fopen ("pgrowth", "w")))
	fprintf (stderr, "cannot open file %s\n", "pgrowth") ;
    fprintf (output, "%10.20e\n", growth) ;
    fclose (output) ;

    if (!(output = fopen ("rcond", "w")))
	fprintf (stderr, "cannot open file %s\n", "rcond") ;
    fprintf (output, "%10.20e\n", rcond) ;
    fclose (output) ;
}

static void do_solve (int n, int *Ap, int *Ai, double *Ax, double *Bx, 
		     double *Bt, int nrhs, int isreal, klu_common *Common)
{
    klu_symbolic *Symbolic = 0 ; 
    klu_numeric *Numeric = 0 ;
    double growth = 0. ;
    double condest = 0. ;

    assert (Ap) ;
    assert (Ai) ;
    assert (Ax) ;

    Symbolic = klu_analyze (n, Ap, Ai, Common) ;
    if (Symbolic == (klu_symbolic *) NULL)
    {
        PRINTF (("klu_analyze failed\n")) ;
	exit (1) ;
    }

    if (!isreal)
    {
	PRINTF (("Complex case\n")) ;
    }
    
    if (isreal)
        Numeric = klu_factor (Ap, Ai, Ax, Symbolic, Common) ;
    else
        Numeric = klu_z_factor (Ap, Ai, Ax, Symbolic, Common) ;

    if (Common->status == KLU_SINGULAR)
    {
        PRINTF (("# singular column : %d\n", Common->singular_col)) ;
	exit (1) ;
    }
    if (Common->status != KLU_OK)
    {
        PRINTF (("klu_factor failed")) ;
	exit (1) ;
    }

    if (isreal)
    {
	klu_refactor (Ap, Ai, Ax, Symbolic, Numeric, Common) ;
    }
    else
    {
	klu_z_refactor (Ap, Ai, Ax, Symbolic, Numeric, Common) ;
    }

    if (Common->status != KLU_OK)
    {
        PRINTF (("klu_refactor failed")) ;
	exit (1) ;
    }

    if (isreal)
    {
	klu_solve (Symbolic, Numeric, n, nrhs, Bx, Common) ;
    }
    else
    {
	klu_z_solve (Symbolic, Numeric, n, nrhs, Bx, Common) ;
    }

    writeResult (Bx, n*nrhs, "x", "xz", isreal) ;

    if (isreal)
    {
	klu_growth (Ap, Ai, Ax, Symbolic, Numeric, &growth, Common) ;
	klu_condest (Ap, Ax, Symbolic, Numeric, &condest, Common) ;
	klu_rcond (Symbolic, Numeric, &rcond, Common) ;
    }
    else
    {
	klu_z_growth (Ap, Ai, Ax, Symbolic, Numeric, &growth, Common) ;
	klu_z_condest (Ap, Ax, Symbolic, Numeric, &condest, Common) ;
	klu_z_rcond (Symbolic, Numeric, &rcond, Common) ;
    }
    writeDiagnostics (growth, condest, rcond) ;

    /* transpose solve */
    if (isreal)
    {
	klu_tsolve (Symbolic, Numeric, n, nrhs, Bt, Common) ;
    }
    else
    {
	klu_z_tsolve (Symbolic, Numeric, n, nrhs, Bt, 1, Common) ;
    }
    
    writeResult (Bt, n*nrhs, "xt", "xtz", isreal) ;
    
    klu_free_symbolic (&Symbolic, Common) ;
    if (isreal)
    {
	klu_free_numeric (&Numeric, Common) ;
    }
    else
    {
	klu_z_free_numeric (&Numeric, Common) ;
    }
}

int main (void)
{
    FILE *file = 0 ;
    int n = 0, nnz = 0, isreal = 0 , nrhs = 0, isRHSreal = 0 ;
    char line [80] ;
    klu_common Common ;
    int *Ap, *Ai ;
    double *Ax, *Bx, *Bt ;
    
    open (&file) ;

    klu_defaults (&Common) ;

    read_options (&Common) ;
    fgets (line, 80, file) ;
    assert (strstr (line, "n, nnz, real, nrhs, isRHSreal")) ;
    
    fgets (line, 80, file) ;
    n = atoi (line) ;
    fgets (line, 80, file) ;
    nnz = atoi (line) ;
    fgets (line, 80, file) ;
    isreal = atoi (line) ;
    fgets (line, 80, file) ;
    nrhs = atoi (line) ;
    fgets (line, 80, file) ;
    isRHSreal = atoi (line) ;

    PRINTF (("tol %f\n", Common.tol)) ;
    PRINTF (("growth %f\n", Common.growth)) ;
    PRINTF (("initmem_amd %f\n", Common.initmem_amd)) ;
    PRINTF (("initmem %f\n", Common.initmem)) ;
    PRINTF (("btf %d\n", Common.btf)) ;
    PRINTF (("ordering %d\n", Common.ordering)) ;
    PRINTF (("scale %d\n", Common.scale)) ;
    PRINTF (("halt_if_singular %d\n", Common.halt_if_singular)) ;
    PRINTF (("n %d\n", n)) ;
    PRINTF (("nnz %d\n", nnz)) ;
    PRINTF (("isreal %d\n", isreal)) ;
    PRINTF (("nrhs %d\n", nrhs)) ;
    PRINTF (("is rhs real %d\n", isRHSreal)) ;

    assert (n > 0) ;
    fgets (line, 80, file) ;
    assert (strstr (line, "column pointers")) ;
    Ap = (int *) malloc ((n + 1) * sizeof (int)) ;
    Ai = (int *) malloc (nnz * sizeof (int)) ;
    if (isreal)
    {
        Ax = (double *) malloc (nnz * sizeof (double)) ;
        Bx = (double *) malloc (nrhs * n * sizeof (double)) ;
        Bt = (double *) malloc (nrhs * n * sizeof (double)) ;
    }
    else
    {
	Ax = (double *) malloc (nnz * 2 * sizeof (double)) ;
	Bx = (double *) malloc (nrhs * n * 2 * sizeof (double)) ;
	Bt = (double *) malloc (nrhs * n * 2 * sizeof (double)) ;
    }

    if (Ap == NULL || Ai == NULL || Ax == NULL || Bx == NULL || Bt == NULL)
    {
	PRINTF (("Malloc failed\n")) ;
	exit (1) ;
    }
    read_ints (file, Ap, n + 1, 0) ;

    fgets (line, 80, file) ;
    assert (strstr (line, "row indices")) ;
    read_ints (file, Ai, nnz, 1) ;
    
    fgets (line, 80, file) ;
    assert (strstr (line, "reals")) ;

    if (isreal)
    {
        read_doubles (file, Ax, nnz) ;
    }
    else
    {
        read_doubles (file, Ax, 2*nnz) ;
    }
    
    fgets (line, 80, file) ;
    assert (strstr (line, "rhs")) ;
    if (isreal)
    {
        read_rhs (file, Bx, Bt, n*nrhs) ;
    }
    else
    {
        read_complex_rhs (file, Bx, Bt, n*nrhs, isRHSreal) ;
    }

/*
#ifndef NPRINT
    PRINTF (("Column pointers\n")) ;
    for (k = 0 ; k < n+1 ; k++)
	PRINTF (("%d\n", Ap [k])) ;
    PRINTF (("row indices\n")) ;
    for (k = 0 ; k < nnz ; k++)
	PRINTF (("%d\n", Ai [k])) ;
    PRINTF (("reals\n")) ;
    if (isreal)
    {
	for (k = 0 ; k < nnz ; k++)
	    PRINTF (("%30.20e\n", Ax [k])) ;
    }
    else
    {
	for (k = 0 ; k < 2 * nnz ; k += 2)
	    PRINTF (("%30.20e   i %30.20e\n", Ax [k], Ax [k+1])) ;
    }

    PRINTF (("rhs\n")) ;
    if (isRHSreal)
    {
	for (k = 0 ; k < nrhs * n ; k++)
	    PRINTF (("%30.20e\n", Bx [k])) ;
    }
    else
    {
	for (k = 0 ; k < 2 * n * nrhs ; k += 2)
	    PRINTF (("%30.20e   i %30.20e\n", Bx [k], Bx [k+1])) ;
    }
#endif */
    close (file) ;    
    
    do_solve (n, Ap, Ai, Ax, Bx, Bt, nrhs, isreal, &Common) ;

    free (Ap) ;
    free (Ai) ;
    free (Ax) ;
    free (Bx) ;
    free (Bt) ;
    
    return (0) ;
}
