LINFACTOR factorize a matrix, or use the factors to solve Ax=b.

Uses LU or CHOL to factorize A, or uses a previously computed factorization to
solve a linear system.  This function automatically selects an LU or Cholesky
factorization, depending on the matrix.  A better method would be for you to
select it yourself.  Note that mldivide uses a faster method for detecting
whether or not A is a candidate for sparse Cholesky factorization (see spsym in
the CHOLMOD package, for example).

Example:

  F = linfactor (A) ;     % factorizes A into the object F

  x = linfactor (F,b) ;   % uses F to solve Ax=b

  norm (A*x-b)

A second output is the time taken by the method, ignoring the overhead of
determining which method to use.  This makes for a fairer comparison between
methods, since normally the user will know if the matrix is supposed to be
symmetric positive definite or not, and whether or not the matrix is sparse.
Also, the overhead here is much higher than mldivide or spsym.

This function has its limitations:

(1) determining whether or not the matrix is symmetric via nnz(A-A') is slow.
mldivide (and spsym in CHOLMOD) do it much faster.

(2) MATLAB really needs a sparse linsolve.  See cs_lsolve, cs_ltsolve, and
cs_usolve in CSparse, for example.

(3) this function really needs to be written as a mexFunction.

(4) the full power of mldivide is not brought to bear.  For example, UMFPACK is
not very fast for sparse tridiagonal matrices.  It's about a factor of four
slower than a specialized tridiagonal solver as used in mldivide.

(5) permuting a sparse vector or matrix is slower in MATLAB than it should be;
a built-in linfactor would reduce this overhead.

(6) mldivide when using UMFPACK uses relaxed partial pivoting and then
iterative refinement.  This leads to sparser LU factors, and typically accurate
results.  linfactor uses sparse LU without iterative refinement.

The primary purpose of this function is to answer The Perennially Asked
Question (or The PAQ for short (*)):  "Why not use x=inv(A)*b to solve Ax=b?
How do I use LU or CHOL to solve Ax=b?"  The full answer is below.  The short
answer to The PAQ (*) is "PAQ=LU ... ;-) ... never EVER use inv(A) to solve
Ax=b."

The secondary purpose of this function is to provide a prototype for some of
the functionality of a true MATLAB built-in linfactor function.

Finally, the third purpose of this function is that you might find it actually
useful for production use, since its syntax is simpler than factorizing the
matrix yourself and then using the factors to solve the system.  

See also lu, chol, mldivide, linsolve, umfpack, cholmod.

Oh, did I tell you never to use inv(A) to solve Ax=b?
