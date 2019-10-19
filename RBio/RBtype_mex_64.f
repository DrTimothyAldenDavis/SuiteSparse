c=======================================================================
c=== RBio/RBtype_mex_64 ================================================
c=======================================================================

c RBio: a MATLAB toolbox for reading and writing sparse matrices in
c Rutherford/Boeing format.
c Copyright (c) 2007, Timothy A. Davis, Univ. of Florida


c-----------------------------------------------------------------------
c RBtype mexFunction:
c-----------------------------------------------------------------------
c
c   [mtype mkind skind] = RBtype (A)
c
c   A: a sparse matrix.   Determines the Rutherford/Boeing type of the
c   matrix.  Very little memory is used (just size(A,2) integer
c   workspace), so this can succeed where a test such as nnz(A-A')==0
c   will fail.
c
c	mkind:	r: (0), A is real, and not binary
c		p: (1), A is binary
c		c: (2), A is complex
c		i: (3), A is integer
c
c	skind:  r: (-1), A is rectangular
c               u: (0), A is unsymmetric (not S, H, Z, below)
c		s: (1), A is symmetric (nnz(A-A.') is 0)
c		h: (2), A is Hermitian (nnz(A-A') is 0)
c		z: (3), A is skew symmetric (nnz(A+A.') is 0)
c
c   mtype is a 3-character string, where mtype(1) is the mkind
c   ('r', 'p', 'c', or 'i').  mtype(2) is the skind ('r', 'u', 's', 'h',
c   or 'z'), and mtype(3) is always 'a'.
c-----------------------------------------------------------------------

	subroutine mexfunction (nargout, pargout, nargin, pargin)
	integer*8
     $	    pargout (*), pargin (*)
	integer*4 nargout, nargin

c	----------------------------------------------------------------
c	MATLAB functions
c	----------------------------------------------------------------

	integer*4 mxClassIDFromClassName,
     $	    mxIsClass, mxIsSparse, mxIsComplex

	integer*8
     $	    mxGetM, mxGetN, mxGetJc, mxGetIr, mxGetPr, mxGetPi,
     $	    mxGetData, mxCreateNumericMatrix, mxCreateDoubleScalar,
     $	    mxCreateString

c	----------------------------------------------------------------
c	local variables
c	----------------------------------------------------------------

	integer*8
     $	    nrow, ncol, nnz, mkind, cp, skind, cpmat,
     $	    Ap, Ai, Ax, Az, kmin, kmax, one
	integer*4 iclass, cmplex, wcmplex
	character mtype*3
	double precision t

c	----------------------------------------------------------------
c	check inputs
c	----------------------------------------------------------------

	if (nargin .ne. 1 .or. nargout .gt. 3) then
	    call mexErrMsgTxt ('[mtype mkind skind] = RBtype (A)')
	endif

c	----------------------------------------------------------------
c	get A
c	----------------------------------------------------------------

	if (mxIsClass (pargin (1), 'double') .ne. 1 .or.
     $	    mxIsSparse (pargin (1)) .ne. 1) then
	    call mexErrMsgTxt ('A must be sparse and double')
	endif
	cmplex = mxIsComplex (pargin (1))
	Ap = mxGetJc (pargin (1))
	Ai = mxGetIr (pargin (1))
	Ax = mxGetPr (pargin (1))
	Az = mxGetPi (pargin (1))
	nrow = mxGetM (pargin (1))
	ncol = mxGetN (pargin (1))

c	----------------------------------------------------------------
c	allocate workspace
c	----------------------------------------------------------------

	iclass = mxClassIDFromClassName ('int64')
	one = 1
	wcmplex = 0
	cpmat = mxCreateNumericMatrix (ncol+1, one, iclass, wcmplex)
	cp = mxGetData (cpmat)

c	----------------------------------------------------------------
c	determine the mtype of A
c	----------------------------------------------------------------

	call RBkind (nrow, ncol, %val(Ap), %val(Ai), %val(Ax),
     $	    %val(Az), cmplex, mkind, skind, mtype, nnz, %val(cp),
     $	    kmin, kmax)

c	----------------------------------------------------------------
c	return the result
c	----------------------------------------------------------------

	pargout (1) = mxCreateString (mtype)
	if (nargout .ge. 2) then
	    t = mkind
	    pargout (2) = mxCreateDoubleScalar (t)
	endif
	if (nargout .ge. 3) then
	    t = skind
	    pargout (3) = mxCreateDoubleScalar (t)
	endif

c	----------------------------------------------------------------
c	free workspace
c	----------------------------------------------------------------

	call mxDestroyArray (%val (cpmat))
	return
	end

