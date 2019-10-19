c=======================================================================
c=== RBio/RBread_mex_32 ================================================
c=======================================================================

c RBio: a MATLAB toolbox for reading and writing sparse matrices in
c Rutherford/Boeing format.
c Copyright (c) 2007, Timothy A. Davis, Univ. of Florida


c-----------------------------------------------------------------------
c RBread mexFunction: read a sparse matrix from a Rutherford/Boeing file
c-----------------------------------------------------------------------
c
c   [A Z title key mtype] = RBread (filename)
c
c   A: a sparse matrix (no explicit zero entries)
c   Z: binary pattern of explicit zero entries in Rutherford/Boeing file.
c	This always has the same size as A, and is always sparse.
c
c   title: the 72-character title string in the file
c   key: the 8-character matrix name in the file
c   mtype: see RBwrite.m for a description.
c
c-----------------------------------------------------------------------

	subroutine mexfunction (nargout, pargout, nargin, pargin)
	integer*4
     $	    pargout (*), pargin (*)
	integer*4 nargout, nargin

c	----------------------------------------------------------------
c	MATLAB functions
c	----------------------------------------------------------------

	integer*4 mxIsChar, mxClassIDFromClassName

	integer*4
     $	    mxGetM, mxGetN, mxCreateSparse, mxGetJc, mxGetData,
     $	    mxGetIr, mxGetPr, mxGetPi, mxGetString, mxCreateString,
     $	    mxCreateNumericMatrix

c	----------------------------------------------------------------
c	local variables
c	----------------------------------------------------------------

	integer*4
     $	    nrow, ncol, nnz, mkind, w, cp,
     $	    skind, wmat, cpmat, Zmat, nw,
     $	    Ap, Ai, Ax, Az, Cmat, Cx, nzeros, info, Zp, Zi, Zx, i,
     $	    nnz2, nnz1, nelnz, one
	integer*4 iclass, cmplex, wcmplex
	character title*72, key*8, mtype*3, ptrfmt*16, indfmt*16,
     $	    valfmt*20, filename*1024

c	----------------------------------------------------------------
c	check inputs
c	----------------------------------------------------------------

	if (nargin .ne. 1 .or. nargout .gt. 5 .or.
     $	    mxIsChar (pargin (1)) .ne. 1) then
	    call mexErrMsgTxt
     $		('Usage: [A Z title key mtype] = RBread (filename)')
	endif

c	----------------------------------------------------------------
c	get filename and open file
c	----------------------------------------------------------------

	if (mxGetString (pargin (1), filename, 1024) .ne. 0) then
	    call mexErrMsgTxt ('filename too long')
	endif
	close (unit = 7)
	open (unit = 7, file = filename, status = 'OLD', err = 998)
	rewind (unit = 7)

c	----------------------------------------------------------------
c	read the header and determine the matrix type
c	----------------------------------------------------------------

	call RBheader (title, key, mtype, nrow, ncol, nnz,
     $	    ptrfmt, indfmt, valfmt, mkind, cmplex, skind, nelnz, info)
	if (nelnz .ne. 0) then
c	    finite-element matrices not supported
	    info = -5
	endif
	call RBerr (info)

c	----------------------------------------------------------------
c	allocate result A
c	----------------------------------------------------------------

	if (skind .gt. 0) then
c	    allocate enough space for upper triangular part (S,H,Z)
	    nnz1 = 2 * nnz
	else
	    nnz1 = nnz
	endif
	nnz1 = max (nnz1, 1)

	pargout (1) = mxCreateSparse (nrow, ncol, nnz1, cmplex)
	Ap = mxGetJc (pargout (1))
	Ai = mxGetIr (pargout (1))
	Ax = mxGetPr (pargout (1))
	Az = mxGetPi (pargout (1))

c	----------------------------------------------------------------
c	allocate workspace
c	----------------------------------------------------------------

	iclass = mxClassIDFromClassName ('int32')
	nw = max (nrow,ncol) + 1
	wcmplex = 0
	one = 1
	wmat = mxCreateNumericMatrix (nw, one, iclass, wcmplex)
	cpmat = mxCreateNumericMatrix (ncol+1, one, iclass, wcmplex)
	w = mxGetData (wmat)
	cp = mxGetData (cpmat)

c	----------------------------------------------------------------
c	read in the sparse matrix
c	----------------------------------------------------------------

	if (mkind .eq. 2) then
c	    complex matrices
	    Cmat = mxCreateNumericMatrix (2 * nnz1, one,
     $		    mxClassIDFromClassName ('double'), wcmplex)
	    Cx = mxGetData (Cmat)
	    call RBcread (nrow, ncol, nnz, ptrfmt, indfmt, valfmt,
     $		mkind, skind, %val (Ap), %val (Ai), %val (Cx), nzeros,
     $		%val (w), %val (cp), info, nw, nnz1)
	else
c	    real, pattern, and integer matrices
	    call RBrread (nrow, ncol, nnz, ptrfmt, indfmt, valfmt,
     $		mkind, skind, %val (Ap), %val (Ai), %val (Ax), nzeros,
     $		%val (w), %val (cp), info, nw, nnz1)
	endif
	call RBerr (info)
	close (unit = 7)

c	----------------------------------------------------------------
c	extract or discard explicit zero entries
c	----------------------------------------------------------------

	if (nargout .ge. 2) then

c	    ------------------------------------------------------------
c	    extract explicit zeros from A and store them in Z
c	    ------------------------------------------------------------

	    pargout (2) = mxCreateSparse
     $		(nrow, ncol, max (nzeros,1), wcmplex)
	    Zp = mxGetJc (pargout (2))
	    Zi = mxGetIr (pargout (2))
	    Zx = mxGetPr (pargout (2))

	    if (mkind .eq. 2) then
c    		complex matrices
		call RBczeros (nrow, ncol, %val (cp),
     $		    %val (Ap), %val (Ai), %val (Cx),
     $		    %val (Zp), %val (Zi), %val (Zx))
	    else
c		real, pattern-only, and integer matrices
		call RBrzeros (nrow, ncol, %val (cp),
     $		    %val (Ap), %val (Ai), %val (Ax),
     $		    %val (Zp), %val (Zi), %val (Zx))
	    endif

c	    convert Z to zero-based
	    call RBmangle (ncol, %val (Zp), %val (Zi), i)

	else

c	    ------------------------------------------------------------
c	    discard explicit zero entries from A (do not keep them)
c	    ------------------------------------------------------------

	    if (mkind .eq. 2) then
c    		complex matrices
		call RBcprune (nrow, ncol,
     $		    %val (Ap), %val (Ai), %val (Cx))
	    else
c		real, pattern-only, and integer matrices
		call RBrprune (nrow, ncol,
     $		    %val (Ap), %val (Ai), %val (Ax))
	    endif

	endif

c	----------------------------------------------------------------
c	convert A to final MATLAB form (zero-based, split complex)
c	----------------------------------------------------------------

	call RBmangle (ncol, %val (Ap), %val (Ai), nnz2)

	if (mkind .eq. 2) then
c	    convert Fortran-style complex values to MATLAB-style
	    call RBcsplit (%val (Cx), %val (Ax), %val (Az), nnz2)
	    call mxDestroyArray (%val (Cmat))
	endif

c	----------------------------------------------------------------
c	return title
c	----------------------------------------------------------------

	if (nargout .ge. 3) then
	    pargout (3) = mxCreateString (title)
	endif

c	----------------------------------------------------------------
c	return key
c	----------------------------------------------------------------

	if (nargout .ge. 4) then
	    pargout (4) = mxCreateString (key)
	endif

c	----------------------------------------------------------------
c	return the matrix type to MATLAB
c	----------------------------------------------------------------

	if (nargout .ge. 5) then
	    pargout (5) = mxCreateString (mtype)
	endif

c	----------------------------------------------------------------
c	free workspace and return
c	----------------------------------------------------------------

	call mxDestroyArray (%val (wmat))
	call mxDestroyArray (%val (cpmat))
	return

c	----------------------------------------------------------------
c	error return
c	----------------------------------------------------------------

998	call mexErrMsgTxt ('error opening file')
	end

