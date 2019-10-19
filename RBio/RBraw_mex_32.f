c=======================================================================
c=== RBio/RBraw_mex_32 =================================================
c=======================================================================

c RBio: a MATLAB toolbox for reading and writing sparse matrices in
c Rutherford/Boeing format.
c Copyright (c) 2007, Timothy A. Davis, Univ. of Florida


c-----------------------------------------------------------------------
c RBraw mexFunction: read the raw contents of a Rutherford/Boeing file
c-----------------------------------------------------------------------
c
c   [mtype Ap Ai Ax title key nrow] = RBraw (filename)
c
c   mtype: Rutherford/Boeing matrix type (psa, rua, rsa, rse, ...)
c   Ap: column pointers (1-based)
c   Ai: row indices (1-based)
c   Ax: numerical values (real, complex, or integer).  Empty for p*a
c	matrices.  A complex matrix is read in as a single double array
c	Ax, where the kth entry has real value Ax(2*k-1) and imaginary
c	value Ax(2*k).
c   title: a string containing the title from the first line of the file
c   key: a string containing the 8-char key, from 1st line of the file
c   nrow: number of rows in the matrix
c
c This function works for both assembled and unassembled (finite-
c element) matrices.  It is also useful for checking the contents of a
c Rutherford/Boeing file in detail, in case the file has invalid column
c pointers, unsorted columns, duplicate entries, entries in the upper
c triangular part of the file for a symmetric matrix, etc.
c
c Example:
c
c   load west0479
c   RBwrite ('mywest', west0479, [ ], 'My west0479 file', 'west0479') ;
c   [mtype Ap Ai Ax title key nrow] = RBraw ('mywest') ;
c
c See also RBfix, RBread, RBreade.
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
     $	    mxGetString, mxCreateString, mxCreateDoubleScalar,
     $	    mxCreateNumericMatrix, mxGetData

c	----------------------------------------------------------------
c	local variables
c	----------------------------------------------------------------

	integer*4
     $	    nrow, ncol, nnz, mkind, info, skind, k, nelnz, one, zero
	integer*4 iclass, cmplex, wcmplex
	character title*72, key*8, mtype*3, ptrfmt*16, indfmt*16,
     $	    valfmt*20, filename*1024
	double precision x

c	----------------------------------------------------------------
c	check inputs
c	----------------------------------------------------------------

	if (nargin .ne. 1 .or. nargout .gt. 7 .or.
     $	    mxIsChar (pargin (1)) .ne. 1) then
	    call mexErrMsgTxt
     $	  ('Usage: [mtype Ap Ai Ax title key nrow] = RBraw (filename)')
	endif

c	----------------------------------------------------------------
c	get filename and open file
c	----------------------------------------------------------------

	if (mxGetString (pargin (1), filename, 1024) .ne. 0) then
	    call mexErrMsgTxt ('filename too long')
	endif
	close (unit = 7)
	open (unit = 7, file = filename, status = 'OLD', err = 998)

c	----------------------------------------------------------------
c	read the header and determine the matrix type
c	----------------------------------------------------------------

	call RBheader (title, key, mtype, nrow, ncol, nnz,
     $	    ptrfmt, indfmt, valfmt,
     $	    mkind, cmplex, skind, nelnz, info)
	call RBerr (info)

c	----------------------------------------------------------------
c	return the matrix type to MATLAB
c	----------------------------------------------------------------

	pargout (1) = mxCreateString (mtype)

c	----------------------------------------------------------------
c	read in the column pointers
c	----------------------------------------------------------------

	iclass = mxClassIDFromClassName ('int32')
	one = 1
	zero = 0
	wcmplex = 0
	if (nargout .ge. 2) then
	    pargout (2) = mxCreateNumericMatrix 
     $		(ncol+1, one, iclass, wcmplex)
	    call RBiread (ptrfmt, ncol+1,
     $		%val(mxGetData (pargout (2))), info)
	    call RBerr (info)
	endif

c	----------------------------------------------------------------
c	read in the row indices
c	----------------------------------------------------------------

	if (nargout .ge. 3) then
	    pargout (3) = mxCreateNumericMatrix
     $		(nnz, one, iclass, wcmplex)
	    call RBiread (indfmt, nnz,
     $		%val(mxGetData (pargout (3))), info)
	    if (info .lt. 0) then
		info = -93
	    endif
	    call RBerr (info)
	endif

c	----------------------------------------------------------------
c	read in the numerical values
c	----------------------------------------------------------------

	if (nelnz .eq. 0) then
	    k = nnz
	else
	    k = nelnz
	endif

	if (nargout .ge. 4) then

	    if (mkind .eq. 1) then

c		pattern-only: create an empty numerical array
		pargout (4) = mxCreateNumericMatrix (zero, zero,
     $		    mxClassIDFromClassName ('double'), wcmplex)

	    elseif (mkind .eq. 3) then

c		read in the numerical values (integer)
		pargout (4) = mxCreateNumericMatrix
     $		    (k, one, iclass, wcmplex)
		call RBiread (valfmt, k,
     $		    %val(mxGetData (pargout (4))), info)
		call RBerr (info)

	    else

c		read in the numerical values (real or complex)
		if (cmplex .eq. 1) then
		    k = 2*k
		endif
		pargout (4) = mxCreateNumericMatrix (k, one,
     $		    mxClassIDFromClassName ('double'), wcmplex)
		call RBxread (valfmt, k,
     $		    %val(mxGetData (pargout (4))), info)
		call RBerr (info)
	    endif

	endif

c	----------------------------------------------------------------
c	return the title
c	----------------------------------------------------------------

	if (nargout .ge. 5) then
	    pargout (5) = mxCreateString (title)
	endif

c	----------------------------------------------------------------
c	return the key
c	----------------------------------------------------------------

	if (nargout .ge. 6) then
	    pargout (6) = mxCreateString (key)
	endif

c	----------------------------------------------------------------
c	return the number of rows
c	----------------------------------------------------------------

	if (nargout .ge. 7) then
	    x = nrow
	    pargout (7) = mxCreateDoubleScalar (x)
	endif

c	----------------------------------------------------------------
c	close file
c	----------------------------------------------------------------

	close (unit = 7)
	return

c	----------------------------------------------------------------
c	error return
c	----------------------------------------------------------------

998	call mexErrMsgTxt ('error opening file')
	end

