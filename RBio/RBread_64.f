c=======================================================================
c=== RBio/RBread_64 ====================================================
c=======================================================================

c RBio: a MATLAB toolbox for reading and writing sparse matrices in
c Rutherford/Boeing format.
c Copyright (c) 2007, Timothy A. Davis, Univ. of Florida


c-----------------------------------------------------------------------
c RBheader:  read Rutherford/Boeing header lines
c-----------------------------------------------------------------------
c
c Rutherford/Boeing file type is a 3-character string:
c
c   (1) R: real, C: complex, P: pattern only, I: integer
c	mkind: R: 0, P: 1, C: 2, I: 3
c
c   (2) S: symmetric, U: unsymmetric, H: Hermitian, Z: skew symmetric,
c	R: rectangular
c	skind: R: -1, U: 0, S: 1, H: 2, Z: 3
c	
c   (3) A: assembled, E: element form
c	nelnz = 0 for A, number of elements for E
c
c	pattern matrices are given numerical values of 1 (except PZA).
c	PZA matrices have +1 in the lower triangular part and -1 in
c	the upper triangular part.
c
c   The matrix is nrow-by-ncol with nnz entries.
c   For symmetric matrices, Ai and Ax are of size 2*nnz (the upper
c	triangular part is constructed).  To skip this construction,
c	pass skind = 0 to RBpattern.
c-----------------------------------------------------------------------

	subroutine RBheader (title, key, mtype, nrow, ncol, nnz,
     $	    ptrfmt, indfmt, valfmt, mkind, cmplex, skind, nelnz, info)

	integer*8
     $	    nrow, ncol, nnz, totcrd, ptrcrd, nelnz,
     $	    indcrd, valcrd, mkind, skind, info
	integer*4 cmplex
	character title*72, key*8, mtype*3, ptrfmt*16, indfmt*16,
     $	    valfmt*20, rhstyp*3
	logical fem

c	----------------------------------------------------------------
c	read header lines 1-4
c	----------------------------------------------------------------

        read (7, 10, err = 91, end = 91)
     $          title, key,
     $          totcrd, ptrcrd, indcrd, valcrd,
     $          mtype, nrow, ncol, nnz, nelnz,
     $          ptrfmt, indfmt, valfmt
10      format (a72, a8 / 4i14 / a3, 11x, 4i14 / 2a16, a20)

	if (nrow .lt. 0 .or. ncol .lt. 0 .or. nnz .lt. 0) then
c	    error: invalid matrix dimensions
	    info = -6
	    return
	endif

c	----------------------------------------------------------------
c	skip the Harwell/Boeing header line 5, if present
c	----------------------------------------------------------------

	read (7, 20, err = 91, end = 91) rhstyp
20	format (a3)
	if ((rhstyp (1:1) .eq. 'F' .or. rhstyp (1:1) .eq. 'f' .or.
     $	    rhstyp (1:1) .eq. 'M' .or. rhstyp (1:1) .eq. 'm')) then
c	    This is the 5th line Harwell/Boeing format.  Ignore it.
	    call mexErrMsgTxt ('Harwell/Boeing RHS ignored')
	else
c	    Backspace one record, since we just read in one row of
c	    the column pointers.
	    backspace (unit = 7)
	endif

c	----------------------------------------------------------------
c	determine if real, pattern, integer, or complex
c	----------------------------------------------------------------

	if (mtype (1:1) .eq. 'R' .or. mtype (1:1) .eq. 'r') then

c	    real
	    mkind = 0
	    cmplex = 0

	elseif (mtype (1:1) .eq. 'P' .or. mtype (1:1) .eq. 'p') then

c	    pattern
	    mkind = 1
	    cmplex = 0

	elseif (mtype (1:1) .eq. 'C' .or. mtype (1:1) .eq. 'c') then

c	    complex
	    mkind = 2
	    cmplex = 1

	elseif (mtype (1:1) .eq. 'I' .or. mtype (1:1) .eq. 'i') then

c	    integer
	    mkind = 3
	    cmplex = 0

	else

c	    error: invalid matrix type
	    info = -5
	    return

	endif

c	----------------------------------------------------------------
c	determine if the upper part must be constructed
c	----------------------------------------------------------------

	if (mtype (2:2) .eq. 'R' .or. mtype (2:2) .eq. 'r') then

c	    rectangular: RRA, PRA, IRA, and CRA matrices
	    skind = -1

	elseif (mtype (2:2) .eq. 'U' .or. mtype (2:2) .eq. 'u') then

c	    unsymmetric: RUA, PUA, IUA, and CUA matrices
	    skind = 0

	elseif (mtype (2:2) .eq. 'S' .or. mtype (2:2) .eq. 's') then

c	    symmetric: RSA, PSA, ISA, and CSA matrices
	    skind = 1

	elseif (mtype (2:2) .eq. 'H' .or. mtype (2:2) .eq. 'h') then

c	    Hermitian: CHA (PHA, IHA, and RHA are valid, but atypical)
	    skind = 2

	elseif (mtype (2:2) .eq. 'Z' .or. mtype (2:2) .eq. 'z') then

c	    skew symmetric: RZA, PZA, IZA, and CZA
	    skind = 3

	else

c	    error: invalid matrix type
	    info = -5
	    return

	endif

c	----------------------------------------------------------------
c	assembled matrices or elemental matrices (**A, **E)
c	----------------------------------------------------------------

	if (mtype (3:3) .eq. 'A' .or. mtype (3:3) .eq. 'a') then

c	    assembled - ignore nelnz
	    fem = .false.
	    nelnz = 0

	elseif (mtype (3:3) .eq. 'E' .or. mtype (3:3) .eq. 'e') then

c	    finite-element
	    fem = .true.
	    continue

	else

c	    error: invalid matrix type
	    info = -5
	    return

	endif

c	----------------------------------------------------------------
c	assembled matrices must be square if skind is not R
c	----------------------------------------------------------------

	if (.not. fem .and. skind .ne. -1 .and. nrow .ne. ncol) then

c	    error: invalid matrix dimensions
	    info = -6
	    return

	endif

c	----------------------------------------------------------------
c	matrix is valid
c	----------------------------------------------------------------

	info = 0
	return

c	----------------------------------------------------------------
c	error reading file
c	----------------------------------------------------------------

91	info = -91
	return
	end


c-----------------------------------------------------------------------
c RBpattern: read the column pointers and row indices
c-----------------------------------------------------------------------

c   w and cp are both of size ncol+1 (undefined on input).
c
c   The matrix is contained in Ap, Ai, and Ax on output (undefined on
c   input).  It has nzeros explicit zero entries.  cp (1..ncol+1) are
c   the column pointers for the matrix Z that will contain all the
c   explicit zero entries.  Ax is not read in (see RBrread and
c   RBcread).
c
c   info is returned as:
c	0	ok
c	-1	invalid column pointers
c	-2	row index out of range
c	-3	duplicate entry
c	-4	entries in upper triangular part of symmetric matrix
c	-5	invalid matrix type
c	-6	invalid dimensions
c	-7	matrix contains unsorted columns
c	-91	error reading file (header) 
c	-92	error reading file (column pointers) 
c	-93	error reading file (row indices) 
c	-94	error reading file (numerical values: A, or sparse b)
c-----------------------------------------------------------------------

	subroutine RBpattern (ptrfmt, indfmt, nrow, ncol, nnz, skind,
     $	    Ap, Ai, w, cp, info, nw)
	integer*8
     $	    nrow, ncol, nnz, skind, info, Ap (ncol+1), Ai (nnz),
     $	    nw, w (nw), cp (ncol+1)
	character ptrfmt*16, indfmt*16
	integer*8
     $	    j, i, p, ilast

c	----------------------------------------------------------------
c	read the pointers and check them
c	----------------------------------------------------------------

	call RBiread (ptrfmt, ncol+1, Ap, info)
	if (info .lt. 0) then
	    return
	endif

	if (Ap (1) .ne. 1 .or. Ap (ncol+1) - 1 .ne. nnz) then
c	    error: invalid matrix (col pointers)
	    info = -1
	    return
	endif
	do 10 j = 2, ncol+1
	    if (Ap (j) .lt. Ap (j-1)) then
c		error: invalid matrix (col pointers)
		info = -1
		return
	    endif
10	continue

c	----------------------------------------------------------------
c	read the row indices and check them
c	----------------------------------------------------------------

	call RBiread (indfmt, nnz, Ai, info)
	if (info .lt. 0) then
	    info = -93
	    return
	endif

	do 20 i = 1, nrow
	    w (i) = -1
20	continue

	do 40 j = 1, ncol
	    ilast = 0
	    do 30 p = Ap (j), Ap (j+1) - 1
		i = Ai (p)
		if (i .lt. 1 .or. i .gt. nrow) then
c		    error: row index out of range
c		    print *, 'column j, rows!', j, i, nrow
		    info = -2
		    return
		endif
		if (w (i) .eq. j) then
c		    error: duplicate entry in matrix
c		    print *, 'column j, duplicate!', j, i
		    info = -3
		    return
		endif
		w (i) = j
		if (i .lt. ilast) then
c		    error: matrix contains unsorted columns
c		    print *, 'column j, unsorted!', j, i, ilast
		    info = -7
		    return
		endif
		ilast = i
30	    continue
40	continue

c	----------------------------------------------------------------
c	construct new column pointers for symmetric matrices
c	----------------------------------------------------------------

	if (skind .gt. 0) then

c	    ------------------------------------------------------------
c	    compute the column counts for the whole matrix
c	    ------------------------------------------------------------

	    do 50 j = 1, ncol+1
		w (j) = 0
50	    continue

	    do 70 j = 1, ncol
		do 60 p = Ap (j), Ap (j+1)-1
		    i = Ai (p)
		    if (i .eq. j) then
c			diagonal entry, only appears as A(j,j)
			w (j) = w (j) + 1
		    elseif (i .gt. j) then
c			entry in lower triangular part, A(i,j) will be
c			duplicated as A(j,i), so count it in both cols
			w (i) = w (i) + 1
			w (j) = w (j) + 1
		    else
c			error: entry in upper triangular part
			info = -4
			return
		    endif
60		continue
70	    continue

c	    ------------------------------------------------------------
c	    compute the new column pointers
c	    ------------------------------------------------------------

	    cp (1) = 1
	    do 80 j = 2, ncol+1
		cp (j) = cp (j-1) + w (j-1)
80	    continue

	endif

c	----------------------------------------------------------------
c	matrix is valid
c	----------------------------------------------------------------

	info = 0
	return
	end


c-----------------------------------------------------------------------
c RBmangle: convert 1-based matrix into 0-based
c-----------------------------------------------------------------------

	subroutine RBmangle (ncol, Ap, Ai, nnz)
	integer*8
     $	    ncol, nnz, p, j, Ap (ncol+1), Ai (*)
	nnz = Ap (ncol + 1) - 1

	do 10 j = 1, ncol+1
	    Ap (j) = Ap (j) - 1
10	continue
	do 20 p = 1, nnz
	    Ai (p) = Ai (p) - 1
20	continue

	return
	end

c-----------------------------------------------------------------------

	subroutine RBiread (ifmt, n, I, info)
	integer*8
     $	    n, I (n), info, p
	character ifmt*16
	info = 0
	read (7, ifmt, err = 92, end = 92) (I (p), p = 1, n)
	return
92	info = -92
	return
	end

c-----------------------------------------------------------------------

	subroutine RBxread (xfmt, n, X, info)
	integer*8
     $	    mkind, n, info, p
	double precision X (n)
	character xfmt*20
	info = 0
	read (7, xfmt, err = 94, end = 94) (X (p), p = 1, n)
	return
94	info = -94
	return
	end


c-----------------------------------------------------------------------
c RBerr: report an error to MATLAB
c-----------------------------------------------------------------------
c
c   info = 0 is OK, info < 0 is a fatal error, info > 0 is a warning

	subroutine RBerr (info)
	integer*8
     $	    info

	if (info .eq. -7) then
	    call mexErrMsgTxt ('matrix contains unsorted columns')

	elseif (info .eq. -1) then
	    call mexErrMsgTxt ('invalid matrix (col pointers)')

	elseif (info .eq. -2) then
	    call mexErrMsgTxt ('row index out of range)')

	elseif (info .eq. -3) then
	    call mexErrMsgTxt ('duplicate entry in matrix')

	elseif (info .eq. -4) then
	    call mexErrMsgTxt ('invalid symmetric matrix')

	elseif (info .eq. -5) then
	    call mexErrMsgTxt ('invalid matrix type')

	elseif (info .eq. -6) then
	    call mexErrMsgTxt ('invalid matrix dimensions')

	elseif (info .eq. -911) then
	    call mexErrMsgTxt ('finite-element form not supported')

	elseif (info .eq. -91) then
	    call mexErrMsgTxt ('error reading file (header)')

	elseif (info .eq. -92) then
	    call mexErrMsgTxt ('error reading file (column pointers)')

	elseif (info .eq. -93) then
	    call mexErrMsgTxt ('error reading file (row indices)')

	elseif (info .eq. -94) then
	    call mexErrMsgTxt ('error reading file (numerical values)')

	elseif (info .eq. -95) then
	    call mexErrMsgTxt ('error reading file (right-hand-side)')

	elseif (info .lt. 0) then
	    print *, info
	    call mexErrMsgTxt ('error (unspecified)')

	elseif (info .gt. 0) then
	    print *, info
	    call mexErrMsgTxt ('warning (unspecified)')

	endif
	return
	end

