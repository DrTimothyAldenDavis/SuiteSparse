c=======================================================================
c=== RBio/RBrread_64 ===================================================
c=======================================================================

c RBio: a MATLAB toolbox for reading and writing sparse matrices in
c Rutherford/Boeing format.
c Copyright (c) 2007, Timothy A. Davis, Univ. of Florida


c-----------------------------------------------------------------------
c RB*read:  read a Rutherford/Boeing matrix
c-----------------------------------------------------------------------

	subroutine RBrread
     $	    (nrow, ncol, nnz, ptrfmt, indfmt, valfmt,
     $	    mkind, skind, Ap, Ai, Ax, nzeros, w, cp, info, nw, nnz1)
	integer*8
     $	    nrow, ncol, nnz, mkind, skind, p, j, i, alen, llen, k,
     $	    ilast, nzeros, info, nw, nnz1
	integer*8
     $	    Ap (ncol+1), Ai (nnz1), w (nw), cp (ncol+1)
	double precision Ax (nnz1), x
	character ptrfmt*16, indfmt*16, valfmt*20
	character*4 s

c	----------------------------------------------------------------
c	read the column pointers and row indices
c	----------------------------------------------------------------

	call RBpattern (ptrfmt, indfmt, nrow, ncol, nnz, skind,
     $	    Ap, Ai, w, cp, info, nw)
	if (info .ne. 0) then
c	    error: pattern is invalid
	    return
	endif

c	----------------------------------------------------------------
c	read the values
c	----------------------------------------------------------------

	if (mkind .eq. 1) then

c	    pattern-only matrix, set all values to 1
	    do 10 i = 1, nnz
		Ax (i) = 1
10	    continue

	elseif (mkind .eq. 3) then

c	    Read in nnz integer values, then convert them to real.
c	    use Ai as workspace.  If the matrix is symmetric, Ai is
c	    twice as big as nnz, so use Ai (nnz+1...2*nnz) as workspace.
c	    Otherwise, for an iua matrix, use Ai (1..nnz) and then
c	    rewind the file and read Ap and Ai back in again.

	    if (skind .le. 0) then
		k = 0
	    else
		k = nnz
	    endif
	    read (7, valfmt, err = 94, end = 94) (Ai(p), p = k+1, k+nnz)
	    do 5 p = 1, nnz
		Ax (p) = Ai (p+k)
5	    continue
	    if (skind .le. 0) then
c		now that Ai has been destroyed, rewind the file and read
c		in Ap and Ai again.  Skip the 4-line header first.
		rewind (unit = 7)
		read (7, 15) (s (k:k), k = 1,4)
15		format (a1 / a1 / a1 / a1)
		call RBpattern (ptrfmt, indfmt, nrow, ncol, nnz, skind,
     $		    Ap, Ai, w, cp, info, nw)
		if (info .ne. 0) then
c		    error: pattern is invalid.  This 'cannot' happen,
c		    because the pattern was already read in above.
		    return
		endif
	    endif

	else

c	    read nnz values
	    read (7, valfmt, err = 94, end = 94) (Ax (p), p = 1, nnz)

	endif

c	----------------------------------------------------------------
c	construct upper triangular part for symmetric matrices
c	----------------------------------------------------------------

c	If skind is zero, then the upper part is not constructed.  This
c	allows the caller to skip this part, and create the upper part
c	of a symmetric (S,H,Z) matrix.  Just pass skind = 0.

	if (skind .gt. 0) then

c	    ------------------------------------------------------------
c	    shift the matrix by adding gaps to the top of each column
c	    ------------------------------------------------------------

	    do 30 j = ncol, 1, -1

c		number of entries in lower tri. part (incl. diagonal)
		llen = Ap (j+1) - Ap (j)

c		number of entries in entire column
		alen = cp (j+1) - cp (j)

c		move the column from Ai (Ap(j) ... Ap(j+1)-1)
c		down to Ai (cp(j+1)-llen ... cp(j+1)-1), leaving a gap
c		at Ai (Ap(j) ... cp(j+1)-llen)

		do 20 k = 1, llen
		    Ai (cp (j+1) - k) = Ai (Ap (j+1) - k)
		    Ax (cp (j+1) - k) = Ax (Ap (j+1) - k)
20		continue
30	    continue

c	    ------------------------------------------------------------
c	    populate the upper triangular part
c	    ------------------------------------------------------------

c	    create temporary column pointers to point to the gaps
	    do 40 j = 1, ncol
		w (j) = cp (j)
40	    continue

	    do 60 j = 1, ncol

c		scan the entries in the lower tri. part, in
c		Ai (cp(j+1)-llen ... cp(j+1)-1)
		llen = Ap (j+1) - Ap (j)
		do 50 k = 1, llen

c		    get the A(i,j) entry in the lower triangular part
		    i = Ai (cp (j+1) - k)
		    x = Ax (cp (j+1) - k)

c		    add A(j,i) as the next entry in column i (excl diag)
		    if (i .ne. j) then
			p = w (i)
			w (i) = w (i) + 1
			Ai (p) = j

			if (skind .eq. 1) then
c			    *SA matrix
			    Ax (p) = x
			elseif (skind .eq. 2) then
c			    *HA matrix
			    Ax (p) = x
			else
c			    *ZA matrix
			    Ax (p) = -x
			endif

		    endif

50		continue
60	    continue

c	    finalize the column pointers
	    do 70 j = 1, ncol+1
		Ap (j) = cp (j)
70	    continue

	endif

c	----------------------------------------------------------------
c	count the number of explicit zeros
c	----------------------------------------------------------------

	nzeros = 0
	do 90 j = 1, ncol
	    cp (j) = nzeros + 1
	    do 80 p = Ap (j), Ap (j+1)-1
		if (Ax (p) .eq. 0) then
		    nzeros = nzeros + 1
		endif
80	    continue
90	continue
	cp (ncol+1) = nzeros + 1

c	----------------------------------------------------------------
c	matrix is valid
c	----------------------------------------------------------------

	info = 0
	return

c	----------------------------------------------------------------
c	error return
c	----------------------------------------------------------------

94	info = -94
	return
	end


c-----------------------------------------------------------------------
c RB*zeros: extract explicit zero entries
c-----------------------------------------------------------------------
c
c   nrow-by-ncol: size of A and Z
c   cp: column pointers of Z on input
c   Ap, Ai, Ax: matrix with zeros on input, pruned on output
c   Zp, Zi, Zx: empty matrix on input, pattern of zeros on output
c
c-----------------------------------------------------------------------

	subroutine RBrzeros
     $	    (nrow, ncol, cp, Ap, Ai, Ax, Zp, Zi, Zx)
	integer*8
     $	    nrow, ncol, Ap (ncol+1), Ai (*), Zp (ncol+1), Zi (*),
     $	    cp (*), i, j, p, pa, pz, p1
	double precision Ax (*), x
	double precision Zx (*)

c	----------------------------------------------------------------
c	copy the column pointers if Z is being constructed
c	----------------------------------------------------------------

	do 10 j = 1, ncol+1
	    Zp (j) = cp (j)
10	continue

c	----------------------------------------------------------------
c	split the matrix
c	----------------------------------------------------------------

	pa = 1
	pz = 1
	do 30 j = 1, ncol
c	    save the new start of column j of A
	    p1 = Ap (j)
	    Ap (j) = pa
	    pz = Zp (j)
c	    split column j of A
	    do 20 p = p1, Ap (j+1)-1
		i = Ai (p)
		x = Ax (p)
		if (x .eq. 0) then
c		    copy into Z
		    Zi (pz) = i
		    Zx (pz) = 1
		    pz = pz + 1
		else
c		    copy into A
		    Ai (pa) = i
		    Ax (pa) = x
		    pa = pa + 1
		endif
20	    continue
30	continue
	Ap (ncol+1) = pa

	return
	end


c-----------------------------------------------------------------------
c RB*prune: discard explicit zero entries
c-----------------------------------------------------------------------
c
c   nrow-by-ncol: size of A
c   Ap, Ai, Ax: matrix with zeros on input, pruned on output
c
c-----------------------------------------------------------------------

	subroutine RBrprune
     $	    (nrow, ncol, Ap, Ai, Ax)
	integer*8
     $	    nrow, ncol, Ap (ncol+1), Ai (*), i, j, p, pa, pz, p1
	double precision Ax (*), x

c	----------------------------------------------------------------
c	prune the matrix
c	----------------------------------------------------------------

	pa = 1
	do 20 j = 1, ncol
c	    save the new start of column j of A
	    p1 = Ap (j)
	    Ap (j) = pa
c	    prune column j of A
	    do 10 p = p1, Ap (j+1)-1
		i = Ai (p)
		x = Ax (p)
		if (x .ne. 0) then
c		    copy into A
		    Ai (pa) = i
		    Ax (pa) = x
		    pa = pa + 1
		endif
10	    continue
20	continue
	Ap (ncol+1) = pa

	return
	end
