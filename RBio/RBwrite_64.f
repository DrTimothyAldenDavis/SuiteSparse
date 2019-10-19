c=======================================================================
c=== RBio/RBwrite_64 ===================================================
c=======================================================================

c RBio: a MATLAB toolbox for reading and writing sparse matrices in
c Rutherford/Boeing format.
c Copyright (c) 2007, Timothy A. Davis, Univ. of Florida


c-----------------------------------------------------------------------
c RBkind: determine the type of a MATLAB matrix
c-----------------------------------------------------------------------
c
c   input: a zero-based MATLAB sparse matrix
c
c	nrow	number of rows of A
c	ncol	number of columns of A
c	Ap	size ncol+1, column pointers
c	Ai	size nnz, row indices (nnz = Ap (ncol+1))
c	Ax	size nnz, real values
c	Az	size nnz, imaginary values (not accessed if A is real)
c	cmplex	1 if A is complex, 0 otherwise
c
c   output:
c	mkind:	r: 0 (real), p: 1 (pattern), c: 2 (complex),
c		i: 3 (integer)
c	skind:  r: -1 (rectangular), u: 0 (unsymmetric), s: 1 symmetric,
c		h: 2 (Hermitian), z: 3 (skew symmetric)
c
c   workspace:
c	munch	size ncol+1, not defined on input or output
c
c   Note that the MATLAB matrix is zero-based (Ap and Ai).  1 must be
c   added whenever they are used (see "1+" in the code below).
c
c   See also SuiteSparse/CHOLMOD/MatrixOps/cholmod_symmetry.c, which
c   also determines if the diagonal is positive.
c-----------------------------------------------------------------------

	subroutine RBkind (nrow, ncol, Ap, Ai, Ax, Az,
     $	    cmplex, mkind, skind, mtype, nnz, munch, kmin, kmax)

	integer*8
     $	    nrow, ncol, Ap (ncol+1), Ai (*), mkind, skind,
     $	    munch (ncol+1), p, i, j, pt, nnz, k, kmin, kmax
	integer*4 cmplex
	double precision Ax (*), Az (*), x_r, x_i, xtr, xti
	logical is_p, is_s, is_h, is_z, is_int
	character mtype*3

c	----------------------------------------------------------------
c	determine numeric type (I*A, R*A, P*A, C*A)
c	----------------------------------------------------------------

c	pattern: if real and all entries are 1.
c	integer: if real and all entries are integers.
c	complex: if cmplex is 1.
c	real: otherwise.

	nnz = 1+ (Ap (ncol+1) - 1)
	kmin = 0
	kmax = 0

	if (cmplex .eq. 1) then
c	    complex matrix (C*A)
	    mtype (1:1) = 'c'
	    mkind = 2
	else
c	    select P** format if all entries are equal to 1
c	    select I** format if all entries are integer and
c	    between -99,999,999 and +999,999,999
	    is_p = .true.
	    is_int = .true.
	    k = dint (Ax (1))
	    kmin = k
	    kmax = k
	    do 10 p = 1, nnz
		if (Ax (p) .ne. 1) then
		    is_p = .false.
		endif
		k = dint (Ax (p))
		kmin = min (kmin, k)
		kmax = max (kmax, k)
		if (k .ne. Ax (p)) then
		    is_int = .false.
		endif
		if (k .le. -99999999 .or. k .ge. 999999999) then
c		    use real format for really big integers
		    is_int = .false.
		endif
		if (.not. is_int .and. .not. is_p) then
		    goto 20
		endif
10		continue
20	    continue
	    if (is_p) then
c		pattern-only matrix (P*A)
		mtype (1:1) = 'p'
		mkind = 1
	    elseif (is_int) then
c		integer matrix (I*A)
		mtype (1:1) = 'i'
		mkind = 3
	    else
c		real matrix (R*A)
		mtype (1:1) = 'r'
		mkind = 0
	    endif
	endif

c	only assembled matrices are handled
	mtype (3:3) = 'a'

c	----------------------------------------------------------------
c	determine symmetry (*RA, *UA, *SA, *HA, *ZA)
c	----------------------------------------------------------------

c	Note that A must have sorted columns for this method to work.
c	This is not checked, since all MATLAB matrices "should" have
c	sorted columns.  Use spcheck(A) to check for this, if needed.

	if (nrow .ne. ncol) then
c	    rectangular matrix (*RA), no need to check values or pattern
	    mtype (2:2) = 'r'
	    skind = -1
	    return
	endif

c	if complex, the matrix is Hermitian until proven otherwise
	is_h = (cmplex .eq. 1)

c	the matrix is symmetric until proven otherwise
	is_s = .true.

c	a non-pattern matrix is skew symmetric until proven otherwise
	is_z = (mkind .ne. 1)

c	if this method returns early, the matrix is unsymmetric
	mtype (2:2) = 'u'
	skind = 0

c	initialize the munch pointers
	do 30 j = 1, ncol
	    munch (j) = 1+ (Ap (j))
30	continue

	do 50 j = 1, ncol

c	    consider all entries not yet munched in column j
	    do 40 p = munch (j), 1+ (Ap (j+1)-1)

		i = 1+ (Ai (p))

		if (i .lt. j) then
c		    entry A(i,j) is unmatched, matrix is unsymmetric
		    return
		endif

c		get the A(j,i) entry, if it exists
		pt = munch (i)

c		munch the A(j,i) entry
		munch (i) = pt + 1

		if (pt .ge. 1+ (Ap (i+1))) then
c		    entry A(j,i) doesn't exist, matrix unsymmetric
		    return
		endif

		if (1+ (Ai (pt)) .ne. j) then
c		    entry A(j,i) doesn't exist, matrix unsymmetric
		    return
		endif

c		A(j,i) exists; check its value with A(i,j)

		if (cmplex .eq. 1) then

c		    get A(i,j)
		    x_r = Ax (p)
		    x_i = Az (p)
c		    get A(j,i)
		    xtr = Ax (pt)
		    xti = Az (pt)
		    if (x_r .ne. xtr .or. x_i .ne. xti) then
c			the matrix cannot be *SA
			is_s = .false.
		    endif
		    if (x_r .ne. -xtr .or. x_i .ne. -xti) then
c			the matrix cannot be *ZA
			is_z = .false.
		    endif
		    if (x_r .ne. xtr .or. x_i .ne. -xti) then
c			the matrix cannot be *HA
			is_h = .false.
		    endif

		else

c		    get A(i,j)
		    x_r = Ax (p)
c		    get A(j,i)
		    xtr = Ax (pt)
		    if (x_r .ne. xtr) then
c			the matrix cannot be *SA
			is_s = .false.
		    endif
		    if (x_r .ne. -xtr) then
c			the matrix cannot be *ZA
			is_z = .false.
		    endif

		endif

		if (.not. (is_s .or. is_z .or. is_h)) then
c		    matrix is unsymmetric; terminate the test
		    return
		endif

40	    continue
50	continue

c	----------------------------------------------------------------
c	return the symmetry
c	----------------------------------------------------------------

	if (is_h) then
c	    Hermitian matrix (*HA)
	    mtype (2:2) = 'h'
	    skind = 2
	elseif (is_s) then
c	    symmetric matrix (*SA)
	    mtype (2:2) = 's'
	    skind = 1
	elseif (is_z) then
c	    skew symmetric matrix (*ZA)
	    mtype (2:2) = 'z'
	    skind = 3
	endif

	return
	end


c-----------------------------------------------------------------------
c RBformat: determine the format required for an array of values
c-----------------------------------------------------------------------
c
c This function ensures that a sufficiently wide format is used that
c can accurately represent the data.  It also ensures that when printed,
c the numerical values all have at least one blank space between them.
c This makes it trivial for a program written in C (say) to read in a
c matrix generated by RBwrite.

c ww, valfmt, valn, and is_int must be defined on input.  They
c are modified on output.
c-----------------------------------------------------------------------

	subroutine RBformat (nnz, x, ww, valfmt, valn, is_int,
     $	    kmin, kmax)
	integer*8
     $	    nnz, i, ww, k, nf (18), valn, nd (9), kmin, kmax
	double precision x (nnz), e, a, b
	logical is_int
	character*20 f (18), d (9), valfmt
	character*80 s

c	----------------------------------------------------------------
c	define all possible formats
c	----------------------------------------------------------------

	f (1)  = '(8E9.1)             '
	f (2)  = '(8E10.2)            '
	f (3)  = '(7E11.3)            '
	f (4)  = '(6E12.4)            '
	f (5)  = '(6E13.5)            '
	f (6)  = '(5E14.6)            '
	f (7)  = '(5E15.7)            '
	f (8)  = '(5E16.8)            '
	f (9)  = '(4E17.9)            '
	f (10) = '(4E18.10)           '
	f (11) = '(4E19.11)           '
	f (12) = '(4E20.12)           '
	f (13) = '(3E21.13)           '
	f (14) = '(3E22.14)           '
	f (15) = '(3E23.15)           '
	f (16) = '(3E24.16)           '
	f (17) = '(3E25.17)           '
	f (18) = '(3E26.18)           '

	nf (1)  = 8
	nf (2)  = 8
	nf (3)  = 7
	nf (4)  = 6
	nf (5)  = 6
	nf (6)  = 5
	nf (7)  = 5
	nf (8)  = 5
	nf (9)  = 4
	nf (10) = 4
	nf (11) = 4
	nf (12) = 4
	nf (13) = 3
	nf (14) = 3
	nf (15) = 3
	nf (16) = 3
	nf (17) = 3
	nf (18) = 3

	if (is_int) then

c	    ------------------------------------------------------------
c	    use an integer format
c	    ------------------------------------------------------------

	    call RBiformat (kmin, kmax, valfmt, valn, ww)

	else

c	    ------------------------------------------------------------
c	    determine if the matrix has huge values or NaN's
c	    ------------------------------------------------------------

	    do 10 i = 1, nnz
		a = abs (x (i))
		if (a .ne. 0) then
		    if (a .ne. a .or. a < 1d-90 .or. a > 1d90) then
			ww = 18
			valfmt = '(2E30.18E3)         '
			valn = 2
			return
		    endif
		endif
10	    continue

c	    ------------------------------------------------------------
c	    find the required precision for a real or complex matrix
c	    ------------------------------------------------------------

	    do 20 i = 1, nnz
		a = x (i)
		do 30 k = ww,18
c		    write the value to a string, read back in, and check
		    write (unit = s, fmt = f(k)) a
		    read  (unit = s, fmt = f(k)) b
		    if (a .eq. b) then
			ww = max (ww, k)
			goto 40
		    endif
30		continue
40		continue
		ww = max (ww, k)
20	    continue

c	    valn is the number of entries per line
	    valfmt = f (ww)
	    valn = nf (ww)

	endif

	return
	end


c-----------------------------------------------------------------------
c RBwrite: write portions of the matrix to the file
c-----------------------------------------------------------------------
c
c   task 0: just count the total number of entries in the matrix
c   task 1: do task 0, and also construct w and cp
c   task 2: write the row indices
c   task 3: write the numerical values
c
c   Note that the MATLAB arrays A and Z are zero-based.  "1+" is added
c   to each use of Ap, Ai, Zp, and Zi.
c-----------------------------------------------------------------------

	subroutine RBwrite (task, nrow, ncol, skind, cmplex, doZ, Ap,
     $	    Ai, Ax, Az, Zp, Zi, mkind,
     $	    indfmt, indn, valfmt, valn, nnz, w, cp)

	integer*8
     $	    task, nrow, ncol, Ap (*), Ai (*), Zp (*), Zi (*),
     $	    cp (*), w (*), nnz, znz, ibuf (80), j, i, nbuf, pa, pz,
     $	    paend, pzend, ia, iz, skind, indn, valn, p, mkind
	integer*4 cmplex
	logical doZ
	double precision xbuf (80), xr, xi, Ax (*), Az (*)
	character valfmt*20, indfmt*20

c	----------------------------------------------------------------
c	determine number of entries in Z
c	----------------------------------------------------------------

	if (doZ) then
	    znz = 1+ (Zp (ncol+1) - 1)
	else
	    znz = 0
	endif

c	clear the nonzero counts
	nnz = 0
	do 10 j = 1, ncol
	    w (j) = 0
10	continue

c	start with an empty buffer
	nbuf = 0

	if (znz .eq. 0) then

c	    ------------------------------------------------------------
c	    no Z present
c	    ------------------------------------------------------------

	    do 30 j = 1, ncol

		do 20 pa = 1+ (Ap (j)), 1+ (Ap (j+1) - 1)

		    i = 1+ (Ai (pa))
		    xr = Ax (pa)
		    if (cmplex .eq. 1) then
			xi = Az (pa)
		    endif

		    if (skind .le. 0 .or. i .ge. j) then

c			consider the (i,j) entry with value (xr,xi)
			nnz = nnz + 1
			if (task .eq. 1) then
c			    only determining nonzero counts
			    w (j) = w (j) + 1
			elseif (task .eq. 2) then
c			    printing the row indices
			    call RBiprint (indfmt, ibuf, nbuf, i, indn)
			elseif (task .eq. 3) then
c			    printing the numerical values
			    call RBxprint (valfmt, xbuf, nbuf, xr,
     $				    valn, mkind)
			    if (cmplex .eq. 1) then
				call RBxprint (valfmt, xbuf, nbuf, xi,
     $				    valn, mkind)
			    endif
			endif

		    endif

20		continue
30	    continue

	else

c	    ------------------------------------------------------------
c	    symmetric, unsymmetric or rectangular matrix, with Z present
c	    ------------------------------------------------------------

	    do 40 j = 1, ncol

c		find the set union of A (:,j) and Z (:,j)

		pa = 1+ (Ap (j))
		pz = 1+ (Zp (j))
		paend = 1+ (Ap (j+1) - 1)
		pzend = 1+ (Zp (j+1) - 1)

c		while entries appear in A or Z
70		continue

c		    get the next entry from A(:,j)
		    if (pa .le. paend) then
			ia = 1+ (Ai (pa))
		    else
			ia = 1+ nrow
		    endif

c		    get the next entry from Z(:,j)
		    if (pz .le. pzend) then
			iz = 1+ (Zi (pz))
		    else
			iz = 1+ nrow
		    endif

c		    exit loop if neither entry is present
		    if (ia .gt. nrow .and. iz .gt. nrow) goto 80

		    if (ia .lt. iz) then
c			get A (i,j)
			i = ia
			xr = Ax (pa)
			if (cmplex .eq. 1) then
			    xi = Az (pa)
			endif
			pa = pa + 1
		    else if (iz .lt. ia) then
c			get Z (i,j)
			i = iz
			xr = 0
			xi = 0
			pz = pz + 1
		    else
c			get A (i,j), and delete its matched Z(i,j)
			i = ia
			xr = Ax (pa)
			if (cmplex .eq. 1) then
			    xi = Az (pa)
			endif
			pa = pa + 1
			pz = pz + 1
		    endif

		    if (skind .le. 0 .or. i .ge. j) then

c			consider the (i,j) entry with value (xr,xi)
			nnz = nnz + 1
			if (task .eq. 1) then
c			    only determining nonzero counts
			    w (j) = w (j) + 1
			elseif (task .eq. 2) then
c			    printing the row indices
			    call RBiprint (indfmt, ibuf, nbuf, i, indn)
			elseif (task .eq. 3) then
c			    printing the numerical values
			    call RBxprint (valfmt, xbuf, nbuf, xr,
     $				    valn, mkind)
			    if (cmplex .eq. 1) then
				call RBxprint (valfmt, xbuf, nbuf, xi,
     $				    valn, mkind)
			    endif
			endif

		    endif

		    goto 70

c		end of while loop
80		continue

40	    continue

	endif

c	----------------------------------------------------------------
c	determine the new column pointers, or finish printing
c	----------------------------------------------------------------

	if (task .eq. 1) then

	    cp (1) = 1
	    do 100 j = 2, ncol+1
		cp (j) = cp (j-1) + w (j-1)
100	    continue

	else if (task .eq. 2) then

	    call RBiflush (indfmt, ibuf, nbuf)

	elseif (task .eq. 3) then

	    call RBxflush (valfmt, xbuf, nbuf, mkind)

	endif

	return
	end


c-----------------------------------------------------------------------
c RBiprint: print a single integer to the file, flush buffer if needed
c-----------------------------------------------------------------------

	subroutine RBiprint (indfmt, ibuf, nbuf, i, indn)
	character indfmt*20
	integer*8
     $	    ibuf (80), nbuf, i, indn
	if (nbuf .ge. indn) then
	    call RBiflush (indfmt, ibuf, nbuf)
	    nbuf = 0
	endif
	nbuf = nbuf + 1
	ibuf (nbuf) = i
	return
	end


c-----------------------------------------------------------------------
c RBiflush: flush the integer buffer to the file
c-----------------------------------------------------------------------

	subroutine RBiflush (indfmt, ibuf, nbuf)
	character indfmt*20
	integer*8
     $	    ibuf (*), nbuf, k
	write (unit = 7, fmt = indfmt, err = 999) (ibuf (k), k = 1,nbuf)
	return
999	call mexErrMsgTxt ('error writing ints')
	return
	end


c-----------------------------------------------------------------------
c RBxprint: print a single real to the file, flush the buffer if needed
c-----------------------------------------------------------------------

	subroutine RBxprint (valfmt, xbuf, nbuf, x, valn, mkind)
	character valfmt*20
	integer*8
     $	    nbuf, valn, mkind
	double precision xbuf (80), x
	if (nbuf .ge. valn) then
	    call RBxflush (valfmt, xbuf, nbuf, mkind)
	    nbuf = 0
	endif
	nbuf = nbuf + 1
	xbuf (nbuf) = x
	return
	end


c-----------------------------------------------------------------------
c RBxflush: flush the real buffer to the file
c-----------------------------------------------------------------------

	subroutine RBxflush (valfmt, xbuf, nbuf, mkind)
	character valfmt*20
	integer*8
     $	    nbuf, k, ibuf (80), mkind
	double precision xbuf (80)
	if (mkind .eq. 3) then
c	    convert to integer first; valfmt is (10I8), for example
	    do 10 k = 1,nbuf
		ibuf (k) = dint (xbuf (k))
10	    continue
	    write (unit = 7, fmt = valfmt, err = 999)
     $		(ibuf (k), k = 1,nbuf)
	else
	    write (unit = 7, fmt = valfmt, err = 999)
     $		(xbuf (k), k = 1,nbuf)
	endif
	return
999	call mexErrMsgTxt ('error writing numerical values')
	return
	end


c-----------------------------------------------------------------------
c RBiformat: determine format for printing an integer
c-----------------------------------------------------------------------

	subroutine RBiformat (kmin, kmax, indfmt, indn, ww)
	integer*8
     $	    n, indn, kmin, kmax, ww
	character*20 indfmt

	if (kmin .ge. 0. and. kmax .le. 9) then
	    indfmt = '(40I2)              '
	    ww = 2
	    indn = 40
	elseif (kmin .ge. -9 .and. kmax .le. 99) then
	    indfmt = '(26I3)              '
	    ww = 3
	    indn = 26
	elseif (kmin .ge. -99 .and. kmax .le. 999) then
	    indfmt = '(20I4)              '
	    ww = 4
	    indn = 20
	elseif (kmin .ge. -999 .and. kmax .le. 9999) then
	    indfmt = '(16I5)              '
	    ww = 5
	    indn = 16
	elseif (kmin .ge. -9999 .and. kmax .le. 99999) then
	    indfmt = '(13I6)              '
	    ww = 6
	    indn = 13
	elseif (kmin .ge. -99999 .and. kmax .le. 999999) then
	    indfmt = '(11I7)              '
	    ww = 7
	    indn = 11
	elseif (kmin .ge. -999999 .and. kmax .le. 9999999) then
	    indfmt = '(10I8)              '
	    ww = 8
	    indn = 10
	elseif (kmin .ge. -9999999 .and. kmax .le. 99999999) then
	    indfmt = '(8I9)               '
	    ww = 9
	    indn = 8
	elseif (kmin .ge. -99999999 .and. kmax .le. 999999999) then
	    indfmt = '(8I10)               '
	    ww = 10
	    indn = 8
	else
	    indfmt = '(5I15)               '
	    ww = 15
	    indn = 5
	endif
	return
	end


c-----------------------------------------------------------------------
c RBcards: determine number of cards required
c-----------------------------------------------------------------------

	subroutine RBcards (nitems, nperline, ncards)
	integer*8
     $	    nitems, nperline, ncards
	if (nitems .eq. 0) then
	    ncards = 0
	else
	    ncards = ((nitems-1) / nperline) + 1
	endif
	return
	end

