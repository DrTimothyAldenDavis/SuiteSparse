c=======================================================================
c=== RBio/RBcsplit_64 ==================================================
c=======================================================================

c RBio: a MATLAB toolbox for reading and writing sparse matrices in
c Rutherford/Boeing format.
c Copyright (c) 2007, Timothy A. Davis, Univ. of Florida


c-----------------------------------------------------------------------
c RBcsplit: split a complex matrix into its real and imaginary parts
c-----------------------------------------------------------------------

	subroutine RBcsplit (Cx, Ax, Az, nnz)
	integer*8
     $	    nnz, i
	complex*16 Cx (*)
	double precision Ax (*), Az (*)
	do 10 i = 1, nnz
	    Ax (i) = dreal (Cx (i))
	    Az (i) = dimag (Cx (i))
10	continue
	return
	end

