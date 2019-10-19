c=======================================================================
c=== RBio/RBint ========================================================
c=======================================================================

c RBio: a MATLAB toolbox for reading and writing sparse matrices in
c Rutherford/Boeing format.
c Copyright (c) 2006, Timothy A. Davis, Univ. Florida.  Version 1.0.


c-----------------------------------------------------------------------
c RBint: determine integer type used by MATLAB
c-----------------------------------------------------------------------

	subroutine RBint (class)
	integer class
	integer mxClassIDFromClassName
	class = mxClassIDFromClassName ('int32')
	return
	end

