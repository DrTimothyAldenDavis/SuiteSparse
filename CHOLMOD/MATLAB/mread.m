function A = mread (filename)
%MREAD: read a sparse matrix from a file in Matrix Market "coord" format.
%   The Matrix Market dense matrix format ("array") is not supported.
%   Unlike MMREAD, only the matrix is returned; the file format is not
%   returned.
%
%   Several extensions to the Matrix Market format are provided.
%   The Matrix Market header line is optional.  If not present, the type
%   is inferred from the data.  A symmetric pattern-only matrix is returned
%   with A(i,i) = 1+length(find(A(:,i))) if it is present in the pattern,
%   and A(i,j) = -1 for off-diagonal entries.  If you want the original
%   Matrix Market matrix in this case, simply use A = spones (mmread (file)).
%
%   See also MMREAD (http://math.nist.gov/MatrixMarket)

%   Copyright 2006, Timothy A. Davis
%   http://www.cise.ufl.edu/research/sparse

error ('mread mexFunction not found') ;
