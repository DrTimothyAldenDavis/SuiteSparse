function cs_print (A,brief)                                                 %#ok
%CS_PRINT print the contents of a sparse matrix.
%   cs_print(A) prints a sparse matrix. cs_print(A,1) prints just a few entries.
%
%   Example:
%       Prob = UFget ('vanHeukelum/cage3') ; A = Prob.A
%       cs_print (A) ;
%
%   See also: DISPLAY.

%   Copyright 2006-2007, Timothy A. Davis.
%   http://www.cise.ufl.edu/research/sparse

error ('cs_print mexFunction not found') ;
