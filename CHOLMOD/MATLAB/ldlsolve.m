function x = ldlsolve (LD,b)						    %#ok
%LDLSOLVE solve LDL'x=b using a sparse LDL' factorization
%
%   Example:
%   x = ldlsolve (LD,b)
%
%   solves the system L*D*L'*x=b for x.  This is equivalent to
%
%   [L,D] = ldlsplit (LD) ;
%   x = L' \ (D \ (L \ b)) ;
%
%   LD is from ldlchol, or as updated by ldlupdate.  You must not modify LD as
%   obtained from ldlchol or ldlupdate prior to passing it to this function.
%   See ldlupdate for more details.
%
%   See also LDLCHOL, LDLUPDATE, LDLSPLIT

%   Copyright 2006-2007, Timothy A. Davis
%   http://www.cise.ufl.edu/research/sparse

error ('ldlsolve mexFunction not found') ;
