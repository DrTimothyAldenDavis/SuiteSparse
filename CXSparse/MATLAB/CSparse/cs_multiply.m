function C = cs_multiply (A,B)						    %#ok
%CS_MULTIPLY sparse matrix multiply.
%   C = cs_multiply(A,B) computes C = A*B.
%
%   Example:
%       Prob1 = UFget ('HB/ibm32') ;        A = Prob1.A ;
%       Prob2 = UFget ('Hamrle/Hamrle1') ;  B = Prob2.A ;
%       C = cs_multiply (A,B) ;
%       D = A*B ;                           % same as C
%
%   See also CS_GAXPY, CS_ADD, MTIMES.

%   Copyright 2006-2007, Timothy A. Davis.
%   http://www.cise.ufl.edu/research/sparse

error ('cs_mult mexFunction not found') ;
