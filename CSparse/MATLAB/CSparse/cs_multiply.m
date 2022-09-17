function C = cs_multiply (A,B)                                              %#ok
%CS_MULTIPLY sparse matrix multiply.
%   C = cs_multiply(A,B) computes C = A*B.
%
%   Example:
%       Prob1 = ssget ('HB/ibm32') ;        A = Prob1.A ;
%       Prob2 = ssget ('Hamrle/Hamrle1') ;  B = Prob2.A ;
%       C = cs_multiply (A,B) ;
%       D = A*B ;                           % same as C
%
%   See also CS_GAXPY, CS_ADD, MTIMES.

% CSparse, Copyright (c) 2006-2022, Timothy A. Davis. All Rights Reserved.
% SPDX-License-Identifier: LGPL-2.1+

error ('cs_mult mexFunction not found') ;
