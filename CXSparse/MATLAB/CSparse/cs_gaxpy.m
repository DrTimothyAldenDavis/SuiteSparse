function z = cs_gaxpy (A,x,y)                                               %#ok
%CS_GAXPY sparse matrix times vector.
%   z = cs_gaxpy(A,x,y) computes z = A*x+y where x and y are full vectors.
%
%   Example:
%       Prob = ssget ('HB/arc130') ; A = Prob.A ; [m n] = size (A) ;
%       x = rand (n,1) ; y = rand (m,1) ;
%       z = cs_gaxpy (A, x, y) ;
%       
%   See also PLUS, MTIMES.

% CXSparse, Copyright (c) 2006-2022, Timothy A. Davis. All Rights Reserved.
% SPDX-License-Identifier: LGPL-2.1+

error ('cs_gaxpy mexFunction not found') ;
