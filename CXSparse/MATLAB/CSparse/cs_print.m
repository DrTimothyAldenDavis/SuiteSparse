function cs_print (A,brief)                                                 %#ok
%CS_PRINT print the contents of a sparse matrix.
%   cs_print(A) prints a sparse matrix. cs_print(A,1) prints just a few entries.
%
%   Example:
%       Prob = ssget ('vanHeukelum/cage3') ; A = Prob.A
%       cs_print (A) ;
%
%   See also: DISPLAY.

% CXSparse, Copyright (c) 2006-2022, Timothy A. Davis. All Rights Reserved.
% SPDX-License-Identifier: LGPL-2.1+

error ('cs_print mexFunction not found') ;
