function d = get_diag (A)
%GET_DIAG extracts the diagonal of a matrix or vector.

% Factorize, Copyright (c) 2011-2012, Timothy A Davis. All Rights Reserved.
% SPDX-License-Identifier: BSD-3-clause

if (isempty (A))
    d = [ ] ;
elseif (isvector (A))
    % diag (A) would return a matrix for this case, which we do not want.
    d = A (1) ;
else
    d = diag (A) ;
end
d = full (d) ;
% assert (isvector (d)) ;
