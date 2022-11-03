function btf_test (nmat)
%BTF_TEST test for BTF
% Requires CSparse (or CXSparse) and ssget
% Example:
%   btf_test
% See also btf, maxtrans, strongcomp, dmperm, ssget,
%   test1, test2, test3, test4, test5, test6.

% BTF, Copyright (c) 2004-2022, University of Florida.  All Rights Reserved.
% Author: Timothy A. Davis.
% SPDX-License-Identifier: LGPL-2.1+

if (nargin < 1)
    nmat = 200 ;
end

test1 (nmat) ;
test2 (nmat) ;
test3 (nmat) ;
test4 (nmat) ;
test5 (nmat) ;
test6 ;

