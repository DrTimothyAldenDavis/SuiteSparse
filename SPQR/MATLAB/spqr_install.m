function spqr_install
%SPQR_INSTALL compile and install SuiteSparseQR
%
% Example:
%   spqr_install                        % compiles using METIS
%
% SuiteSparseQR relies on CHOLMOD, AMD, and COLAMD, and can optionally use
% CCOLAMD, CAMD, and METIS as well.  By default, CCOLAMD, CAMD, and METIS are
% used.  METIS is assumed to be in the ../../CHOLMOD/SuiteSparse_metis
% directory.  If not present there, it is not used.
%
% You can only use spqr_install while in the SuiteSparseQR/MATLAB directory.
%
% See also spqr, spqr_solve, spqr_qmult.

% SPQR, Copyright (c) 2008-2022, Timothy A Davis. All Rights Reserved.
% SPDX-License-Identifier: GPL-2.0+

if (nargin < 1)
    tbb = 0 ;
end

% compile SuiteSparseQR and add to the path
spqr_make ;
spqr_path = pwd ;
addpath (spqr_path)

fprintf ('\nThe following path has been added.  You may wish to add it\n') ;
fprintf ('permanently, using the MATLAB pathtool command.\n') ;
fprintf ('%s\n', spqr_path) ;

fprintf ('\nTo try your new mexFunctions, try this command:\n') ;
fprintf ('spqr_demo\n') ;


