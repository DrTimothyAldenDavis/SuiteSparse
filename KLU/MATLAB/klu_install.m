function klu_install (with_cholmod)
%KLU_INSTALL compiles and installs the KLU, BTF, AMD, and COLAMD mexFunctions
%
% Example:
%   klu_install         % compiles KLU without CHOLMOD
%   klu_install (1)     % with CHOLMOD, CCAMD, CCOLAMD, and METIS
%
% KLU relies on AMD, COLAMD, and BTF for its ordering options, and can
% optionally use CHOLMOD, CCOLAMD, CAMD, and METIS as well.  By default,
% CHOLMOD, CCOLAMD, CAMD, and METIS are not used.
%
% See http://www-users.cs.umn.edu/~karypis/metis for a copy of METIS 4.0.1.
%
% You must type the klu_install command while in the KLU/MATLAB directory.
%
% See also klu, btf

% Copyright 2004-2009 Timothy A. Davis, Univ. of Florida

if (nargin < 1)
    with_cholmod = 0 ;
end

% compile KLU and add to the path
klu_make (with_cholmod) ;
klu_path = pwd ;
addpath (klu_path)

fprintf ('\nNow compiling the AMD, COLAMD, and BTF mexFunctions:\n') ;

% compile BTF and add to the path
cd ../../BTF/MATLAB
btf_make
btf_path = pwd ;
addpath (btf_path)

% compile AMD and add to the path
cd ../../AMD/MATLAB
amd_make
amd_path = pwd ;
addpath (amd_path)

% compile COLAMD and add to the path
cd ../../COLAMD/MATLAB
colamd_make
colamd_path = pwd ;
addpath (colamd_path)

cd (klu_path)

fprintf ('\nThe following paths have been added.  You may wish to add them\n') ;
fprintf ('permanently, using the MATLAB pathtool command.\n') ;
fprintf ('%s\n', klu_path) ;
fprintf ('%s\n', amd_path) ;
fprintf ('%s\n', colamd_path) ;
fprintf ('%s\n', btf_path) ;

fprintf ('\nTo try your new mexFunctions, cut-and-paste this command:\n') ;
fprintf ('klu_demo, btf_demo, amd_demo, colamd_demo\n') ;
