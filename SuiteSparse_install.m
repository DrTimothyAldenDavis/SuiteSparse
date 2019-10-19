function SuiteSparse_install (do_demo)
%SuiteSparse_install: compiles and installs all of SuiteSparse
% A Suite of Sparse matrix packages, authored or co-authored by Tim Davis, Univ.
% Florida. You must be in the same directory as SuiteSparse_install to use this.
%
% Packages in SuiteSparse:
%
% UMFPACK        sparse LU factorization (multifrontal)
% CHOLMOD        sparse Cholesky factorization, and many other operations
% AMD            sparse symmetric approximate minimum degree ordering
% COLAMD         sparse column approximate minimum degree ordering
% CAMD           constrained AMD
% CCOLAMD        constrained COLAMD
% CSparse        a Concise Sparse matrix package (32-bit/real only)
% CXSparse       extended version of CSparse (32-bit/64-bit/real/complex)
% UFget          interface to UF Sparse Matrix Collection (MATLAB 7.0 or later)
% KLU            sparse LU factorization (left-looking)
% BTF            permutation to block triangular form (like dmperm)
% LDL            sparse LDL' factorization
% UFcollection   tools for managing the UF Sparse Matrix Collection
% RBio           read/write Rutherford/Boeing files (requires Fortran compiler)
% SSMULT         sparse matrix times sparse matrix
% MESHND         2D and 3D regular mesh generation and nested dissection
% LINFACTOR      illustrates the use of LU and CHOL (MATLAB 7.3 or later)
% MATLAB_Tools   various simple m-files and demos
% SuiteSparseQR  sparse QR factorization
%
% CXSparse is installed in place of CSparse; cd to CSparse/MATLAB and type
% cs_install if you wish to use the latter.  Since Microsoft Windows does not
% support ANSI C99, CXSparse does not support complex matrices on Windows.
%
% Except where noted, all packages work on MATLAB 6.1 or later.  They have not
% been tested on earlier versions, but they might work there.  Please let me
% know if you try SuiteSparse on MATLAB 6.0 or earlier, whether it works or not.
%
% Example:
%    SuiteSparse_install
%    help SuiteSparse      % for more details
%
% See also AMD, COLAMD, CAMD, CCOLAMD, CHOLMOD, UMFPACK, CSPARSE, CXSPARSE,
%      UFget, RBio, UFcollection, KLU, BTF, MESHND, SSMULT, LINFACTOR,
%      SuiteSparse, SPQR, PATHTOOL, PATH.

% Copyright 1990-2008, Timothy A. Davis.
% http://www.cise.ufl.edu/research/sparse
% In collaboration with Patrick Amestoy, Yanqing Chen, Iain Duff, John Gilbert,
% Steve Hadfield, Bill Hager, Stefan Larimore, Esmond Ng, Eka Palamadai, and
% Siva Rajamanickam.

paths = { } ;
SuiteSparse = pwd ;

% add MATLAB_Tools to the path (for getversion)
cd ([SuiteSparse '/MATLAB_Tools']) ;
paths = add_to_path (paths, pwd) ;
cd (SuiteSparse) ;

% determine the MATLAB version (6.1, 6.5, 7.0, ...)
v = getversion ;
pc = ispc ;

% check if METIS 4.0.1 is present where it's supposed to be
have_metis = exist ('metis-4.0', 'dir') ;
if (~have_metis)
    fprintf ('SPQR, CHOLMOD, and KLU optionally use METIS 4.0.1.  Download\n') ;
    fprintf ('it from http://glaros.dtc.umn.edu/gkhome/views/metis\n');
    fprintf ('and place the metis-4.0 directory in this directory.\n') ;
    input ('or hit enter to continue without METIS: ', 's') ;
    fprintf ('Now compiling without METIS...\n\n') ;
end

% print the introduction
help SuiteSparse_install

fprintf ('MATLAB version %g (%s)\n', v, version) ;

% add SuiteSparse to the path
fprintf ('\nPlease wait while SuiteSparse is compiled and installed...\n') ;
paths = add_to_path (paths, SuiteSparse) ;

% compile and install UMFPACK
try
    cd ([SuiteSparse '/UMFPACK/MATLAB']) ;
    paths = add_to_path (paths, pwd) ;
    umfpack_make
catch                                                                       %#ok
    disp (lasterr) ;
    try
	fprintf ('Trying to install with lcc_lib/libmwlapack.lib instead\n') ;
	umfpack_make ('lcc_lib/libmwlapack.lib') ;
    catch                                                                   %#ok
        disp (lasterr) ;
	fprintf ('UMFPACK not installed\n') ;
    end
end

% compile and install CHOLMOD
try
    % determine whether or not to compile CHOLMOD
    cd ([SuiteSparse '/CHOLMOD/MATLAB']) ;
    paths = add_to_path (paths, pwd) ;
    if (have_metis)
       cholmod_make
    else
       cholmod_make ('no metis') ;
    end
catch                                                                       %#ok
    disp (lasterr) ;
    fprintf ('CHOLMOD not installed\n') ;
end

% compile and install AMD
try
    cd ([SuiteSparse '/AMD/MATLAB']) ;
    paths = add_to_path (paths, pwd) ;
    amd_make
catch                                                                       %#ok
    disp (lasterr) ;
    fprintf ('AMD not installed\n') ;
end

% compile and install COLAMD
try
    cd ([SuiteSparse '/COLAMD/MATLAB']) ;
    paths = add_to_path (paths, pwd) ;
    colamd_make
catch                                                                       %#ok
    disp (lasterr) ;
    fprintf ('COLAMD not installed\n') ;
end

% compile and install CCOLAMD
try
    cd ([SuiteSparse '/CCOLAMD/MATLAB']) ;
    paths = add_to_path (paths, pwd) ;
    ccolamd_make
catch                                                                       %#ok
    disp (lasterr) ;
    fprintf ('CCOLAMD not installed\n') ;
end

% compile and install CAMD
try
    cd ([SuiteSparse '/CAMD/MATLAB']) ;
    paths = add_to_path (paths, pwd) ;
    camd_make
catch                                                                       %#ok
    disp (lasterr) ;
    fprintf ('CAMD not installed\n') ;
end

% compile and install CXSparse and UFget
try
    cd ([SuiteSparse '/CXSparse/MATLAB/CSparse']) ;
    paths = add_to_path (paths, [SuiteSparse '/CXSparse/MATLAB/CSparse']) ;
    paths = add_to_path (paths, [SuiteSparse '/CXSparse/MATLAB/Demo']) ;
    if (v >= 7.0)
	paths = add_to_path (paths, [SuiteSparse '/CXSparse/MATLAB/UFget']) ;
	fprintf ('UFget installed successfully\n') ;
    else
	fprintf ('UFget skipped; requires MATLAB 7.0 or later\n') ;
    end
    if (pc)
	% Windows does not support ANSI C99 complex, which CXSparse requires
	fprintf ('Compiling CXSparse without complex support\n') ;
	cs_make (1, 0) ;
    else
	cs_make (1) ;
    end
catch                                                                       %#ok
    disp (lasterr) ;
    fprintf ('CXSparse not installed\n') ;
end

% compile and install LDL
try
    cd ([SuiteSparse '/LDL/MATLAB']) ;
    paths = add_to_path (paths, pwd) ;
    ldl_make
catch                                                                       %#ok
    disp (lasterr) ;
    fprintf ('LDL not installed\n') ;
end

% compile and install BTF
try
    cd ([SuiteSparse '/BTF/MATLAB']) ;
    paths = add_to_path (paths, pwd) ;
    btf_make
catch                                                                       %#ok
    disp (lasterr) ;
    fprintf ('BTF not installed\n') ;
end

% compile and install KLU
try
    cd ([SuiteSparse '/KLU/MATLAB']) ;
    paths = add_to_path (paths, pwd) ;
    klu_make (have_metis) ;
catch                                                                       %#ok
    disp (lasterr) ;
    fprintf ('KLU not installed\n') ;
end

% compile and install SSMULT
try
    cd ([SuiteSparse '/SSMULT']) ;
    paths = add_to_path (paths, pwd) ;
    ssmult_install (0) ;
catch                                                                       %#ok
    disp (lasterr) ;
    fprintf ('SSMULT not installed\n') ;
end

% compile and install UFcollection
try
    % do not try to compile with large-file I/O for MATLAB 6.5 or earlier
    cd ([SuiteSparse '/UFcollection']) ;
    paths = add_to_path (paths, pwd) ;
    UFcollection_install (v < 7.0) ;
catch                                                                       %#ok
    disp (lasterr) ;
    fprintf ('UFcollection not installed\n') ;
end

% install LINFACTOR, MESHND, MATLAB_Tools/*
try
    cd ([SuiteSparse '/MATLAB_Tools/Factorize']) ;
    paths = add_to_path (paths, pwd) ;
    cd ([SuiteSparse '/MESHND']) ;
    paths = add_to_path (paths, pwd) ;
    if (v > 7.2)
        % LINFACTOR requires MATLAB 7.3 or later
        cd ([SuiteSparse '/LINFACTOR']) ;
        paths = add_to_path (paths, pwd) ;
        fprintf ('LINFACTOR installed\n') ;
    end
    cd ([SuiteSparse '/MATLAB_Tools/find_components']) ;
    paths = add_to_path (paths, pwd) ;
    cd ([SuiteSparse '/MATLAB_Tools/GEE']) ;
    paths = add_to_path (paths, pwd) ;
    cd ([SuiteSparse '/MATLAB_Tools/shellgui']) ;
    paths = add_to_path (paths, pwd) ;
    cd ([SuiteSparse '/MATLAB_Tools/waitmex']) ;
    paths = add_to_path (paths, pwd) ;
    cd ([SuiteSparse '/MATLAB_Tools/spok']) ;
    paths = add_to_path (paths, pwd) ;
    mex spok.c spok_mex.c
    fprintf ('LINFACTOR, MESHND, MATLAB_Tools installed\n') ;
catch                                                                       %#ok
    disp (lasterr) ;
    fprintf ('LINFACTOR, MESHND, and/or MATLAB_Tools not installed\n') ;
end

% compile and install SuiteSparseQR
try
    if (pc)
        fprintf ('Note that SuiteSparseQR will not compile with the lcc\n') ;
        fprintf ('compiler provided with MATLAB on Windows\n') ;
    end
    cd ([SuiteSparse '/SPQR/MATLAB']) ;
    paths = add_to_path (paths, pwd) ;
    if (have_metis)
       spqr_make
    else
       spqr_make ('no metis') ;
    end
catch                                                                       %#ok
    disp (lasterr) ;                                                        %#ok
    fprintf ('SuiteSparseQR not installed\n') ;
end

% compile and install RBio (not on Windows ... no default Fortran compiler)
if (~pc)
    try
	cd ([SuiteSparse '/RBio']) ;
	RBmake
	paths = add_to_path (paths, pwd) ;
    catch                                                                   %#ok
	disp (lasterr) ;                                                    %#ok
	fprintf ('RBio not installed (Fortran compiler required).\n') ;
    end
end

% post-install wrapup

cd (SuiteSparse)
fprintf ('SuiteSparse is now installed.\n') ;

if (nargin < 1)
    % ask if demo should be run
    y = input ('Hit enter to run the SuiteSparse demo (or "n" to quit): ', 's') ;
    if (isempty (y))
        y = 'y' ;
    end
    do_demo = (y (1) ~= 'n') ;
end
if (do_demo)
    try
	SuiteSparse_demo ;
    catch                                                                   %#ok
        disp (lasterr) ;
	fprintf ('SuiteSparse demo failed\n') ;
    end
end

fprintf ('\nSuiteSparse installation is complete.  The following paths\n') ;
fprintf ('have been added for this session.  Use pathtool to add them\n') ;
fprintf ('permanently.  If you cannot save the new path because of file\n');
fprintf ('permissions, then add these commands to your startup.m file.\n') ;
fprintf ('Type "doc startup" and "doc pathtool" for more information.\n\n') ;
for k = 1:length (paths)
    fprintf ('addpath %s\n', paths {k}) ;
end
cd (SuiteSparse)

fprintf ('\nSuiteSparse for MATLAB %g installation complete\n', getversion) ;

%-------------------------------------------------------------------------------
function paths = add_to_path (paths, newpath)
% add a path
addpath (newpath) ;
paths = [paths { newpath } ] ;						    %#ok
