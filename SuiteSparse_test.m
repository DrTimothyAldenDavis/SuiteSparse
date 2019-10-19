function SuiteSparse_test
% SuiteSparse_test exhaustive test of all SuiteSparse packages
%
% Your current directory must be SuiteSparse for this function to work.
% SuiteSparse_install must be run prior to running this test.  Warning:
% this test takes a *** long **** time.
%
% Example:
%   SuiteSparse_test
%
% See also SuiteSparse_install, SuiteSparse_demo.

% Copyright 2007, Tim Davis, University of Florida

npackages = 13 ;
h = waitbar (0, 'SuiteSparse test:') ;
SuiteSparse = pwd ;

v = getversion ;
if (v < 7)
    error ('SuiteSparse_test requires MATLAB 7.0 or later') ;
end

% if at UF, ensure pre-installed UF Sparse Matrix Collection is used
uf = { '/cise/homes/davis/Install/UFget', 'd:/UFget', '/share/UFget', ...
    '/windows/UFget' } ;
for k = 1:length(uf)
    if (exist (uf {k}, 'dir'))
        addpath (uf {k}) ;
        break ;
    end
end

try

    %---------------------------------------------------------------------------
    % CSparse (32-bit MATLAB only)
    %---------------------------------------------------------------------------

    if (isempty (strfind (computer, '64')))
        % compile and install CSparse (not installed by SuiteSparse_install)
        waitbar (1/(npackages+1), h, 'SuiteSparse test: CSparse') ;
        addpath ([SuiteSparse '/CSparse/MATLAB/CSparse']) ;
        addpath ([SuiteSparse '/CSparse/MATLAB/Demo']) ;
        cd ([SuiteSparse '/CSparse/MATLAB/CSparse']) ;
        cs_make ;

        % test CSparse
        cd ([SuiteSparse '/CSparse/MATLAB/Test']) ;
        testall ;

        % uninstall CSparse by removing it from path
        rmpath ([SuiteSparse '/CSparse/MATLAB/CSparse']) ;
        rmpath ([SuiteSparse '/CSparse/MATLAB/Demo']) ;
    end

    %---------------------------------------------------------------------------
    % CXSparse
    %---------------------------------------------------------------------------

    waitbar (2/(npackages+1), h, 'SuiteSparse test: CXSparse') ;
    cd ([SuiteSparse '/CXSparse/MATLAB/Test']) ;
    testall ;

    %---------------------------------------------------------------------------
    % COLAMD
    %---------------------------------------------------------------------------

    waitbar (3/(npackages+1), h, 'SuiteSparse test: COLAMD') ;
    cd ([SuiteSparse '/COLAMD/MATLAB']) ;
    colamd_test ;

    %---------------------------------------------------------------------------
    % CCOLAMD
    %---------------------------------------------------------------------------

    waitbar (4/(npackages+1), h, 'SuiteSparse test: CCOLAMD') ;
    cd ([SuiteSparse '/CCOLAMD/MATLAB']) ;
    ccolamd_test ;

    %---------------------------------------------------------------------------
    % UMFPACK
    %---------------------------------------------------------------------------

    waitbar (5/(npackages+1), h, 'SuiteSparse test: UMFPACK') ;
    cd ([SuiteSparse '/UMFPACK/MATLAB']) ;
    umfpack_test (800) ;

    %---------------------------------------------------------------------------
    % CHOLMOD
    %---------------------------------------------------------------------------

    waitbar (6/(npackages+1), h, 'SuiteSparse test: CHOLMOD') ;
    cd ([SuiteSparse '/CHOLMOD/MATLAB/Test']) ;
    cholmod_test ;

    %---------------------------------------------------------------------------
    % BTF
    %---------------------------------------------------------------------------

    waitbar (7/(npackages+1), h, 'SuiteSparse test: BTF') ;
    cd ([SuiteSparse '/BTF/MATLAB/Test']) ;
    btf_test ;

    %---------------------------------------------------------------------------
    % KLU
    %---------------------------------------------------------------------------

    waitbar (8/(npackages+1), h, 'SuiteSparse test: KLU') ;
    cd ([SuiteSparse '/KLU/MATLAB/Test']) ;
    klu_test ;

    %---------------------------------------------------------------------------
    % LDL
    %---------------------------------------------------------------------------

    waitbar (9/(npackages+1), h, 'SuiteSparse test: LDL') ;
    cd ([SuiteSparse '/LDL/MATLAB']) ;
    ldlmain2 ;
    ldltest ;

    %---------------------------------------------------------------------------
    % LINFACTOR:  MATLAB 7.3 (R2006b) or later required
    %---------------------------------------------------------------------------

    if (v > 7.2)
        waitbar (10/(npackages+1), h, 'SuiteSparse test: LINFACTOR') ;
        cd ([SuiteSparse '/LINFACTOR']) ;
        lintests ;
    end

    %---------------------------------------------------------------------------
    % MESHND
    %---------------------------------------------------------------------------

    waitbar (11/(npackages+1), h, 'SuiteSparse test: MESHND') ;
    cd ([SuiteSparse '/MESHND']) ;
    meshnd_quality ;

    %---------------------------------------------------------------------------
    % SSMULT
    %---------------------------------------------------------------------------

    waitbar (12/(npackages+1), h, 'SuiteSparse test: SSMULT') ;
    cd ([SuiteSparse '/SSMULT']) ;
    ssmult_test ;

    %---------------------------------------------------------------------------
    % MATLAB_Tools
    %---------------------------------------------------------------------------

    waitbar (13/(npackages+1), h, 'SuiteSparse test: MATLAB Tools') ;
    cd ([SuiteSparse '/MATLAB_Tools']) ;
    figure (1) ;
    clf
    pagerankdemo (1000) ;
    figure (2) ;
    clf
    seashell ;
    shellgui ;
    cd ([SuiteSparse '/MATLAB_Tools/waitmex']) ;
    waitmex ;
    url = 'http://www.cise.ufl.edu/research/sparse' ;
    fprintf ('<a href="%s">Click here for more details</a>\n', url) ;
    hprintf ('or see <a href="%s">\n', url) ;

    %---------------------------------------------------------------------------
    % AMD, CAMD, UFcollection, UFget
    %---------------------------------------------------------------------------

    % no exhaustive tests; tested via other packages

catch                                                                       %#ok

    %---------------------------------------------------------------------------
    % test failure
    %---------------------------------------------------------------------------

    cd (SuiteSparse) ;
    disp (lasterr) ;                                                        %#ok
    fprintf ('SuiteSparse test: FAILED\n') ;
    return

end

%-------------------------------------------------------------------------------
% test OK
%-------------------------------------------------------------------------------

close (h) ;
fprintf ('SuiteSparse test: OK\n') ;
cd (SuiteSparse) ;
