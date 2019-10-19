%RBMAKE compile the RBio toolbox for use in MATLAB
% Compiles the Fortran mexFunctions RBread, RBwrite, RBtype, and RBraw.
%
% Example:
%
%   RBmake
%
% See also RBread, RBwrite, RBtype, RBraw, RBinstall.
%
% Copyright 2007, Timothy A. Davis

if (~isempty (strfind (computer, '64')))
    fprintf ('Compiling 64-bit version of RBio.\n') ;
    try
        % try with -largeArrayDims (will fail on old MATLAB versions)
        mex -O -largeArrayDims -output RBread RBread_mex_64.f RBread_64.f ...
            RBrread_64.f RBcread_64.f RBcsplit_64.f
        mex -O -largeArrayDims -output RBtype RBtype_mex_64.f RBwrite_64.f
        mex -O -largeArrayDims -output RBwrite RBwrite_mex_64.f RBwrite_64.f
        mex -O -largeArrayDims -output RBraw RBraw_mex_64.f RBread_64.f
    catch
        % try without -largeArrayDims (will fail on recent MATLAB versions)
        mex -O -output RBread RBread_mex_64.f RBread_64.f ...
            RBrread_64.f RBcread_64.f RBcsplit_64.f
        mex -O -output RBtype RBtype_mex_64.f RBwrite_64.f
        mex -O -output RBwrite RBwrite_mex_64.f RBwrite_64.f
        mex -O -output RBraw RBraw_mex_64.f RBread_64.f
    end
else
    fprintf ('Compiling 32-bit version of RBio.\n') ;
    mex -O -output RBread RBread_mex_32.f RBread_32.f RBrread_32.f ...
	RBcread_32.f RBcsplit_32.f
    mex -O -output RBtype RBtype_mex_32.f RBwrite_32.f
    mex -O -output RBwrite RBwrite_mex_32.f RBwrite_32.f
    mex -O -output RBraw RBraw_mex_32.f RBread_32.f
end

fprintf ('Note: Fortran compiler is required; this will fail otherwise...\n') ;

fprintf ('RBio successfully compiled.\n') ;
