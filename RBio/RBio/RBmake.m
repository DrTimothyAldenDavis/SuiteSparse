function RBmake
%RBMAKE compile the RBio toolbox for use in MATLAB
% Compiles the Fortran mexFunctions RBread, RBwrite, RBtype, and RBraw.
%
% Example:
%
%   RBmake
%
% See also RBread, RBwrite, RBtype, RBraw, RBinstall.
%
% Copyright 2009, Timothy A. Davis

if (~isempty (strfind (computer, '64')))
    try
        % try with -largeArrayDims (will fail on old MATLAB versions)
        mex -O -largeArrayDims RBread.c  RBerror.c ../Source/RBio.c ../../UFconfig/UFconfig.c -I../../UFconfig -I../Include
        mex -O -largeArrayDims RBwrite.c RBerror.c ../Source/RBio.c ../../UFconfig/UFconfig.c -I../../UFconfig -I../Include
        mex -O -largeArrayDims RBraw.c   RBerror.c ../Source/RBio.c ../../UFconfig/UFconfig.c -I../../UFconfig -I../Include
        mex -O -largeArrayDims RBtype.c  RBerror.c ../Source/RBio.c ../../UFconfig/UFconfig.c -I../../UFconfig -I../Include
    catch %#ok<CTCH>
        % try without -largeArrayDims (will fail on recent MATLAB versions)
        mex -O RBread.c  RBerror.c ../Source/RBio.c ../../UFconfig/UFconfig.c -I../../UFconfig -I../Include
        mex -O RBwrite.c RBerror.c ../Source/RBio.c ../../UFconfig/UFconfig.c -I../../UFconfig -I../Include
        mex -O RBraw.c   RBerror.c ../Source/RBio.c ../../UFconfig/UFconfig.c -I../../UFconfig -I../Include
        mex -O RBtype.c  RBerror.c ../Source/RBio.c ../../UFconfig/UFconfig.c -I../../UFconfig -I../Include
    end
else
    mex -O RBread.c  RBerror.c ../Source/RBio.c ../../UFconfig/UFconfig.c -I../../UFconfig -I../Include
    mex -O RBwrite.c RBerror.c ../Source/RBio.c ../../UFconfig/UFconfig.c -I../../UFconfig -I../Include
    mex -O RBraw.c   RBerror.c ../Source/RBio.c ../../UFconfig/UFconfig.c -I../../UFconfig -I../Include
    mex -O RBtype.c  RBerror.c ../Source/RBio.c ../../UFconfig/UFconfig.c -I../../UFconfig -I../Include
end

fprintf ('RBio successfully compiled.\n') ;
