function [err, errnan] = gbtest_err (A, B)
%GBTEST_ERR compare two matrices
%
% err = gbtest_err (A, B)
%  
% Returns the norm (A-B,1), ignoring inf's and nan's.
% Also tests the result of isinf and isnan for A and B.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2020, All Rights
% Reserved. http://suitesparse.com.  See GraphBLAS/Doc/License.txt.

err = 0 ;
errnan = false ;

X = isnan (A) ;
Y = isnan (B) ;
if (~gbtest_eq (X, Y))
    errnan = true ;
end
if (nnz (X) > 0)
    A (X) = 0 ;
end
if (nnz (Y) > 0)
    B (Y) = 0 ;
end

X = isinf (A) ;
Y = isinf (B) ;
if (~gbtest_eq (X, Y))
    errnan = true ;
end
if (nnz (X) > 0)
    A (X) = 0 ;
end
if (nnz (Y) > 0)
    B (Y) = 0 ;
end

%{
if (~gbtest_eq (isfinite (A), isfinite (B)))
    errnan = true ;
end

if (isreal (A) ~= isreal (B))
    if (nnz (imag (A)) ~= nnz (imag (B)))
        err = 99 ;
    end
end
%}

%{
if (~isreal (A) || ~isreal (B))

    if (~gbtest_eq (isnan (real (A)), isnan (real (B))))
        % error ('isnan differs') ;
        err = 4 ;
    elseif (~gbtest_eq (isinf (real (A)), isinf (real (B))))
        % error ('isinf differs') ;
        err = 5 ;
    elseif (~gbtest_eq (isfinite (real (A)), isfinite (real (B))))
        % error ('isfinite differs') ;
        err = 6 ;
    end

    if (~gbtest_eq (isnan (imag (A)), isnan (imag (B))))
        % error ('isnan differs') ;
        err = 7 ;
    elseif (~gbtest_eq (isinf (imag (A)), isinf (imag (B))))
        % error ('isinf differs') ;
        err = 8 ;
    elseif (~gbtest_eq (isfinite (imag (A)), isfinite (imag (B))))
        % error ('isfinite differs') ;
        err = 9 ;
    end

end
%}

A (~isfinite (A)) = 0 ;
B (~isfinite (B)) = 0 ;
if (err == 0)
    err = GrB.normdiff (A, B, 1) ;
end

