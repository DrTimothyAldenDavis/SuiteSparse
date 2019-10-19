function resid (A, x, b)
% resid (A, x, b), print the relative residual,
% norm (A*x-b,inf) / (norm(A,1)*norm(x,inf) + norm(b,inf))

fprintf ('resid: %8.2e\n', ...
    norm (A*x-b,inf) / (norm(A,1)*norm(x,inf) + norm(b,inf))) ;
