function private_stream = spqr_repeatable (seed)
%SPQR_REPEATABLE ensure repeatable results, or use the default random stream.
% Not user-callable.

% Copyright 2011, Leslie Foster and Timothy A Davis

if (seed == 1)
    % use a new strearm with the default random seed
    private_stream = RandStream ('mt19937ar') ;
elseif (seed > 1)
    % use a new stream with a specific random seed
    private_stream = RandStream ('mt19937ar', 'seed', seed) ;
else
    % do not use the private stream
    private_stream = [ ] ;
end

