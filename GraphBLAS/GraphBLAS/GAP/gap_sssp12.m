function path_length = gap_sssp12 (source, A, delta)
%GAP_SSSP12 single source shortest path, via delta stepping, for GAP
%
% A is square, unsymmetric, int32, and stored by row.  It is assumed that all
% its explicit entries are > 0.  The method is based on LAGraph_sssp12.

%-------------------------------------------------------------------------------
% check inputs
%-------------------------------------------------------------------------------

if (~isequal (GrB.type (A), 'int32'))
    % FUTURE: allow for different types of A
    error ('A must be int32') ;
end

if (~GrB.isbyrow (A))
    % FUTURE: extend to handle A by column
    error ('A must be stored by row') ;
end

[m, n] = size (A) ;
if (m ~= n)
    error ('A must be square') ;
end

delta = int32 (delta) ;

%-------------------------------------------------------------------------------
% initializations
%-------------------------------------------------------------------------------

empty = GrB (1, n, 'int32', 'by row') ;

% tmasked: a sparse vector containing the path lengths currently being computed
tmasked = empty ;
tmasked (source) = 0 ;

% t (i) = path length from source to node i, as a dense vector
t = empty ;
t (:) = int32 (inf) ;
t (source) = 0 ;

% s = nodes found in this pass
s = empty ;
s (source) = true ;

% AL = entries in A that are <= delta
AL = GrB.select (A, '<=', delta) ;

% AH = entries in A that are > delta
AH = GrB.select (A, '>' , delta) ;
AH_nvals = GrB.entries (AH) ;

i = int32 (0) ;

desc_s = struct ('mask', 'structural') ;

inf32 = int32 (inf) ;
do_LT_first = true ;

%-------------------------------------------------------------------------------
% SSSP iterations
%-------------------------------------------------------------------------------

while (GrB.entries (tmasked) > 0)

    % tmasked = select (tmasked < (i+1)*delta)
    uBound = (i+1) * delta ;
    tmasked = GrB.select (tmasked, '<', uBound) ;

    %---------------------------------------------------------------------------
    % inner iterations
    %---------------------------------------------------------------------------

    while (GrB.entries (tmasked) > 0)

        % tReq = tmasked * AL, using the min.plus semiring
        tReq = GrB.mxm (tmasked, 'min.+', AL) ;

        % s = s | spones (tmasked)
        s = GrB.eadd (s, 'pair.logical', tmasked) ;

        if (GrB.entries (tReq) == 0)
            % if tReq has no entries, no need to continue
            break ;
        end

        % tless = (tReq .< t), and drop zeros so it can be a structural mask
        tless = GrB.prune (GrB.emult (tReq, '<', t)) ;
        if (GrB.entries (tless) == 0)
            % if tless has no entries, no need to continue
            break ;
        end

        % tmasked<tless> = select (tReq < (i+1)*delta)
        tmasked = GrB.select (empty, tless, tReq, '<', uBound, desc_s) ;

        % t<tless> = tReq
        t = GrB.assign (t, tless, tReq, desc_s) ;
    end

    %---------------------------------------------------------------------------
    % next outer iteration
    %---------------------------------------------------------------------------

    if (AH_nvals > 0)

        % tmasked<s> = t
        tmasked = GrB.assign (empty, s, t, desc_s) ;

        % tReq = tmasked * AH using the min.plus semiring
        tReq = GrB.mxm (tmasked, 'min.+', AH) ;

        % tless = (tReq .< t)
        tless = GrB.emult (tReq, '<', t) ;

        % t<tless> = tReq
        t = GrB.assign (t, tless, tReq) ;
    end

    % prepare for next set of inner iterations
    i = i + 1 ;
    lBound = i * delta ;

    % tmasked = select (lBound <= t < inf)
    if (do_LT_first)
        tmasked = GrB.select (t, '<', inf32) ;
        n1 = GrB.entries (tmasked) ;
        tmasked = GrB.select (tmasked, '>=', lBound) ;
        if ((n-n1) < (n1-GrB.entries (tmasked)))
            % reverse the order for future iterations
            do_LT_first = false ;
        end
    else
        tmasked = GrB.select (t, '>=', lBound) ;
        tmasked = GrB.select (tmasked, '<', inf32) ;
    end

    % clear s for the next set of inner iterations
    s = empty ;
end

% return result
path_length = t ;

