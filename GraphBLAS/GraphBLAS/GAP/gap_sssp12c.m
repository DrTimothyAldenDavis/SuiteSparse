function path_length = gap_sssp12c (source, A, delta)
%GAP_SSSP12c single source shortest path, via delta stepping, for GAP
%
% A is square, unsymmetric, int32, and stored by row.  It is assumed to have
% only positive entries.  The method is based on LAGraph_sssp12c.  This is
% slower than gap_sssp12. 

%-------------------------------------------------------------------------------
% check inputs
%-------------------------------------------------------------------------------

if (~isequal (GrB.type (A), 'int32'))
    error ('A must be int32') ;
end

if (~GrB.isbyrow (A))
    error ('A must be ''by row''') ;
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

% t (i) = path length from source to node i
t = empty ;
t (:) = int32 (inf) ;
t (source) = 0 ;

% s = nodes found in this pass
s = empty ;
s (source) = true ;

% reach (i) = true if node i can be reached from the source node
reach = empty ;
reach (:) = false ;
reach (source) = true ;

remain = true ;

% AL = entries in A that are <= delta
AL = GrB.select (A, '<=', delta) ;

% AH = entries in A that are > delta
AH = GrB.select (A, '>' , delta) ;

i = int32 (0) ;

desc_s  = struct ('mask', 'structural') ;
desc_rs = struct ('mask', 'structural', 'out', 'replace') ;

% fprintf ('\nINIT===================================\n') ;
% AL
% AH
% reach
% s

%-------------------------------------------------------------------------------
% SSSP iterations
%-------------------------------------------------------------------------------

while (remain)

% fprintf ('\ni = %d ================================\n', i) ;

    % tmasked = select (t < (i+1)*delta)
    uBound = (i+1) * delta ;
    tmasked = GrB.assign (empty, reach, t) ;
    tmasked = GrB.select (tmasked, '<', uBound) ;

% tmasked

    %---------------------------------------------------------------------------
    % continue while the current bucket B [i] is not empty
    %---------------------------------------------------------------------------

    while (GrB.entries (tmasked) > 0)

% fprintf ('\n inner -------------------------------------------\n') ;

        % tReq = tmasked * AL, using the min.plus semiring
        tReq = GrB.mxm (tmasked, 'min.+', AL) ;
% tReq

        % s = s | spones (tmasked)
        s = GrB.eadd (s, 'pair.logical', tmasked) ;
% s

        if (GrB.entries (tReq) == 0)
            % if tReq is empty, no need to continue
            break ;
        end

        % tless<tReq> = (tReq < t)
        tless = GrB.eadd (empty, tReq, tReq, '<', t, desc_s) ;
% tless

        % remove explicit zeros from tless to use it as a structural mask
        tless = GrB.prune (tless) ;
% tless
        if (GrB.entries (tless) == 0)
            % if tless is empty, no need to continue
            break ;
        end

        % update reachable node list/mask
        % reach<tless> = true
        reach = GrB.assign (reach, tless, true, desc_s) ;
% reach

        % tmasked<tless> = select (i*delta <= tReq < (i+1)*delta)
        tmasked = GrB.select (empty, tless, tReq, '<', uBound, desc_s) ;
% tmasked

        % t<tless> = tReq
        t = GrB.assign (t, tless, tReq, desc_s) ;
% t
    end

    %---------------------------------------------------------------------------
    % outer iterations
    %---------------------------------------------------------------------------

% fprintf ('\nnext outer loop ------------------------------------\n') ;

    % tmasked<s> = t
    tmasked = GrB.assign (tmasked, s, t, desc_rs) ;
% tmasked

    % tReq = tmasked * AH using the min.plus semiring
    tReq = GrB.mxm (tmasked, 'min.+', AH) ;
% tReq

    % t = min (t, tReq) ;
    tless = GrB.eadd (empty, tReq, tReq, '<', t, desc_s) ;
% tless
    t = GrB.assign (t, tless, tReq) ;
% t

    % update reachable node list/mask
    reach = GrB.assign (reach, tless, true) ;
% reach

    % remove previous buckets
    % reach<s> = false
    reach = GrB.assign (reach, s, false, desc_s) ;
    remain = any (reach) ;
% reach

    % clear s for the next loop
    s = empty ;
    i = i + 1 ;
end

% return result
path_length = t ;
% path_length
