function gap_sssp
%GAP_SSSP run SSSP for the GAP benchmark

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2020, All Rights Reserved.
% http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

rng ('default') ;

% warmup, to make sure GrB library is loaded
C = GrB (1) * GrB (1) + 1 ;
clear C

index = ssget ;
f = find (index.nrows == index.ncols & index.nnz > 5e6 & index.isReal) ;
[~,i] = sort (index.nnz (f)) ;
matrices = f (i) ;

% tiny test matrices:
matrices = { 'cover', 'HB/jagmesh7' } ;
deltas = [ 100 100 ] ;

% test matrices for laptop:
matrices = { 'HB/west0067', 'SNAP/roadNet-CA' , ...
    'SNAP/com-Orkut', 'LAW/indochina-2004' } ;
deltas = [ 100 100 100 100 ] ;

% the GAP test matrices:
matrices = {
    'GAP/GAP-kron'
    'GAP/GAP-urand'
    'GAP/GAP-twitter'
    'GAP/GAP-web'
    'GAP/GAP-road'
    } ;
deltas = [ 27 35 51 150 200000 ] ;

[status, result] = system ('hostname') ;
clear status
if (isequal (result (1:5), 'hyper'))
    fprintf ('hypersparse: %d threads\n', GrB.threads (40)) ;
elseif (isequal (result (1:5), 'slash'))
    fprintf ('slash: %d threads\n', GrB.threads (8)) ;
elseif (isequal (result (1:9), 'backslash'))
    fprintf ('backslash: %d threads\n', GrB.threads (24)) ;
else
    fprintf ('default: %d threads\n', GrB.threads) ;
end
clear result

threads = GrB.threads ;
threads = [threads threads/2]

good = '/home/davis/sparse/LAGraph/Test/SSSP/' ;

for k = 1:length(matrices)

    %---------------------------------------------------------------------------
    % get the GAP problem
    %---------------------------------------------------------------------------

    id = matrices {k} ;
    fprintf ('\nmatrix: %s\n', id) ;
    GrB.burble (0) ;
    t1 = tic ;
    clear A Prob
    if (isequal (id, 'cover'))
        A = mread ('cover.mtx') ;
        Prob.A = A ;
        Prob.name = 'cover' ;
    else
        Prob = ssget (id, index) ;
    end
    t1 = toc (t1) ; ;
    fprintf ('load time: %g sec\n', t1) ;
    t1 = tic ;
    A = abs (GrB (Prob.A, 'by row', 'int32')) ;
    A = GrB.prune (A) ;
    n = size (A,1) ;
    try
        sources = Prob.aux.sources ;
    catch
        try
            sources = mread (sprintf ('%s/sources_%d.mtx', good, n)) ;
            sources = full (sources) ;
        catch
            sources = randperm (n, 64) ;
        end
    end
    name = Prob.name ;
    clear Prob
    fprintf ('\n%s: nodes: %g million  nvals: %g million\n', ...
        name, n / 1e6, nnz (A) / 1e6) ;
    t1 = toc (t1) ;
    fprintf ('init time: %g sec\n', t1) ;
    % whos

    [i, j, x] = find (A) ;
    % figure (1)
    % histogram (x)
    % drawnow
    fprintf ('edgeweights: min: %g med: %g max: %g\n', ...  
        min (x), median (x), max (x)) ;
    clear i j x

    delta = deltas (k) ;
    fprintf ('delta for this matrix: %d\n', delta) ;

    %---------------------------------------------------------------------------
    % compute the SSSP for each source node
    %---------------------------------------------------------------------------

    for nthreads = threads
        GrB.threads (nthreads) ;

        fprintf ('\ngap_sssp tests: %d threads\n', nthreads) ;

        tot12 = 0 ;
        tot12c = 0 ;
        for trial = 1:length(sources)
            source = sources (trial)  ;
            % fprintf ('source: %d\n', source) ;

            % gap_sssp12c
            %{
            tstart = tic ;
            path_length = gap_sssp12c (source, A, delta) ;
            t = toc (tstart) ;
            tot12c = tot12c + t ;
            fprintf ('trial: %2d source: %8d GrB SSSP12c time: %8.3f\n', ...
                trial, source, t) ;
            path_length = GrB.prune (path_length) ;

            % check result
            try
                tstart = tic ;
                pgood = GrB (mread (sprintf ('%s/pathlen_%02d_%d.mtx', ...
                    good, trial-1, n))) ;
                pgood = pgood' ;
                err = norm (pgood - path_length) / norm (pgood) ;
                t = toc (tstart) ;
                nzdiff = GrB.entries (path_length) - GrB.entries (pgood) ;
            catch
                err = 0 ;
                nzdiff = 0 ;
            end
    %       fprintf ('err: %g (time %g sec) entries %d %d diff %d\n', err, t, ...
    %           GrB.entries (path_length),  GrB.entries (pgood), ...
    %           GrB.entries (path_length) - GrB.entries (pgood)) ;
            assert (err == 0) ;
            assert (nzdiff == 0) ;
            %}

            % gap_sssp12
            tstart = tic ;
            path_len2 = gap_sssp12 (source, A, delta) ;
            t = toc (tstart) ;
            tot12 = tot12 + t ;
            fprintf ('trial: %2d source: %8d GrB SSSP12  time: %8.3f\n', ...
                trial, source, t) ;
            path_len2 = GrB.prune (path_len2) ;
            %{
            try
                err = norm (pgood - path_len2) / norm (pgood) ;
                nzdiff = GrB.entries (path_len2) - GrB.entries (pgood) ;
            catch
            end
            assert (err == 0) ;
            assert (nzdiff == 0) ;
            assert (isequal (path_length, path_len2)) ;
            %}

            clear path_length path_len2 pgood
        end

        ntrials = trial ;

        fprintf ('avg GrB SSSP12c time:  %10.3f (%d trials)\n', ...
            tot12c/ntrials, ntrials) ;

        fprintf ('avg GrB SSSP12  time:  %10.3f (%d trials)\n', ...
            tot12/ntrials, ntrials) ;

    end

    clear A
end

