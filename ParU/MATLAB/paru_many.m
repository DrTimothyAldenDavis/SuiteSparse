function paru_many
%PARU_MANY test many matrices with ParU
%
% Usage: paru_many
%
% Requires ssget and umfpack in the SuiteSparse meta-package.
%
% See also paru, paru_make, paru_demo, paru_tiny, mldivide, ssget, umfpack.

% ParU, Copyright (c) 2022-2024, Mohsen Aznaveh and Timothy A. Davis,
% All Rights Reserved.
% SPDX-License-Identifier: GPL-3.0-or-later

addpath ('../../UMFPACK/MATLAB') ;

% get all real square unsymmetric matrices in the SuiteSparse Collection,
% that are not candidates for a Cholesky factorization.
index = ssget ;
test_matrices = find ((index.nrows == index.ncols) & index.isReal ...
    & (index.sprank == index.nrows) & ~index.cholcand ...
    & index.numerical_symmetry < 1) ;

% these matrices are too large, causing MATLAB and/or paru to fail or to take
% far too much time.  Some are graphs that are not meant to represent a sparse
% linear system, and suffer very high fill-in if factorized.
too_large = [
    2575
    2576
    982
    2456
    1297
    2778
    2788
    2462
    1370
    913
    2779
    2460
    2439
    2267
    2649
    2380
    1396
    1397
    1901
    1902
    1903
    1904
    1905
    2847
    2386
    2843
     916
    2844
    ] ;

% these matrices are singular or nearly so:
singular_matrices = [
          56
        1525
        1479
        1527
        1528
         185
        1495
        1485
        1516
          59
          58
         186
         238
        1515
        1481
        1486
        1474
        1501
        1480
        1499
        2399
        1489
        1496
        2097
        1491
        1482
        1519
        1490
        1498
        2149
        1493
         274
         109
        2396
        1492
        1497
        2457
         264
        1484
         317
        1475
        1494
        2400
        1477
        1478
         158
         273
        2417
        2389
        2416
        1523
        1476
         282
        2357
        2365
        1117
        1154
        1156
        1157
        1158
        1159
        1160
        1161
        1162
        1163
        1164
        1165
        1166
        1167
        1168
        2361
        2369
        1488
        1458
        2392
        2015
        2358
        2366
        1465
        1172
         212
        1467
        1469
        2362
        2370
        2429
        2431
         240
        2407
        1529
        1511
        1522
        2359
        2367
        1518
        2420
        2363
        2371
        2401
        1055
        2705
        2331
         843
        1471
        1483
        2428
        2689
         314
        2195
        2196
        1463
        1239
         230
        2360
        2368
        2364
        2372
        1521
         348
        2408
         429
        1186
         511
         512
         514
        2419
        1507
        2418
        1530
        2404
        2531
        1533
        1466
        1468
        1470
         347
        2422
        1456
        1464
        1512
        2402
         375
         244
        2777
         345
         459
         456
         457
        2311
         458
          68
        1181
         755
        1500
        2790
        2785
         519
         520
        2784
        2312
         754
        2320
        2297
        2398
        2310
        2309
         525
         526
        2350
        2791
        1841
        2308
        2351
         211
        1508
        1185
        2352
        2348
        2256
        1891
        2323
        2405
        2340
        2343
        2339
        2342
        2346
        2341
        2353
         209
         210
        2104
        1178
        2299
        2698
        1534
        2314
        2425
        1245
        2354
        2313
        2324
        2355
        1622
        1234
        1217
        2430
        1472
        2347
        1242
        2349
        2699
        1176
        2356
        1215
         410
        1175
        1216
        1179
        2700
        2433
        2704
        2315
        2413
        2432
        2393
        2390
        2440
        2527
        1624
        2288
        2322
        2643
        1424
        2589
        1473
        2592
        2636
        2620
         819
        1531
        1247
        1248
        2434
        2344
        2345
        2438
        2316
        2627
        1180
        1459
        2296
        2797
        2634
        2435
         954
        1505
         376
        2582
        1182
         740
         742
        2611
        2298
        2515
        2793
        2394
        2391
        2444
        1173
        2644
         956
        2199
        2200
        1517
         744
        2499
         346
        1174
        2332
        1462
        1461
        2524
        2333
        2602
        2588
        1502
        2395
        1503
        2293
        2290
         746
        2645
        2641
        2295
        2622
        2614
        2289
        2292
        2631
        2787
        2637
         955
         560
        2284
        2326
         957
         958
        2327
        2328
        1246
        1238
        1513
         562
        2625
        2792
        2635
        1369
        1377
        2609
        2607
         564
        2630
        2500
        2601
         566
        2594
        1244
         748
        2600
        2640
        2598
        2056
        2135
        2816
        2612
        2613
        2606
        2325
        2034
        2646
        2621
         960
        2584
        2286
        2610
        2795
        1510
        2628
        2287
        2587
        2618
        2599
        2638
        2530
        2574
        2837
        2593
        2639
        2597
        2623
        2585
         959
        2654
        2629
        2604
        2583
        2304
        2617
        2596
        1418
        2626
        2608
        2591
        2836
        2501
         750
        2302
        2603
        2448
        2461
        2834
        1231
        2605
        1251
        2615
        2616
        2838
        1417
        2619
        2595
         979
        2303
        2590
        2378
        2318
        2502
        2510
        2379
        2305
        2441
        2798
        2306
        2307
        2586
        1399
        2319
        2633
        2624
        2516
        2489
        2291
        2301
        1867
        2839
         285
        2267
        2268
        2662
        2277
        2840
        2841
        1141
        1155
        2701
         455
        2377
        1335
        1339
        2550
        2551
        2835
    ] ;

% these matrices cause METIS to fail
skip_metis = [1373 1396 1397] ;

% skip these matrices (too large, or singular):
skip = [too_large ; singular_matrices] ;

test_matrices = setdiff (test_matrices, skip, 'stable') ;

% just one matrix:
% test_matrices = 2279 ;

% sort matrices by nnz(A)
nz = index.nnz (test_matrices) ;
[~,p] = sort (nz) ;
test_matrices = test_matrices (p) ;

% matrices used for the results in the ACM TOMS paper:
acm_toms_matrices = [
     447    % Goodwin/rim
    2241    % TSOPF/TSOPF_RS_b39_c30
     286    % ATandT/twotone
    1334    % VanVelzen/std1_Jac2
    2376    % Williams/mac_econ_fwd500
    1336    % VanVelzen/std1_Jac3
    2237    % TSOPF/TSOPF_RS_b300_c1
     750    % Mallya/lhr71
    1338    % VanVelzen/Zd_Jac2
     827    % Vavasis/av41092
    1342    % VanVelzen/Zd_Jac6
    1858    % QLi/crashbasis
     812    % Simon/bbmat
    1340    % VanVelzen/Zd_Jac3
    2377    % Williams/mc2depi
    2238    % TSOPF/TSOPF_RS_b300_c2
    2243    % TSOPF/TSOPF_RS_b678_c1
    2239    % TSOPF/TSOPF_RS_b300_c3
    1867    % Muite/Chebyshev4
    1201    % Hamrle/Hamrle3
     285    % ATandT/pre2
    2235    % TSOPF/TSOPF_RS_b2052_c1
     896    % Norris/torso1
    2244    % TSOPF/TSOPF_RS_b678_c2
    2219    % TSOPF/TSOPF_RS_b2383
    2844    % VLSI/vas_stokes_4M
    2845    % VLSI/stokes
] ;

% just use the ACM TOMS matrices:
% test_matrices = acm_toms_matrices ;

nmat = length (test_matrices) ;

% warmup to make sure the paru mexFunction is loaded:
paru_demo

% start with this matrix:
first = 1 ;
% first = find (test_matrices == 916) ;

fprintf ('testing %d matrices:\n', nmat) ;
for k = first:nmat
    id = test_matrices (k) ;
    fprintf ('%4d %4d: %s/%s nz %d\n', k, id, ...
        index.Group {id}, index.Name {id}, index.nnz (id)) ;
end
fprintf ('Hit enter to continue:\n') ;
pause
fprintf ('\n') ;

save test_matrices

clear opts_metis
opts_metis.ordering = 'metis' ;

clear opts_metis_guard
opts_metis_guard.ordering = 'metis_guard' ;

% umfpack scaling default is SUM, paru default is MAX;
umf_scale_max = umfpack ;
umf_scale_max.scale = 'max' ;

if (first > 1)

    % continue the experiments from a prior run
    if (isequal (test_matrices, acm_toms_matrices))
        % load the results for the ACM TOMS paper:
        load Stats_ACM_TOMS
    else
        % load results for many matrices:
        load Stats
        % backslash_stats umfpack_stats umfpack_maxscale_stats paru_amd_stats paru_meg_stats paru_met_stats
    end

else

    % record statistics for further analysis
    backslash_stats.time = inf (nmat,1) ;

    % UMFPACK with defaults (sum scaling)
    umfpack_stats.time = inf (nmat,1) ;
    umfpack_stats.factorization_time = inf (nmat,1) ;
    umfpack_stats.analysis_time = inf (nmat,1) ;
    umfpack_stats.solve_time = inf (nmat,1) ;
    umfpack_stats.flops = zeros (nmat,1) ;  % in factorization
    umfpack_stats.nnzLU = zeros (nmat,1) ;  % nnz (L+U), including singletons
    umfpack_stats.flops_exact = zeros (nmat,1) ;  % in factorization
    umfpack_stats.nnzLU_exact = zeros (nmat,1) ;  % nnz (L+U), including singletons
    umfpack_stats.strategy_used = char (nmat,3) ; % sym, uns
    umfpack_stats.ordering_used = char (nmat,3) ; % amd, met, col, non
    umfpack_stats.scaling = char (nmat,3) ; % sum, max, non

    % UMFPACK with max scaling
    umfpack_maxscale_stats.time = inf (nmat,1) ;
    umfpack_maxscale_stats.factorization_time = inf (nmat,1) ;
    umfpack_maxscale_stats.analysis_time = inf (nmat,1) ;
    umfpack_maxscale_stats.solve_time = inf (nmat,1) ;
    umfpack_maxscale_stats.flops = zeros (nmat,1) ;  % in factorization
    umfpack_maxscale_stats.nnzLU = zeros (nmat,1) ;  % nnz (L+U), incl singletons
    umfpack_maxscale_stats.flops_exact = zeros (nmat,1) ;  % in factorization
    umfpack_maxscale_stats.nnzLU_exact = zeros (nmat,1) ;  % nnz (L+U), incl singletons
    umfpack_maxscale_stats.strategy_used = char (nmat,3) ; % sym, uns
    umfpack_maxscale_stats.ordering_used = char (nmat,3) ; % amd, met, col, non
    umfpack_maxscale_stats.scaling = char (nmat,3) ; % sum, max, non

    % ParU with default (in MATLAB) ordering: AMD/COLAMD
    paru_amd_stats.time = inf (nmat,1) ;
    paru_amd_stats.factorization_time = inf (nmat,1) ;
    paru_amd_stats.analysis_time = inf (nmat,1) ;
    paru_amd_stats.solve_time = inf (nmat,1) ;
    paru_amd_stats.flops = zeros (nmat,1) ;  % in factorization
    paru_amd_stats.nnzLU = zeros (nmat,1) ;  % nnz (L+U), including singletons
    paru_amd_stats.flops_exact = zeros (nmat,1) ;  % in factorization
    paru_amd_stats.nnzLU_exact = zeros (nmat,1) ;  % nnz (L+U), including singletons
    paru_amd_stats.strategy_used = char (nmat,3) ; % sym, uns
    paru_amd_stats.ordering_used = char (nmat,3) ; % amd, met, col, non
    paru_amd_stats.scaling = char (nmat,3) ; % sum, max, non

    % ParU with METIS_GUARD ordering
    paru_meg_stats.time = inf (nmat,1) ;
    paru_meg_stats.factorization_time = inf (nmat,1) ;
    paru_meg_stats.analysis_time = inf (nmat,1) ;
    paru_meg_stats.solve_time = inf (nmat,1) ;
    paru_meg_stats.flops = zeros (nmat,1) ;  % in factorization
    paru_meg_stats.nnzLU = zeros (nmat,1) ;  % nnz (L+U), including singletons
    paru_meg_stats.flops_exact = zeros (nmat,1) ;  % in factorization
    paru_meg_stats.nnzLU_exact = zeros (nmat,1) ;  % nnz (L+U), including singletons
    paru_meg_stats.strategy_used = char (nmat,3) ; % sym, uns
    paru_meg_stats.ordering_used = char (nmat,3) ; % amd, met, col, non
    paru_meg_stats.scaling = char (nmat,3) ; % sum, max, non

    % ParU with METIS ordering
    paru_met_stats.time = inf (nmat,1) ;
    paru_met_stats.factorization_time = inf (nmat,1) ;
    paru_met_stats.analysis_time = inf (nmat,1) ;
    paru_met_stats.solve_time = inf (nmat,1) ;
    paru_met_stats.flops = zeros (nmat,1) ;  % in factorization
    paru_met_stats.nnzLU = zeros (nmat,1) ;  % nnz (L+U), including singletons
    paru_met_stats.flops_exact = zeros (nmat,1) ;  % in factorization
    paru_met_stats.nnzLU_exact = zeros (nmat,1) ;  % nnz (L+U), including singletons
    paru_met_stats.strategy_used = char (nmat,3) ; % sym, uns
    paru_met_stats.ordering_used = char (nmat,3) ; % amd, met, col, non
    paru_met_stats.scaling = char (nmat,3) ; % sum, max, non
end


% see if ParU is compiled with -DDEVELOPER
try
    [x,stats,P,Q,R] = paru (sparse (1), 1) ;
    dev = true ;
catch me
    dev = false ;
end

% solve each matrix
for k = first:nmat
    id = test_matrices (k) ;
    fprintf ('\n--------------------------------------------------------------------------------\n') ;
    fprintf ('%4d %4d: %s/%s nz %d\n', k, id, ...
        index.Group {id}, index.Name {id}, index.nnz (id)) ;

    % get the problem
    Prob = ssget (id, index) ;
    A = Prob.A ;
    clear Prob
    if (~isreal (A))
        error ('!') ;
    end
    n = size (A,1) ;
    xtrue = rand (n,1) ;
    b = A*xtrue ;
    anorm = norm (A, 1) ;

    % try x=A\b (usually via UMFPACK)
    lastwarn ('') ;
    t1 = tic ;
    x = A\b ;
    t_backslash = toc (t1) ;
    backslash_stats.time (k) = t_backslash ;

    % try other methods if the matrix is not singular
    [~, lastid] = lastwarn ;
    lastwarn ('') ;
    if (isempty (lastid))

        % print results of x=A\b
        resid = norm (A*x-b,1) / anorm ;
        fprintf ('A\\b      resid %8.2e time: %10.2f sec\n', ...
            resid, t_backslash) ;

        % try the UMFPACK mexFunction with default options
        t1 = tic ;
        [x, stats] = umfpack (A, '\', b) ;
        t_umfpack = toc (t1) ;
        resid = norm (A*x-b,1) / anorm ;
        fprintf ('UMF:sum  resid %8.2e time: %10.2f sec               ', ...
            resid, t_umfpack) ;
        fprintf ('order: %10.2f factor: %10.2f solve: %10.2f sec ', ...
            stats.analysis_time, stats.factorization_time, stats.solve_time) ;
        speedup = t_backslash / t_umfpack ;
        if (speedup > 1)
            fprintf ('speedup: %8.2f', speedup) ;
        else
            fprintf ('       : %8.2f', speedup) ;
        end
        strat = stats.strategy_used (1:3) ;
        fprintf (' ordering: %s %s\n', strat, stats.ordering_used) ;

        % record UMFPACK stats
        umfpack_stats.time (k) = t_umfpack ;
        umfpack_stats.factorization_time (k) = stats.factorization_time ;
        umfpack_stats.analysis_time (k) = stats.analysis_time ;
        umfpack_stats.solve_time (k) = stats.solve_time ;
        umfpack_stats.flops (k) = stats.factorization_flop_count ;
        umfpack_stats.nnzLU (k) = stats.nnz_in_L_plus_U ;
        umfpack_stats.strategy_used (k,1:3) = strat ;
        umfpack_stats.ordering_used (k,1:3) = stats.ordering_used (1:3) ;

        if (dev)
            tt = tic ;
            % get the exact flop count and nnz(L+U) for UMFPACK with SUM scale
            [L,U,P,Q,R] = umfpack (A) ;
            umfpack_stats.flops_exact (k) = luflop (L, U) ;
            umfpack_stats.nnzLU_exact (k) = nnz (L) + nnz (U) - n ;
            clear L U P Q R
            tt = toc (tt) ;
            fprintf ('    flops %g %g nnzLU %g %g (%g sec)\n', ...
                umfpack_stats.flops (k), ...
                umfpack_stats.flops_exact (k), ...
                umfpack_stats.nnzLU (k), ...
                umfpack_stats.nnzLU_exact (k), tt) ;
        end

        % try the UMFPACK mexFunction with MAX scaling
        t1 = tic ;
        [x, stats] = umfpack (A, '\', b, umf_scale_max) ;
        t_umfpack = toc (t1) ;
        resid = norm (A*x-b,1) / anorm ;
        fprintf ('UMF:max  resid %8.2e time: %10.2f sec               ', ...
            resid, t_umfpack) ;
        fprintf ('order: %10.2f factor: %10.2f solve: %10.2f sec ', ...
            stats.analysis_time, stats.factorization_time, stats.solve_time) ;
        speedup = t_backslash / t_umfpack ;
        if (speedup > 1)
            fprintf ('speedup: %8.2f', speedup) ;
        else
            fprintf ('       : %8.2f', speedup) ;
        end
        strat = stats.strategy_used (1:3) ;
        fprintf (' ordering: %s %s\n', strat, stats.ordering_used) ;

        % record UMFPACK stats
        umfpack_maxscale_stats.time (k) = t_umfpack ;
        umfpack_maxscale_stats.factorization_time (k) = ...
            stats.factorization_time ;
        umfpack_maxscale_stats.analysis_time (k) = stats.analysis_time ;
        umfpack_maxscale_stats.solve_time (k) = stats.solve_time ;
        umfpack_maxscale_stats.flops (k) = stats.factorization_flop_count ;
        umfpack_maxscale_stats.nnzLU (k) = stats.nnz_in_L_plus_U ;
        umfpack_maxscale_stats.strategy_used (k,1:3) = strat ;
        umfpack_maxscale_stats.ordering_used (k,1:3) = ...
            stats.ordering_used (1:3) ;

        if (dev)
            % get the exact flop count and nnz(L+U) for UMFPACK with MAX scale
            tt = tic ;
            [L,U,P,Q,R] = umfpack (A, umf_scale_max) ;
            umfpack_maxscale_stats.flops_exact (k) = luflop (L, U) ;
            umfpack_maxscale_stats.nnzLU_exact (k) = nnz (L) + nnz (U) - n ;
            clear L U P Q R
            tt = toc (tt) ;
            fprintf ('    flops %g %g nnzLU %g %g (%g sec)\n', ...
                umfpack_maxscale_stats.flops (k), ...
                umfpack_maxscale_stats.flops_exact (k), ...
                umfpack_maxscale_stats.nnzLU (k), ...
                umfpack_maxscale_stats.nnzLU_exact (k), tt) ;
        end

        % try the ParU mexFunction with default options:
        % ordering: AMD for symmetric strategy, COLAMD for unsymmetric
        t2 = tic ;
        [x, stats] = paru (A,b) ;
        t_paru = toc (t2) ;
        resid = norm (A*x-b,1) / anorm ;
        fprintf ('ParU:max resid %8.2e time: %10.2f sec (AMD/COLAMD)  ', ...
            resid, t_paru) ;
        fprintf ('order: %10.2f factor: %10.2f solve: %10.2f sec ', ...
            stats.analysis_time, stats.factorization_time, stats.solve_time) ;
        speedup = t_backslash / t_paru ;
        if (speedup > 1)
            fprintf ('speedup: %8.2f', speedup) ;
        else
            fprintf ('       : %8.2f', speedup) ;
        end
        strat = stats.strategy_used (1:3) ;
        fprintf (' ordering: %s %s\n', strat, stats.ordering_used) ;

        % record ParU stats with AMD
        paru_amd_stats.time (k) = t_paru ;
        paru_amd_stats.factorization_time (k) = stats.factorization_time ;
        paru_amd_stats.analysis_time (k) = stats.analysis_time ;
        paru_amd_stats.solve_time (k) = stats.solve_time ;
        paru_amd_stats.flops (k) = stats.factorization_flop_count ;
        paru_amd_stats.nnzLU (k) = stats.lnz + stats.unz - n ;
        paru_amd_stats.strategy_used (k,1:3) = strat ;
        paru_amd_stats.ordering_used (k,1:3) = stats.ordering_used (1:3) ;

        if (dev)
            tt = tic ;
            % get the exact flop count and nnz(L+U) for ParU with AMD
            [x, stats, P, Q, R] = paru (A,b) ;
            S = spdiags (R, 0, n, n) \ A ;
            [L,U] = lu (S (P+1,Q+1), 0) ;
            assert (nnz (triu (L, 1)) == 0) ;
            paru_amd_stats.flops_exact (k) = luflop (L, U) ;
            paru_amd_stats.nnzLU_exact (k) = nnz (L) + nnz (U) - n ;
            clear L U P Q R S
            tt = toc (tt) ;
            fprintf ('    flops %g %g nnzLU %g %g (%g sec)\n', ...
                paru_amd_stats.flops (k), ...
                paru_amd_stats.flops_exact (k), ...
                paru_amd_stats.nnzLU (k), ...
                paru_amd_stats.nnzLU_exact (k), tt) ;
        end

        % try the ParU mexFunction with ordering: METIS_guard
        t2 = tic ;
        [x, stats] = paru (A,b,opts_metis_guard) ;
        t_paru = toc (t2) ;
        resid = norm (A*x-b,1) / anorm ;
        fprintf ('ParU:max resid %8.2e time: %10.2f sec (METIS_guard) ', ...
            resid, t_paru) ;
        fprintf ('order: %10.2f factor: %10.2f solve: %10.2f sec ', ...
            stats.analysis_time, stats.factorization_time, stats.solve_time) ;
        speedup = t_backslash / t_paru ;
        if (speedup > 1)
            fprintf ('speedup: %8.2f', speedup) ;
        else
            fprintf ('       : %8.2f', speedup) ;
        end
        strat = stats.strategy_used (1:3) ;
        fprintf (' ordering: %s %s\n', strat, stats.ordering_used) ;

        % record ParU stats with METIS_GUARD
        paru_meg_stats.time (k) = t_paru ;
        paru_meg_stats.factorization_time (k) = stats.factorization_time ;
        paru_meg_stats.analysis_time (k) = stats.analysis_time ;
        paru_meg_stats.solve_time (k) = stats.solve_time ;
        paru_meg_stats.flops (k) = stats.factorization_flop_count ;
        paru_meg_stats.nnzLU (k) = stats.lnz + stats.unz - n ;
        paru_meg_stats.strategy_used (k,1:3) = strat ;
        paru_meg_stats.ordering_used (k,1:3) = stats.ordering_used (1:3) ;

        if (dev)
            % get the exact flop count and nnz(L+U) for ParU with METIS_GUARD
            tt = tic ;
            [x, stats, P, Q, R] = paru (A,b,opts_metis_guard) ;
            S = spdiags (R, 0, n, n) \ A ;
            [L,U] = lu (S (P+1,Q+1), 0) ;
            assert (nnz (triu (L, 1)) == 0) ;
            paru_meg_stats.flops_exact (k) = luflop (L, U) ;
            paru_meg_stats.nnzLU_exact (k) = nnz (L) + nnz (U) - n ;
            clear L U P Q R S
            tt = toc (tt) ;
            fprintf ('    flops %g %g nnzLU %g %g (%g sec)\n', ...
                paru_meg_stats.flops (k), ...
                paru_meg_stats.flops_exact (k), ...
                paru_meg_stats.nnzLU (k), ...
                paru_meg_stats.nnzLU_exact (k), tt) ;
        end

        % try the ParU mexFunction with ordering: METIS
        % ordering: METIS; usually slower overall when considering x=A\b, but
        % can result in faster numeric factorization time, particularly for
        % large matrices that represent a 2D or 3D mesh.
        do_metis = (~any (skip_metis == id)) ;
        if (do_metis)
            t2 = tic ;
            [x, stats] = paru (A,b,opts_metis) ;
            t_paru = toc (t2) ;
            resid = norm (A*x-b,1) / anorm ;
            fprintf ('ParU:max resid %8.2e time: %10.2f sec (METIS)       ', ...
                resid, t_paru) ;
            fprintf ('order: %10.2f factor: %10.2f solve: %10.2f sec ', ...
                stats.analysis_time, stats.factorization_time, stats.solve_time) ;
            speedup = t_backslash / t_paru ;
            if (speedup > 1)
                fprintf ('speedup: %8.2f', speedup) ;
            else
                fprintf ('       : %8.2f', speedup) ;
            end
            strat = stats.strategy_used (1:3) ;
            fprintf (' ordering: %s %s\n', strat, stats.ordering_used) ;

            % record ParU stats with METIS
            paru_met_stats.time (k) = t_paru ;
            paru_met_stats.factorization_time (k) = stats.factorization_time ;
            paru_met_stats.analysis_time (k) = stats.analysis_time ;
            paru_met_stats.solve_time (k) = stats.solve_time ;
            paru_met_stats.flops (k) = stats.factorization_flop_count ;
            paru_met_stats.nnzLU (k) = stats.lnz + stats.unz - n ;
            paru_met_stats.strategy_used (k,1:3) = strat ;
            paru_met_stats.ordering_used (k,1:3) = stats.ordering_used (1:3) ;

            if (dev)
                % get the exact flop count and nnz(L+U) for ParU with METIS
                tt = tic ;
                [x, stats, P, Q, R] = paru (A,b,opts_metis) ;
                S = spdiags (R, 0, n, n) \ A ;
                [L,U] = lu (S (P+1,Q+1), 0) ;
                assert (nnz (triu (L, 1)) == 0) ;
                paru_met_stats.flops_exact (k) = luflop (L, U) ;
                paru_met_stats.nnzLU_exact (k) = nnz (L) + nnz (U) - n ;
                clear L U P Q R S
                tt = toc (tt) ;
                fprintf ('    flops %g %g nnzLU %g %g (%g sec)\n', ...
                    paru_met_stats.flops (k), ...
                    paru_met_stats.flops_exact (k), ...
                    paru_met_stats.nnzLU (k), ...
                    paru_met_stats.nnzLU_exact (k), tt) ;
            end

        else
            fprintf ('ParU (METIS) skipped\n') ;
        end

    else

        % add this singular matrix to the list of matrices to skip
        skip = [skip ; id] ; %#ok<AGROW>
        save skip_set skip

    end

    if (isequal (test_matrices, acm_toms_matrices))
        % save the results for the ACM TOMS paper:
        save Stats_ACM_TOMS backslash_stats umfpack_stats umfpack_maxscale_stats paru_amd_stats paru_meg_stats paru_met_stats
    else
        % save results for many matrices:
        save Stats backslash_stats umfpack_stats umfpack_maxscale_stats paru_amd_stats paru_meg_stats paru_met_stats
    end

end

