function paru_many
%PARU_MANY test many matrices with ParU
%
% Usage: paru_many
%
% Requires ssget in the SuiteSparse meta-package.
%
% See also paru, paru_make, paru_demo, paru_tiny, mldivide, ssget.

% ParU, Copyright (c) 2022-2024, Mohsen Aznaveh and Timothy A. Davis,
% All Rights Reserved.
% SPDX-License-Identifier: GPL-3.0-or-later

% FIXME: add the umfpack mexFunction, to get its analyze/factorize/solve times
% FIXME: save results to a *.mat file for further analysis,
%   once umfpack mexFunction, stats.lnz, stats.unz, and stats.flops
%   are added.

% get all real square matrices in the SuiteSparse Collection,
% that are not candidates for a Cholesky factorization.
index = ssget ;
test_matrices = find (index.nrows == index.ncols & index.isReal & ~index.cholcand) ;

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
    ] ;

% these matrices cause METIS to fail
skip_metis = [1373] ; %#ok<NBRAK2>

% skip these matrices (too large, or singluar):
skip = [too_large ; singular_matrices] ;

test_matrices = setdiff (test_matrices, skip, 'stable') ;

% sort matrices by nnz(A)
nz = index.nnz (test_matrices) ;
[~,p] = sort (nz) ;
test_matrices = test_matrices (p) ;
nmat = length (test_matrices) ;

% warmup to make sure the paru mexFunction is loaded:
paru_demo

% start with this matrix:
first = 1 ;
% first = find (test_matrices == 1235) ;

fprintf ('testing %d matrices:\n', nmat) ;
for k = first:nmat
    id = test_matrices (k) ;
    fprintf ('%4d %4d: %s/%s nz %d\n', k, id, ...
        index.Group {id}, index.Name {id}, index.nnz (id)) ;
end
fprintf ('Hit enter to continue:\n') ;
pause
fprintf ('\n') ;

clear opts_metis
opts_metis.ordering = 'metis' ;

clear opts_metis_guard
opts_metis_guard.ordering = 'metis_guard' ;

for k = first:nmat
    id = test_matrices (k) ;
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

    % try x=paru(A,b), but only if the matrix is not singular
    [~, lastid] = lastwarn ;
    lastwarn ('') ;
    if (isempty (lastid))

        % print results of x=A\b
        resid = norm (A*x-b,1) / anorm ;
        fprintf ('A\\b  resid %8.2e time: %10.2f sec\n', resid, t_backslash) ;

        % default options:
        % ordering: AMD for symmetric strategy, COLAMD for unsymmetric
        t2 = tic ;
        [x2, stats] = paru (A,b) ;
        t_paru = toc (t2) ;
        resid2 = norm (A*x2-b,1) / anorm ;
        fprintf ('ParU resid %8.2e time: %10.2f sec (AMD/COLAMD)  ', ...
            resid2, t_paru) ;
        fprintf ('order: %10.2f factor: %10.2f solve: %10.2f sec ', ...
            stats.analysis_time, stats.factorize_time, stats.solve_time) ;
        speedup = t_backslash / t_paru ;
        if (speedup > 1)
            fprintf ('speedup: %8.2f', speedup) ;
        else
            fprintf ('       : %8.2f', speedup) ;
        end
        fprintf (' ordering: %s\n', stats.ordering) ;

        % ordering: METIS_guard
        t2 = tic ;
        [x2, stats] = paru (A,b,opts_metis_guard) ;
        t_paru = toc (t2) ;
        resid2 = norm (A*x2-b,1) / anorm ;
        fprintf ('ParU resid %8.2e time: %10.2f sec (METIS_guard) ', ...
            resid2, t_paru) ;
        fprintf ('order: %10.2f factor: %10.2f solve: %10.2f sec ', ...
            stats.analysis_time, stats.factorize_time, stats.solve_time) ;
        speedup = t_backslash / t_paru ;
        if (speedup > 1)
            fprintf ('speedup: %8.2f', speedup) ;
        else
            fprintf ('       : %8.2f', speedup) ;
        end
        fprintf (' ordering: %s\n', stats.ordering) ;

        % ordering: METIS; usually slower overall when considering x=A\b, but
        % can result in faster numeric factorization time, particularly for
        % large matrices that represent a 2D or 3D mesh.
        do_metis = (~any (skip_metis == id)) ;
        if (do_metis)
            t2 = tic ;
            [x2, stats] = paru (A,b,opts_metis) ;
            t_paru = toc (t2) ;
            resid2 = norm (A*x2-b,1) / anorm ;
            fprintf ('ParU resid %8.2e time: %10.2f sec (METIS)       ', ...
                resid2, t_paru) ;
            fprintf ('order: %10.2f factor: %10.2f solve: %10.2f sec ', ...
                stats.analysis_time, stats.factorize_time, stats.solve_time) ;
            speedup = t_backslash / t_paru ;
            if (speedup > 1)
                fprintf ('speedup: %8.2f', speedup) ;
            else
                fprintf ('       : %8.2f', speedup) ;
            end
            fprintf (' ordering: %s\n', stats.ordering) ;
        else
            fprintf ('ParU (METIS) skipped\n') ;
        end

    else

        % add this singular matrix to the list of matrices to skip
        skip = [skip ; id] ; %#ok<AGROW>
        save skip_set skip

    end
end


