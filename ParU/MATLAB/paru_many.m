function paru_many
%PARU_MANY: test many matrices with ParU
%
% Usage: paru_many
%
% See also paru, paru_make, paru_demo, paru_tiny, mldivide.

% ParU, Copyright (c) 2022-2024, Mohsen Aznaveh and Timothy A. Davis,
% All Rights Reserved.
% SPDX-License-Identifier: GPL-3.0-or-later

% FIXME: add the umfpack mexFunction, so we can consider
% its analyze/factorize/solve times

index = ssget ;
square = find (index.nrows == index.ncols & index.isReal & ~index.cholcand) ;

% these matrices are too large, causing MATLAB itself to terminate:
too_large = [
    2575
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
    ] ;

% skip these matrices (too large, or singluar):
skip = [too_large ; singular_matrices] ;

square = setdiff (square, skip, 'stable') ;
nz = index.nnz (square) ;
[~,p] = sort (nz) ;
square = square (p) ;

nmat = length (square) ;

% warmup to make sure the paru mexFunction is loaded:
paru_demo

fprintf ('testing %d matrices:\n', nmat) ;
for k = 1:nmat
    id = square (k) ;
    fprintf ('%4d %4d: %s/%s nz %d\n', k, id, ...
        index.Group {id}, index.Name {id}, index.nnz (id)) ;
end

rng ('default') ;

clear opts_metis
opts_metis.ordering = 'metis' ;

ok_list = [ ] ;

for k = 1:nmat
    id = square (k) ;
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
    [lastmsg, lastid] = lastwarn ;
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
        fprintf ('ParU resid %8.2e time: %10.2f sec (default) ', ...
            resid2, t_paru) ;
        fprintf ('order: %10.2f factor: %10.2f solve: %10.2f sec ', ...
            stats.analysis_time, stats.factorize_time, stats.solve_time) ;
        speedup = t_backslash / t_paru ;
        if (speedup > 1)
            fprintf ('speedup: %8.2f\n', speedup) ;
        else
            fprintf ('       : %8.2f\n', speedup) ;
        end

        % ordering: METIS; usually slower overall when considering x=A\b, but
        % can result in faster numeric factorization time, particularly for
        % large problems.
        t2 = tic ;
        [x2, stats] = paru (A,b,opts_metis) ;
        t_paru = toc (t2) ;
        resid2 = norm (A*x2-b,1) / anorm ;
        fprintf ('ParU resid %8.2e time: %10.2f sec (METIS)   ', ...
            resid2, t_paru) ;
        fprintf ('order: %10.2f factor: %10.2f solve: %10.2f sec ', ...
            stats.analysis_time, stats.factorize_time, stats.solve_time) ;

        speedup = t_backslash / t_paru ;
        if (speedup > 1)
            fprintf ('speedup: %8.2f\n', speedup) ;
        else
            fprintf ('       : %8.2f\n', speedup) ;
        end

    else
        skip = [skip ; id] ;
        save skip_set skip
    end

end

