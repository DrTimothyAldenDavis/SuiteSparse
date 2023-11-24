%BLAS_DUMP_RESULTS analyze results from BLAS_DUMP
% use with 'make large' or go_nd12k

% Used in supernodal factorization:
%
%   0   dsyrk ("L", "N",    n k 0   lda ldc 0
%   1   ssyrk ("L", "N",    n k 0   lda ldc 0
%   2   zherk ("L", "N",    n k 0   lda ldc 0
%   3   cherk ("L", "N",    n k 0   lda ldc 0
%
%   4   dgemm ("N", "C",    m n k   lda ldb ldc
%   5   sgemm ("N", "C",    m n k   lda ldb ldc
%   6   zgemm ("N", "C",    m n k   lda ldb ldc
%   7   cgemm ("N", "C",    m n k   lda ldb ldc
%
%   8   dpotrf ("L",        n 0 0   lda 0 0
%   9   spotrf ("L",        n 0 0   lda 0 0
%   10  zpotrf ("L",        n 0 0   lda 0 0
%   11  cpotrf ("L",        n 0 0   lda 0 0
%
%   12  dtrsm ("R", "L", "C", "N",      m n 0   lda ldb 0
%   13  strsm ("R", "L", "C", "N",      m n 0   lda ldb 0
%   14  ztrsm ("R", "L", "C", "N",      m n 0   lda ldb 0
%   15  ctrsm ("R", "L", "C", "N",      m n 0   lda ldb 0
%
% Used in supernodal forward/backsolve (not benchmarked):
%
%   16  dgemm ("N", "N",
%   17  sgemm ("N", "N",
%   18  zgemm ("N", "N",
%   19  cgemm ("N", "N",
%
%   20  dgemm ("C", "N",
%   21  sgemm ("C", "N",
%   22  zgemm ("C", "N",
%   23  cgemm ("C", "N",
%
%   24  dgemv ("N",
%   25  sgemv ("N",
%   26  zgemv ("N",
%   27  cgemv ("N",
%
%   28  dgemv ("C",
%   29  sgemv ("C",
%   30  zgemv ("C",
%   31  cgemv ("C",
%
%   32  dtrsm ("L", "L", "N", "N",
%   33  strsm ("L", "L", "N", "N",
%   34  ztrsm ("L", "L", "N", "N",
%   35  ctrsm ("L", "L", "N", "N",
%
%   36  dtrsm ("L", "L", "C", "N",
%   37  strsm ("L", "L", "C", "N",
%   38  ztrsm ("L", "L", "C", "N",
%   39  ctrsm ("L", "L", "C", "N",
%
%   40  dtrsv ("L", "N", "N",
%   41  strsv ("L", "N", "N",
%   42  ztrsv ("L", "N", "N",
%   43  ctrsv ("L", "N", "N",
%
%   44  dtrsv ("L", "C", "N",
%   45  strsv ("L", "C", "N",
%   46  ztrsv ("L", "C", "N",
%   47  ctrsv ("L", "C", "N",

clear all

% from 'make large', not in MATLAB
outside = load ('blas_dump.txt') ;

% from 'go_nd12k' in MATLAB
inside = load ('MATLAB/blas_dump.txt') ;

%    Each line contains the 10 space-delimited values:
%
%    1   kernel      see blas_legend.txt (0 to 15)
%    2   4 or 8      sizeof (blas integer), always 8 for MATLAB
%    3   4 or 8      size of the index in a sparse matrix, always 8 for MATLAB;
%                    cholmod_*i_simple will report 4.
%    4:6   m,n,k       sizes of the matrices (0 if not used)
%    7:9  lda,ldb,ldc leading dimensions (0 if not used)
%    10    time        time for this call to the BLAS / LAPACK

%-------------------------------------------------------------------------------
% outside MATLAB, double results:
%-------------------------------------------------------------------------------

% analyze   time:      2.135 sec
% factorize time:      5.085 sec
% solve     time:      0.106 sec
% total     time:      7.326 sec
% CHOLMOD BLAS statistics:
% SYRK  CPU calls         1783 time   1.3997e+00
% GEMM  CPU calls         1450 time   1.1142e+00
% POTRF CPU calls          334 time   9.9486e-01
% TRSM  CPU calls          333 time   6.7501e-01
% total time in the BLAS:   4.1838e+00

dsyrk_outside  = outside (find (outside (:,1) == 0), :) ;
dgemm_outside  = outside (find (outside (:,1) == 4), :) ;
dpotrf_outside = outside (find (outside (:,1) == 8), :) ;
dtrsm_outside  = outside (find (outside (:,1) == 12), :) ;

dsyrk_outside_time  = dsyrk_outside (:,10) ;
dgemm_outside_time  = dgemm_outside (:,10) ;
dpotrf_outside_time = dpotrf_outside (:,10) ;
dtrsm_outside_time  = dtrsm_outside (:,10) ;

fprintf ('\n') ;
fprintf ('total dsyrk  time, outside: %g\n', sum (dsyrk_outside_time)) ;
fprintf ('total dgemm  time, outside: %g\n', sum (dgemm_outside_time)) ;
fprintf ('total dpotrf time, outside: %g\n', sum (dpotrf_outside_time)) ;
fprintf ('total dtrsm  time, outside: %g\n', sum (dtrsm_outside_time)) ;
fprintf ('total BLAS   time, outside: %g\n', ...
 sum (dsyrk_outside_time) + sum (dgemm_outside_time) + ...
 sum (dpotrf_outside_time) + sum (dtrsm_outside_time)) ;

% get work for dsyrk/ssyrk
n_syrk = dsyrk_outside (:,4) ;
k_syrk = dsyrk_outside (:,5) ;
syrk_work = n_syrk .* n_syrk .* (k_syrk+1) ;

% get work for dgemm/sgemm
m_gemm = dgemm_outside (:,4) ;
n_gemm = dgemm_outside (:,5) ;
k_gemm = dgemm_outside (:,6) ;
gemm_work = (m_gemm .* n_gemm .* k_gemm) * 2 ;

% get work for dpotrf/spotrf
n_potrf = dpotrf_outside (:,4) ;
potrf_work = (n_potrf .* n_potrf .* n_potrf) / 3 ;

% get work for dtrsm/strsm
m_trsm = dtrsm_outside (:,4) ;
n_trsm = dtrsm_outside (:,5) ;
trsm_work = m_trsm .* n_trsm .* n_trsm ;

%-------------------------------------------------------------------------------
% outside MATLAB, single results:
%-------------------------------------------------------------------------------

% analyze   time:      2.375 sec
% factorize time:      2.814 sec
% solve     time:      0.046 sec
% total     time:      5.236 sec
% CHOLMOD BLAS statistics:
% SYRK  CPU calls         1783 time   7.9215e-01
% GEMM  CPU calls         1450 time   5.9799e-01
% POTRF CPU calls          334 time   5.0774e-01
% TRSM  CPU calls          333 time   3.5276e-01
% total time in the BLAS:   2.2506e+00

ssyrk_outside  = outside (find (outside (:,1) == 1), :) ;
sgemm_outside  = outside (find (outside (:,1) == 5), :) ;
spotrf_outside = outside (find (outside (:,1) == 9), :) ;
strsm_outside  = outside (find (outside (:,1) == 13), :) ;

ssyrk_outside_time  = ssyrk_outside (:,10) ;
sgemm_outside_time  = sgemm_outside (:,10) ;
spotrf_outside_time = spotrf_outside (:,10) ;
strsm_outside_time  = strsm_outside (:,10) ;

fprintf ('\n') ;
fprintf ('total ssyrk  time, outside: %g\n', sum (ssyrk_outside_time)) ;
fprintf ('total sgemm  time, outside: %g\n', sum (sgemm_outside_time)) ;
fprintf ('total spotrf time, outside: %g\n', sum (spotrf_outside_time)) ;
fprintf ('total strsm  time, outside: %g\n', sum (strsm_outside_time)) ;
fprintf ('total BLAS   time, outside: %g\n', ...
 sum (ssyrk_outside_time) + sum (sgemm_outside_time) + ...
 sum (spotrf_outside_time) + sum (strsm_outside_time)) ;

%-------------------------------------------------------------------------------
% inside MATLAB, double results:
%-------------------------------------------------------------------------------

% time for x=A\b (in double):        9.82203 sec
% analyze time: 2.62621 sec
% factorize time: 7.25898 sec
% solve time: 0.125075 sec
% CHOLMOD BLAS statistics:
% SYRK  CPU calls         1783 time   1.9567e+00
% GEMM  CPU calls         1450 time   1.3923e+00
% POTRF CPU calls          334 time   1.5183e+00
% TRSM  CPU calls          333 time   8.0667e-01
% total time in the BLAS:   5.6740e+00
% time for x=cholmod2(A,b) (double): 10.0717 sec

dsyrk_inside  = inside (find (inside (:,1) == 0), :) ;
dgemm_inside  = inside (find (inside (:,1) == 4), :) ;
dpotrf_inside = inside (find (inside (:,1) == 8), :) ;
dtrsm_inside  = inside (find (inside (:,1) == 12), :) ;

dsyrk_inside_time  = dsyrk_inside (:,10) ;
dgemm_inside_time  = dgemm_inside (:,10) ;
dpotrf_inside_time = dpotrf_inside (:,10) ;
dtrsm_inside_time  = dtrsm_inside (:,10) ;

fprintf ('\n') ;
fprintf ('total dsyrk  time,  inside: %g\n', sum (dsyrk_inside_time)) ;
fprintf ('total dgemm  time,  inside: %g\n', sum (dgemm_inside_time)) ;
fprintf ('total dpotrf time,  inside: %g\n', sum (dpotrf_inside_time)) ;
fprintf ('total dtrsm  time,  inside: %g\n', sum (dtrsm_inside_time)) ;
fprintf ('total BLAS   time,  inside: %g\n', ...
 sum (dsyrk_inside_time) + sum (dgemm_inside_time) + ...
 sum (dpotrf_inside_time) + sum (dtrsm_inside_time)) ;

%-------------------------------------------------------------------------------
% inside MATLAB, single results:
%-------------------------------------------------------------------------------

% analyze time: 2.63795 sec
% factorize time: 142.123 sec
% solve time: 0.0919384 sec
% CHOLMOD BLAS statistics:
% SYRK  CPU calls         1783 time   5.9033e+01
% GEMM  CPU calls         1450 time   2.3369e+01
% POTRF CPU calls          334 time   4.0921e+01
% TRSM  CPU calls          333 time   1.7772e+01
% total time in the BLAS:   1.4109e+02
% time for x=cholmod2(A,b) (single): 144.914 sec

ssyrk_inside  = inside (find (inside (:,1) == 1), :) ;
sgemm_inside  = inside (find (inside (:,1) == 5), :) ;
spotrf_inside = inside (find (inside (:,1) == 9), :) ;
strsm_inside  = inside (find (inside (:,1) == 13), :) ;

ssyrk_inside_time  = ssyrk_inside (:,10) ;
sgemm_inside_time  = sgemm_inside (:,10) ;
spotrf_inside_time = spotrf_inside (:,10) ;
strsm_inside_time  = strsm_inside (:,10) ;

fprintf ('\n') ;
fprintf ('total ssyrk  time,  inside: %g\n', sum (ssyrk_inside_time)) ;
fprintf ('total sgemm  time,  inside: %g\n', sum (sgemm_inside_time)) ;
fprintf ('total spotrf time,  inside: %g\n', sum (spotrf_inside_time)) ;
fprintf ('total strsm  time,  inside: %g\n', sum (strsm_inside_time)) ;
fprintf ('total BLAS   time,  inside: %g\n', ...
 sum (ssyrk_inside_time) + sum (sgemm_inside_time) + ...
 sum (spotrf_inside_time) + sum (strsm_inside_time)) ;

whos

%-------------------------------------------------------------------------------
% compare results: 
%-------------------------------------------------------------------------------

%-------------------------------------------------------------------------------
% figure 1 
%-------------------------------------------------------------------------------

% x axis is each call to the BLAS/LAPACK, in order.
% y axis is the cumulative sum of the run time in that BLAS/LAPACK method.

figure (1)
clf

subplot (2,2,1)
nsyrk = length (ssyrk_outside)
semilogy ((1:nsyrk), cumsum (ssyrk_inside_time), 'r') ;
hold on
semilogy ((1:nsyrk), cumsum (ssyrk_outside_time), 'g') ;
semilogy ((1:nsyrk), cumsum (dsyrk_inside_time), 'b') ;
semilogy ((1:nsyrk), cumsum (dsyrk_outside_time), 'k') ;
xlabel ('*syrk, in order of call') ;
ylabel ('cumulative syrk time (log10 scale)') ;
legend ('ssyrk inside', 'ssyrk outside', ...
         'dsyrk inside', 'dsyrk outside', ...
         'Location', 'SouthEast') ;

subplot (2,2,2)
ngemm = length (sgemm_outside)
semilogy ((1:ngemm), cumsum (sgemm_inside_time), 'r') ;
hold on
semilogy ((1:ngemm), cumsum (sgemm_outside_time), 'g') ;
semilogy ((1:ngemm), cumsum (dgemm_inside_time), 'b') ;
semilogy ((1:ngemm), cumsum (dgemm_outside_time), 'k') ;
xlabel ('*gemm, in order of call') ;
ylabel ('cumulative gemm time (log10 scale)') ;
legend ('sgemm inside', 'sgemm outside', ...
         'dgemm inside', 'dgemm outside', ...
         'Location', 'SouthEast') ;

subplot (2,2,3)
npotrf = length (spotrf_outside)
semilogy ((1:npotrf), cumsum (spotrf_inside_time), 'r') ;
hold on
semilogy ((1:npotrf), cumsum (spotrf_outside_time), 'g') ;
semilogy ((1:npotrf), cumsum (dpotrf_inside_time), 'b') ;
semilogy ((1:npotrf), cumsum (dpotrf_outside_time), 'k') ;
xlabel ('*potrf, in order of call') ;
ylabel ('cumulative potrf time (log10 scale)') ;
legend ('spotrf inside', 'spotrf outside', ...
         'dpotrf inside', 'dpotrf outside', ...
         'Location', 'SouthEast') ;

subplot (2,2,4)
ntrsm = length (strsm_outside)
semilogy ((1:ntrsm), cumsum (strsm_inside_time), 'r') ;
hold on
semilogy ((1:ntrsm), cumsum (strsm_outside_time), 'g') ;
semilogy ((1:ntrsm), cumsum (dtrsm_inside_time), 'b') ;
semilogy ((1:ntrsm), cumsum (dtrsm_outside_time), 'k') ;
xlabel ('*trsm, in order of call') ;
ylabel ('cumulative *trsm time (log10 scale)') ;
legend ('strsm inside', 'strsm outside', ...
         'dtrsm inside', 'dtrsm outside', ...
         'Location', 'SouthEast') ;

%-------------------------------------------------------------------------------
% figure 2
%-------------------------------------------------------------------------------

% x axis is amount of work in the BLAS/LAPACK call
% y axis is the run time of that call inside

figure (2)
clf

subplot (2,2,1)
loglog (syrk_work, ssyrk_inside_time, 'ro') ;
hold on
loglog (syrk_work, ssyrk_outside_time, 'go') ;
xlabel ('syrk work (log scale)') ;
ylabel ('ssyrk time (log scale)') ;
legend ('inside', 'outside', 'Location', 'SouthEast') ;

subplot (2,2,2)
loglog (gemm_work, sgemm_inside_time, 'ro') ;
hold on
loglog (gemm_work, sgemm_outside_time, 'go') ;
xlabel ('gemm work (log scale)') ;
ylabel ('sgemm time (log scale)') ;
legend ('inside', 'outside', 'Location', 'SouthEast') ;

subplot (2,2,3)
loglog (potrf_work, spotrf_inside_time, 'ro') ;
hold on
loglog (potrf_work, spotrf_outside_time, 'go') ;
xlabel ('potrf work (log scale)') ;
ylabel ('potrf time (log scale)') ;
legend ('inside', 'outside', 'Location', 'SouthEast') ;

subplot (2,2,4)
loglog (trsm_work, strsm_inside_time, 'ro') ;
hold on
loglog (trsm_work, strsm_outside_time, 'go') ;
xlabel ('trsm work (log scale)') ;
ylabel ('strsm time (log scale)') ;
legend ('inside', 'outside', 'Location', 'SouthEast') ;

%-------------------------------------------------------------------------------
% figure 3
%-------------------------------------------------------------------------------

% x axis is amount of work in the BLAS/LAPACK call
% y axis is the run time of that call inside divided by outside

figure (3)
clf

subplot (2,2,1)
loglog (syrk_work, ssyrk_inside_time ./ ssyrk_outside_time, 'bo') ;
xlabel ('syrk work (log scale)') ;
ylabel ('ssyrk time (inside/outside)') ;

subplot (2,2,2)
loglog (gemm_work, sgemm_inside_time ./ sgemm_outside_time, 'bo') ;
xlabel ('gemm work (log scale)') ;
ylabel ('sgemm time (inside/outside)') ;

subplot (2,2,3)
loglog (potrf_work, spotrf_inside_time ./ spotrf_outside_time, 'bo') ;
xlabel ('potrf work (log scale)') ;
ylabel ('spotrf time (inside/outside)') ;

subplot (2,2,4)
loglog (trsm_work, strsm_inside_time ./ strsm_outside_time, 'bo') ;
xlabel ('trsm work (log scale)') ;
ylabel ('strsm time (inside/outside)') ;

%-------------------------------------------------------------------------------
% worst cases
%-------------------------------------------------------------------------------

ssyrk_rel  = (ssyrk_inside_time ./ ssyrk_outside_time) ;
sgemm_rel  = (sgemm_inside_time ./ sgemm_outside_time) ;
spotrf_rel = (spotrf_inside_time ./ spotrf_outside_time) ;
strsm_rel  = (strsm_inside_time ./ strsm_outside_time) ;

[bad, i1] = max (ssyrk_rel) ;
[bad, i2] = max (sgemm_rel) ;
[bad, i3] = max (spotrf_rel) ;
[bad, i4] = max (strsm_rel) ;

fprintf ('\nWorst case ssyrk:\n') ;
fprintf ('    n %4d k %4d         inside: %12.6f sec  outside: %12.6f sec\n', ...
    n_syrk (i1), k_syrk (i1), ssyrk_inside_time (i1), ssyrk_outside_time (i1)) ;
fprintf ('    same in dsyrk: inside %12.6f sec, outside %12.6f\n', ...
    dsyrk_inside_time (i1), dsyrk_outside_time (i1)) ;

fprintf ('\nWorst case sgemm:\n') ;
fprintf ('    m %4d n %4d k %4d  inside: %12.6f sec  outside: %12.6f sec\n', ...
    m_gemm (i2), n_gemm (i2), k_gemm (i2), sgemm_inside_time (i2), sgemm_outside_time (i2)) ;
fprintf ('    same in dgemm: inside %12.6f sec, outside %12.6f\n', ...
    dgemm_inside_time (i2), dgemm_outside_time (i2)) ;

fprintf ('\nWorst case spotrf:\n') ;
fprintf ('    n %4d                inside: %12.6f sec  outside: %12.6f sec\n', ...
    n_potrf (i3), spotrf_inside_time (i3), spotrf_outside_time (i3)) ;
fprintf ('    same in dpotrm: inside %12.6f sec, outside %12.6f\n', ...
    dpotrf_inside_time (i3), dpotrf_outside_time (i3)) ;

fprintf ('\nWorst case strsm:\n') ;
fprintf ('    m %4d n %4d         inside: %12.6f sec  outside: %12.6f sec\n', ...
    m_trsm (i4), n_trsm (i4), strsm_inside_time (i4), strsm_outside_time (i4)) ;
fprintf ('    same in strrm: inside %12.6f sec, outside %12.6f\n', ...
    dtrsm_inside_time (i4), dtrsm_outside_time (i4)) ;


