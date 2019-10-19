function [out1,out2,P] = spqr (A,arg2,arg3)                                 %#ok
%SPQR multithreaded multifrontal rank-revealing sparse QR.
%For a sparse m-by-n matrix A, and sparse or full B of size m-by-k:
%
%   R = spqr (A)              Q-less QR
%   R = spqr (A,0)            economy variant (size(R,1) = min(m,n))
%   R = spqr (A,opts)         as above, with non-default options
%
%   [Q,R] = spqr (A)          A=Q*R
%   [Q,R] = spqr (A,0)        economy variant (size(Q,2) = size(R,1) = min(m,n))
%   [Q,R] = spqr (A,opts)     A=Q*R, with non-default options
%
%   [Q,R,P] = spqr (A)        A*P=Q*R where P reduces fill-in
%   [Q,R,P] = spqr (A,0)      economy variant (size(Q,2) = size(R,1) = min(m,n))
%   [Q,R,P] = spqr (A,opts)   as above, with non-default options
%
%   [C,R] = spqr (A,B)        as R=spqr(A), also returns C=Q'*B
%   [C,R] = spqr (A,B,0)      economy variant (size(C,1) = size(R,1) = min(m,n))
%   [C,R] = spqr (A,B,opts)   as above, with non-default options
%
%   [C,R,P] = spqr (A,B)      as R=spqr(A*P), also returns C=Q'*B
%   [C,R,P] = spqr (A,B,0)    economy variant (size(C,1) = size(R,1) = min(m,n))
%   [C,R,P] = spqr (A,B,opts) as above, with non-default options
%
% P is chosen to reduce fill-in and to return R in upper trapezoidal form if A
% is estimated to have less than full rank.  opts provides non-default options.
% Q can be optionally returned in Householder form, which is far sparser than
% returning Q as a sparse matrix.
%
% Example:
%   The least-squares solution of an overdetermined system A*x=b with
%   m > n can be found in at least one of seven ways (in increasing order of
%   efficiency):
%
%      x = pinv(full(A)) * b ;
%      [Q,R] = spqr (A) ; x = R\(Q'*b) ;
%      [Q,R,P] = spqr (A) ; x = P*(R\(Q'*b)) ;
%      [Q,R,P] = spqr (A,struct('Q','Householder')) ; x=P*(R\spqr_qmult(Q,b,0));
%      [c,R,P] = spqr (A,b) ; x = P*(R\c) ;
%      [c,R,p] = spqr (A,b,0) ; x = (R\c) ; x (p) = x ;
%      x = spqr_solve (A,b) ;
%  
%   The minimum-norm solution of an underdetermined system A*x=b with
%   m < n can be found in one of five ways (in increasing order of efficiency):
%
%      x = pinv(full(A)) * b ;
%      [Q,R] = spqr (A') ; x = Q*(R'\b) ;
%      [Q,R,P] = spqr (A') ; x = Q*(R'\(P'*b)) ;
%      [Q,R,P] = spqr(A',struct('Q','Householder'));x=spqr_qmult(Q,R'\(P'*b),1);
%      x = spqr_solve (A,b,struct('solution','min2norm')) ;
%
% Entries not present in opts are set to their defaults:
%
%   opts.tol:   columns with norm <= tol are treated as zero. The default is
%       20 * (m+n) * eps * sqrt(max(diag(A'*A))).
%
%   opts.econ:  number of rows of R and columns of Q to return.  m is the
%       default.  n gives the standard economy form.  A value less than the
%       estimated rank r is set to r, so opts.econ = 0 gives the "rank-sized"
%       factorization, where size(R,1) == nnz(diag(R)) == r.
%
%   opts.ordering: a string describing what ordering method to use.
%       Let [m n] = size (S) where S is obtained by removing singletons from A.
%       'default': the default ordering: COLAMD if m<=2*n; otherwise try AMD.
%           If the flops for chol(SP'SP) / nnz(R) >= 500 and nnz(R)/nnz(S)>=5
%           with the ordering P found by AMD, then try METIS, and take the
%           best ordering found.
%       'amd': AMD(S'*S)
%       'colamd': COLAMD(S)
%       'metis': METIS(S'*S), only if METIS is installed
%       'best': try all three (AMD, COLAMD, METIS) and take the best
%       'bestamd': try AMD and COLAMD and take the best
%       'fixed': P=I; this is the only option if P is not present in the output
%       'natural': singleton removal only
%       The singleton pre-ordering permutes A prior to factorization into the
%       form [A11 A12 ; 0 A22] where A11 is upper triangular with
%       all(abs(diag(A11)) > opts.tol) (see spqr_singletons).
%
%   opts.Q: a string describing how Q is returned.  The default is 'discard'
%       if Q is not present in the output, or 'matrix' otherwise.  If Q is
%       present and opts.Q is 'discard', then Q=[] is returned (thus
%       R=spqr(A*P) is [Q,R,P]=spqr(A) where spqr finds P and Q is discarded
%       instead). 'matrix' returns Q as a sparse matrix where A=Q*R or A*P=Q*R.
%       'Householder' returns Q as a struct containing the Householder
%       reflections applied to A to obtain R, resulting in a far sparser Q than
%       the 'matrix' option.  
%
%   opts.permutation: a string describing how P is to be returned.  The default
%       is 'matrix', so that A*P=Q*R.  'vector' gives A(:,P)=Q*R instead.
%
%   opts.spumoni: acts just like spparms('spumoni',k).
%
%   opts.grain, opts.small, opts.nthreads: multitasking control (if compiled
%       with TBB); the scheduler tries to ensure that all parallel tasks have
%       at least max (total_flops / opts.grain, opts.small) flops.  No TBB
%       parallelism is exploited if opts.grain = 1.  opts.nthreads gives the
%       number of threads to use for TBB (which is different than the number of
%       threads used by the BLAS).  opts.nthreads <= 0 means to let TBB
%       determine the number of threads (normally equal to the number of
%       cores); otherwise, use exactly opts.nthreads threads.  Defaults: 1,
%       1e6, and 0, respectively.  TBB is disabled by default since it
%       conflicts with BLAS multithreading.  If you enable TBB, be sure to
%       disable BLAS multithreading with maxNumCompThreads(1), or choose
%       opts.nthreads * (number of BLAS threads) equal to the number of cores.
%       A good value of opts.grain is twice that of opts.nthreads.  If TBB
%       parallelism is enabled, the METIS ordering normally gives the best
%       speedup for large problems.
%
%   opts.min2norm: used by spqr_solve; 'basic' (default), or 'min2norm'.
%       Determines the kind of solution that spqr_solve computes for
%       underdetermined systems.  Has no effect for least-squares problems;
%       ignored by spqr itself.
%
% See also SPQR_QMULT, SPQR_SOLVE, LU, NULL, ORTH, QRDELETE, QRINSERT, QRUPDATE,
%       SPQR_SINGLETONS.

%   Copyright 2008-2009, Timothy A. Davis
%   http://www.cise.ufl.edu/research/sparse

type spqr
error ('spqr mexFunction not found') ;
