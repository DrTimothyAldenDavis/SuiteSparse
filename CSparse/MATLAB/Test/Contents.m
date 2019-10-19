% CSparse "textbook" MATLAB M-files and mexFunctions, related to CSparse but
% not a part of CSparse itself.
%
%   M-files:
%
%   chol_downdate  - downdate a Cholesky factorization.
%   chol_left      - left-looking Cholesky factorization.
%   chol_left2     - left-looking Cholesky factorization, more details.
%   chol_right     - right-looking Cholesky factorization.
%   chol_super     - left-looking "supernodal" Cholesky factorization.
%   chol_up        - up-looking Cholesky factorization.
%   chol_update    - update a Cholesky factorization.
%   chol_updown    - update or downdate a Cholesky factorization.
%   cond1est       - 1-norm condition estimate.
%   cs_fiedler     - the Fiedler vector of a connected graph.
%   givens2        - find a Givens rotation.
%   house          - find a Householder reflection.
%   lu_left        - left-looking LU factorization.
%   lu_right       - right-looking LU factorization.
%   lu_rightp      - right-looking LU factorization, with partial pivoting.
%   lu_rightpr     - recursive right-looking LU, with partial pivoting.
%   lu_rightr      - recursive right-looking LU.
%   norm1est       - 1-norm estimate.
%   qr_givens      - Givens-rotation QR factorization.
%   qr_givens_full - Givens-rotation QR factorization, for full matrices.
%   qr_left        - left-looking Householder QR factorization.
%   qr_right       - right-looking Householder QR factorization.
%
%   mexFunctions:
%
%   cs_frand       - generate a random finite-element matrix
%   cs_ipvec       - x(p)=b
%   cs_maxtransr   - recursive maximum matching algorithm
%   cs_pvec        - x=b(p)
%   cs_reach       - non-recursive (interface to CSparse cs_reach)
%   cs_reachr      - recursive reach (using a recursive depth-first search)
%   cs_rowcnt      - row counts for sparse Cholesky
%   cs_sqr         - symbolic QR ordering and analysis
%   cs_sparse2     - same as cs_sparse, to test cs_entry function
