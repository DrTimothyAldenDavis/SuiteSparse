#-------------------------------------------------------------------------------
# SPEX/Python/spex_python_demo.py: demo of 3 backslash functions with different input
#                           matrices
#-------------------------------------------------------------------------------

# SPEX: (c) 2022-2026, Christopher Lourenco, Jinhao Chen,
# Lorena Mejia Domenzain, Erick Moreno-Centeno, and Timothy A. Davis.
# All Rights Reserved.
# SPDX-License-Identifier: GPL-2.0-or-later or LGPL-3.0-or-later

#------------------------------------------------------------------------------

# SPEX is a package for solving sparse linear systems of equations
# with a roundoff-free integer-preserving method.  The result is
# always exact, unless the matrix A is perfectly singular.

# Import SPEX
import SPEXpy as SPEX
from SPEXpy import Options

# Import scientific computing
import numpy as np
from numpy.random import default_rng
from scipy.sparse import csc_matrix
from scipy.sparse import random
from scipy import stats


##--------------------------------------------------------------------------
## Cholesky
##--------------------------------------------------------------------------

# Create A and B
print("Demoing the Cholesky interface")
row = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])
col = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2])
data = np.array([4, 12, -16, 12, 37, -43, -16, -43, 98],dtype=np.float64)
A1=csc_matrix((data, (row, col)), shape=(3, 3))
b1=np.ones(3,dtype=np.float64)

# Solve
x=SPEX.cholesky_backslash(A1,b1)
print("\nThe Cholesky solution is:\n")
print(x)


##--------------------------------------------------------------------------
## LU
##--------------------------------------------------------------------------

# Generate a random sparse matrix A and populate b
print("Demoing the LU interface")
n=10
rng = default_rng()
rvs = stats.poisson(25, loc=10).rvs
S = random(n, n, density=0.7, random_state=rng, data_rvs=rvs)
S2=S+np.eye(n)
A2=csc_matrix(S2)
b2=np.ones(n,dtype=np.float64)

# Solve
options=Options("string")
x=SPEX.lu_backslash(A2,b2,options)
print("\nThe LU solution is:\n")
print(x)

##--------------------------------------------------------------------------
## QR
##--------------------------------------------------------------------------

# Generate a random sparse matrix A and populate b
print("Demoing the QR solution")
n=7
m=10
rng = default_rng()
rvs = stats.poisson(25, loc=10).rvs
S = random(m, n, density=0.7, random_state=rng, data_rvs=rvs)
S2=S+np.eye(m, n)
A3=csc_matrix(S2)
b3=np.ones(m,dtype=np.float64)

# Solve
options=Options("string")
x=SPEX.qr_backslash(A3,b3,options)
print("\nThe QR solution is:\n")
print(x)

print("\nDemoing rank")
print("\nThe rank of square matrix 1 is:")
r = SPEX.rank(A1)
print(r)

print("\nThe rank of square matrix 2 is:")
r = SPEX.rank(A2)
print(r)

print("\nThe rank of rectangular matrix is:")
r = SPEX.rank(A3)
print(r)


##--------------------------------------------------------------------------
## Backslash
##--------------------------------------------------------------------------

print("Demoing SPEX backslash")
# Use the previous matrices

# Solve
x=SPEX.backslash(A1,b1)
print("\nLDL good")
x=SPEX.backslash(A2,b2)
print("\nLU good")
x=SPEX.backslash(A3,b3)
print("\nQR good")
print("All set, you're ready to go!")
#SPEX.backslash always returns the output as float64


##--------------------------------------------------------------------------
## Regression tests for the Python interface (SPEX 3.4.0)
##--------------------------------------------------------------------------
# Deterministic tests supplement the illustrative/random examples above.
# Run from SPEX/Python with the native Python environment used to load the
# compiled SPEX bridge library (the same environment as for the demo above).
# No additional Python packages are required beyond the demo's dependencies.

from fractions import Fraction
import traceback

failures = []


def check(name, test):
    """Run every regression test, reporting a failure without hiding later tests."""
    try:
        test()
    except Exception:
        failures.append(name)
        print("FAIL:", name)
        traceback.print_exc()
    else:
        print("PASS:", name)


def assert_solution(actual, expected):
    np.testing.assert_allclose(
        np.asarray(actual, dtype=np.float64),
        np.asarray(expected, dtype=np.float64),
        rtol=1e-11, atol=1e-11)


def test_cholesky_spd():
    A = csc_matrix([[4., 1.], [1., 3.]])
    b = np.array([1., 2.], dtype=np.float64)
    assert_solution(SPEX.cholesky_backslash(A, b), [1./11., 7./11.])


def test_lu():
    A = csc_matrix([[2., 1.], [0., 3.]])
    b = np.array([5., 6.], dtype=np.float64)
    assert_solution(SPEX.lu_backslash(A, b), [1.5, 2.])
    assert_solution(SPEX.backslash(A, b), [1.5, 2.])


def test_ldl_bridge():
    # The original package does not export a public Python ldl_backslash.
    # Exercise case 4 of the C bridge directly on a symmetric indefinite A.
    A = csc_matrix([[1., 2.], [2., 1.]])
    b = np.array([3., 3.], dtype=np.float64)
    assert_solution(SPEX.spex_connect(A, b, 2, False, 4), [1., 1.])
    assert_solution(SPEX.backslash(A, b), [1., 1.])


def test_ldl_to_lu_fallback():
    # A is symmetric, but LDL hits a zero leading pivot; LU should succeed.
    A = csc_matrix([[0., 1.], [1., 0.]])
    b = np.array([2., 3.], dtype=np.float64)
    assert_solution(SPEX.backslash(A, b), [3., 2.])


def test_tall_full_rank_qr():
    A = csc_matrix([[1., 0.], [0., 1.], [1., 1.]])
    b = np.array([1., 2., 4.], dtype=np.float64)
    expected = [4./3., 7./3.]
    assert_solution(SPEX.qr_backslash(A, b), expected)
    assert_solution(SPEX.backslash(A, b), expected)


def test_tall_rank_deficient_qr():
    # Rank 2, 3 columns: QR returns a basic *least-squares* solution.
    A = csc_matrix([[1., 0., 1.],
                    [0., 1., 1.],
                    [1., 1., 2.],
                    [0., 0., 0.]])
    b = np.array([1., 2., 4., 3.], dtype=np.float64)
    x = np.asarray(SPEX.qr_backslash(A, b), dtype=np.float64)
    assert len(x) == A.shape[1]
    assert SPEX.rank(A) == 2
    # Least-squares optimality: A.T @ (A @ x - b) = 0.
    np.testing.assert_allclose(A.T @ (A @ x - b), np.zeros(A.shape[1]),
                               rtol=0, atol=1e-10)
    # Basic: no more than rank(A) nonzero components (basis may vary by ordering).
    assert np.count_nonzero(np.abs(x) > 1e-10) <= 2


def test_rank_zero_qr():
    # An explicitly stored-zero CSC pattern avoids requiring empty-pattern support.
    A = csc_matrix((np.array([0., 0.]),
                    (np.array([0, 1]), np.array([0, 1]))), shape=(3, 2))
    b = np.array([1., 2., 3.], dtype=np.float64)
    assert SPEX.rank(A) == 0
    assert_solution(SPEX.qr_backslash(A, b), [0., 0.])
    assert_solution(SPEX.backslash(A, b), [0., 0.])


def test_one_column_zero_qr():
    # Exercises the new n == 1 / rank == 0 branch in QR factorization and solve.
    A = csc_matrix((np.array([0.]),
                    (np.array([0]), np.array([0]))), shape=(3, 1))
    b = np.array([1., 2., 3.], dtype=np.float64)
    assert_solution(SPEX.qr_backslash(A, b), [0.])


def test_wide_full_rank_qr():
    A = csc_matrix([[1., 0., 1.], [0., 1., 1.]])
    b = np.array([2., 1.], dtype=np.float64)
    # Minimum-norm solution is A.T (A A.T)^(-1) b.
    expected = A.T @ np.linalg.solve((A @ A.T).toarray(), b)
    assert_solution(SPEX.qr_backslash(A, b), expected)
    assert_solution(SPEX.backslash(A, b), expected)
    np.testing.assert_allclose(A @ SPEX.qr_backslash(A, b), b,
                               rtol=1e-11, atol=1e-11)


def test_wide_rank_deficient_error():
    A = csc_matrix([[1., 1., 0.], [2., 2., 0.]])
    b = np.array([1., 2.], dtype=np.float64)
    try:
        SPEX.qr_backslash(A, b)
    except SPEX.SPEX_error:
        return  # SPEX_SINGULAR (raised by the Python wrapper)
    raise AssertionError("Expected SPEX_qr_backslash to reject rank-deficient wide A")


def test_square_singular_error():
    A = csc_matrix([[1., 2.], [2., 4.]])
    b = np.array([1., 2.], dtype=np.float64)
    try:
        SPEX.backslash(A, b)
    except SPEX.SPEX_error:
        return  # expected error for square singular A
    raise AssertionError("Expected SPEX_backslash to reject square singular A")


def test_exact_rational_strings():
    A = csc_matrix([[2., 0.], [0., 3.]])
    b = np.array([1., 1.], dtype=np.float64)
    out = SPEX.lu_backslash(A, b, Options("string", "colamd"))
    values = [Fraction(item.decode("ascii") if isinstance(item, bytes)
                       else str(item)) for item in out]
    assert values == [Fraction(1, 2), Fraction(1, 3)]


def test_ordering_options():
    A = csc_matrix([[3., 1., 0.], [1., 4., 1.], [0., 1., 2.]])
    b = np.array([4., 6., 3.], dtype=np.float64)
    expected = np.linalg.solve(A.toarray(), b)
    for ordering in ("none", "colamd", "amd"):
        assert_solution(SPEX.lu_backslash(
            A, b, Options("double", ordering)), expected)
        assert_solution(SPEX.qr_backslash(
            A, b, Options("double", ordering)), expected)


def test_rank():
    assert SPEX.rank(csc_matrix([[1., 0.], [0., 1.]])) == 2
    assert SPEX.rank(csc_matrix([[1., 2.], [2., 4.]])) == 1
    A = csc_matrix([[1., 2., 3.], [2., 4., 6.]])
    assert SPEX.rank(A) == 1


def test_bad_rhs_dimension():
    A = csc_matrix([[1., 0.], [0., 1.], [1., 1.]])
    b = np.array([1., 2.], dtype=np.float64)  # wrong size: needs 3 rows
    try:
        SPEX.qr_backslash(A, b)
    except SPEX.SPEX_error:
        return
    raise AssertionError("Expected an input-dimension error")


def test_repeated_calls():
    # Smoke test for state/cleanup. Passing does NOT prove absence of C memory leaks.
    A = csc_matrix([[2., 0.], [0., 3.]])
    b = np.array([1., 1.], dtype=np.float64)
    for _ in range(100):
        assert_solution(SPEX.backslash(A, b), [0.5, 1./3.])
    for _ in range(25):
        out = SPEX.lu_backslash(A, b, Options("string", "colamd"))
        assert len(out) == 2


print("\n" + "=" * 64)
print("SPEX Python regression tests")
print("=" * 64)

check("Cholesky: SPD solve", test_cholesky_spd)
check("LU: direct and general backslash", test_lu)
check("LDL: direct C bridge and general backslash", test_ldl_bridge)
check("Default backslash: LDL to LU fallback", test_ldl_to_lu_fallback)
check("Tall QR: full-rank least squares", test_tall_full_rank_qr)
check("Tall QR: rank-deficient basic least squares", test_tall_rank_deficient_qr)
check("Tall QR: all-zero matrix (rank 0)", test_rank_zero_qr)
check("Tall QR: one zero column (rank 0)", test_one_column_zero_qr)
check("Wide QR: full-row-rank minimum norm", test_wide_full_rank_qr)
check("Wide QR: rank-deficient error", test_wide_rank_deficient_error)
check("Square singular: error", test_square_singular_error)
check("Exact rational string output", test_exact_rational_strings)
check("Ordering: none, COLAMD, AMD", test_ordering_options)
check("Exact rank: square and rectangular", test_rank)
check("Bad RHS dimensions: error", test_bad_rhs_dimension)
check("Repeated calls: state/output cleanup smoke test", test_repeated_calls)

if failures:
    print("\nFAILED: {} of {} regression tests".format(len(failures), 16))
    print("Failures: " + ", ".join(failures))
    raise SystemExit(1)

print("\nALL 16 PYTHON REGRESSION TESTS PASSED")
