/* ========================================================================== */
/* === MAXTRANS ============================================================= */
/* ========================================================================== */

/* Finds a column permutation that maximizes the number of entries on the
 * diagonal of a sparse matrix.  See btf.h for more information.
 */ 

#include "btf.h"
#include "btf_internal.h"

#ifndef RECURSIVE

/* ========================================================================== */
/* === dfs: non-recursive version (default) ================================= */
/* ========================================================================== */

/* Perform a depth-first-search starting at column k, to find an augmenting
 * path.  An augmenting path is a sequence of row/column pairs (i1,k), (i2,j1),
 * (i3,j2), ..., (i(s+1), js), such that all of the following properties hold:
 *
 *	* column k is not matched to any row
 *	* entries in the path are nonzero
 *	* the pairs (i1,j1), (i2,j2), (i3,j3) ..., (is,js) have been 
 *	    previously matched to each other
 *	* (i(s+1), js) is nonzero, and row i(s+1) is not matched to any column
 *
 * Once this path is found, the matching can be changed to the set of pairs
 * path.  An augmenting path is a sequence of row/column pairs
 *
 *	(i1,k), (i2,j1), (i3,j2), ..., (i(s+1), js)
 *
 * Once a row is matched with a column it remains matched with some column, but
 * not necessarily the column it was first matched with.
 *
 * This routine is very similar to the dfs routine in klu_kernel.c, in the
 * KLU sparse LU factorization package.
 *
 * The algorithm is based on the paper "On Algorithms for obtaining a maximum
 * transversal" by Iain Duff, ACM Trans. Mathematical Software, vol 7, no. 1,
 * pp. 315-330, and "Algorithm 575: Permutations for a zero-free diagonal",
 * same issue, pp. 387-390.  The code here is a new implementation of that
 * algorithm, with different data structures and control flow.  After writing
 * this code, I carefully compared my algorithm with MC21A/B (ACM Algorithm 575)
 * Some of the comparisons are partial because I didn't dig deeply into all of
 * the details of MC21A/B, such as how the stack is maintained.  The following
 * arguments are essentially identical between this code and MC21A:
 *
 * maxtrans	MC21A,B
 * --------	-------
 * n		N	    identical
 * k		JORD	    identical
 * Ap		IP	    column / row pointers
 * Ai		ICN	    row / column indices
 * Ap[n]	LICN	    length of index array (# of nonzeros in A)
 * Match	IPERM	    output column / row permutation
 * nfound	NUMNZ	    # of nonzeros on diagonal of permuted matrix
 * Flag		CV	    mark a node as visited by the depth-first-search
 *
 * The following are different, but analogous:
 *
 * Cheap	ARP	    indicates what part of the a column / row has
 *			    already been matched.
 *
 * The following arguments are very different:
 *
 * -		LENR	    # of entries in each row/column (unused in maxtrans)
 * Pstack	OUT	    Pstack keeps track of where we are in the depth-
 *			    first-search scan of column j.  I think that OUT
 *			    plays a similar role in MC21B, but I'm unsure.
 * Istack	PR	    keeps track of the rows in the path.  PR is a link
 *			    list, though, whereas Istack is a stack.  Maxtrans
 *			    does not use any link lists.
 * Jstack	OUT? PR?    the stack for nodes in the path (unsure)
 *
 * The following control structures are roughly comparable:
 *
 * maxtrans			MC21B
 * --------			-----
 * for (k = 0 ; k < n ; k++)	DO 100 JORD=1,N
 * while (head >= 0)		DO 70 K=1,JORD
 * for (p = Cheap [j] ; ...)	DO 20 II=IN1,IN2
 * for (p = head ; ...)		DO 90 K=1,JORD
 */

static int dfs
(
    int k,		/* which stage of the main loop we're in */
    int Ap [ ],		/* column pointers, size n+1 */
    int Ai [ ],		/* row indices, size nz = Ap [n] */
    int Match [ ],	/* size n,  Match [i] = j if col j matched to i */
    int Cheap [ ],	/* rows Ai [Ap [j] .. Cheap [j]-1] alread matched */
    int Flag [ ],	/* Flag [j] = k if j already visited this stage */
    int Istack [ ],	/* size n.  Row index stack. */
    int Jstack [ ],	/* size n.  Column index stack. */
    int Pstack [ ]	/* size n.  Keeps track of position in adjacency list */
)
{
    /* local variables, but "global" to all DFS levels: */
    int found ;	/* true if match found.  */
    int head ;	/* top of stack */

    /* variables that are purely local to any one DFS level: */
    int j2 ;	/* the next DFS goes to node j2 */
    int pend ;	/* one past the end of the adjacency list for node j */

    /* variables that need to be pushed then popped from the stack: */
    int i ;	/* the row tentatively matched to i if DFS successful */
    int j ;	/* the DFS is at the current node j */
    int p ;	/* current index into the adj. list for node j */
    /* the variables i, j, and p are stacked in Istack, Jstack, and Pstack */

    /* start a DFS to find a match for column k */
    found = FALSE ;
    i = EMPTY ;
    head = 0 ;
    Jstack [0] = k ;
    ASSERT (Flag [k] != k) ;

    while (head >= 0)
    {
	j = Jstack [head] ;
	pend = Ap [j+1] ;

	if (Flag [j] != k)	    /* a node is not yet visited */
	{

	    /* -------------------------------------------------------------- */
	    /* prework for node j */
	    /* -------------------------------------------------------------- */

	    /* first time that j has been visited */
	    Flag [j] = k ;
	    /* cheap assignment: find the next unmatched row in col j */
	    for (p = Cheap [j] ; p < pend && !found ; p++)
	    {
		i = Ai [p] ;
		found = (Match [i] == EMPTY) ;
	    }
	    Cheap [j] = p ;

	    /* -------------------------------------------------------------- */

	    /* prepare for DFS */
	    if (found)
	    {
		/* end of augmenting path, column j matched with row i */
		Istack [head] = i ;
		break ;
	    }
	    /* set Pstack [head] to the first entry in column j to scan */
	    Pstack [head] = Ap [j] ;
	}

	/* ------------------------------------------------------------------ */
	/* DFS for nodes adjacent to j */
	/* ------------------------------------------------------------------ */

	/* If cheap assignment not made, continue the dfs.  All rows in column
	 * j are already matched.  Add the adjacent nodes to the stack by
	 * iterating through until finding another non-visited node. */
	for (p = Pstack [head] ; p < pend ; p++)
	{
	    i = Ai [p] ;
	    j2 = Match [i] ;
	    ASSERT (j2 != EMPTY) ;
	    if (Flag [j2] != k)
	    {
		/* Node j2 is not yet visited, start a dfs on node j2.
		 * Keep track of where we left off in the scan of adjacency list
		 * of node j so we can restart j where we left off. */
		Pstack [head] = p + 1 ;
		/* Push j2 onto the stack and immediately break
		 * so we can recurse on node j2.  Also keep track of row i
		 * which (if this dfs works) will be matched with the
		 * current node j. */
		Istack [head] = i ;
		Jstack [++head] = j2 ;
		break ;
	    }
	}

	/* node j is done, but the postwork is postponed - see below */
	if (p == pend)
	{
	    /* If all adjacent nodes of j are already visited, pop j from
	     * stack and continue.  We failed to find a match. */
	    head-- ;
	}
    }

    /* postwork for all nodes j in the stack */
    /* unwind the path and make the corresponding matches */
    if (found)
    {
	for (p = head ; p >= 0 ; p--)
	{
	    j = Jstack [p] ;
	    i = Istack [p] ;

	    /* -------------------------------------------------------------- */
	    /* postwork for node j */
	    /* -------------------------------------------------------------- */
	    /* if found, match row i with column j */
	    Match [i] = j ;
	}
    }
    return (found) ;
}

#else

/* ========================================================================== */
/* === dfs: recursive version (only for illustration) ======================= */
/* ========================================================================== */

/* The following is a recursive version of dfs, which computes identical results
 * as the non-recursive dfs.  It is included here because it is easier to read.
 * Compare the comments in the code below with the identical comments in the
 * non-recursive code above, and that will help you see the correlation between
 * the two routines.
 *
 * This routine can cause stack overflow, and is thus not recommended for heavy
 * usage, particularly for large matrices.  To help in delaying stack overflow,
 * global variables are used, reducing the amount of information each call to
 * dfs places on the call/return stack (the integers i, j, p, and the return
 * address).  To try this version, compile the code with -DRECURSIVE or include
 * the following line at the top of this file:
#define RECURSIVE
 */

static int found, *Ap, *Ai, *Match, k, *Cheap, *Flag ;

static void dfs
(
    int j		/* at column j in the DFS */
)
{
    int p, i ;

    /* ---------------------------------------------------------------------- */
    /* prework for node j */
    /* ---------------------------------------------------------------------- */

    /* first time that j has been visited */
    Flag [j] = k ;

    /* cheap assignment: find the next unmatched row in col j */
    for (p = Cheap [j] ; p < Ap [j+1] && !found ; p++)
    {
	i = Ai [p] ;
	found = (Match [i] == EMPTY) ;
    }
    Cheap [j] = p ;

    /* ---------------------------------------------------------------------- */
    /* DFS for nodes adjacent to j */
    /* ---------------------------------------------------------------------- */

    /* If cheap assignment not made, continue the dfs.  All rows in column
     * j are already matched.  Add the adjacent nodes to the stack by
     * iterating through until finding another non-visited node. */
    for (p = Ap [j] ; p < Ap [j+1] && !found ; p++)
    {
	i = Ai [p] ;
	if (Flag [Match [i]] != k)
	{
	    dfs (Match [i]) ;
	}
    }

    /* ---------------------------------------------------------------------- */
    /* postwork for node j */
    /* ---------------------------------------------------------------------- */

    /* if found, match row i with column j */
    if (found)
    {
	Match [i] = j ;
    }
}

#endif


/* ========================================================================== */
/* === maxtrans ============================================================= */
/* ========================================================================== */

#ifndef RECURSIVE

int maxtrans	    /* returns # of columns in the matching */
(
    int nrow,	    /* A is nrow-by-ncol in compressed column form */
    int ncol,
    int Ap [ ],	    /* size ncol+1 */
    int Ai [ ],	    /* size nz = Ap [ncol] */

    /* output, not defined on input */
    int Match [ ],  /* size nrow.  Match [i] = j if column j matched to row i */

    /* workspace, not defined on input or output */
    int Work [ ]    /* size 5*ncol */
)

#else

int maxtrans	    /* recursive version - same as above except for Work size */
(
    int nrow,
    int ncol,
    int Ap_in [ ],
    int Ai_in [ ],
    int Match_in [ ],
    int Work [ ]    /* size 2*ncol */
)

#endif

{
    int i, j, nfound, nbadcol ;

#ifndef RECURSIVE
    int k, *Cheap, *Flag, *Istack, *Jstack, *Pstack ;
#else
    Ap = Ap_in ;
    Ai = Ai_in ;
    Match = Match_in ;
#endif

    /* ---------------------------------------------------------------------- */
    /* get workspace and initialize */
    /* ---------------------------------------------------------------------- */

    Cheap  = Work ; Work += ncol ;
    Flag   = Work ; Work += ncol ;

#ifndef RECURSIVE
    /* stack for non-recursive dfs */
    Istack = Work ; Work += ncol ;
    Jstack = Work ; Work += ncol ;
    Pstack = Work ;
#endif

    /* in column j, rows Ai [Ap [j] .. Cheap [j]-1] are known to be matched */
    for (j = 0 ; j < ncol ; j++)
    {
	Cheap [j] = Ap [j] ;
	Flag [j] = EMPTY ; 
    }

    /* all rows and columns are currently unmatched */
    for (i = 0 ; i < nrow ; i++)
    {
	Match [i] = EMPTY ;
    }

    /* ---------------------------------------------------------------------- */
    /* find a matching row for each column k */
    /* ---------------------------------------------------------------------- */

    nfound = 0 ;
    for (k = 0 ; k < ncol ; k++)
    {
	/* find an augmenting path to match some row i to column k */
#ifndef RECURSIVE
	/* non-recursive dfs (default) */
	if (dfs (k, Ap, Ai, Match, Cheap, Flag, Istack, Jstack, Pstack))
#else
	/* recursive dfs (for illustration only) */
	found = FALSE ;
	dfs (k) ;
	if (found)
#endif
	{
	    /* we found it.  Match [i] = k for some row i has been done. */
	    nfound++ ;
	}
    }

    /* ---------------------------------------------------------------------- */
    /* complete the permutation if the matrix is structurally singular */
    /* ---------------------------------------------------------------------- */

    /* note that at this point, all entries in Flag are less than ncol */

    if (nfound < ncol)
    {
	/* flag all matched columns */
	for (i = 0 ; i < nrow ; i++)
	{
	    j = Match [i] ;
	    if (j != EMPTY)
	    {
		/* printf ("i is OK %d matched to j %d\n", i, j) ; */
		/* row i and column j are matched */
		Flag [j] = ncol ;
	    }
	}
	/* make a list of all unmatched columns (use Cheap as workspace) */
	nbadcol = 0 ;
	for (j = 0 ; j < ncol ; j++)
	{
	    if (Flag [j] != ncol)
	    {
		Cheap [nbadcol++] = j ;
		/* printf ("j is matched to nobody %d\n", j) ; */
	    }
	}
	ASSERT (nfound + nbadcol == ncol) ;
	/* make an assignment for each unmatched row */
	for (i = 0 ; i < nrow ; i++)
	{
	    if (Match [i] == EMPTY && nbadcol > 0)
	    {
		/* get an unmatched column j */
		j = Cheap [--nbadcol] ;
		/* assign j to row i and flag the entry by "flipping" it */
		Match [i] = MAXTRANS_FLIP (j) ;
		/* printf ("need to force i %d with j %d\n", i, j) ; */
	    }
	}
    }

    /* ---------------------------------------------------------------------- */
    /* return the Match, and the # of matches made */
    /* ---------------------------------------------------------------------- */

    /* At this point, row i is matched to j = Match [i] if j >= 0.  i is an
     * unmatched row if Match [i] < 0.
     *
     * the permutation can be recovered as follows: Row i is
     * matched with column j, where j = MAXTRANS_UNFLIP (Match [i]) and where j
     * will always be in the valid range 0 to ncol-1.  The entry A(i,j) is zero
     * if MAXTRANS_ISFLIPPED (Match [i]) is true, and nonzero otherwise.  nfound
     * is the number of entries in the Match array that are non-negative. */

    return (nfound) ;
}
