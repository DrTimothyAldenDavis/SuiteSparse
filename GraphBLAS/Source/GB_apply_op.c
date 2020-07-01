//------------------------------------------------------------------------------
// GB_apply_op: typecast and apply a unary operator to an array
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2020, All Rights Reserved.
// http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

//------------------------------------------------------------------------------

// Cx = op ((xtype) Ax)

// Cx and Ax may be aliased.
// Compare with GB_transpose_op.c

#include "GB_apply.h"
#include "GB_binop.h"
#include "GB_unused.h"
#ifndef GBCOMPACT
#include "GB_unop__include.h"
#include "GB_binop__include.h"
#endif

void GB_apply_op            // apply a unary operator, Cx = op ((xtype) Ax)
(
    GB_void *Cx,                    // output array, of type op->ztype
        const GrB_UnaryOp op1,          // unary operator to apply
        const GrB_BinaryOp op2,         // binary operator to apply
        const GxB_Scalar scalar,        // scalar to bind to binary operator
        bool binop_bind1st,             // if true, binop(x,Ax) else binop(Ax,y)
    const GB_void *Ax,              // input array, of type Atype
    const GrB_Type Atype,           // type of Ax
    const int64_t anz,              // size of Ax and Cx
    GB_Context Context
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    ASSERT (Cx != NULL) ;
    ASSERT (Ax != NULL) ;
    ASSERT (anz >= 0) ;
    ASSERT (Atype != NULL) ;
    ASSERT (op1 != NULL || op2 != NULL) ;

    //--------------------------------------------------------------------------
    // determine the number of threads to use
    //--------------------------------------------------------------------------

    GB_GET_NTHREADS_MAX (nthreads_max, chunk, Context) ;
    int nthreads = GB_nthreads (anz, chunk, nthreads_max) ;

    //--------------------------------------------------------------------------
    // apply the operator
    //--------------------------------------------------------------------------

    if (op1 != NULL)
    {

        //----------------------------------------------------------------------
        // built-in unary operator
        //----------------------------------------------------------------------

        GrB_UnaryOp op = op1 ;

        #ifndef GBCOMPACT
        bool no_typecasting = (Atype == op->xtype)
            || (op->opcode == GB_IDENTITY_opcode)
            || (op->opcode == GB_ONE_opcode) ;

        if (no_typecasting)
        { 

            // only two workers are allowed to do their own typecasting from
            // the Atype to the xtype of the operator: IDENTITY and ONE.  For
            // all others, the input type Atype must match the op->xtype of the
            // operator.  If this check isn't done, abs.fp32 with fc32 input
            // will map to abs.fc32, based on the type of the input Ax, which is
            // the wrong operator.

            //------------------------------------------------------------------
            // define the worker for the switch factory
            //------------------------------------------------------------------

            #define GB_unop_apply(op,zname,aname) \
                GB_unop_apply_ ## op ## zname ## aname

            #define GB_WORKER(op,zname,ztype,aname,atype)               \
            {                                                           \
                GrB_Info info = GB_unop_apply (op,zname,aname)          \
                    ((ztype *) Cx, (const atype *) Ax, anz, nthreads) ; \
                if (info == GrB_SUCCESS) return ;                       \
            }                                                           \
            break ;

            //------------------------------------------------------------------
            // launch the switch factory
            //------------------------------------------------------------------

            #include "GB_unop_factory.c"
        }
        #endif

        //----------------------------------------------------------------------
        // generic worker: typecast and apply a unary operator
        //----------------------------------------------------------------------

        GB_BURBLE_N (anz, "generic ") ;

        size_t asize = Atype->size ;
        size_t zsize = op->ztype->size ;
        size_t xsize = op->xtype->size ;
        GB_cast_function
            cast_A_to_X = GB_cast_factory (op->xtype->code, Atype->code) ;
        GxB_unary_function fop = op->function ;

        int64_t p ;
        #pragma omp parallel for num_threads(nthreads) schedule(static)
        for (p = 0 ; p < anz ; p++)
        { 
            // xwork = (xtype) Ax [p]
            GB_void xwork [GB_VLA(xsize)] ;
            cast_A_to_X (xwork, Ax +(p*asize), asize) ;
            // Cx [p] = fop (xwork)
            fop (Cx +(p*zsize), xwork) ;
        }

    }
    else
    {

        //----------------------------------------------------------------------
        // built-in binary operator
        //----------------------------------------------------------------------

        GB_Opcode opcode = op2->opcode ;
        GB_Type_code xcode, ycode, zcode ;
        bool op_is_first  = opcode == GB_FIRST_opcode ;
        bool op_is_second = opcode == GB_SECOND_opcode ;
        bool op_is_pair   = opcode == GB_PAIR_opcode ;

        size_t asize = Atype->size ;
        size_t ssize = scalar->type->size ;
        size_t zsize = op2->ztype->size ;
        size_t xsize = op2->xtype->size ;
        size_t ysize = op2->ytype->size ;

        GB_Type_code scode = scalar->type->code ;
        xcode = op2->xtype->code ;
        ycode = op2->ytype->code ;

        // typecast the scalar to the operator input
        bool ignore_scalar = false ;
        size_t ssize_cast ;
        GB_Type_code scode_cast ;
        if (binop_bind1st)
        { 
            ssize_cast = xsize ;
            scode_cast = xcode ;
            ignore_scalar = op_is_second || op_is_pair ;
        }
        else
        { 
            ssize_cast = ysize ;
            scode_cast = ycode ;
            ignore_scalar = op_is_first  || op_is_pair ;
        }
        GB_void swork [GB_VLA(ssize_cast)] ;
        GB_void *scalarx = (GB_void *) scalar->x ;
        if (scode_cast != scode && !ignore_scalar)
        { 
            // typecast the scalar to the operator input, in swork
            GB_cast_function cast_s = GB_cast_factory (scode_cast, scode) ;
            cast_s (swork, scalar->x, ssize) ;
            scalarx = swork ;
        }

        #ifndef GBCOMPACT
        if (binop_bind1st)
        {

            //------------------------------------------------------------------
            // z = op(scalar,Ax)
            //------------------------------------------------------------------

            if (GB_binop_builtin (
                op2->xtype, ignore_scalar,
                Atype,      op_is_first  || op_is_pair,
                op2, false, &opcode, &xcode, &ycode, &zcode))
            { 

                //--------------------------------------------------------------
                // define the worker for the switch factory
                //--------------------------------------------------------------

                #define GB_bind1st(op,xname) GB_bind1st_ ## op ## xname

                #define GB_BINOP_WORKER(op,xname)                       \
                {                                                       \
                    if (GB_bind1st (op, xname) (Cx, scalarx, Ax,        \
                            anz, nthreads) == GrB_SUCCESS) return ;     \
                }                                                       \
                break ;

                //--------------------------------------------------------------
                // launch the switch factory
                //--------------------------------------------------------------

                #define GB_NO_SECOND
                #define GB_NO_PAIR
                #include "GB_binop_factory.c"
            }
        }
        else
        {

            //------------------------------------------------------------------
            // z = op(Ax,scalar)
            //------------------------------------------------------------------

            if (GB_binop_builtin (
                Atype,      op_is_second || op_is_pair,
                op2->ytype, ignore_scalar,
                op2, false, &opcode, &xcode, &ycode, &zcode))
            { 

                //--------------------------------------------------------------
                // define the worker for the switch factory
                //--------------------------------------------------------------

                #define GB_bind2nd(op,xname) GB_bind2nd_ ## op ## xname
                #undef  GB_BINOP_WORKER
                #define GB_BINOP_WORKER(op,xname)                       \
                {                                                       \
                    if (GB_bind2nd (op, xname) (Cx, Ax, scalarx,        \
                            anz, nthreads) == GrB_SUCCESS) return ;     \
                }                                                       \
                break ;

                //--------------------------------------------------------------
                // launch the switch factory
                //--------------------------------------------------------------

                #define GB_NO_FIRST
                #define GB_NO_PAIR
                #include "GB_binop_factory.c"
            }
        }
        #endif

        //----------------------------------------------------------------------
        // generic worker: typecast and apply a binary operator
        //----------------------------------------------------------------------

        GB_BURBLE_N (anz, "generic ") ;
        GB_Type_code acode = Atype->code ;
        GxB_binary_function fop = op2->function ;

        if (binop_bind1st)
        { 
            // Cx = op (scalar,Ax)
            GB_cast_function cast_A_to_Y = GB_cast_factory (ycode, acode) ;
            int64_t p ;
            #pragma omp parallel for num_threads(nthreads) schedule(static)
            for (p = 0 ; p < anz ; p++)
            {
                // ywork = (ytype) Ax [p]
                GB_void ywork [GB_VLA(ysize)] ;
                cast_A_to_Y (ywork, Ax +(p*asize), asize) ;
                // Cx [p] = fop (xwork, ywork)
                fop (Cx +(p*zsize), scalarx, ywork) ;
            }
        }
        else
        { 
            // Cx = op (Ax,scalar)
            GB_cast_function cast_A_to_X = GB_cast_factory (xcode, acode) ;
            int64_t p ;
            #pragma omp parallel for num_threads(nthreads) schedule(static)
            for (p = 0 ; p < anz ; p++)
            {
                // xwork = (xtype) Ax [p]
                GB_void xwork [GB_VLA(xsize)] ;
                cast_A_to_X (xwork, Ax +(p*asize), asize) ;
                // Cx [p] = fop (xwork, ywork)
                fop (Cx +(p*zsize), xwork, scalarx) ;
            }
        }
    }
}

