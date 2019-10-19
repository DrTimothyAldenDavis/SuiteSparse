
#include <unistd.h>
#include <sys/times.h>

#define TINY_TIME 1e-4

void my_tic (double stats [2])
{
    /* Return the current time */
    /* stats [0]: current wallclock time, in seconds */
    /* stats [1]: user + system time for the process, in seconds */

    struct tms t ;
    double ticks ;

    ticks = (double) sysconf (_SC_CLK_TCK) ;
    stats [0] = (double) times (&t) / ticks ;
    stats [1] = (double) (t.tms_utime + t.tms_stime) / ticks ;

    /* if time is tiny, just return zero */
    if (stats [0] < TINY_TIME) stats [0] = 0 ;
    if (stats [1] < TINY_TIME) stats [1] = 0 ;
}

void my_toc (double stats [2])
{
    /* Return the current time since the last call to umfpack_tic. */
    /* On input, stats holds the values returned by umfpack_tic. */
    /* On ouput, stats holds the time since the last umfpack_tic. */

    double done [2] ;
    my_tic (done) ;

    stats [0] = done [0] - stats [0] ;
    stats [1] = done [1] - stats [1] ;

    if (stats [0] < 0) stats [0] = 0 ;
    if (stats [1] < 0) stats [1] = 0 ;

}


