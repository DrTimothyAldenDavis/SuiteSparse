
typedef long unsigned int size_t;
typedef int wchar_t;

typedef struct
{
    int quot;
    int rem;
} div_t;

typedef struct
{
    long int quot;
    long int rem;
} ldiv_t;

__extension__ typedef struct
{
    long long int quot;
    long long int rem;
} lldiv_t;

extern size_t __ctype_get_mb_cur_max (void) __attribute__ ((__nothrow__));

extern double atof (__const char *__nptr)
    __attribute__ ((__nothrow__)) __attribute__ ((__pure__)) __attribute__ ((__nonnull__ (1)));

extern int atoi (__const char *__nptr)
    __attribute__ ((__nothrow__)) __attribute__ ((__pure__)) __attribute__ ((__nonnull__ (1)));

extern long int atol (__const char *__nptr)
    __attribute__ ((__nothrow__)) __attribute__ ((__pure__)) __attribute__ ((__nonnull__ (1)));

__extension__ extern long long int atoll (__const char *__nptr)
    __attribute__ ((__nothrow__)) __attribute__ ((__pure__)) __attribute__ ((__nonnull__ (1)));

extern double strtod (__const char *__restrict __nptr,
		      char **__restrict __endptr) __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (1)));

extern float strtof (__const char *__restrict __nptr,
		     char **__restrict __endptr) __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (1)));

extern long double strtold (__const char *__restrict __nptr,
			    char **__restrict __endptr) __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (1)));

extern long int strtol (__const char *__restrict __nptr,
			char **__restrict __endptr, int __base)
    __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (1)));

extern unsigned long int strtoul (__const char *__restrict __nptr,
				  char **__restrict __endptr, int __base)
    __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (1)));

__extension__
    extern long long int strtoq (__const char *__restrict __nptr,
				 char **__restrict __endptr, int __base)
    __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (1)));

__extension__
    extern unsigned long long int strtouq (__const char *__restrict __nptr,
					   char **__restrict __endptr, int __base)
    __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (1)));

__extension__
    extern long long int strtoll (__const char *__restrict __nptr,
				  char **__restrict __endptr, int __base)
    __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (1)));

__extension__
    extern unsigned long long int strtoull (__const char *__restrict __nptr,
					    char **__restrict __endptr, int __base)
    __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (1)));

extern char *l64a (long int __n) __attribute__ ((__nothrow__));

extern long int a64l (__const char *__s)
    __attribute__ ((__nothrow__)) __attribute__ ((__pure__)) __attribute__ ((__nonnull__ (1)));

typedef unsigned char __u_char;
typedef unsigned short int __u_short;
typedef unsigned int __u_int;
typedef unsigned long int __u_long;

typedef signed char __int8_t;
typedef unsigned char __uint8_t;
typedef signed short int __int16_t;
typedef unsigned short int __uint16_t;
typedef signed int __int32_t;
typedef unsigned int __uint32_t;

typedef signed long int __int64_t;
typedef unsigned long int __uint64_t;

typedef long int __quad_t;
typedef unsigned long int __u_quad_t;

typedef unsigned long int __dev_t;
typedef unsigned int __uid_t;
typedef unsigned int __gid_t;
typedef unsigned long int __ino_t;
typedef unsigned long int __ino64_t;
typedef unsigned int __mode_t;
typedef unsigned long int __nlink_t;
typedef long int __off_t;
typedef long int __off64_t;
typedef int __pid_t;
typedef struct
{
    int __val[2];
} __fsid_t;
typedef long int __clock_t;
typedef unsigned long int __rlim_t;
typedef unsigned long int __rlim64_t;
typedef unsigned int __id_t;
typedef long int __time_t;
typedef unsigned int __useconds_t;
typedef long int __suseconds_t;

typedef int __daddr_t;
typedef long int __swblk_t;
typedef int __key_t;

typedef int __clockid_t;

typedef void *__timer_t;

typedef long int __blksize_t;

typedef long int __blkcnt_t;
typedef long int __blkcnt64_t;

typedef unsigned long int __fsblkcnt_t;
typedef unsigned long int __fsblkcnt64_t;

typedef unsigned long int __fsfilcnt_t;
typedef unsigned long int __fsfilcnt64_t;

typedef long int __ssize_t;

typedef __off64_t __loff_t;
typedef __quad_t *__qaddr_t;
typedef char *__caddr_t;

typedef long int __intptr_t;

typedef unsigned int __socklen_t;

typedef __u_char u_char;
typedef __u_short u_short;
typedef __u_int u_int;
typedef __u_long u_long;
typedef __quad_t quad_t;
typedef __u_quad_t u_quad_t;
typedef __fsid_t fsid_t;

typedef __loff_t loff_t;

typedef __ino_t ino_t;
typedef __dev_t dev_t;

typedef __gid_t gid_t;

typedef __mode_t mode_t;

typedef __nlink_t nlink_t;

typedef __uid_t uid_t;

typedef __off_t off_t;
typedef __pid_t pid_t;

typedef __id_t id_t;

typedef __ssize_t ssize_t;

typedef __daddr_t daddr_t;
typedef __caddr_t caddr_t;

typedef __key_t key_t;

typedef __time_t time_t;

typedef __clockid_t clockid_t;
typedef __timer_t timer_t;

typedef unsigned long int ulong;
typedef unsigned short int ushort;
typedef unsigned int uint;
typedef int int8_t __attribute__ ((__mode__ (__QI__)));
typedef int int16_t __attribute__ ((__mode__ (__HI__)));
typedef int int32_t __attribute__ ((__mode__ (__SI__)));
typedef int int64_t __attribute__ ((__mode__ (__DI__)));

typedef unsigned int u_int8_t __attribute__ ((__mode__ (__QI__)));
typedef unsigned int u_int16_t __attribute__ ((__mode__ (__HI__)));
typedef unsigned int u_int32_t __attribute__ ((__mode__ (__SI__)));
typedef unsigned int u_int64_t __attribute__ ((__mode__ (__DI__)));

typedef int register_t __attribute__ ((__mode__ (__word__)));

typedef int __sig_atomic_t;

typedef struct
{
    unsigned long int __val[(1024 / (8 * sizeof (unsigned long int)))];
} __sigset_t;

typedef __sigset_t sigset_t;

struct timespec
{
    __time_t tv_sec;
    long int tv_nsec;
};

struct timeval
{
    __time_t tv_sec;
    __suseconds_t tv_usec;
};

typedef __suseconds_t suseconds_t;

typedef long int __fd_mask;
typedef struct
{

    __fd_mask __fds_bits[1024 / (8 * (int) sizeof (__fd_mask))];

}
fd_set;

typedef __fd_mask fd_mask;

extern int select (int __nfds, fd_set * __restrict __readfds,
		   fd_set * __restrict __writefds,
		   fd_set * __restrict __exceptfds, struct timeval *__restrict __timeout);
extern int pselect (int __nfds, fd_set * __restrict __readfds,
		    fd_set * __restrict __writefds,
		    fd_set * __restrict __exceptfds,
		    const struct timespec *__restrict __timeout, const __sigset_t * __restrict __sigmask);

__extension__ extern unsigned int gnu_dev_major (unsigned long long int __dev) __attribute__ ((__nothrow__));
__extension__ extern unsigned int gnu_dev_minor (unsigned long long int __dev) __attribute__ ((__nothrow__));
__extension__
    extern unsigned long long int gnu_dev_makedev (unsigned int __major,
						   unsigned int __minor) __attribute__ ((__nothrow__));
typedef __blkcnt_t blkcnt_t;

typedef __fsblkcnt_t fsblkcnt_t;

typedef __fsfilcnt_t fsfilcnt_t;
typedef unsigned long int pthread_t;

typedef union
{
    char __size[56];
    long int __align;
} pthread_attr_t;

typedef struct __pthread_internal_list
{
    struct __pthread_internal_list *__prev;
    struct __pthread_internal_list *__next;
} __pthread_list_t;
typedef union
{
    struct __pthread_mutex_s
    {
	int __lock;
	unsigned int __count;
	int __owner;

	unsigned int __nusers;

	int __kind;

	int __spins;
	__pthread_list_t __list;
    } __data;
    char __size[40];
    long int __align;
} pthread_mutex_t;

typedef union
{
    char __size[4];
    int __align;
} pthread_mutexattr_t;

typedef union
{
    struct
    {
	int __lock;
	unsigned int __futex;
	__extension__ unsigned long long int __total_seq;
	__extension__ unsigned long long int __wakeup_seq;
	__extension__ unsigned long long int __woken_seq;
	void *__mutex;
	unsigned int __nwaiters;
	unsigned int __broadcast_seq;
    } __data;
    char __size[48];
    __extension__ long long int __align;
} pthread_cond_t;

typedef union
{
    char __size[4];
    int __align;
} pthread_condattr_t;

typedef unsigned int pthread_key_t;

typedef int pthread_once_t;

typedef union
{

    struct
    {
	int __lock;
	unsigned int __nr_readers;
	unsigned int __readers_wakeup;
	unsigned int __writer_wakeup;
	unsigned int __nr_readers_queued;
	unsigned int __nr_writers_queued;
	int __writer;
	int __shared;
	unsigned long int __pad1;
	unsigned long int __pad2;

	unsigned int __flags;
    } __data;
    char __size[56];
    long int __align;
} pthread_rwlock_t;

typedef union
{
    char __size[8];
    long int __align;
} pthread_rwlockattr_t;

typedef volatile int pthread_spinlock_t;

typedef union
{
    char __size[32];
    long int __align;
} pthread_barrier_t;

typedef union
{
    char __size[4];
    int __align;
} pthread_barrierattr_t;

extern long int random (void) __attribute__ ((__nothrow__));

extern void srandom (unsigned int __seed) __attribute__ ((__nothrow__));

extern char *initstate (unsigned int __seed, char *__statebuf,
			size_t __statelen) __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (2)));

extern char *setstate (char *__statebuf) __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (1)));

struct random_data
{
    int32_t *fptr;
    int32_t *rptr;
    int32_t *state;
    int rand_type;
    int rand_deg;
    int rand_sep;
    int32_t *end_ptr;
};

extern int random_r (struct random_data *__restrict __buf,
		     int32_t * __restrict __result) __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (1, 2)));

extern int srandom_r (unsigned int __seed, struct random_data *__buf)
    __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (2)));

extern int initstate_r (unsigned int __seed, char *__restrict __statebuf,
			size_t __statelen,
			struct random_data *__restrict __buf)
    __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (2, 4)));

extern int setstate_r (char *__restrict __statebuf,
		       struct random_data *__restrict __buf)
    __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (1, 2)));

extern int rand (void) __attribute__ ((__nothrow__));

extern void srand (unsigned int __seed) __attribute__ ((__nothrow__));

extern int rand_r (unsigned int *__seed) __attribute__ ((__nothrow__));

extern double drand48 (void) __attribute__ ((__nothrow__));
extern double erand48 (unsigned short int __xsubi[3]) __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (1)));

extern long int lrand48 (void) __attribute__ ((__nothrow__));
extern long int nrand48 (unsigned short int __xsubi[3]) __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (1)));

extern long int mrand48 (void) __attribute__ ((__nothrow__));
extern long int jrand48 (unsigned short int __xsubi[3]) __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (1)));

extern void srand48 (long int __seedval) __attribute__ ((__nothrow__));
extern unsigned short int *seed48 (unsigned short int __seed16v[3])
    __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (1)));
extern void lcong48 (unsigned short int __param[7]) __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (1)));

struct drand48_data
{
    unsigned short int __x[3];
    unsigned short int __old_x[3];
    unsigned short int __c;
    unsigned short int __init;
    unsigned long long int __a;
};

extern int drand48_r (struct drand48_data *__restrict __buffer,
		      double *__restrict __result) __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (1, 2)));
extern int erand48_r (unsigned short int __xsubi[3],
		      struct drand48_data *__restrict __buffer,
		      double *__restrict __result) __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (1, 2)));

extern int lrand48_r (struct drand48_data *__restrict __buffer,
		      long int *__restrict __result) __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (1, 2)));
extern int nrand48_r (unsigned short int __xsubi[3],
		      struct drand48_data *__restrict __buffer,
		      long int *__restrict __result) __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (1, 2)));

extern int mrand48_r (struct drand48_data *__restrict __buffer,
		      long int *__restrict __result) __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (1, 2)));
extern int jrand48_r (unsigned short int __xsubi[3],
		      struct drand48_data *__restrict __buffer,
		      long int *__restrict __result) __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (1, 2)));

extern int srand48_r (long int __seedval, struct drand48_data *__buffer)
    __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (2)));

extern int seed48_r (unsigned short int __seed16v[3],
		     struct drand48_data *__buffer) __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (1, 2)));

extern int lcong48_r (unsigned short int __param[7],
		      struct drand48_data *__buffer) __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (1, 2)));

extern void *malloc (size_t __size) __attribute__ ((__nothrow__)) __attribute__ ((__malloc__));

extern void *calloc (size_t __nmemb, size_t __size) __attribute__ ((__nothrow__)) __attribute__ ((__malloc__));

extern void *realloc (void *__ptr, size_t __size)
    __attribute__ ((__nothrow__)) __attribute__ ((__warn_unused_result__));

extern void free (void *__ptr) __attribute__ ((__nothrow__));

extern void cfree (void *__ptr) __attribute__ ((__nothrow__));

extern void *alloca (size_t __size) __attribute__ ((__nothrow__));

extern void *valloc (size_t __size) __attribute__ ((__nothrow__)) __attribute__ ((__malloc__));

extern int posix_memalign (void **__memptr, size_t __alignment, size_t __size)
    __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (1)));

extern void abort (void) __attribute__ ((__nothrow__)) __attribute__ ((__noreturn__));

extern int atexit (void (*__func) (void)) __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (1)));

extern int on_exit (void (*__func) (int __status, void *__arg), void *__arg)
    __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (1)));

extern void exit (int __status) __attribute__ ((__nothrow__)) __attribute__ ((__noreturn__));

extern void _Exit (int __status) __attribute__ ((__nothrow__)) __attribute__ ((__noreturn__));

extern char *getenv (__const char *__name) __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (1)));

extern char *__secure_getenv (__const char *__name) __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (1)));

extern int putenv (char *__string) __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (1)));

extern int setenv (__const char *__name, __const char *__value, int __replace)
    __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (2)));

extern int unsetenv (__const char *__name) __attribute__ ((__nothrow__));

extern int clearenv (void) __attribute__ ((__nothrow__));
extern char *mktemp (char *__template) __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (1)));
extern int mkstemp (char *__template) __attribute__ ((__nonnull__ (1)));
extern int mkstemps (char *__template, int __suffixlen) __attribute__ ((__nonnull__ (1)));
extern char *mkdtemp (char *__template) __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (1)));

extern int system (__const char *__command);

extern char *realpath (__const char *__restrict __name, char *__restrict __resolved) __attribute__ ((__nothrow__));

typedef int (*__compar_fn_t) (__const void *, __const void *);

extern void *bsearch (__const void *__key, __const void *__base,
		      size_t __nmemb, size_t __size, __compar_fn_t __compar) __attribute__ ((__nonnull__ (1, 2, 5)));

extern void qsort (void *__base, size_t __nmemb, size_t __size,
		   __compar_fn_t __compar) __attribute__ ((__nonnull__ (1, 4)));
extern int abs (int __x) __attribute__ ((__nothrow__)) __attribute__ ((__const__));
extern long int labs (long int __x) __attribute__ ((__nothrow__)) __attribute__ ((__const__));

__extension__ extern long long int llabs (long long int __x) __attribute__ ((__nothrow__)) __attribute__ ((__const__));

extern div_t div (int __numer, int __denom) __attribute__ ((__nothrow__)) __attribute__ ((__const__));
extern ldiv_t ldiv (long int __numer, long int __denom) __attribute__ ((__nothrow__)) __attribute__ ((__const__));

__extension__ extern lldiv_t lldiv (long long int __numer,
				    long long int __denom) __attribute__ ((__nothrow__)) __attribute__ ((__const__));

extern char *ecvt (double __value, int __ndigit, int *__restrict __decpt,
		   int *__restrict __sign) __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (3, 4)));

extern char *fcvt (double __value, int __ndigit, int *__restrict __decpt,
		   int *__restrict __sign) __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (3, 4)));

extern char *gcvt (double __value, int __ndigit, char *__buf)
    __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (3)));

extern char *qecvt (long double __value, int __ndigit,
		    int *__restrict __decpt, int *__restrict __sign)
    __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (3, 4)));
extern char *qfcvt (long double __value, int __ndigit,
		    int *__restrict __decpt, int *__restrict __sign)
    __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (3, 4)));
extern char *qgcvt (long double __value, int __ndigit, char *__buf)
    __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (3)));

extern int ecvt_r (double __value, int __ndigit, int *__restrict __decpt,
		   int *__restrict __sign, char *__restrict __buf,
		   size_t __len) __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (3, 4, 5)));
extern int fcvt_r (double __value, int __ndigit, int *__restrict __decpt,
		   int *__restrict __sign, char *__restrict __buf,
		   size_t __len) __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (3, 4, 5)));

extern int qecvt_r (long double __value, int __ndigit,
		    int *__restrict __decpt, int *__restrict __sign,
		    char *__restrict __buf, size_t __len)
    __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (3, 4, 5)));
extern int qfcvt_r (long double __value, int __ndigit,
		    int *__restrict __decpt, int *__restrict __sign,
		    char *__restrict __buf, size_t __len)
    __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (3, 4, 5)));

extern int mblen (__const char *__s, size_t __n) __attribute__ ((__nothrow__));

extern int mbtowc (wchar_t * __restrict __pwc, __const char *__restrict __s, size_t __n) __attribute__ ((__nothrow__));

extern int wctomb (char *__s, wchar_t __wchar) __attribute__ ((__nothrow__));

extern size_t mbstowcs (wchar_t * __restrict __pwcs,
			__const char *__restrict __s, size_t __n) __attribute__ ((__nothrow__));

extern size_t wcstombs (char *__restrict __s,
			__const wchar_t * __restrict __pwcs, size_t __n) __attribute__ ((__nothrow__));

extern int rpmatch (__const char *__response) __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (1)));
extern int posix_openpt (int __oflag);
extern int getloadavg (double __loadavg[], int __nelem) __attribute__ ((__nothrow__)) __attribute__ ((__nonnull__ (1)));

typedef struct SuiteSparse_config_struct
{
    void *(*malloc_memory) (size_t);
    void *(*realloc_memory) (void *, size_t);
    void (*free_memory) (void *);
    void *(*calloc_memory) (size_t, size_t);

} SuiteSparse_config;

void *SuiteSparse_malloc (size_t nitems, size_t size_of_item, int *ok, SuiteSparse_config * config);

void *SuiteSparse_free (void *p, SuiteSparse_config * config);

void SuiteSparse_tic (double tic[2]);

double SuiteSparse_toc (double tic[2]);

double SuiteSparse_time (void);

void *SuiteSparse_malloc (size_t nitems, size_t size_of_item, int *ok, SuiteSparse_config * config)
{
    void *p;
    if (nitems < 1)
	nitems = 1;
    if (nitems * size_of_item != ((double) nitems) * size_of_item)
    {

	*ok = 0;
	return (((void *) 0));
    }
    if (!config || config->malloc_memory == ((void *) 0))
    {

	p = (void *) malloc (nitems * size_of_item);
    }
    else
    {

	p = (void *) (config->malloc_memory) (nitems * size_of_item);
    }
    *ok = (p != ((void *) 0));
    return (p);
}

void *SuiteSparse_free (void *p, SuiteSparse_config * config)
{
    if (p)
    {
	if (!config || config->free_memory == ((void *) 0))
	{

	    free (p);
	}
	else
	{

	    (config->free_memory) (p);
	}
    }
    return (((void *) 0));
}

typedef __clock_t clock_t;

struct tm
{
    int tm_sec;
    int tm_min;
    int tm_hour;
    int tm_mday;
    int tm_mon;
    int tm_year;
    int tm_wday;
    int tm_yday;
    int tm_isdst;

    long int tm_gmtoff;
    __const char *tm_zone;

};

struct itimerspec
{
    struct timespec it_interval;
    struct timespec it_value;
};

struct sigevent;

extern clock_t clock (void) __attribute__ ((__nothrow__));

extern time_t time (time_t * __timer) __attribute__ ((__nothrow__));

extern double difftime (time_t __time1, time_t __time0) __attribute__ ((__nothrow__)) __attribute__ ((__const__));

extern time_t mktime (struct tm *__tp) __attribute__ ((__nothrow__));

extern size_t strftime (char *__restrict __s, size_t __maxsize,
			__const char *__restrict __format,
			__const struct tm *__restrict __tp) __attribute__ ((__nothrow__));

typedef struct __locale_struct
{

    struct locale_data *__locales[13];

    const unsigned short int *__ctype_b;
    const int *__ctype_tolower;
    const int *__ctype_toupper;

    const char *__names[13];
} *__locale_t;

typedef __locale_t locale_t;

extern size_t strftime_l (char *__restrict __s, size_t __maxsize,
			  __const char *__restrict __format,
			  __const struct tm *__restrict __tp, __locale_t __loc) __attribute__ ((__nothrow__));

extern struct tm *gmtime (__const time_t * __timer) __attribute__ ((__nothrow__));

extern struct tm *localtime (__const time_t * __timer) __attribute__ ((__nothrow__));

extern struct tm *gmtime_r (__const time_t * __restrict __timer,
			    struct tm *__restrict __tp) __attribute__ ((__nothrow__));

extern struct tm *localtime_r (__const time_t * __restrict __timer,
			       struct tm *__restrict __tp) __attribute__ ((__nothrow__));

extern char *asctime (__const struct tm *__tp) __attribute__ ((__nothrow__));

extern char *ctime (__const time_t * __timer) __attribute__ ((__nothrow__));

extern char *asctime_r (__const struct tm *__restrict __tp, char *__restrict __buf) __attribute__ ((__nothrow__));

extern char *ctime_r (__const time_t * __restrict __timer, char *__restrict __buf) __attribute__ ((__nothrow__));

extern char *__tzname[2];
extern int __daylight;
extern long int __timezone;

extern char *tzname[2];

extern void tzset (void) __attribute__ ((__nothrow__));

extern int daylight;
extern long int timezone;

extern int stime (__const time_t * __when) __attribute__ ((__nothrow__));
extern time_t timegm (struct tm *__tp) __attribute__ ((__nothrow__));

extern time_t timelocal (struct tm *__tp) __attribute__ ((__nothrow__));

extern int dysize (int __year) __attribute__ ((__nothrow__)) __attribute__ ((__const__));
extern int nanosleep (__const struct timespec *__requested_time, struct timespec *__remaining);

extern int clock_getres (clockid_t __clock_id, struct timespec *__res) __attribute__ ((__nothrow__));

extern int clock_gettime (clockid_t __clock_id, struct timespec *__tp) __attribute__ ((__nothrow__));

extern int clock_settime (clockid_t __clock_id, __const struct timespec *__tp) __attribute__ ((__nothrow__));

extern int clock_nanosleep (clockid_t __clock_id, int __flags, __const struct timespec *__req, struct timespec *__rem);

extern int clock_getcpuclockid (pid_t __pid, clockid_t * __clock_id) __attribute__ ((__nothrow__));

extern int timer_create (clockid_t __clock_id,
			 struct sigevent *__restrict __evp,
			 timer_t * __restrict __timerid) __attribute__ ((__nothrow__));

extern int timer_delete (timer_t __timerid) __attribute__ ((__nothrow__));

extern int timer_settime (timer_t __timerid, int __flags,
			  __const struct itimerspec *__restrict __value,
			  struct itimerspec *__restrict __ovalue) __attribute__ ((__nothrow__));

extern int timer_gettime (timer_t __timerid, struct itimerspec *__value) __attribute__ ((__nothrow__));

extern int timer_getoverrun (timer_t __timerid) __attribute__ ((__nothrow__));

void SuiteSparse_tic (double tic[2])
{

    struct timespec t;
    clock_gettime (1, &t);
    tic[0] = (double) (t.tv_sec);
    tic[1] = (double) (t.tv_nsec);
}

double SuiteSparse_toc (double tic[2])
{
    double toc[2];
    SuiteSparse_tic (toc);
    return ((toc[0] - tic[0]) + 1e-9 * (toc[1] - tic[1]));
}

double SuiteSparse_time (void)
{
    double toc[2];
    SuiteSparse_tic (toc);
    return (toc[0] + 1e-9 * toc[1]);
}
