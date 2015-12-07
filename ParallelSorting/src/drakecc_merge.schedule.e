# 1 "merge.schedule.c"
# 1 "<built-in>"
# 1 "<command-line>"
# 1 "/usr/include/stdc-predef.h" 1 3 4
# 1 "<command-line>" 2
# 1 "merge.schedule.c"
# 1 "/usr/include/stdlib.h" 1 3 4
# 24 "/usr/include/stdlib.h" 3 4
# 1 "/usr/include/features.h" 1 3 4
# 374 "/usr/include/features.h" 3 4
# 1 "/usr/include/x86_64-linux-gnu/sys/cdefs.h" 1 3 4
# 385 "/usr/include/x86_64-linux-gnu/sys/cdefs.h" 3 4
# 1 "/usr/include/x86_64-linux-gnu/bits/wordsize.h" 1 3 4
# 386 "/usr/include/x86_64-linux-gnu/sys/cdefs.h" 2 3 4
# 375 "/usr/include/features.h" 2 3 4
# 398 "/usr/include/features.h" 3 4
# 1 "/usr/include/x86_64-linux-gnu/gnu/stubs.h" 1 3 4
# 10 "/usr/include/x86_64-linux-gnu/gnu/stubs.h" 3 4
# 1 "/usr/include/x86_64-linux-gnu/gnu/stubs-64.h" 1 3 4
# 11 "/usr/include/x86_64-linux-gnu/gnu/stubs.h" 2 3 4
# 399 "/usr/include/features.h" 2 3 4
# 25 "/usr/include/stdlib.h" 2 3 4







# 1 "/usr/lib/gcc/x86_64-linux-gnu/4.8/include/stddef.h" 1 3 4
# 212 "/usr/lib/gcc/x86_64-linux-gnu/4.8/include/stddef.h" 3 4
typedef long unsigned int size_t;
# 324 "/usr/lib/gcc/x86_64-linux-gnu/4.8/include/stddef.h" 3 4
typedef int wchar_t;
# 33 "/usr/include/stdlib.h" 2 3 4








# 1 "/usr/include/x86_64-linux-gnu/bits/waitflags.h" 1 3 4
# 50 "/usr/include/x86_64-linux-gnu/bits/waitflags.h" 3 4
typedef enum
{
  P_ALL,
  P_PID,
  P_PGID
} idtype_t;
# 42 "/usr/include/stdlib.h" 2 3 4
# 1 "/usr/include/x86_64-linux-gnu/bits/waitstatus.h" 1 3 4
# 64 "/usr/include/x86_64-linux-gnu/bits/waitstatus.h" 3 4
# 1 "/usr/include/endian.h" 1 3 4
# 36 "/usr/include/endian.h" 3 4
# 1 "/usr/include/x86_64-linux-gnu/bits/endian.h" 1 3 4
# 37 "/usr/include/endian.h" 2 3 4
# 60 "/usr/include/endian.h" 3 4
# 1 "/usr/include/x86_64-linux-gnu/bits/byteswap.h" 1 3 4
# 27 "/usr/include/x86_64-linux-gnu/bits/byteswap.h" 3 4
# 1 "/usr/include/x86_64-linux-gnu/bits/types.h" 1 3 4
# 27 "/usr/include/x86_64-linux-gnu/bits/types.h" 3 4
# 1 "/usr/include/x86_64-linux-gnu/bits/wordsize.h" 1 3 4
# 28 "/usr/include/x86_64-linux-gnu/bits/types.h" 2 3 4


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
# 121 "/usr/include/x86_64-linux-gnu/bits/types.h" 3 4
# 1 "/usr/include/x86_64-linux-gnu/bits/typesizes.h" 1 3 4
# 122 "/usr/include/x86_64-linux-gnu/bits/types.h" 2 3 4


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
typedef struct { int __val[2]; } __fsid_t;
typedef long int __clock_t;
typedef unsigned long int __rlim_t;
typedef unsigned long int __rlim64_t;
typedef unsigned int __id_t;
typedef long int __time_t;
typedef unsigned int __useconds_t;
typedef long int __suseconds_t;

typedef int __daddr_t;
typedef int __key_t;


typedef int __clockid_t;


typedef void * __timer_t;


typedef long int __blksize_t;




typedef long int __blkcnt_t;
typedef long int __blkcnt64_t;


typedef unsigned long int __fsblkcnt_t;
typedef unsigned long int __fsblkcnt64_t;


typedef unsigned long int __fsfilcnt_t;
typedef unsigned long int __fsfilcnt64_t;


typedef long int __fsword_t;

typedef long int __ssize_t;


typedef long int __syscall_slong_t;

typedef unsigned long int __syscall_ulong_t;



typedef __off64_t __loff_t;
typedef __quad_t *__qaddr_t;
typedef char *__caddr_t;


typedef long int __intptr_t;


typedef unsigned int __socklen_t;
# 28 "/usr/include/x86_64-linux-gnu/bits/byteswap.h" 2 3 4
# 1 "/usr/include/x86_64-linux-gnu/bits/wordsize.h" 1 3 4
# 29 "/usr/include/x86_64-linux-gnu/bits/byteswap.h" 2 3 4






# 1 "/usr/include/x86_64-linux-gnu/bits/byteswap-16.h" 1 3 4
# 36 "/usr/include/x86_64-linux-gnu/bits/byteswap.h" 2 3 4
# 44 "/usr/include/x86_64-linux-gnu/bits/byteswap.h" 3 4
static __inline unsigned int
__bswap_32 (unsigned int __bsx)
{
  return __builtin_bswap32 (__bsx);
}
# 108 "/usr/include/x86_64-linux-gnu/bits/byteswap.h" 3 4
static __inline __uint64_t
__bswap_64 (__uint64_t __bsx)
{
  return __builtin_bswap64 (__bsx);
}
# 61 "/usr/include/endian.h" 2 3 4
# 65 "/usr/include/x86_64-linux-gnu/bits/waitstatus.h" 2 3 4

union wait
  {
    int w_status;
    struct
      {

 unsigned int __w_termsig:7;
 unsigned int __w_coredump:1;
 unsigned int __w_retcode:8;
 unsigned int:16;







      } __wait_terminated;
    struct
      {

 unsigned int __w_stopval:8;
 unsigned int __w_stopsig:8;
 unsigned int:16;






      } __wait_stopped;
  };
# 43 "/usr/include/stdlib.h" 2 3 4
# 67 "/usr/include/stdlib.h" 3 4
typedef union
  {
    union wait *__uptr;
    int *__iptr;
  } __WAIT_STATUS __attribute__ ((__transparent_union__));
# 95 "/usr/include/stdlib.h" 3 4


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


# 139 "/usr/include/stdlib.h" 3 4
extern size_t __ctype_get_mb_cur_max (void) __attribute__ ((__nothrow__ , __leaf__)) __attribute__ ((__warn_unused_result__));




extern double atof (const char *__nptr)
     __attribute__ ((__nothrow__ , __leaf__)) __attribute__ ((__pure__)) __attribute__ ((__nonnull__ (1))) __attribute__ ((__warn_unused_result__));

extern int atoi (const char *__nptr)
     __attribute__ ((__nothrow__ , __leaf__)) __attribute__ ((__pure__)) __attribute__ ((__nonnull__ (1))) __attribute__ ((__warn_unused_result__));

extern long int atol (const char *__nptr)
     __attribute__ ((__nothrow__ , __leaf__)) __attribute__ ((__pure__)) __attribute__ ((__nonnull__ (1))) __attribute__ ((__warn_unused_result__));





__extension__ extern long long int atoll (const char *__nptr)
     __attribute__ ((__nothrow__ , __leaf__)) __attribute__ ((__pure__)) __attribute__ ((__nonnull__ (1))) __attribute__ ((__warn_unused_result__));





extern double strtod (const char *__restrict __nptr,
        char **__restrict __endptr)
     __attribute__ ((__nothrow__ , __leaf__)) __attribute__ ((__nonnull__ (1)));





extern float strtof (const char *__restrict __nptr,
       char **__restrict __endptr) __attribute__ ((__nothrow__ , __leaf__)) __attribute__ ((__nonnull__ (1)));

extern long double strtold (const char *__restrict __nptr,
       char **__restrict __endptr)
     __attribute__ ((__nothrow__ , __leaf__)) __attribute__ ((__nonnull__ (1)));





extern long int strtol (const char *__restrict __nptr,
   char **__restrict __endptr, int __base)
     __attribute__ ((__nothrow__ , __leaf__)) __attribute__ ((__nonnull__ (1)));

extern unsigned long int strtoul (const char *__restrict __nptr,
      char **__restrict __endptr, int __base)
     __attribute__ ((__nothrow__ , __leaf__)) __attribute__ ((__nonnull__ (1)));




__extension__
extern long long int strtoq (const char *__restrict __nptr,
        char **__restrict __endptr, int __base)
     __attribute__ ((__nothrow__ , __leaf__)) __attribute__ ((__nonnull__ (1)));

__extension__
extern unsigned long long int strtouq (const char *__restrict __nptr,
           char **__restrict __endptr, int __base)
     __attribute__ ((__nothrow__ , __leaf__)) __attribute__ ((__nonnull__ (1)));





__extension__
extern long long int strtoll (const char *__restrict __nptr,
         char **__restrict __endptr, int __base)
     __attribute__ ((__nothrow__ , __leaf__)) __attribute__ ((__nonnull__ (1)));

__extension__
extern unsigned long long int strtoull (const char *__restrict __nptr,
     char **__restrict __endptr, int __base)
     __attribute__ ((__nothrow__ , __leaf__)) __attribute__ ((__nonnull__ (1)));

# 276 "/usr/include/stdlib.h" 3 4

extern __inline __attribute__ ((__gnu_inline__)) int
__attribute__ ((__nothrow__ , __leaf__)) atoi (const char *__nptr)
{
  return (int) strtol (__nptr, (char **) ((void *)0), 10);
}
extern __inline __attribute__ ((__gnu_inline__)) long int
__attribute__ ((__nothrow__ , __leaf__)) atol (const char *__nptr)
{
  return strtol (__nptr, (char **) ((void *)0), 10);
}




__extension__ extern __inline __attribute__ ((__gnu_inline__)) long long int
__attribute__ ((__nothrow__ , __leaf__)) atoll (const char *__nptr)
{
  return strtoll (__nptr, (char **) ((void *)0), 10);
}

# 305 "/usr/include/stdlib.h" 3 4
extern char *l64a (long int __n) __attribute__ ((__nothrow__ , __leaf__)) __attribute__ ((__warn_unused_result__));


extern long int a64l (const char *__s)
     __attribute__ ((__nothrow__ , __leaf__)) __attribute__ ((__pure__)) __attribute__ ((__nonnull__ (1))) __attribute__ ((__warn_unused_result__));




# 1 "/usr/include/x86_64-linux-gnu/sys/types.h" 1 3 4
# 27 "/usr/include/x86_64-linux-gnu/sys/types.h" 3 4






typedef __u_char u_char;
typedef __u_short u_short;
typedef __u_int u_int;
typedef __u_long u_long;
typedef __quad_t quad_t;
typedef __u_quad_t u_quad_t;
typedef __fsid_t fsid_t;




typedef __loff_t loff_t;



typedef __ino_t ino_t;
# 60 "/usr/include/x86_64-linux-gnu/sys/types.h" 3 4
typedef __dev_t dev_t;




typedef __gid_t gid_t;




typedef __mode_t mode_t;




typedef __nlink_t nlink_t;




typedef __uid_t uid_t;





typedef __off_t off_t;
# 98 "/usr/include/x86_64-linux-gnu/sys/types.h" 3 4
typedef __pid_t pid_t;





typedef __id_t id_t;




typedef __ssize_t ssize_t;





typedef __daddr_t daddr_t;
typedef __caddr_t caddr_t;





typedef __key_t key_t;
# 132 "/usr/include/x86_64-linux-gnu/sys/types.h" 3 4
# 1 "/usr/include/time.h" 1 3 4
# 57 "/usr/include/time.h" 3 4


typedef __clock_t clock_t;



# 73 "/usr/include/time.h" 3 4


typedef __time_t time_t;



# 91 "/usr/include/time.h" 3 4
typedef __clockid_t clockid_t;
# 103 "/usr/include/time.h" 3 4
typedef __timer_t timer_t;
# 133 "/usr/include/x86_64-linux-gnu/sys/types.h" 2 3 4
# 146 "/usr/include/x86_64-linux-gnu/sys/types.h" 3 4
# 1 "/usr/lib/gcc/x86_64-linux-gnu/4.8/include/stddef.h" 1 3 4
# 147 "/usr/include/x86_64-linux-gnu/sys/types.h" 2 3 4



typedef unsigned long int ulong;
typedef unsigned short int ushort;
typedef unsigned int uint;
# 194 "/usr/include/x86_64-linux-gnu/sys/types.h" 3 4
typedef int int8_t __attribute__ ((__mode__ (__QI__)));
typedef int int16_t __attribute__ ((__mode__ (__HI__)));
typedef int int32_t __attribute__ ((__mode__ (__SI__)));
typedef int int64_t __attribute__ ((__mode__ (__DI__)));


typedef unsigned int u_int8_t __attribute__ ((__mode__ (__QI__)));
typedef unsigned int u_int16_t __attribute__ ((__mode__ (__HI__)));
typedef unsigned int u_int32_t __attribute__ ((__mode__ (__SI__)));
typedef unsigned int u_int64_t __attribute__ ((__mode__ (__DI__)));

typedef int register_t __attribute__ ((__mode__ (__word__)));
# 219 "/usr/include/x86_64-linux-gnu/sys/types.h" 3 4
# 1 "/usr/include/x86_64-linux-gnu/sys/select.h" 1 3 4
# 30 "/usr/include/x86_64-linux-gnu/sys/select.h" 3 4
# 1 "/usr/include/x86_64-linux-gnu/bits/select.h" 1 3 4
# 22 "/usr/include/x86_64-linux-gnu/bits/select.h" 3 4
# 1 "/usr/include/x86_64-linux-gnu/bits/wordsize.h" 1 3 4
# 23 "/usr/include/x86_64-linux-gnu/bits/select.h" 2 3 4
# 31 "/usr/include/x86_64-linux-gnu/sys/select.h" 2 3 4


# 1 "/usr/include/x86_64-linux-gnu/bits/sigset.h" 1 3 4
# 22 "/usr/include/x86_64-linux-gnu/bits/sigset.h" 3 4
typedef int __sig_atomic_t;




typedef struct
  {
    unsigned long int __val[(1024 / (8 * sizeof (unsigned long int)))];
  } __sigset_t;
# 34 "/usr/include/x86_64-linux-gnu/sys/select.h" 2 3 4



typedef __sigset_t sigset_t;





# 1 "/usr/include/time.h" 1 3 4
# 120 "/usr/include/time.h" 3 4
struct timespec
  {
    __time_t tv_sec;
    __syscall_slong_t tv_nsec;
  };
# 44 "/usr/include/x86_64-linux-gnu/sys/select.h" 2 3 4

# 1 "/usr/include/x86_64-linux-gnu/bits/time.h" 1 3 4
# 30 "/usr/include/x86_64-linux-gnu/bits/time.h" 3 4
struct timeval
  {
    __time_t tv_sec;
    __suseconds_t tv_usec;
  };
# 46 "/usr/include/x86_64-linux-gnu/sys/select.h" 2 3 4


typedef __suseconds_t suseconds_t;





typedef long int __fd_mask;
# 64 "/usr/include/x86_64-linux-gnu/sys/select.h" 3 4
typedef struct
  {






    __fd_mask __fds_bits[1024 / (8 * (int) sizeof (__fd_mask))];


  } fd_set;






typedef __fd_mask fd_mask;
# 96 "/usr/include/x86_64-linux-gnu/sys/select.h" 3 4

# 106 "/usr/include/x86_64-linux-gnu/sys/select.h" 3 4
extern int select (int __nfds, fd_set *__restrict __readfds,
     fd_set *__restrict __writefds,
     fd_set *__restrict __exceptfds,
     struct timeval *__restrict __timeout);
# 118 "/usr/include/x86_64-linux-gnu/sys/select.h" 3 4
extern int pselect (int __nfds, fd_set *__restrict __readfds,
      fd_set *__restrict __writefds,
      fd_set *__restrict __exceptfds,
      const struct timespec *__restrict __timeout,
      const __sigset_t *__restrict __sigmask);





# 1 "/usr/include/x86_64-linux-gnu/bits/select2.h" 1 3 4
# 24 "/usr/include/x86_64-linux-gnu/bits/select2.h" 3 4
extern long int __fdelt_chk (long int __d);
extern long int __fdelt_warn (long int __d)
  __attribute__((__warning__ ("bit outside of fd_set selected")));
# 129 "/usr/include/x86_64-linux-gnu/sys/select.h" 2 3 4



# 220 "/usr/include/x86_64-linux-gnu/sys/types.h" 2 3 4


# 1 "/usr/include/x86_64-linux-gnu/sys/sysmacros.h" 1 3 4
# 24 "/usr/include/x86_64-linux-gnu/sys/sysmacros.h" 3 4


__extension__
extern unsigned int gnu_dev_major (unsigned long long int __dev)
     __attribute__ ((__nothrow__ , __leaf__)) __attribute__ ((__const__));
__extension__
extern unsigned int gnu_dev_minor (unsigned long long int __dev)
     __attribute__ ((__nothrow__ , __leaf__)) __attribute__ ((__const__));
__extension__
extern unsigned long long int gnu_dev_makedev (unsigned int __major,
            unsigned int __minor)
     __attribute__ ((__nothrow__ , __leaf__)) __attribute__ ((__const__));


__extension__ extern __inline __attribute__ ((__gnu_inline__)) __attribute__ ((__const__)) unsigned int
__attribute__ ((__nothrow__ , __leaf__)) gnu_dev_major (unsigned long long int __dev)
{
  return ((__dev >> 8) & 0xfff) | ((unsigned int) (__dev >> 32) & ~0xfff);
}

__extension__ extern __inline __attribute__ ((__gnu_inline__)) __attribute__ ((__const__)) unsigned int
__attribute__ ((__nothrow__ , __leaf__)) gnu_dev_minor (unsigned long long int __dev)
{
  return (__dev & 0xff) | ((unsigned int) (__dev >> 12) & ~0xff);
}

__extension__ extern __inline __attribute__ ((__gnu_inline__)) __attribute__ ((__const__)) unsigned long long int
__attribute__ ((__nothrow__ , __leaf__)) gnu_dev_makedev (unsigned int __major, unsigned int __minor)
{
  return ((__minor & 0xff) | ((__major & 0xfff) << 8)
   | (((unsigned long long int) (__minor & ~0xff)) << 12)
   | (((unsigned long long int) (__major & ~0xfff)) << 32));
}


# 223 "/usr/include/x86_64-linux-gnu/sys/types.h" 2 3 4





typedef __blksize_t blksize_t;






typedef __blkcnt_t blkcnt_t;



typedef __fsblkcnt_t fsblkcnt_t;



typedef __fsfilcnt_t fsfilcnt_t;
# 270 "/usr/include/x86_64-linux-gnu/sys/types.h" 3 4
# 1 "/usr/include/x86_64-linux-gnu/bits/pthreadtypes.h" 1 3 4
# 21 "/usr/include/x86_64-linux-gnu/bits/pthreadtypes.h" 3 4
# 1 "/usr/include/x86_64-linux-gnu/bits/wordsize.h" 1 3 4
# 22 "/usr/include/x86_64-linux-gnu/bits/pthreadtypes.h" 2 3 4
# 60 "/usr/include/x86_64-linux-gnu/bits/pthreadtypes.h" 3 4
typedef unsigned long int pthread_t;


union pthread_attr_t
{
  char __size[56];
  long int __align;
};

typedef union pthread_attr_t pthread_attr_t;





typedef struct __pthread_internal_list
{
  struct __pthread_internal_list *__prev;
  struct __pthread_internal_list *__next;
} __pthread_list_t;
# 90 "/usr/include/x86_64-linux-gnu/bits/pthreadtypes.h" 3 4
typedef union
{
  struct __pthread_mutex_s
  {
    int __lock;
    unsigned int __count;
    int __owner;

    unsigned int __nusers;



    int __kind;

    short __spins;
    short __elision;
    __pthread_list_t __list;
# 124 "/usr/include/x86_64-linux-gnu/bits/pthreadtypes.h" 3 4
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
# 211 "/usr/include/x86_64-linux-gnu/bits/pthreadtypes.h" 3 4
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
# 271 "/usr/include/x86_64-linux-gnu/sys/types.h" 2 3 4



# 315 "/usr/include/stdlib.h" 2 3 4






extern long int random (void) __attribute__ ((__nothrow__ , __leaf__));


extern void srandom (unsigned int __seed) __attribute__ ((__nothrow__ , __leaf__));





extern char *initstate (unsigned int __seed, char *__statebuf,
   size_t __statelen) __attribute__ ((__nothrow__ , __leaf__)) __attribute__ ((__nonnull__ (2)));



extern char *setstate (char *__statebuf) __attribute__ ((__nothrow__ , __leaf__)) __attribute__ ((__nonnull__ (1)));







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
       int32_t *__restrict __result) __attribute__ ((__nothrow__ , __leaf__)) __attribute__ ((__nonnull__ (1, 2)));

extern int srandom_r (unsigned int __seed, struct random_data *__buf)
     __attribute__ ((__nothrow__ , __leaf__)) __attribute__ ((__nonnull__ (2)));

extern int initstate_r (unsigned int __seed, char *__restrict __statebuf,
   size_t __statelen,
   struct random_data *__restrict __buf)
     __attribute__ ((__nothrow__ , __leaf__)) __attribute__ ((__nonnull__ (2, 4)));

extern int setstate_r (char *__restrict __statebuf,
         struct random_data *__restrict __buf)
     __attribute__ ((__nothrow__ , __leaf__)) __attribute__ ((__nonnull__ (1, 2)));






extern int rand (void) __attribute__ ((__nothrow__ , __leaf__));

extern void srand (unsigned int __seed) __attribute__ ((__nothrow__ , __leaf__));




extern int rand_r (unsigned int *__seed) __attribute__ ((__nothrow__ , __leaf__));







extern double drand48 (void) __attribute__ ((__nothrow__ , __leaf__));
extern double erand48 (unsigned short int __xsubi[3]) __attribute__ ((__nothrow__ , __leaf__)) __attribute__ ((__nonnull__ (1)));


extern long int lrand48 (void) __attribute__ ((__nothrow__ , __leaf__));
extern long int nrand48 (unsigned short int __xsubi[3])
     __attribute__ ((__nothrow__ , __leaf__)) __attribute__ ((__nonnull__ (1)));


extern long int mrand48 (void) __attribute__ ((__nothrow__ , __leaf__));
extern long int jrand48 (unsigned short int __xsubi[3])
     __attribute__ ((__nothrow__ , __leaf__)) __attribute__ ((__nonnull__ (1)));


extern void srand48 (long int __seedval) __attribute__ ((__nothrow__ , __leaf__));
extern unsigned short int *seed48 (unsigned short int __seed16v[3])
     __attribute__ ((__nothrow__ , __leaf__)) __attribute__ ((__nonnull__ (1)));
extern void lcong48 (unsigned short int __param[7]) __attribute__ ((__nothrow__ , __leaf__)) __attribute__ ((__nonnull__ (1)));





struct drand48_data
  {
    unsigned short int __x[3];
    unsigned short int __old_x[3];
    unsigned short int __c;
    unsigned short int __init;
    __extension__ unsigned long long int __a;

  };


extern int drand48_r (struct drand48_data *__restrict __buffer,
        double *__restrict __result) __attribute__ ((__nothrow__ , __leaf__)) __attribute__ ((__nonnull__ (1, 2)));
extern int erand48_r (unsigned short int __xsubi[3],
        struct drand48_data *__restrict __buffer,
        double *__restrict __result) __attribute__ ((__nothrow__ , __leaf__)) __attribute__ ((__nonnull__ (1, 2)));


extern int lrand48_r (struct drand48_data *__restrict __buffer,
        long int *__restrict __result)
     __attribute__ ((__nothrow__ , __leaf__)) __attribute__ ((__nonnull__ (1, 2)));
extern int nrand48_r (unsigned short int __xsubi[3],
        struct drand48_data *__restrict __buffer,
        long int *__restrict __result)
     __attribute__ ((__nothrow__ , __leaf__)) __attribute__ ((__nonnull__ (1, 2)));


extern int mrand48_r (struct drand48_data *__restrict __buffer,
        long int *__restrict __result)
     __attribute__ ((__nothrow__ , __leaf__)) __attribute__ ((__nonnull__ (1, 2)));
extern int jrand48_r (unsigned short int __xsubi[3],
        struct drand48_data *__restrict __buffer,
        long int *__restrict __result)
     __attribute__ ((__nothrow__ , __leaf__)) __attribute__ ((__nonnull__ (1, 2)));


extern int srand48_r (long int __seedval, struct drand48_data *__buffer)
     __attribute__ ((__nothrow__ , __leaf__)) __attribute__ ((__nonnull__ (2)));

extern int seed48_r (unsigned short int __seed16v[3],
       struct drand48_data *__buffer) __attribute__ ((__nothrow__ , __leaf__)) __attribute__ ((__nonnull__ (1, 2)));

extern int lcong48_r (unsigned short int __param[7],
        struct drand48_data *__buffer)
     __attribute__ ((__nothrow__ , __leaf__)) __attribute__ ((__nonnull__ (1, 2)));









extern void *malloc (size_t __size) __attribute__ ((__nothrow__ , __leaf__)) __attribute__ ((__malloc__)) __attribute__ ((__warn_unused_result__));

extern void *calloc (size_t __nmemb, size_t __size)
     __attribute__ ((__nothrow__ , __leaf__)) __attribute__ ((__malloc__)) __attribute__ ((__warn_unused_result__));










extern void *realloc (void *__ptr, size_t __size)
     __attribute__ ((__nothrow__ , __leaf__)) __attribute__ ((__warn_unused_result__));

extern void free (void *__ptr) __attribute__ ((__nothrow__ , __leaf__));




extern void cfree (void *__ptr) __attribute__ ((__nothrow__ , __leaf__));



# 1 "/usr/include/alloca.h" 1 3 4
# 24 "/usr/include/alloca.h" 3 4
# 1 "/usr/lib/gcc/x86_64-linux-gnu/4.8/include/stddef.h" 1 3 4
# 25 "/usr/include/alloca.h" 2 3 4







extern void *alloca (size_t __size) __attribute__ ((__nothrow__ , __leaf__));






# 493 "/usr/include/stdlib.h" 2 3 4





extern void *valloc (size_t __size) __attribute__ ((__nothrow__ , __leaf__)) __attribute__ ((__malloc__)) __attribute__ ((__warn_unused_result__));




extern int posix_memalign (void **__memptr, size_t __alignment, size_t __size)
     __attribute__ ((__nothrow__ , __leaf__)) __attribute__ ((__nonnull__ (1))) __attribute__ ((__warn_unused_result__));
# 513 "/usr/include/stdlib.h" 3 4


extern void abort (void) __attribute__ ((__nothrow__ , __leaf__)) __attribute__ ((__noreturn__));



extern int atexit (void (*__func) (void)) __attribute__ ((__nothrow__ , __leaf__)) __attribute__ ((__nonnull__ (1)));
# 530 "/usr/include/stdlib.h" 3 4





extern int on_exit (void (*__func) (int __status, void *__arg), void *__arg)
     __attribute__ ((__nothrow__ , __leaf__)) __attribute__ ((__nonnull__ (1)));






extern void exit (int __status) __attribute__ ((__nothrow__ , __leaf__)) __attribute__ ((__noreturn__));













extern void _Exit (int __status) __attribute__ ((__nothrow__ , __leaf__)) __attribute__ ((__noreturn__));






extern char *getenv (const char *__name) __attribute__ ((__nothrow__ , __leaf__)) __attribute__ ((__nonnull__ (1))) __attribute__ ((__warn_unused_result__));

# 578 "/usr/include/stdlib.h" 3 4
extern int putenv (char *__string) __attribute__ ((__nothrow__ , __leaf__)) __attribute__ ((__nonnull__ (1)));





extern int setenv (const char *__name, const char *__value, int __replace)
     __attribute__ ((__nothrow__ , __leaf__)) __attribute__ ((__nonnull__ (2)));


extern int unsetenv (const char *__name) __attribute__ ((__nothrow__ , __leaf__)) __attribute__ ((__nonnull__ (1)));






extern int clearenv (void) __attribute__ ((__nothrow__ , __leaf__));
# 606 "/usr/include/stdlib.h" 3 4
extern char *mktemp (char *__template) __attribute__ ((__nothrow__ , __leaf__)) __attribute__ ((__nonnull__ (1)));
# 620 "/usr/include/stdlib.h" 3 4
extern int mkstemp (char *__template) __attribute__ ((__nonnull__ (1))) __attribute__ ((__warn_unused_result__));
# 642 "/usr/include/stdlib.h" 3 4
extern int mkstemps (char *__template, int __suffixlen) __attribute__ ((__nonnull__ (1))) __attribute__ ((__warn_unused_result__));
# 663 "/usr/include/stdlib.h" 3 4
extern char *mkdtemp (char *__template) __attribute__ ((__nothrow__ , __leaf__)) __attribute__ ((__nonnull__ (1))) __attribute__ ((__warn_unused_result__));
# 712 "/usr/include/stdlib.h" 3 4





extern int system (const char *__command) __attribute__ ((__warn_unused_result__));

# 734 "/usr/include/stdlib.h" 3 4
extern char *realpath (const char *__restrict __name,
         char *__restrict __resolved) __attribute__ ((__nothrow__ , __leaf__)) __attribute__ ((__warn_unused_result__));






typedef int (*__compar_fn_t) (const void *, const void *);
# 752 "/usr/include/stdlib.h" 3 4



extern void *bsearch (const void *__key, const void *__base,
        size_t __nmemb, size_t __size, __compar_fn_t __compar)
     __attribute__ ((__nonnull__ (1, 2, 5))) __attribute__ ((__warn_unused_result__));


# 1 "/usr/include/x86_64-linux-gnu/bits/stdlib-bsearch.h" 1 3 4
# 19 "/usr/include/x86_64-linux-gnu/bits/stdlib-bsearch.h" 3 4
extern __inline __attribute__ ((__gnu_inline__)) void *
bsearch (const void *__key, const void *__base, size_t __nmemb, size_t __size,
  __compar_fn_t __compar)
{
  size_t __l, __u, __idx;
  const void *__p;
  int __comparison;

  __l = 0;
  __u = __nmemb;
  while (__l < __u)
    {
      __idx = (__l + __u) / 2;
      __p = (void *) (((const char *) __base) + (__idx * __size));
      __comparison = (*__compar) (__key, __p);
      if (__comparison < 0)
 __u = __idx;
      else if (__comparison > 0)
 __l = __idx + 1;
      else
 return (void *) __p;
    }

  return ((void *)0);
}
# 761 "/usr/include/stdlib.h" 2 3 4




extern void qsort (void *__base, size_t __nmemb, size_t __size,
     __compar_fn_t __compar) __attribute__ ((__nonnull__ (1, 4)));
# 775 "/usr/include/stdlib.h" 3 4
extern int abs (int __x) __attribute__ ((__nothrow__ , __leaf__)) __attribute__ ((__const__)) __attribute__ ((__warn_unused_result__));
extern long int labs (long int __x) __attribute__ ((__nothrow__ , __leaf__)) __attribute__ ((__const__)) __attribute__ ((__warn_unused_result__));



__extension__ extern long long int llabs (long long int __x)
     __attribute__ ((__nothrow__ , __leaf__)) __attribute__ ((__const__)) __attribute__ ((__warn_unused_result__));







extern div_t div (int __numer, int __denom)
     __attribute__ ((__nothrow__ , __leaf__)) __attribute__ ((__const__)) __attribute__ ((__warn_unused_result__));
extern ldiv_t ldiv (long int __numer, long int __denom)
     __attribute__ ((__nothrow__ , __leaf__)) __attribute__ ((__const__)) __attribute__ ((__warn_unused_result__));




__extension__ extern lldiv_t lldiv (long long int __numer,
        long long int __denom)
     __attribute__ ((__nothrow__ , __leaf__)) __attribute__ ((__const__)) __attribute__ ((__warn_unused_result__));

# 812 "/usr/include/stdlib.h" 3 4
extern char *ecvt (double __value, int __ndigit, int *__restrict __decpt,
     int *__restrict __sign) __attribute__ ((__nothrow__ , __leaf__)) __attribute__ ((__nonnull__ (3, 4))) __attribute__ ((__warn_unused_result__));




extern char *fcvt (double __value, int __ndigit, int *__restrict __decpt,
     int *__restrict __sign) __attribute__ ((__nothrow__ , __leaf__)) __attribute__ ((__nonnull__ (3, 4))) __attribute__ ((__warn_unused_result__));




extern char *gcvt (double __value, int __ndigit, char *__buf)
     __attribute__ ((__nothrow__ , __leaf__)) __attribute__ ((__nonnull__ (3))) __attribute__ ((__warn_unused_result__));




extern char *qecvt (long double __value, int __ndigit,
      int *__restrict __decpt, int *__restrict __sign)
     __attribute__ ((__nothrow__ , __leaf__)) __attribute__ ((__nonnull__ (3, 4))) __attribute__ ((__warn_unused_result__));
extern char *qfcvt (long double __value, int __ndigit,
      int *__restrict __decpt, int *__restrict __sign)
     __attribute__ ((__nothrow__ , __leaf__)) __attribute__ ((__nonnull__ (3, 4))) __attribute__ ((__warn_unused_result__));
extern char *qgcvt (long double __value, int __ndigit, char *__buf)
     __attribute__ ((__nothrow__ , __leaf__)) __attribute__ ((__nonnull__ (3))) __attribute__ ((__warn_unused_result__));




extern int ecvt_r (double __value, int __ndigit, int *__restrict __decpt,
     int *__restrict __sign, char *__restrict __buf,
     size_t __len) __attribute__ ((__nothrow__ , __leaf__)) __attribute__ ((__nonnull__ (3, 4, 5)));
extern int fcvt_r (double __value, int __ndigit, int *__restrict __decpt,
     int *__restrict __sign, char *__restrict __buf,
     size_t __len) __attribute__ ((__nothrow__ , __leaf__)) __attribute__ ((__nonnull__ (3, 4, 5)));

extern int qecvt_r (long double __value, int __ndigit,
      int *__restrict __decpt, int *__restrict __sign,
      char *__restrict __buf, size_t __len)
     __attribute__ ((__nothrow__ , __leaf__)) __attribute__ ((__nonnull__ (3, 4, 5)));
extern int qfcvt_r (long double __value, int __ndigit,
      int *__restrict __decpt, int *__restrict __sign,
      char *__restrict __buf, size_t __len)
     __attribute__ ((__nothrow__ , __leaf__)) __attribute__ ((__nonnull__ (3, 4, 5)));






extern int mblen (const char *__s, size_t __n) __attribute__ ((__nothrow__ , __leaf__));


extern int mbtowc (wchar_t *__restrict __pwc,
     const char *__restrict __s, size_t __n) __attribute__ ((__nothrow__ , __leaf__));


extern int wctomb (char *__s, wchar_t __wchar) __attribute__ ((__nothrow__ , __leaf__));



extern size_t mbstowcs (wchar_t *__restrict __pwcs,
   const char *__restrict __s, size_t __n) __attribute__ ((__nothrow__ , __leaf__));

extern size_t wcstombs (char *__restrict __s,
   const wchar_t *__restrict __pwcs, size_t __n)
     __attribute__ ((__nothrow__ , __leaf__));








extern int rpmatch (const char *__response) __attribute__ ((__nothrow__ , __leaf__)) __attribute__ ((__nonnull__ (1))) __attribute__ ((__warn_unused_result__));
# 899 "/usr/include/stdlib.h" 3 4
extern int getsubopt (char **__restrict __optionp,
        char *const *__restrict __tokens,
        char **__restrict __valuep)
     __attribute__ ((__nothrow__ , __leaf__)) __attribute__ ((__nonnull__ (1, 2, 3))) __attribute__ ((__warn_unused_result__));
# 951 "/usr/include/stdlib.h" 3 4
extern int getloadavg (double __loadavg[], int __nelem)
     __attribute__ ((__nothrow__ , __leaf__)) __attribute__ ((__nonnull__ (1)));


# 1 "/usr/include/x86_64-linux-gnu/bits/stdlib-float.h" 1 3 4
# 24 "/usr/include/x86_64-linux-gnu/bits/stdlib-float.h" 3 4

extern __inline __attribute__ ((__gnu_inline__)) double
__attribute__ ((__nothrow__ , __leaf__)) atof (const char *__nptr)
{
  return strtod (__nptr, (char **) ((void *)0));
}

# 956 "/usr/include/stdlib.h" 2 3 4



# 1 "/usr/include/x86_64-linux-gnu/bits/stdlib.h" 1 3 4
# 23 "/usr/include/x86_64-linux-gnu/bits/stdlib.h" 3 4
extern char *__realpath_chk (const char *__restrict __name,
        char *__restrict __resolved,
        size_t __resolvedlen) __attribute__ ((__nothrow__ , __leaf__)) __attribute__ ((__warn_unused_result__));
extern char *__realpath_alias (const char *__restrict __name, char *__restrict __resolved) __asm__ ("" "realpath") __attribute__ ((__nothrow__ , __leaf__))

                                                 __attribute__ ((__warn_unused_result__));
extern char *__realpath_chk_warn (const char *__restrict __name, char *__restrict __resolved, size_t __resolvedlen) __asm__ ("" "__realpath_chk") __attribute__ ((__nothrow__ , __leaf__))


                                                __attribute__ ((__warn_unused_result__))
     __attribute__((__warning__ ("second argument of realpath must be either NULL or at " "least PATH_MAX bytes long buffer")))
                                      ;

extern __inline __attribute__ ((__always_inline__)) __attribute__ ((__gnu_inline__)) __attribute__ ((__artificial__)) __attribute__ ((__warn_unused_result__)) char *
__attribute__ ((__nothrow__ , __leaf__)) realpath (const char *__restrict __name, char *__restrict __resolved)
{
  if (__builtin_object_size (__resolved, 2 > 1) != (size_t) -1)
    {




      return __realpath_chk (__name, __resolved, __builtin_object_size (__resolved, 2 > 1));
    }

  return __realpath_alias (__name, __resolved);
}


extern int __ptsname_r_chk (int __fd, char *__buf, size_t __buflen,
       size_t __nreal) __attribute__ ((__nothrow__ , __leaf__)) __attribute__ ((__nonnull__ (2)));
extern int __ptsname_r_alias (int __fd, char *__buf, size_t __buflen) __asm__ ("" "ptsname_r") __attribute__ ((__nothrow__ , __leaf__))

     __attribute__ ((__nonnull__ (2)));
extern int __ptsname_r_chk_warn (int __fd, char *__buf, size_t __buflen, size_t __nreal) __asm__ ("" "__ptsname_r_chk") __attribute__ ((__nothrow__ , __leaf__))


     __attribute__ ((__nonnull__ (2))) __attribute__((__warning__ ("ptsname_r called with buflen bigger than " "size of buf")))
                   ;

extern __inline __attribute__ ((__always_inline__)) __attribute__ ((__gnu_inline__)) __attribute__ ((__artificial__)) int
__attribute__ ((__nothrow__ , __leaf__)) ptsname_r (int __fd, char *__buf, size_t __buflen)
{
  if (__builtin_object_size (__buf, 2 > 1) != (size_t) -1)
    {
      if (!__builtin_constant_p (__buflen))
 return __ptsname_r_chk (__fd, __buf, __buflen, __builtin_object_size (__buf, 2 > 1));
      if (__buflen > __builtin_object_size (__buf, 2 > 1))
 return __ptsname_r_chk_warn (__fd, __buf, __buflen, __builtin_object_size (__buf, 2 > 1));
    }
  return __ptsname_r_alias (__fd, __buf, __buflen);
}


extern int __wctomb_chk (char *__s, wchar_t __wchar, size_t __buflen)
  __attribute__ ((__nothrow__ , __leaf__)) __attribute__ ((__warn_unused_result__));
extern int __wctomb_alias (char *__s, wchar_t __wchar) __asm__ ("" "wctomb") __attribute__ ((__nothrow__ , __leaf__))
              __attribute__ ((__warn_unused_result__));

extern __inline __attribute__ ((__always_inline__)) __attribute__ ((__gnu_inline__)) __attribute__ ((__artificial__)) __attribute__ ((__warn_unused_result__)) int
__attribute__ ((__nothrow__ , __leaf__)) wctomb (char *__s, wchar_t __wchar)
{







  if (__builtin_object_size (__s, 2 > 1) != (size_t) -1 && 16 > __builtin_object_size (__s, 2 > 1))
    return __wctomb_chk (__s, __wchar, __builtin_object_size (__s, 2 > 1));
  return __wctomb_alias (__s, __wchar);
}


extern size_t __mbstowcs_chk (wchar_t *__restrict __dst,
         const char *__restrict __src,
         size_t __len, size_t __dstlen) __attribute__ ((__nothrow__ , __leaf__));
extern size_t __mbstowcs_alias (wchar_t *__restrict __dst, const char *__restrict __src, size_t __len) __asm__ ("" "mbstowcs") __attribute__ ((__nothrow__ , __leaf__))


                                  ;
extern size_t __mbstowcs_chk_warn (wchar_t *__restrict __dst, const char *__restrict __src, size_t __len, size_t __dstlen) __asm__ ("" "__mbstowcs_chk") __attribute__ ((__nothrow__ , __leaf__))



     __attribute__((__warning__ ("mbstowcs called with dst buffer smaller than len " "* sizeof (wchar_t)")))
                        ;

extern __inline __attribute__ ((__always_inline__)) __attribute__ ((__gnu_inline__)) __attribute__ ((__artificial__)) size_t
__attribute__ ((__nothrow__ , __leaf__)) mbstowcs (wchar_t *__restrict __dst, const char *__restrict __src, size_t __len)

{
  if (__builtin_object_size (__dst, 2 > 1) != (size_t) -1)
    {
      if (!__builtin_constant_p (__len))
 return __mbstowcs_chk (__dst, __src, __len,
          __builtin_object_size (__dst, 2 > 1) / sizeof (wchar_t));

      if (__len > __builtin_object_size (__dst, 2 > 1) / sizeof (wchar_t))
 return __mbstowcs_chk_warn (__dst, __src, __len,
         __builtin_object_size (__dst, 2 > 1) / sizeof (wchar_t));
    }
  return __mbstowcs_alias (__dst, __src, __len);
}


extern size_t __wcstombs_chk (char *__restrict __dst,
         const wchar_t *__restrict __src,
         size_t __len, size_t __dstlen) __attribute__ ((__nothrow__ , __leaf__));
extern size_t __wcstombs_alias (char *__restrict __dst, const wchar_t *__restrict __src, size_t __len) __asm__ ("" "wcstombs") __attribute__ ((__nothrow__ , __leaf__))


                                  ;
extern size_t __wcstombs_chk_warn (char *__restrict __dst, const wchar_t *__restrict __src, size_t __len, size_t __dstlen) __asm__ ("" "__wcstombs_chk") __attribute__ ((__nothrow__ , __leaf__))



     __attribute__((__warning__ ("wcstombs called with dst buffer smaller than len")));

extern __inline __attribute__ ((__always_inline__)) __attribute__ ((__gnu_inline__)) __attribute__ ((__artificial__)) size_t
__attribute__ ((__nothrow__ , __leaf__)) wcstombs (char *__restrict __dst, const wchar_t *__restrict __src, size_t __len)

{
  if (__builtin_object_size (__dst, 2 > 1) != (size_t) -1)
    {
      if (!__builtin_constant_p (__len))
 return __wcstombs_chk (__dst, __src, __len, __builtin_object_size (__dst, 2 > 1));
      if (__len > __builtin_object_size (__dst, 2 > 1))
 return __wcstombs_chk_warn (__dst, __src, __len, __builtin_object_size (__dst, 2 > 1));
    }
  return __wcstombs_alias (__dst, __src, __len);
}
# 960 "/usr/include/stdlib.h" 2 3 4
# 968 "/usr/include/stdlib.h" 3 4

# 2 "merge.schedule.c" 2

# 1 "/home/TDDD56/usr/include/drake/schedule.h" 1 3
# 22 "/home/TDDD56/usr/include/drake/schedule.h" 3
# 1 "/home/TDDD56/usr/include/drake/task.h" 1 3
# 22 "/home/TDDD56/usr/include/drake/task.h" 3
# 1 "/usr/lib/gcc/x86_64-linux-gnu/4.8/include/stddef.h" 1 3 4
# 147 "/usr/lib/gcc/x86_64-linux-gnu/4.8/include/stddef.h" 3 4
typedef long int ptrdiff_t;
# 23 "/home/TDDD56/usr/include/drake/task.h" 2 3




enum task_status {TASK_INVALID, TASK_INIT, TASK_START, TASK_RUN, TASK_KILLED, TASK_ZOMBIE, TASK_DESTROY};

typedef enum task_status task_status_t;


typedef unsigned int task_id;


struct link;

typedef struct link link_t;

typedef link_t* link_tp;



# 1 "/home/TDDD56/usr/include/pelib/structure.h" 1 3
# 24 "/home/TDDD56/usr/include/pelib/structure.h" 3
# 1 "/usr/include/stdio.h" 1 3 4
# 29 "/usr/include/stdio.h" 3 4




# 1 "/usr/lib/gcc/x86_64-linux-gnu/4.8/include/stddef.h" 1 3 4
# 34 "/usr/include/stdio.h" 2 3 4
# 44 "/usr/include/stdio.h" 3 4
struct _IO_FILE;



typedef struct _IO_FILE FILE;





# 64 "/usr/include/stdio.h" 3 4
typedef struct _IO_FILE __FILE;
# 74 "/usr/include/stdio.h" 3 4
# 1 "/usr/include/libio.h" 1 3 4
# 31 "/usr/include/libio.h" 3 4
# 1 "/usr/include/_G_config.h" 1 3 4
# 15 "/usr/include/_G_config.h" 3 4
# 1 "/usr/lib/gcc/x86_64-linux-gnu/4.8/include/stddef.h" 1 3 4
# 16 "/usr/include/_G_config.h" 2 3 4




# 1 "/usr/include/wchar.h" 1 3 4
# 82 "/usr/include/wchar.h" 3 4
typedef struct
{
  int __count;
  union
  {

    unsigned int __wch;



    char __wchb[4];
  } __value;
} __mbstate_t;
# 21 "/usr/include/_G_config.h" 2 3 4
typedef struct
{
  __off_t __pos;
  __mbstate_t __state;
} _G_fpos_t;
typedef struct
{
  __off64_t __pos;
  __mbstate_t __state;
} _G_fpos64_t;
# 32 "/usr/include/libio.h" 2 3 4
# 49 "/usr/include/libio.h" 3 4
# 1 "/usr/lib/gcc/x86_64-linux-gnu/4.8/include/stdarg.h" 1 3 4
# 40 "/usr/lib/gcc/x86_64-linux-gnu/4.8/include/stdarg.h" 3 4
typedef __builtin_va_list __gnuc_va_list;
# 50 "/usr/include/libio.h" 2 3 4
# 144 "/usr/include/libio.h" 3 4
struct _IO_jump_t; struct _IO_FILE;
# 154 "/usr/include/libio.h" 3 4
typedef void _IO_lock_t;





struct _IO_marker {
  struct _IO_marker *_next;
  struct _IO_FILE *_sbuf;



  int _pos;
# 177 "/usr/include/libio.h" 3 4
};


enum __codecvt_result
{
  __codecvt_ok,
  __codecvt_partial,
  __codecvt_error,
  __codecvt_noconv
};
# 245 "/usr/include/libio.h" 3 4
struct _IO_FILE {
  int _flags;




  char* _IO_read_ptr;
  char* _IO_read_end;
  char* _IO_read_base;
  char* _IO_write_base;
  char* _IO_write_ptr;
  char* _IO_write_end;
  char* _IO_buf_base;
  char* _IO_buf_end;

  char *_IO_save_base;
  char *_IO_backup_base;
  char *_IO_save_end;

  struct _IO_marker *_markers;

  struct _IO_FILE *_chain;

  int _fileno;



  int _flags2;

  __off_t _old_offset;



  unsigned short _cur_column;
  signed char _vtable_offset;
  char _shortbuf[1];



  _IO_lock_t *_lock;
# 293 "/usr/include/libio.h" 3 4
  __off64_t _offset;
# 302 "/usr/include/libio.h" 3 4
  void *__pad1;
  void *__pad2;
  void *__pad3;
  void *__pad4;
  size_t __pad5;

  int _mode;

  char _unused2[15 * sizeof (int) - 4 * sizeof (void *) - sizeof (size_t)];

};


typedef struct _IO_FILE _IO_FILE;


struct _IO_FILE_plus;

extern struct _IO_FILE_plus _IO_2_1_stdin_;
extern struct _IO_FILE_plus _IO_2_1_stdout_;
extern struct _IO_FILE_plus _IO_2_1_stderr_;
# 338 "/usr/include/libio.h" 3 4
typedef __ssize_t __io_read_fn (void *__cookie, char *__buf, size_t __nbytes);







typedef __ssize_t __io_write_fn (void *__cookie, const char *__buf,
     size_t __n);







typedef int __io_seek_fn (void *__cookie, __off64_t *__pos, int __w);


typedef int __io_close_fn (void *__cookie);
# 390 "/usr/include/libio.h" 3 4
extern int __underflow (_IO_FILE *);
extern int __uflow (_IO_FILE *);
extern int __overflow (_IO_FILE *, int);
# 434 "/usr/include/libio.h" 3 4
extern int _IO_getc (_IO_FILE *__fp);
extern int _IO_putc (int __c, _IO_FILE *__fp);
extern int _IO_feof (_IO_FILE *__fp) __attribute__ ((__nothrow__ , __leaf__));
extern int _IO_ferror (_IO_FILE *__fp) __attribute__ ((__nothrow__ , __leaf__));

extern int _IO_peekc_locked (_IO_FILE *__fp);





extern void _IO_flockfile (_IO_FILE *) __attribute__ ((__nothrow__ , __leaf__));
extern void _IO_funlockfile (_IO_FILE *) __attribute__ ((__nothrow__ , __leaf__));
extern int _IO_ftrylockfile (_IO_FILE *) __attribute__ ((__nothrow__ , __leaf__));
# 464 "/usr/include/libio.h" 3 4
extern int _IO_vfscanf (_IO_FILE * __restrict, const char * __restrict,
   __gnuc_va_list, int *__restrict);
extern int _IO_vfprintf (_IO_FILE *__restrict, const char *__restrict,
    __gnuc_va_list);
extern __ssize_t _IO_padn (_IO_FILE *, int, __ssize_t);
extern size_t _IO_sgetn (_IO_FILE *, void *, size_t);

extern __off64_t _IO_seekoff (_IO_FILE *, __off64_t, int, int);
extern __off64_t _IO_seekpos (_IO_FILE *, __off64_t, int);

extern void _IO_free_backup_area (_IO_FILE *) __attribute__ ((__nothrow__ , __leaf__));
# 75 "/usr/include/stdio.h" 2 3 4




typedef __gnuc_va_list va_list;
# 108 "/usr/include/stdio.h" 3 4


typedef _G_fpos_t fpos_t;




# 164 "/usr/include/stdio.h" 3 4
# 1 "/usr/include/x86_64-linux-gnu/bits/stdio_lim.h" 1 3 4
# 165 "/usr/include/stdio.h" 2 3 4



extern struct _IO_FILE *stdin;
extern struct _IO_FILE *stdout;
extern struct _IO_FILE *stderr;







extern int remove (const char *__filename) __attribute__ ((__nothrow__ , __leaf__));

extern int rename (const char *__old, const char *__new) __attribute__ ((__nothrow__ , __leaf__));




extern int renameat (int __oldfd, const char *__old, int __newfd,
       const char *__new) __attribute__ ((__nothrow__ , __leaf__));








extern FILE *tmpfile (void) __attribute__ ((__warn_unused_result__));
# 209 "/usr/include/stdio.h" 3 4
extern char *tmpnam (char *__s) __attribute__ ((__nothrow__ , __leaf__)) __attribute__ ((__warn_unused_result__));





extern char *tmpnam_r (char *__s) __attribute__ ((__nothrow__ , __leaf__)) __attribute__ ((__warn_unused_result__));
# 227 "/usr/include/stdio.h" 3 4
extern char *tempnam (const char *__dir, const char *__pfx)
     __attribute__ ((__nothrow__ , __leaf__)) __attribute__ ((__malloc__)) __attribute__ ((__warn_unused_result__));








extern int fclose (FILE *__stream);




extern int fflush (FILE *__stream);

# 252 "/usr/include/stdio.h" 3 4
extern int fflush_unlocked (FILE *__stream);
# 266 "/usr/include/stdio.h" 3 4






extern FILE *fopen (const char *__restrict __filename,
      const char *__restrict __modes) __attribute__ ((__warn_unused_result__));




extern FILE *freopen (const char *__restrict __filename,
        const char *__restrict __modes,
        FILE *__restrict __stream) __attribute__ ((__warn_unused_result__));
# 295 "/usr/include/stdio.h" 3 4

# 306 "/usr/include/stdio.h" 3 4
extern FILE *fdopen (int __fd, const char *__modes) __attribute__ ((__nothrow__ , __leaf__)) __attribute__ ((__warn_unused_result__));
# 319 "/usr/include/stdio.h" 3 4
extern FILE *fmemopen (void *__s, size_t __len, const char *__modes)
  __attribute__ ((__nothrow__ , __leaf__)) __attribute__ ((__warn_unused_result__));




extern FILE *open_memstream (char **__bufloc, size_t *__sizeloc) __attribute__ ((__nothrow__ , __leaf__)) __attribute__ ((__warn_unused_result__));






extern void setbuf (FILE *__restrict __stream, char *__restrict __buf) __attribute__ ((__nothrow__ , __leaf__));



extern int setvbuf (FILE *__restrict __stream, char *__restrict __buf,
      int __modes, size_t __n) __attribute__ ((__nothrow__ , __leaf__));





extern void setbuffer (FILE *__restrict __stream, char *__restrict __buf,
         size_t __size) __attribute__ ((__nothrow__ , __leaf__));


extern void setlinebuf (FILE *__stream) __attribute__ ((__nothrow__ , __leaf__));








extern int fprintf (FILE *__restrict __stream,
      const char *__restrict __format, ...);




extern int printf (const char *__restrict __format, ...);

extern int sprintf (char *__restrict __s,
      const char *__restrict __format, ...) __attribute__ ((__nothrow__));





extern int vfprintf (FILE *__restrict __s, const char *__restrict __format,
       __gnuc_va_list __arg);




extern int vprintf (const char *__restrict __format, __gnuc_va_list __arg);

extern int vsprintf (char *__restrict __s, const char *__restrict __format,
       __gnuc_va_list __arg) __attribute__ ((__nothrow__));





extern int snprintf (char *__restrict __s, size_t __maxlen,
       const char *__restrict __format, ...)
     __attribute__ ((__nothrow__)) __attribute__ ((__format__ (__printf__, 3, 4)));

extern int vsnprintf (char *__restrict __s, size_t __maxlen,
        const char *__restrict __format, __gnuc_va_list __arg)
     __attribute__ ((__nothrow__)) __attribute__ ((__format__ (__printf__, 3, 0)));

# 412 "/usr/include/stdio.h" 3 4
extern int vdprintf (int __fd, const char *__restrict __fmt,
       __gnuc_va_list __arg)
     __attribute__ ((__format__ (__printf__, 2, 0)));
extern int dprintf (int __fd, const char *__restrict __fmt, ...)
     __attribute__ ((__format__ (__printf__, 2, 3)));








extern int fscanf (FILE *__restrict __stream,
     const char *__restrict __format, ...) __attribute__ ((__warn_unused_result__));




extern int scanf (const char *__restrict __format, ...) __attribute__ ((__warn_unused_result__));

extern int sscanf (const char *__restrict __s,
     const char *__restrict __format, ...) __attribute__ ((__nothrow__ , __leaf__));
# 443 "/usr/include/stdio.h" 3 4
extern int fscanf (FILE *__restrict __stream, const char *__restrict __format, ...) __asm__ ("" "__isoc99_fscanf")

                          __attribute__ ((__warn_unused_result__));
extern int scanf (const char *__restrict __format, ...) __asm__ ("" "__isoc99_scanf")
                         __attribute__ ((__warn_unused_result__));
extern int sscanf (const char *__restrict __s, const char *__restrict __format, ...) __asm__ ("" "__isoc99_sscanf") __attribute__ ((__nothrow__ , __leaf__))

                      ;
# 463 "/usr/include/stdio.h" 3 4








extern int vfscanf (FILE *__restrict __s, const char *__restrict __format,
      __gnuc_va_list __arg)
     __attribute__ ((__format__ (__scanf__, 2, 0))) __attribute__ ((__warn_unused_result__));





extern int vscanf (const char *__restrict __format, __gnuc_va_list __arg)
     __attribute__ ((__format__ (__scanf__, 1, 0))) __attribute__ ((__warn_unused_result__));


extern int vsscanf (const char *__restrict __s,
      const char *__restrict __format, __gnuc_va_list __arg)
     __attribute__ ((__nothrow__ , __leaf__)) __attribute__ ((__format__ (__scanf__, 2, 0)));
# 494 "/usr/include/stdio.h" 3 4
extern int vfscanf (FILE *__restrict __s, const char *__restrict __format, __gnuc_va_list __arg) __asm__ ("" "__isoc99_vfscanf")



     __attribute__ ((__format__ (__scanf__, 2, 0))) __attribute__ ((__warn_unused_result__));
extern int vscanf (const char *__restrict __format, __gnuc_va_list __arg) __asm__ ("" "__isoc99_vscanf")

     __attribute__ ((__format__ (__scanf__, 1, 0))) __attribute__ ((__warn_unused_result__));
extern int vsscanf (const char *__restrict __s, const char *__restrict __format, __gnuc_va_list __arg) __asm__ ("" "__isoc99_vsscanf") __attribute__ ((__nothrow__ , __leaf__))



     __attribute__ ((__format__ (__scanf__, 2, 0)));
# 522 "/usr/include/stdio.h" 3 4









extern int fgetc (FILE *__stream);
extern int getc (FILE *__stream);





extern int getchar (void);

# 550 "/usr/include/stdio.h" 3 4
extern int getc_unlocked (FILE *__stream);
extern int getchar_unlocked (void);
# 561 "/usr/include/stdio.h" 3 4
extern int fgetc_unlocked (FILE *__stream);











extern int fputc (int __c, FILE *__stream);
extern int putc (int __c, FILE *__stream);





extern int putchar (int __c);

# 594 "/usr/include/stdio.h" 3 4
extern int fputc_unlocked (int __c, FILE *__stream);







extern int putc_unlocked (int __c, FILE *__stream);
extern int putchar_unlocked (int __c);






extern int getw (FILE *__stream);


extern int putw (int __w, FILE *__stream);








extern char *fgets (char *__restrict __s, int __n, FILE *__restrict __stream)
     __attribute__ ((__warn_unused_result__));
# 638 "/usr/include/stdio.h" 3 4
extern char *gets (char *__s) __attribute__ ((__warn_unused_result__)) __attribute__ ((__deprecated__));


# 665 "/usr/include/stdio.h" 3 4
extern __ssize_t __getdelim (char **__restrict __lineptr,
          size_t *__restrict __n, int __delimiter,
          FILE *__restrict __stream) __attribute__ ((__warn_unused_result__));
extern __ssize_t getdelim (char **__restrict __lineptr,
        size_t *__restrict __n, int __delimiter,
        FILE *__restrict __stream) __attribute__ ((__warn_unused_result__));







extern __ssize_t getline (char **__restrict __lineptr,
       size_t *__restrict __n,
       FILE *__restrict __stream) __attribute__ ((__warn_unused_result__));








extern int fputs (const char *__restrict __s, FILE *__restrict __stream);





extern int puts (const char *__s);






extern int ungetc (int __c, FILE *__stream);






extern size_t fread (void *__restrict __ptr, size_t __size,
       size_t __n, FILE *__restrict __stream) __attribute__ ((__warn_unused_result__));




extern size_t fwrite (const void *__restrict __ptr, size_t __size,
        size_t __n, FILE *__restrict __s);

# 737 "/usr/include/stdio.h" 3 4
extern size_t fread_unlocked (void *__restrict __ptr, size_t __size,
         size_t __n, FILE *__restrict __stream) __attribute__ ((__warn_unused_result__));
extern size_t fwrite_unlocked (const void *__restrict __ptr, size_t __size,
          size_t __n, FILE *__restrict __stream);








extern int fseek (FILE *__stream, long int __off, int __whence);




extern long int ftell (FILE *__stream) __attribute__ ((__warn_unused_result__));




extern void rewind (FILE *__stream);

# 773 "/usr/include/stdio.h" 3 4
extern int fseeko (FILE *__stream, __off_t __off, int __whence);




extern __off_t ftello (FILE *__stream) __attribute__ ((__warn_unused_result__));
# 792 "/usr/include/stdio.h" 3 4






extern int fgetpos (FILE *__restrict __stream, fpos_t *__restrict __pos);




extern int fsetpos (FILE *__stream, const fpos_t *__pos);
# 815 "/usr/include/stdio.h" 3 4

# 824 "/usr/include/stdio.h" 3 4


extern void clearerr (FILE *__stream) __attribute__ ((__nothrow__ , __leaf__));

extern int feof (FILE *__stream) __attribute__ ((__nothrow__ , __leaf__)) __attribute__ ((__warn_unused_result__));

extern int ferror (FILE *__stream) __attribute__ ((__nothrow__ , __leaf__)) __attribute__ ((__warn_unused_result__));




extern void clearerr_unlocked (FILE *__stream) __attribute__ ((__nothrow__ , __leaf__));
extern int feof_unlocked (FILE *__stream) __attribute__ ((__nothrow__ , __leaf__)) __attribute__ ((__warn_unused_result__));
extern int ferror_unlocked (FILE *__stream) __attribute__ ((__nothrow__ , __leaf__)) __attribute__ ((__warn_unused_result__));








extern void perror (const char *__s);






# 1 "/usr/include/x86_64-linux-gnu/bits/sys_errlist.h" 1 3 4
# 26 "/usr/include/x86_64-linux-gnu/bits/sys_errlist.h" 3 4
extern int sys_nerr;
extern const char *const sys_errlist[];
# 854 "/usr/include/stdio.h" 2 3 4




extern int fileno (FILE *__stream) __attribute__ ((__nothrow__ , __leaf__)) __attribute__ ((__warn_unused_result__));




extern int fileno_unlocked (FILE *__stream) __attribute__ ((__nothrow__ , __leaf__)) __attribute__ ((__warn_unused_result__));
# 873 "/usr/include/stdio.h" 3 4
extern FILE *popen (const char *__command, const char *__modes) __attribute__ ((__warn_unused_result__));





extern int pclose (FILE *__stream);





extern char *ctermid (char *__s) __attribute__ ((__nothrow__ , __leaf__));
# 913 "/usr/include/stdio.h" 3 4
extern void flockfile (FILE *__stream) __attribute__ ((__nothrow__ , __leaf__));



extern int ftrylockfile (FILE *__stream) __attribute__ ((__nothrow__ , __leaf__)) __attribute__ ((__warn_unused_result__));


extern void funlockfile (FILE *__stream) __attribute__ ((__nothrow__ , __leaf__));
# 934 "/usr/include/stdio.h" 3 4
# 1 "/usr/include/x86_64-linux-gnu/bits/stdio.h" 1 3 4
# 43 "/usr/include/x86_64-linux-gnu/bits/stdio.h" 3 4
extern __inline __attribute__ ((__gnu_inline__)) int
getchar (void)
{
  return _IO_getc (stdin);
}




extern __inline __attribute__ ((__gnu_inline__)) int
fgetc_unlocked (FILE *__fp)
{
  return (__builtin_expect (((__fp)->_IO_read_ptr >= (__fp)->_IO_read_end), 0) ? __uflow (__fp) : *(unsigned char *) (__fp)->_IO_read_ptr++);
}





extern __inline __attribute__ ((__gnu_inline__)) int
getc_unlocked (FILE *__fp)
{
  return (__builtin_expect (((__fp)->_IO_read_ptr >= (__fp)->_IO_read_end), 0) ? __uflow (__fp) : *(unsigned char *) (__fp)->_IO_read_ptr++);
}


extern __inline __attribute__ ((__gnu_inline__)) int
getchar_unlocked (void)
{
  return (__builtin_expect (((stdin)->_IO_read_ptr >= (stdin)->_IO_read_end), 0) ? __uflow (stdin) : *(unsigned char *) (stdin)->_IO_read_ptr++);
}




extern __inline __attribute__ ((__gnu_inline__)) int
putchar (int __c)
{
  return _IO_putc (__c, stdout);
}




extern __inline __attribute__ ((__gnu_inline__)) int
fputc_unlocked (int __c, FILE *__stream)
{
  return (__builtin_expect (((__stream)->_IO_write_ptr >= (__stream)->_IO_write_end), 0) ? __overflow (__stream, (unsigned char) (__c)) : (unsigned char) (*(__stream)->_IO_write_ptr++ = (__c)));
}





extern __inline __attribute__ ((__gnu_inline__)) int
putc_unlocked (int __c, FILE *__stream)
{
  return (__builtin_expect (((__stream)->_IO_write_ptr >= (__stream)->_IO_write_end), 0) ? __overflow (__stream, (unsigned char) (__c)) : (unsigned char) (*(__stream)->_IO_write_ptr++ = (__c)));
}


extern __inline __attribute__ ((__gnu_inline__)) int
putchar_unlocked (int __c)
{
  return (__builtin_expect (((stdout)->_IO_write_ptr >= (stdout)->_IO_write_end), 0) ? __overflow (stdout, (unsigned char) (__c)) : (unsigned char) (*(stdout)->_IO_write_ptr++ = (__c)));
}
# 124 "/usr/include/x86_64-linux-gnu/bits/stdio.h" 3 4
extern __inline __attribute__ ((__gnu_inline__)) int
__attribute__ ((__nothrow__ , __leaf__)) feof_unlocked (FILE *__stream)
{
  return (((__stream)->_flags & 0x10) != 0);
}


extern __inline __attribute__ ((__gnu_inline__)) int
__attribute__ ((__nothrow__ , __leaf__)) ferror_unlocked (FILE *__stream)
{
  return (((__stream)->_flags & 0x20) != 0);
}
# 935 "/usr/include/stdio.h" 2 3 4


# 1 "/usr/include/x86_64-linux-gnu/bits/stdio2.h" 1 3 4
# 23 "/usr/include/x86_64-linux-gnu/bits/stdio2.h" 3 4
extern int __sprintf_chk (char *__restrict __s, int __flag, size_t __slen,
     const char *__restrict __format, ...) __attribute__ ((__nothrow__ , __leaf__));
extern int __vsprintf_chk (char *__restrict __s, int __flag, size_t __slen,
      const char *__restrict __format,
      __gnuc_va_list __ap) __attribute__ ((__nothrow__ , __leaf__));


extern __inline __attribute__ ((__always_inline__)) __attribute__ ((__gnu_inline__)) __attribute__ ((__artificial__)) int
__attribute__ ((__nothrow__ , __leaf__)) sprintf (char *__restrict __s, const char *__restrict __fmt, ...)
{
  return __builtin___sprintf_chk (__s, 2 - 1,
      __builtin_object_size (__s, 2 > 1), __fmt, __builtin_va_arg_pack ());
}






extern __inline __attribute__ ((__always_inline__)) __attribute__ ((__gnu_inline__)) __attribute__ ((__artificial__)) int
__attribute__ ((__nothrow__ , __leaf__)) vsprintf (char *__restrict __s, const char *__restrict __fmt, __gnuc_va_list __ap)

{
  return __builtin___vsprintf_chk (__s, 2 - 1,
       __builtin_object_size (__s, 2 > 1), __fmt, __ap);
}



extern int __snprintf_chk (char *__restrict __s, size_t __n, int __flag,
      size_t __slen, const char *__restrict __format,
      ...) __attribute__ ((__nothrow__ , __leaf__));
extern int __vsnprintf_chk (char *__restrict __s, size_t __n, int __flag,
       size_t __slen, const char *__restrict __format,
       __gnuc_va_list __ap) __attribute__ ((__nothrow__ , __leaf__));


extern __inline __attribute__ ((__always_inline__)) __attribute__ ((__gnu_inline__)) __attribute__ ((__artificial__)) int
__attribute__ ((__nothrow__ , __leaf__)) snprintf (char *__restrict __s, size_t __n, const char *__restrict __fmt, ...)

{
  return __builtin___snprintf_chk (__s, __n, 2 - 1,
       __builtin_object_size (__s, 2 > 1), __fmt, __builtin_va_arg_pack ());
}






extern __inline __attribute__ ((__always_inline__)) __attribute__ ((__gnu_inline__)) __attribute__ ((__artificial__)) int
__attribute__ ((__nothrow__ , __leaf__)) vsnprintf (char *__restrict __s, size_t __n, const char *__restrict __fmt, __gnuc_va_list __ap)

{
  return __builtin___vsnprintf_chk (__s, __n, 2 - 1,
        __builtin_object_size (__s, 2 > 1), __fmt, __ap);
}





extern int __fprintf_chk (FILE *__restrict __stream, int __flag,
     const char *__restrict __format, ...);
extern int __printf_chk (int __flag, const char *__restrict __format, ...);
extern int __vfprintf_chk (FILE *__restrict __stream, int __flag,
      const char *__restrict __format, __gnuc_va_list __ap);
extern int __vprintf_chk (int __flag, const char *__restrict __format,
     __gnuc_va_list __ap);


extern __inline __attribute__ ((__always_inline__)) __attribute__ ((__gnu_inline__)) __attribute__ ((__artificial__)) int
fprintf (FILE *__restrict __stream, const char *__restrict __fmt, ...)
{
  return __fprintf_chk (__stream, 2 - 1, __fmt,
   __builtin_va_arg_pack ());
}

extern __inline __attribute__ ((__always_inline__)) __attribute__ ((__gnu_inline__)) __attribute__ ((__artificial__)) int
printf (const char *__restrict __fmt, ...)
{
  return __printf_chk (2 - 1, __fmt, __builtin_va_arg_pack ());
}







extern __inline __attribute__ ((__always_inline__)) __attribute__ ((__gnu_inline__)) __attribute__ ((__artificial__)) int
vprintf (const char *__restrict __fmt, __gnuc_va_list __ap)
{

  return __vfprintf_chk (stdout, 2 - 1, __fmt, __ap);



}

extern __inline __attribute__ ((__always_inline__)) __attribute__ ((__gnu_inline__)) __attribute__ ((__artificial__)) int
vfprintf (FILE *__restrict __stream,
   const char *__restrict __fmt, __gnuc_va_list __ap)
{
  return __vfprintf_chk (__stream, 2 - 1, __fmt, __ap);
}


extern int __dprintf_chk (int __fd, int __flag, const char *__restrict __fmt,
     ...) __attribute__ ((__format__ (__printf__, 3, 4)));
extern int __vdprintf_chk (int __fd, int __flag,
      const char *__restrict __fmt, __gnuc_va_list __arg)
     __attribute__ ((__format__ (__printf__, 3, 0)));


extern __inline __attribute__ ((__always_inline__)) __attribute__ ((__gnu_inline__)) __attribute__ ((__artificial__)) int
dprintf (int __fd, const char *__restrict __fmt, ...)
{
  return __dprintf_chk (__fd, 2 - 1, __fmt,
   __builtin_va_arg_pack ());
}





extern __inline __attribute__ ((__always_inline__)) __attribute__ ((__gnu_inline__)) __attribute__ ((__artificial__)) int
vdprintf (int __fd, const char *__restrict __fmt, __gnuc_va_list __ap)
{
  return __vdprintf_chk (__fd, 2 - 1, __fmt, __ap);
}
# 227 "/usr/include/x86_64-linux-gnu/bits/stdio2.h" 3 4
extern char *__gets_chk (char *__str, size_t) __attribute__ ((__warn_unused_result__));
extern char *__gets_warn (char *__str) __asm__ ("" "gets")
     __attribute__ ((__warn_unused_result__)) __attribute__((__warning__ ("please use fgets or getline instead, gets can't " "specify buffer size")))
                               ;

extern __inline __attribute__ ((__always_inline__)) __attribute__ ((__gnu_inline__)) __attribute__ ((__artificial__)) __attribute__ ((__warn_unused_result__)) char *
gets (char *__str)
{
  if (__builtin_object_size (__str, 2 > 1) != (size_t) -1)
    return __gets_chk (__str, __builtin_object_size (__str, 2 > 1));
  return __gets_warn (__str);
}


extern char *__fgets_chk (char *__restrict __s, size_t __size, int __n,
     FILE *__restrict __stream) __attribute__ ((__warn_unused_result__));
extern char *__fgets_alias (char *__restrict __s, int __n, FILE *__restrict __stream) __asm__ ("" "fgets")

                                        __attribute__ ((__warn_unused_result__));
extern char *__fgets_chk_warn (char *__restrict __s, size_t __size, int __n, FILE *__restrict __stream) __asm__ ("" "__fgets_chk")


     __attribute__ ((__warn_unused_result__)) __attribute__((__warning__ ("fgets called with bigger size than length " "of destination buffer")))
                                 ;

extern __inline __attribute__ ((__always_inline__)) __attribute__ ((__gnu_inline__)) __attribute__ ((__artificial__)) __attribute__ ((__warn_unused_result__)) char *
fgets (char *__restrict __s, int __n, FILE *__restrict __stream)
{
  if (__builtin_object_size (__s, 2 > 1) != (size_t) -1)
    {
      if (!__builtin_constant_p (__n) || __n <= 0)
 return __fgets_chk (__s, __builtin_object_size (__s, 2 > 1), __n, __stream);

      if ((size_t) __n > __builtin_object_size (__s, 2 > 1))
 return __fgets_chk_warn (__s, __builtin_object_size (__s, 2 > 1), __n, __stream);
    }
  return __fgets_alias (__s, __n, __stream);
}

extern size_t __fread_chk (void *__restrict __ptr, size_t __ptrlen,
      size_t __size, size_t __n,
      FILE *__restrict __stream) __attribute__ ((__warn_unused_result__));
extern size_t __fread_alias (void *__restrict __ptr, size_t __size, size_t __n, FILE *__restrict __stream) __asm__ ("" "fread")


            __attribute__ ((__warn_unused_result__));
extern size_t __fread_chk_warn (void *__restrict __ptr, size_t __ptrlen, size_t __size, size_t __n, FILE *__restrict __stream) __asm__ ("" "__fread_chk")




     __attribute__ ((__warn_unused_result__)) __attribute__((__warning__ ("fread called with bigger size * nmemb than length " "of destination buffer")))
                                 ;

extern __inline __attribute__ ((__always_inline__)) __attribute__ ((__gnu_inline__)) __attribute__ ((__artificial__)) __attribute__ ((__warn_unused_result__)) size_t
fread (void *__restrict __ptr, size_t __size, size_t __n,
       FILE *__restrict __stream)
{
  if (__builtin_object_size (__ptr, 0) != (size_t) -1)
    {
      if (!__builtin_constant_p (__size)
   || !__builtin_constant_p (__n)
   || (__size | __n) >= (((size_t) 1) << (8 * sizeof (size_t) / 2)))
 return __fread_chk (__ptr, __builtin_object_size (__ptr, 0), __size, __n, __stream);

      if (__size * __n > __builtin_object_size (__ptr, 0))
 return __fread_chk_warn (__ptr, __builtin_object_size (__ptr, 0), __size, __n, __stream);
    }
  return __fread_alias (__ptr, __size, __n, __stream);
}
# 327 "/usr/include/x86_64-linux-gnu/bits/stdio2.h" 3 4
extern size_t __fread_unlocked_chk (void *__restrict __ptr, size_t __ptrlen,
        size_t __size, size_t __n,
        FILE *__restrict __stream) __attribute__ ((__warn_unused_result__));
extern size_t __fread_unlocked_alias (void *__restrict __ptr, size_t __size, size_t __n, FILE *__restrict __stream) __asm__ ("" "fread_unlocked")


                     __attribute__ ((__warn_unused_result__));
extern size_t __fread_unlocked_chk_warn (void *__restrict __ptr, size_t __ptrlen, size_t __size, size_t __n, FILE *__restrict __stream) __asm__ ("" "__fread_unlocked_chk")




     __attribute__ ((__warn_unused_result__)) __attribute__((__warning__ ("fread_unlocked called with bigger size * nmemb than " "length of destination buffer")))
                                        ;

extern __inline __attribute__ ((__always_inline__)) __attribute__ ((__gnu_inline__)) __attribute__ ((__artificial__)) __attribute__ ((__warn_unused_result__)) size_t
fread_unlocked (void *__restrict __ptr, size_t __size, size_t __n,
  FILE *__restrict __stream)
{
  if (__builtin_object_size (__ptr, 0) != (size_t) -1)
    {
      if (!__builtin_constant_p (__size)
   || !__builtin_constant_p (__n)
   || (__size | __n) >= (((size_t) 1) << (8 * sizeof (size_t) / 2)))
 return __fread_unlocked_chk (__ptr, __builtin_object_size (__ptr, 0), __size, __n,
         __stream);

      if (__size * __n > __builtin_object_size (__ptr, 0))
 return __fread_unlocked_chk_warn (__ptr, __builtin_object_size (__ptr, 0), __size, __n,
       __stream);
    }


  if (__builtin_constant_p (__size)
      && __builtin_constant_p (__n)
      && (__size | __n) < (((size_t) 1) << (8 * sizeof (size_t) / 2))
      && __size * __n <= 8)
    {
      size_t __cnt = __size * __n;
      char *__cptr = (char *) __ptr;
      if (__cnt == 0)
 return 0;

      for (; __cnt > 0; --__cnt)
 {
   int __c = (__builtin_expect (((__stream)->_IO_read_ptr >= (__stream)->_IO_read_end), 0) ? __uflow (__stream) : *(unsigned char *) (__stream)->_IO_read_ptr++);
   if (__c == (-1))
     break;
   *__cptr++ = __c;
 }
      return (__cptr - (char *) __ptr) / __size;
    }

  return __fread_unlocked_alias (__ptr, __size, __n, __stream);
}
# 938 "/usr/include/stdio.h" 2 3 4






# 25 "/home/TDDD56/usr/include/pelib/structure.h" 2 3
# 1 "/usr/lib/gcc/x86_64-linux-gnu/4.8/include/stddef.h" 1 3 4
# 26 "/home/TDDD56/usr/include/pelib/structure.h" 2 3

# 1 "/home/TDDD56/usr/include/pelib/template.h" 1 3
# 28 "/home/TDDD56/usr/include/pelib/structure.h" 2 3
# 59 "/home/TDDD56/usr/include/pelib/structure.h" 3
link_t *
pelib_link_t_alloc_struct();





int
pelib_link_t_alloc_buffer(link_t* obj, size_t n);







int
pelib_link_t_set_buffer(link_t* obj, void* buffer, size_t n);



link_t *
pelib_link_t_alloc();





link_t *
pelib_link_t_alloc_collection(size_t n);






link_t *
pelib_link_t_alloc_from(void* buffer, size_t n);





int
pelib_link_t_init(link_t *obj);





int
pelib_link_t_copy(link_t src, link_t* dst);





int
pelib_link_t_free_struct(link_t *obj);





int
pelib_link_t_free_buffer(link_t *);




int
pelib_link_t_free(link_t *);




int
pelib_link_t_destroy(link_t);




int
pelib_link_t_compare(link_t a, link_t b);






FILE*
pelib_link_t_printf(FILE* str, link_t obj);







FILE*
pelib_link_t_printf_detail(FILE* str, link_t obj, int lvl);






size_t
pelib_link_t_fwrite(link_t obj, FILE* str);






size_t
pelib_link_t_fread(link_t* obj, FILE* str);


char*
pelib_link_t_string(link_t);





char*
pelib_link_t_string_detail(link_t, int);
# 44 "/home/TDDD56/usr/include/drake/task.h" 2 3




# 1 "/home/TDDD56/usr/include/pelib/structure.h" 1 3
# 25 "/home/TDDD56/usr/include/pelib/structure.h" 3
# 1 "/usr/lib/gcc/x86_64-linux-gnu/4.8/include/stddef.h" 1 3 4
# 26 "/home/TDDD56/usr/include/pelib/structure.h" 2 3
# 59 "/home/TDDD56/usr/include/pelib/structure.h" 3
link_tp *
pelib_link_tp_alloc_struct();





int
pelib_link_tp_alloc_buffer(link_tp* obj, size_t n);







int
pelib_link_tp_set_buffer(link_tp* obj, void* buffer, size_t n);



link_tp *
pelib_link_tp_alloc();





link_tp *
pelib_link_tp_alloc_collection(size_t n);






link_tp *
pelib_link_tp_alloc_from(void* buffer, size_t n);





int
pelib_link_tp_init(link_tp *obj);





int
pelib_link_tp_copy(link_tp src, link_tp* dst);





int
pelib_link_tp_free_struct(link_tp *obj);





int
pelib_link_tp_free_buffer(link_tp *);




int
pelib_link_tp_free(link_tp *);




int
pelib_link_tp_destroy(link_tp);




int
pelib_link_tp_compare(link_tp a, link_tp b);






FILE*
pelib_link_tp_printf(FILE* str, link_tp obj);







FILE*
pelib_link_tp_printf_detail(FILE* str, link_tp obj, int lvl);






size_t
pelib_link_tp_fwrite(link_tp obj, FILE* str);






size_t
pelib_link_tp_fread(link_tp* obj, FILE* str);


char*
pelib_link_tp_string(link_tp);





char*
pelib_link_tp_string_detail(link_tp, int);
# 49 "/home/TDDD56/usr/include/drake/task.h" 2 3




# 1 "/home/TDDD56/usr/include/pelib/array.h" 1 3
# 45 "/home/TDDD56/usr/include/pelib/array.h" 3
struct array_link_tp
  {
    size_t capacity;
    size_t length;
    link_tp* data;
  };
typedef struct array_link_tp array_link_tp_t;


# 1 "/home/TDDD56/usr/include/pelib/structure.h" 1 3
# 25 "/home/TDDD56/usr/include/pelib/structure.h" 3
# 1 "/usr/lib/gcc/x86_64-linux-gnu/4.8/include/stddef.h" 1 3 4
# 26 "/home/TDDD56/usr/include/pelib/structure.h" 2 3
# 59 "/home/TDDD56/usr/include/pelib/structure.h" 3
array_link_tp_t *
pelib_array_link_tp_t_alloc_struct();





int
pelib_array_link_tp_t_alloc_buffer(array_link_tp_t* obj, size_t n);







int
pelib_array_link_tp_t_set_buffer(array_link_tp_t* obj, void* buffer, size_t n);



array_link_tp_t *
pelib_array_link_tp_t_alloc();





array_link_tp_t *
pelib_array_link_tp_t_alloc_collection(size_t n);






array_link_tp_t *
pelib_array_link_tp_t_alloc_from(void* buffer, size_t n);





int
pelib_array_link_tp_t_init(array_link_tp_t *obj);





int
pelib_array_link_tp_t_copy(array_link_tp_t src, array_link_tp_t* dst);





int
pelib_array_link_tp_t_free_struct(array_link_tp_t *obj);





int
pelib_array_link_tp_t_free_buffer(array_link_tp_t *);




int
pelib_array_link_tp_t_free(array_link_tp_t *);




int
pelib_array_link_tp_t_destroy(array_link_tp_t);




int
pelib_array_link_tp_t_compare(array_link_tp_t a, array_link_tp_t b);






FILE*
pelib_array_link_tp_t_printf(FILE* str, array_link_tp_t obj);







FILE*
pelib_array_link_tp_t_printf_detail(FILE* str, array_link_tp_t obj, int lvl);






size_t
pelib_array_link_tp_t_fwrite(array_link_tp_t obj, FILE* str);






size_t
pelib_array_link_tp_t_fread(array_link_tp_t* obj, FILE* str);


char*
pelib_array_link_tp_t_string(array_link_tp_t);





char*
pelib_array_link_tp_t_string_detail(array_link_tp_t, int);
# 55 "/home/TDDD56/usr/include/pelib/array.h" 2 3


array_link_tp_t* pelib_array_link_tp_loadfilename(char*);


array_link_tp_t* pelib_array_link_tp_loadfilenamebinary(char*);


array_link_tp_t* pelib_array_link_tp_preloadfilenamebinary(char*);





array_link_tp_t* pelib_array_link_tp_loadfilenamewindowbinary(char*, size_t from, size_t to);

int
pelib_array_link_tp_storefilename(array_link_tp_t*, char*);

int
pelib_array_link_tp_storefilenamebinary(array_link_tp_t*, char*);

int
pelib_array_link_tp_checkascending(array_link_tp_t*);

link_tp
pelib_array_link_tp_read(array_link_tp_t*, size_t i);

int
pelib_array_link_tp_write(array_link_tp_t*, size_t i, link_tp elem);



int
pelib_array_link_tp_append(array_link_tp_t*, link_tp elem);

size_t
pelib_array_link_tp_length(array_link_tp_t*);

size_t
pelib_array_link_tp_capacity(array_link_tp_t*);



int
pelib_array_link_tp_compare(array_link_tp_t* a1, array_link_tp_t* a2);
# 54 "/home/TDDD56/usr/include/drake/task.h" 2 3



struct cross_link;

typedef struct cross_link cross_link_t;

typedef cross_link_t* cross_link_tp;



# 1 "/home/TDDD56/usr/include/pelib/structure.h" 1 3
# 25 "/home/TDDD56/usr/include/pelib/structure.h" 3
# 1 "/usr/lib/gcc/x86_64-linux-gnu/4.8/include/stddef.h" 1 3 4
# 26 "/home/TDDD56/usr/include/pelib/structure.h" 2 3
# 59 "/home/TDDD56/usr/include/pelib/structure.h" 3
cross_link_t *
pelib_cross_link_t_alloc_struct();





int
pelib_cross_link_t_alloc_buffer(cross_link_t* obj, size_t n);







int
pelib_cross_link_t_set_buffer(cross_link_t* obj, void* buffer, size_t n);



cross_link_t *
pelib_cross_link_t_alloc();





cross_link_t *
pelib_cross_link_t_alloc_collection(size_t n);






cross_link_t *
pelib_cross_link_t_alloc_from(void* buffer, size_t n);





int
pelib_cross_link_t_init(cross_link_t *obj);





int
pelib_cross_link_t_copy(cross_link_t src, cross_link_t* dst);





int
pelib_cross_link_t_free_struct(cross_link_t *obj);





int
pelib_cross_link_t_free_buffer(cross_link_t *);




int
pelib_cross_link_t_free(cross_link_t *);




int
pelib_cross_link_t_destroy(cross_link_t);




int
pelib_cross_link_t_compare(cross_link_t a, cross_link_t b);






FILE*
pelib_cross_link_t_printf(FILE* str, cross_link_t obj);







FILE*
pelib_cross_link_t_printf_detail(FILE* str, cross_link_t obj, int lvl);






size_t
pelib_cross_link_t_fwrite(cross_link_t obj, FILE* str);






size_t
pelib_cross_link_t_fread(cross_link_t* obj, FILE* str);


char*
pelib_cross_link_t_string(cross_link_t);





char*
pelib_cross_link_t_string_detail(cross_link_t, int);
# 66 "/home/TDDD56/usr/include/drake/task.h" 2 3




# 1 "/home/TDDD56/usr/include/pelib/structure.h" 1 3
# 25 "/home/TDDD56/usr/include/pelib/structure.h" 3
# 1 "/usr/lib/gcc/x86_64-linux-gnu/4.8/include/stddef.h" 1 3 4
# 26 "/home/TDDD56/usr/include/pelib/structure.h" 2 3
# 59 "/home/TDDD56/usr/include/pelib/structure.h" 3
cross_link_tp *
pelib_cross_link_tp_alloc_struct();





int
pelib_cross_link_tp_alloc_buffer(cross_link_tp* obj, size_t n);







int
pelib_cross_link_tp_set_buffer(cross_link_tp* obj, void* buffer, size_t n);



cross_link_tp *
pelib_cross_link_tp_alloc();





cross_link_tp *
pelib_cross_link_tp_alloc_collection(size_t n);






cross_link_tp *
pelib_cross_link_tp_alloc_from(void* buffer, size_t n);





int
pelib_cross_link_tp_init(cross_link_tp *obj);





int
pelib_cross_link_tp_copy(cross_link_tp src, cross_link_tp* dst);





int
pelib_cross_link_tp_free_struct(cross_link_tp *obj);





int
pelib_cross_link_tp_free_buffer(cross_link_tp *);




int
pelib_cross_link_tp_free(cross_link_tp *);




int
pelib_cross_link_tp_destroy(cross_link_tp);




int
pelib_cross_link_tp_compare(cross_link_tp a, cross_link_tp b);






FILE*
pelib_cross_link_tp_printf(FILE* str, cross_link_tp obj);







FILE*
pelib_cross_link_tp_printf_detail(FILE* str, cross_link_tp obj, int lvl);






size_t
pelib_cross_link_tp_fwrite(cross_link_tp obj, FILE* str);






size_t
pelib_cross_link_tp_fread(cross_link_tp* obj, FILE* str);


char*
pelib_cross_link_tp_string(cross_link_tp);





char*
pelib_cross_link_tp_string_detail(cross_link_tp, int);
# 71 "/home/TDDD56/usr/include/drake/task.h" 2 3




# 1 "/home/TDDD56/usr/include/pelib/array.h" 1 3
# 45 "/home/TDDD56/usr/include/pelib/array.h" 3
struct array_cross_link_tp
  {
    size_t capacity;
    size_t length;
    cross_link_tp* data;
  };
typedef struct array_cross_link_tp array_cross_link_tp_t;


# 1 "/home/TDDD56/usr/include/pelib/structure.h" 1 3
# 25 "/home/TDDD56/usr/include/pelib/structure.h" 3
# 1 "/usr/lib/gcc/x86_64-linux-gnu/4.8/include/stddef.h" 1 3 4
# 26 "/home/TDDD56/usr/include/pelib/structure.h" 2 3
# 59 "/home/TDDD56/usr/include/pelib/structure.h" 3
array_cross_link_tp_t *
pelib_array_cross_link_tp_t_alloc_struct();





int
pelib_array_cross_link_tp_t_alloc_buffer(array_cross_link_tp_t* obj, size_t n);







int
pelib_array_cross_link_tp_t_set_buffer(array_cross_link_tp_t* obj, void* buffer, size_t n);



array_cross_link_tp_t *
pelib_array_cross_link_tp_t_alloc();





array_cross_link_tp_t *
pelib_array_cross_link_tp_t_alloc_collection(size_t n);






array_cross_link_tp_t *
pelib_array_cross_link_tp_t_alloc_from(void* buffer, size_t n);





int
pelib_array_cross_link_tp_t_init(array_cross_link_tp_t *obj);





int
pelib_array_cross_link_tp_t_copy(array_cross_link_tp_t src, array_cross_link_tp_t* dst);





int
pelib_array_cross_link_tp_t_free_struct(array_cross_link_tp_t *obj);





int
pelib_array_cross_link_tp_t_free_buffer(array_cross_link_tp_t *);




int
pelib_array_cross_link_tp_t_free(array_cross_link_tp_t *);




int
pelib_array_cross_link_tp_t_destroy(array_cross_link_tp_t);




int
pelib_array_cross_link_tp_t_compare(array_cross_link_tp_t a, array_cross_link_tp_t b);






FILE*
pelib_array_cross_link_tp_t_printf(FILE* str, array_cross_link_tp_t obj);







FILE*
pelib_array_cross_link_tp_t_printf_detail(FILE* str, array_cross_link_tp_t obj, int lvl);






size_t
pelib_array_cross_link_tp_t_fwrite(array_cross_link_tp_t obj, FILE* str);






size_t
pelib_array_cross_link_tp_t_fread(array_cross_link_tp_t* obj, FILE* str);


char*
pelib_array_cross_link_tp_t_string(array_cross_link_tp_t);





char*
pelib_array_cross_link_tp_t_string_detail(array_cross_link_tp_t, int);
# 55 "/home/TDDD56/usr/include/pelib/array.h" 2 3


array_cross_link_tp_t* pelib_array_cross_link_tp_loadfilename(char*);


array_cross_link_tp_t* pelib_array_cross_link_tp_loadfilenamebinary(char*);


array_cross_link_tp_t* pelib_array_cross_link_tp_preloadfilenamebinary(char*);





array_cross_link_tp_t* pelib_array_cross_link_tp_loadfilenamewindowbinary(char*, size_t from, size_t to);

int
pelib_array_cross_link_tp_storefilename(array_cross_link_tp_t*, char*);

int
pelib_array_cross_link_tp_storefilenamebinary(array_cross_link_tp_t*, char*);

int
pelib_array_cross_link_tp_checkascending(array_cross_link_tp_t*);

cross_link_tp
pelib_array_cross_link_tp_read(array_cross_link_tp_t*, size_t i);

int
pelib_array_cross_link_tp_write(array_cross_link_tp_t*, size_t i, cross_link_tp elem);



int
pelib_array_cross_link_tp_append(array_cross_link_tp_t*, cross_link_tp elem);

size_t
pelib_array_cross_link_tp_length(array_cross_link_tp_t*);

size_t
pelib_array_cross_link_tp_capacity(array_cross_link_tp_t*);



int
pelib_array_cross_link_tp_compare(array_cross_link_tp_t* a1, array_cross_link_tp_t* a2);
# 76 "/home/TDDD56/usr/include/drake/task.h" 2 3



struct processor;

typedef struct processor processor_t;


struct task
{

 task_id id;

  processor_t* core;

 array_link_tp_t *succ;

  array_link_tp_t *pred;

 array_cross_link_tp_t *sink;

  array_cross_link_tp_t *source;

 int frequency;

  task_status_t status;

 char *name;

  int (*init)(struct task*, void*);

 int (*start)(struct task*);

  int (*run)(struct task*);

 int (*kill)(struct task*);

  int (*destroy)(struct task*);

 unsigned long long int start_time, stop_time, start_presort, stop_presort, check_time, push_time, work_time, check_errors, check_recv, check_putback, check_feedback, put_reset, put_pop, put_send, check_wait, push_wait, work_wait, work_read, work_write;

  unsigned long long int step_init, step_start, step_check, step_work, step_push, step_killed, step_zombie, step_transition;
};

typedef struct task task_t;

typedef struct task* task_tp;



# 1 "/home/TDDD56/usr/include/pelib/structure.h" 1 3
# 25 "/home/TDDD56/usr/include/pelib/structure.h" 3
# 1 "/usr/lib/gcc/x86_64-linux-gnu/4.8/include/stddef.h" 1 3 4
# 26 "/home/TDDD56/usr/include/pelib/structure.h" 2 3
# 59 "/home/TDDD56/usr/include/pelib/structure.h" 3
task_t *
pelib_task_t_alloc_struct();





int
pelib_task_t_alloc_buffer(task_t* obj, size_t n);







int
pelib_task_t_set_buffer(task_t* obj, void* buffer, size_t n);



task_t *
pelib_task_t_alloc();





task_t *
pelib_task_t_alloc_collection(size_t n);






task_t *
pelib_task_t_alloc_from(void* buffer, size_t n);





int
pelib_task_t_init(task_t *obj);





int
pelib_task_t_copy(task_t src, task_t* dst);





int
pelib_task_t_free_struct(task_t *obj);





int
pelib_task_t_free_buffer(task_t *);




int
pelib_task_t_free(task_t *);




int
pelib_task_t_destroy(task_t);




int
pelib_task_t_compare(task_t a, task_t b);






FILE*
pelib_task_t_printf(FILE* str, task_t obj);







FILE*
pelib_task_t_printf_detail(FILE* str, task_t obj, int lvl);






size_t
pelib_task_t_fwrite(task_t obj, FILE* str);






size_t
pelib_task_t_fread(task_t* obj, FILE* str);


char*
pelib_task_t_string(task_t);





char*
pelib_task_t_string_detail(task_t, int);
# 127 "/home/TDDD56/usr/include/drake/task.h" 2 3




# 1 "/home/TDDD56/usr/include/pelib/structure.h" 1 3
# 25 "/home/TDDD56/usr/include/pelib/structure.h" 3
# 1 "/usr/lib/gcc/x86_64-linux-gnu/4.8/include/stddef.h" 1 3 4
# 26 "/home/TDDD56/usr/include/pelib/structure.h" 2 3
# 59 "/home/TDDD56/usr/include/pelib/structure.h" 3
task_tp *
pelib_task_tp_alloc_struct();





int
pelib_task_tp_alloc_buffer(task_tp* obj, size_t n);







int
pelib_task_tp_set_buffer(task_tp* obj, void* buffer, size_t n);



task_tp *
pelib_task_tp_alloc();





task_tp *
pelib_task_tp_alloc_collection(size_t n);






task_tp *
pelib_task_tp_alloc_from(void* buffer, size_t n);





int
pelib_task_tp_init(task_tp *obj);





int
pelib_task_tp_copy(task_tp src, task_tp* dst);





int
pelib_task_tp_free_struct(task_tp *obj);





int
pelib_task_tp_free_buffer(task_tp *);




int
pelib_task_tp_free(task_tp *);




int
pelib_task_tp_destroy(task_tp);




int
pelib_task_tp_compare(task_tp a, task_tp b);






FILE*
pelib_task_tp_printf(FILE* str, task_tp obj);







FILE*
pelib_task_tp_printf_detail(FILE* str, task_tp obj, int lvl);






size_t
pelib_task_tp_fwrite(task_tp obj, FILE* str);






size_t
pelib_task_tp_fread(task_tp* obj, FILE* str);


char*
pelib_task_tp_string(task_tp);





char*
pelib_task_tp_string_detail(task_tp, int);
# 132 "/home/TDDD56/usr/include/drake/task.h" 2 3




# 1 "/home/TDDD56/usr/include/pelib/array.h" 1 3
# 45 "/home/TDDD56/usr/include/pelib/array.h" 3
struct array_task_tp
  {
    size_t capacity;
    size_t length;
    task_tp* data;
  };
typedef struct array_task_tp array_task_tp_t;


# 1 "/home/TDDD56/usr/include/pelib/structure.h" 1 3
# 25 "/home/TDDD56/usr/include/pelib/structure.h" 3
# 1 "/usr/lib/gcc/x86_64-linux-gnu/4.8/include/stddef.h" 1 3 4
# 26 "/home/TDDD56/usr/include/pelib/structure.h" 2 3
# 59 "/home/TDDD56/usr/include/pelib/structure.h" 3
array_task_tp_t *
pelib_array_task_tp_t_alloc_struct();





int
pelib_array_task_tp_t_alloc_buffer(array_task_tp_t* obj, size_t n);







int
pelib_array_task_tp_t_set_buffer(array_task_tp_t* obj, void* buffer, size_t n);



array_task_tp_t *
pelib_array_task_tp_t_alloc();





array_task_tp_t *
pelib_array_task_tp_t_alloc_collection(size_t n);






array_task_tp_t *
pelib_array_task_tp_t_alloc_from(void* buffer, size_t n);





int
pelib_array_task_tp_t_init(array_task_tp_t *obj);





int
pelib_array_task_tp_t_copy(array_task_tp_t src, array_task_tp_t* dst);





int
pelib_array_task_tp_t_free_struct(array_task_tp_t *obj);





int
pelib_array_task_tp_t_free_buffer(array_task_tp_t *);




int
pelib_array_task_tp_t_free(array_task_tp_t *);




int
pelib_array_task_tp_t_destroy(array_task_tp_t);




int
pelib_array_task_tp_t_compare(array_task_tp_t a, array_task_tp_t b);






FILE*
pelib_array_task_tp_t_printf(FILE* str, array_task_tp_t obj);







FILE*
pelib_array_task_tp_t_printf_detail(FILE* str, array_task_tp_t obj, int lvl);






size_t
pelib_array_task_tp_t_fwrite(array_task_tp_t obj, FILE* str);






size_t
pelib_array_task_tp_t_fread(array_task_tp_t* obj, FILE* str);


char*
pelib_array_task_tp_t_string(array_task_tp_t);





char*
pelib_array_task_tp_t_string_detail(array_task_tp_t, int);
# 55 "/home/TDDD56/usr/include/pelib/array.h" 2 3


array_task_tp_t* pelib_array_task_tp_loadfilename(char*);


array_task_tp_t* pelib_array_task_tp_loadfilenamebinary(char*);


array_task_tp_t* pelib_array_task_tp_preloadfilenamebinary(char*);





array_task_tp_t* pelib_array_task_tp_loadfilenamewindowbinary(char*, size_t from, size_t to);

int
pelib_array_task_tp_storefilename(array_task_tp_t*, char*);

int
pelib_array_task_tp_storefilenamebinary(array_task_tp_t*, char*);

int
pelib_array_task_tp_checkascending(array_task_tp_t*);

task_tp
pelib_array_task_tp_read(array_task_tp_t*, size_t i);

int
pelib_array_task_tp_write(array_task_tp_t*, size_t i, task_tp elem);



int
pelib_array_task_tp_append(array_task_tp_t*, task_tp elem);

size_t
pelib_array_task_tp_length(array_task_tp_t*);

size_t
pelib_array_task_tp_capacity(array_task_tp_t*);



int
pelib_array_task_tp_compare(array_task_tp_t* a1, array_task_tp_t* a2);
# 137 "/home/TDDD56/usr/include/drake/task.h" 2 3



int
drake_task_depleted(task_tp task);
# 23 "/home/TDDD56/usr/include/drake/schedule.h" 2 3






typedef struct {

 size_t id;

 double start_time;

 double frequency;
} drake_schedule_task_t;


typedef struct {

 size_t core_number;

 size_t task_number;

 char **task_name;

 double stage_time;

 size_t *tasks_in_core;

 size_t *consumers_in_core;

 size_t *producers_in_core;

 size_t *consumers_in_task;

 size_t *producers_in_task;

 size_t **consumers_id;

 size_t **producers_id;

 size_t *remote_consumers_in_task;

 size_t *remote_producers_in_task;

 drake_schedule_task_t **schedule;
} drake_schedule_t;
# 80 "/home/TDDD56/usr/include/drake/schedule.h" 3
void* drake_function_merge(size_t id, task_status_t state);

void drake_schedule_init_merge(drake_schedule_t*);

void drake_schedule_destroy_merge(drake_schedule_t*);

int drake_task_number_merge();

char* drake_task_name_merge(size_t);
# 4 "merge.schedule.c" 2



# 1 "/home/TDDD56/usr/include/drake/node.h" 1 3
# 23 "/home/TDDD56/usr/include/drake/node.h" 3
# 1 "/usr/lib/gcc/x86_64-linux-gnu/4.8/include/stddef.h" 1 3 4
# 24 "/home/TDDD56/usr/include/drake/node.h" 2 3





# 1 "/home/TDDD56/usr/include/drake/task.h" 1 3
# 22 "/home/TDDD56/usr/include/drake/task.h" 3
# 1 "/usr/lib/gcc/x86_64-linux-gnu/4.8/include/stddef.h" 1 3 4
# 23 "/home/TDDD56/usr/include/drake/task.h" 2 3
# 30 "/home/TDDD56/usr/include/drake/node.h" 2 3
# 77 "/home/TDDD56/usr/include/drake/node.h" 3
int merge_init_task_1(task_t *, void *aux);




int merge_start_task_1(task_t *);



int merge_run_task_1(task_t *);




int merge_kill_task_1(task_t *);




int merge_destroy_task_1(task_t *);
# 8 "merge.schedule.c" 2




# 1 "/home/TDDD56/usr/include/drake/node.h" 1 3
# 23 "/home/TDDD56/usr/include/drake/node.h" 3
# 1 "/usr/lib/gcc/x86_64-linux-gnu/4.8/include/stddef.h" 1 3 4
# 24 "/home/TDDD56/usr/include/drake/node.h" 2 3





# 1 "/home/TDDD56/usr/include/drake/task.h" 1 3
# 22 "/home/TDDD56/usr/include/drake/task.h" 3
# 1 "/usr/lib/gcc/x86_64-linux-gnu/4.8/include/stddef.h" 1 3 4
# 23 "/home/TDDD56/usr/include/drake/task.h" 2 3
# 30 "/home/TDDD56/usr/include/drake/node.h" 2 3
# 77 "/home/TDDD56/usr/include/drake/node.h" 3
int merge_init_task_2(task_t *, void *aux);




int merge_start_task_2(task_t *);



int merge_run_task_2(task_t *);




int merge_kill_task_2(task_t *);




int merge_destroy_task_2(task_t *);
# 13 "merge.schedule.c" 2




# 1 "/home/TDDD56/usr/include/drake/node.h" 1 3
# 23 "/home/TDDD56/usr/include/drake/node.h" 3
# 1 "/usr/lib/gcc/x86_64-linux-gnu/4.8/include/stddef.h" 1 3 4
# 24 "/home/TDDD56/usr/include/drake/node.h" 2 3





# 1 "/home/TDDD56/usr/include/drake/task.h" 1 3
# 22 "/home/TDDD56/usr/include/drake/task.h" 3
# 1 "/usr/lib/gcc/x86_64-linux-gnu/4.8/include/stddef.h" 1 3 4
# 23 "/home/TDDD56/usr/include/drake/task.h" 2 3
# 30 "/home/TDDD56/usr/include/drake/node.h" 2 3
# 77 "/home/TDDD56/usr/include/drake/node.h" 3
int merge_init_task_3(task_t *, void *aux);




int merge_start_task_3(task_t *);



int merge_run_task_3(task_t *);




int merge_kill_task_3(task_t *);




int merge_destroy_task_3(task_t *);
# 18 "merge.schedule.c" 2




# 1 "/home/TDDD56/usr/include/drake/node.h" 1 3
# 23 "/home/TDDD56/usr/include/drake/node.h" 3
# 1 "/usr/lib/gcc/x86_64-linux-gnu/4.8/include/stddef.h" 1 3 4
# 24 "/home/TDDD56/usr/include/drake/node.h" 2 3





# 1 "/home/TDDD56/usr/include/drake/task.h" 1 3
# 22 "/home/TDDD56/usr/include/drake/task.h" 3
# 1 "/usr/lib/gcc/x86_64-linux-gnu/4.8/include/stddef.h" 1 3 4
# 23 "/home/TDDD56/usr/include/drake/task.h" 2 3
# 30 "/home/TDDD56/usr/include/drake/node.h" 2 3
# 77 "/home/TDDD56/usr/include/drake/node.h" 3
int merge_init_task_4(task_t *, void *aux);




int merge_start_task_4(task_t *);



int merge_run_task_4(task_t *);




int merge_kill_task_4(task_t *);




int merge_destroy_task_4(task_t *);
# 23 "merge.schedule.c" 2




# 1 "/home/TDDD56/usr/include/drake/node.h" 1 3
# 23 "/home/TDDD56/usr/include/drake/node.h" 3
# 1 "/usr/lib/gcc/x86_64-linux-gnu/4.8/include/stddef.h" 1 3 4
# 24 "/home/TDDD56/usr/include/drake/node.h" 2 3





# 1 "/home/TDDD56/usr/include/drake/task.h" 1 3
# 22 "/home/TDDD56/usr/include/drake/task.h" 3
# 1 "/usr/lib/gcc/x86_64-linux-gnu/4.8/include/stddef.h" 1 3 4
# 23 "/home/TDDD56/usr/include/drake/task.h" 2 3
# 30 "/home/TDDD56/usr/include/drake/node.h" 2 3
# 77 "/home/TDDD56/usr/include/drake/node.h" 3
int merge_init_task_5(task_t *, void *aux);




int merge_start_task_5(task_t *);



int merge_run_task_5(task_t *);




int merge_kill_task_5(task_t *);




int merge_destroy_task_5(task_t *);
# 28 "merge.schedule.c" 2




# 1 "/home/TDDD56/usr/include/drake/node.h" 1 3
# 23 "/home/TDDD56/usr/include/drake/node.h" 3
# 1 "/usr/lib/gcc/x86_64-linux-gnu/4.8/include/stddef.h" 1 3 4
# 24 "/home/TDDD56/usr/include/drake/node.h" 2 3





# 1 "/home/TDDD56/usr/include/drake/task.h" 1 3
# 22 "/home/TDDD56/usr/include/drake/task.h" 3
# 1 "/usr/lib/gcc/x86_64-linux-gnu/4.8/include/stddef.h" 1 3 4
# 23 "/home/TDDD56/usr/include/drake/task.h" 2 3
# 30 "/home/TDDD56/usr/include/drake/node.h" 2 3
# 77 "/home/TDDD56/usr/include/drake/node.h" 3
int merge_init_task_6(task_t *, void *aux);




int merge_start_task_6(task_t *);



int merge_run_task_6(task_t *);




int merge_kill_task_6(task_t *);




int merge_destroy_task_6(task_t *);
# 33 "merge.schedule.c" 2




# 1 "/home/TDDD56/usr/include/drake/node.h" 1 3
# 23 "/home/TDDD56/usr/include/drake/node.h" 3
# 1 "/usr/lib/gcc/x86_64-linux-gnu/4.8/include/stddef.h" 1 3 4
# 24 "/home/TDDD56/usr/include/drake/node.h" 2 3





# 1 "/home/TDDD56/usr/include/drake/task.h" 1 3
# 22 "/home/TDDD56/usr/include/drake/task.h" 3
# 1 "/usr/lib/gcc/x86_64-linux-gnu/4.8/include/stddef.h" 1 3 4
# 23 "/home/TDDD56/usr/include/drake/task.h" 2 3
# 30 "/home/TDDD56/usr/include/drake/node.h" 2 3
# 77 "/home/TDDD56/usr/include/drake/node.h" 3
int merge_init_task_7(task_t *, void *aux);




int merge_start_task_7(task_t *);



int merge_run_task_7(task_t *);




int merge_kill_task_7(task_t *);




int merge_destroy_task_7(task_t *);
# 38 "merge.schedule.c" 2




# 1 "/home/TDDD56/usr/include/drake/node.h" 1 3
# 23 "/home/TDDD56/usr/include/drake/node.h" 3
# 1 "/usr/lib/gcc/x86_64-linux-gnu/4.8/include/stddef.h" 1 3 4
# 24 "/home/TDDD56/usr/include/drake/node.h" 2 3





# 1 "/home/TDDD56/usr/include/drake/task.h" 1 3
# 22 "/home/TDDD56/usr/include/drake/task.h" 3
# 1 "/usr/lib/gcc/x86_64-linux-gnu/4.8/include/stddef.h" 1 3 4
# 23 "/home/TDDD56/usr/include/drake/task.h" 2 3
# 30 "/home/TDDD56/usr/include/drake/node.h" 2 3
# 77 "/home/TDDD56/usr/include/drake/node.h" 3
int merge_init_task_8(task_t *, void *aux);




int merge_start_task_8(task_t *);



int merge_run_task_8(task_t *);




int merge_kill_task_8(task_t *);




int merge_destroy_task_8(task_t *);
# 43 "merge.schedule.c" 2




# 1 "/home/TDDD56/usr/include/drake/node.h" 1 3
# 23 "/home/TDDD56/usr/include/drake/node.h" 3
# 1 "/usr/lib/gcc/x86_64-linux-gnu/4.8/include/stddef.h" 1 3 4
# 24 "/home/TDDD56/usr/include/drake/node.h" 2 3





# 1 "/home/TDDD56/usr/include/drake/task.h" 1 3
# 22 "/home/TDDD56/usr/include/drake/task.h" 3
# 1 "/usr/lib/gcc/x86_64-linux-gnu/4.8/include/stddef.h" 1 3 4
# 23 "/home/TDDD56/usr/include/drake/task.h" 2 3
# 30 "/home/TDDD56/usr/include/drake/node.h" 2 3
# 77 "/home/TDDD56/usr/include/drake/node.h" 3
int merge_init_task_9(task_t *, void *aux);




int merge_start_task_9(task_t *);



int merge_run_task_9(task_t *);




int merge_kill_task_9(task_t *);




int merge_destroy_task_9(task_t *);
# 48 "merge.schedule.c" 2




# 1 "/home/TDDD56/usr/include/drake/node.h" 1 3
# 23 "/home/TDDD56/usr/include/drake/node.h" 3
# 1 "/usr/lib/gcc/x86_64-linux-gnu/4.8/include/stddef.h" 1 3 4
# 24 "/home/TDDD56/usr/include/drake/node.h" 2 3





# 1 "/home/TDDD56/usr/include/drake/task.h" 1 3
# 22 "/home/TDDD56/usr/include/drake/task.h" 3
# 1 "/usr/lib/gcc/x86_64-linux-gnu/4.8/include/stddef.h" 1 3 4
# 23 "/home/TDDD56/usr/include/drake/task.h" 2 3
# 30 "/home/TDDD56/usr/include/drake/node.h" 2 3
# 77 "/home/TDDD56/usr/include/drake/node.h" 3
int merge_init_task_10(task_t *, void *aux);




int merge_start_task_10(task_t *);



int merge_run_task_10(task_t *);




int merge_kill_task_10(task_t *);




int merge_destroy_task_10(task_t *);
# 53 "merge.schedule.c" 2




# 1 "/home/TDDD56/usr/include/drake/node.h" 1 3
# 23 "/home/TDDD56/usr/include/drake/node.h" 3
# 1 "/usr/lib/gcc/x86_64-linux-gnu/4.8/include/stddef.h" 1 3 4
# 24 "/home/TDDD56/usr/include/drake/node.h" 2 3





# 1 "/home/TDDD56/usr/include/drake/task.h" 1 3
# 22 "/home/TDDD56/usr/include/drake/task.h" 3
# 1 "/usr/lib/gcc/x86_64-linux-gnu/4.8/include/stddef.h" 1 3 4
# 23 "/home/TDDD56/usr/include/drake/task.h" 2 3
# 30 "/home/TDDD56/usr/include/drake/node.h" 2 3
# 77 "/home/TDDD56/usr/include/drake/node.h" 3
int merge_init_task_11(task_t *, void *aux);




int merge_start_task_11(task_t *);



int merge_run_task_11(task_t *);




int merge_kill_task_11(task_t *);




int merge_destroy_task_11(task_t *);
# 58 "merge.schedule.c" 2




# 1 "/home/TDDD56/usr/include/drake/node.h" 1 3
# 23 "/home/TDDD56/usr/include/drake/node.h" 3
# 1 "/usr/lib/gcc/x86_64-linux-gnu/4.8/include/stddef.h" 1 3 4
# 24 "/home/TDDD56/usr/include/drake/node.h" 2 3





# 1 "/home/TDDD56/usr/include/drake/task.h" 1 3
# 22 "/home/TDDD56/usr/include/drake/task.h" 3
# 1 "/usr/lib/gcc/x86_64-linux-gnu/4.8/include/stddef.h" 1 3 4
# 23 "/home/TDDD56/usr/include/drake/task.h" 2 3
# 30 "/home/TDDD56/usr/include/drake/node.h" 2 3
# 77 "/home/TDDD56/usr/include/drake/node.h" 3
int merge_init_task_12(task_t *, void *aux);




int merge_start_task_12(task_t *);



int merge_run_task_12(task_t *);




int merge_kill_task_12(task_t *);




int merge_destroy_task_12(task_t *);
# 63 "merge.schedule.c" 2




# 1 "/home/TDDD56/usr/include/drake/node.h" 1 3
# 23 "/home/TDDD56/usr/include/drake/node.h" 3
# 1 "/usr/lib/gcc/x86_64-linux-gnu/4.8/include/stddef.h" 1 3 4
# 24 "/home/TDDD56/usr/include/drake/node.h" 2 3





# 1 "/home/TDDD56/usr/include/drake/task.h" 1 3
# 22 "/home/TDDD56/usr/include/drake/task.h" 3
# 1 "/usr/lib/gcc/x86_64-linux-gnu/4.8/include/stddef.h" 1 3 4
# 23 "/home/TDDD56/usr/include/drake/task.h" 2 3
# 30 "/home/TDDD56/usr/include/drake/node.h" 2 3
# 77 "/home/TDDD56/usr/include/drake/node.h" 3
int merge_init_task_13(task_t *, void *aux);




int merge_start_task_13(task_t *);



int merge_run_task_13(task_t *);




int merge_kill_task_13(task_t *);




int merge_destroy_task_13(task_t *);
# 68 "merge.schedule.c" 2




# 1 "/home/TDDD56/usr/include/drake/node.h" 1 3
# 23 "/home/TDDD56/usr/include/drake/node.h" 3
# 1 "/usr/lib/gcc/x86_64-linux-gnu/4.8/include/stddef.h" 1 3 4
# 24 "/home/TDDD56/usr/include/drake/node.h" 2 3





# 1 "/home/TDDD56/usr/include/drake/task.h" 1 3
# 22 "/home/TDDD56/usr/include/drake/task.h" 3
# 1 "/usr/lib/gcc/x86_64-linux-gnu/4.8/include/stddef.h" 1 3 4
# 23 "/home/TDDD56/usr/include/drake/task.h" 2 3
# 30 "/home/TDDD56/usr/include/drake/node.h" 2 3
# 77 "/home/TDDD56/usr/include/drake/node.h" 3
int merge_init_task_14(task_t *, void *aux);




int merge_start_task_14(task_t *);



int merge_run_task_14(task_t *);




int merge_kill_task_14(task_t *);




int merge_destroy_task_14(task_t *);
# 73 "merge.schedule.c" 2




# 1 "/home/TDDD56/usr/include/drake/node.h" 1 3
# 23 "/home/TDDD56/usr/include/drake/node.h" 3
# 1 "/usr/lib/gcc/x86_64-linux-gnu/4.8/include/stddef.h" 1 3 4
# 24 "/home/TDDD56/usr/include/drake/node.h" 2 3





# 1 "/home/TDDD56/usr/include/drake/task.h" 1 3
# 22 "/home/TDDD56/usr/include/drake/task.h" 3
# 1 "/usr/lib/gcc/x86_64-linux-gnu/4.8/include/stddef.h" 1 3 4
# 23 "/home/TDDD56/usr/include/drake/task.h" 2 3
# 30 "/home/TDDD56/usr/include/drake/node.h" 2 3
# 77 "/home/TDDD56/usr/include/drake/node.h" 3
int merge_init_task_15(task_t *, void *aux);




int merge_start_task_15(task_t *);



int merge_run_task_15(task_t *);




int merge_kill_task_15(task_t *);




int merge_destroy_task_15(task_t *);
# 78 "merge.schedule.c" 2


int drake_task_number_merge()
{
 return 15;
}

char* drake_task_name_merge(size_t index)
{
 switch(index - 1)
 {
  case 0:
   return "task_1";
  break;
  case 1:
   return "task_2";
  break;
  case 2:
   return "task_3";
  break;
  case 3:
   return "task_4";
  break;
  case 4:
   return "task_5";
  break;
  case 5:
   return "task_6";
  break;
  case 6:
   return "task_7";
  break;
  case 7:
   return "task_8";
  break;
  case 8:
   return "task_9";
  break;
  case 9:
   return "task_10";
  break;
  case 10:
   return "task_11";
  break;
  case 11:
   return "task_12";
  break;
  case 12:
   return "task_13";
  break;
  case 13:
   return "task_14";
  break;
  case 14:
   return "task_15";
  break;
  default:
   return "invalid task id";
  break;
 }
}

void drake_schedule_init_merge(drake_schedule_t* schedule)
{
 schedule->core_number = 1;
 schedule->task_number = 15;
 schedule->stage_time = 0;

 schedule->tasks_in_core = malloc(sizeof(size_t) * schedule->core_number);

 schedule->task_name = malloc(sizeof(size_t*) * schedule->task_number);
 schedule->task_name[0] = "task_1";
 schedule->task_name[1] = "task_2";
 schedule->task_name[2] = "task_3";
 schedule->task_name[3] = "task_4";
 schedule->task_name[4] = "task_5";
 schedule->task_name[5] = "task_6";
 schedule->task_name[6] = "task_7";
 schedule->task_name[7] = "task_8";
 schedule->task_name[8] = "task_9";
 schedule->task_name[9] = "task_10";
 schedule->task_name[10] = "task_11";
 schedule->task_name[11] = "task_12";
 schedule->task_name[12] = "task_13";
 schedule->task_name[13] = "task_14";
 schedule->task_name[14] = "task_15";

 schedule->tasks_in_core[0] = 15;

 schedule->consumers_in_core = malloc(sizeof(size_t) * schedule->core_number);
 schedule->consumers_in_core[0] = 0;

 schedule->producers_in_core = malloc(sizeof(size_t) * schedule->core_number);
 schedule->producers_in_core[0] = 0;

 schedule->consumers_in_task = malloc(sizeof(size_t) * schedule->task_number);
 schedule->consumers_in_task[0] = 0;
 schedule->consumers_in_task[1] = 1;
 schedule->consumers_in_task[2] = 1;
 schedule->consumers_in_task[3] = 1;
 schedule->consumers_in_task[4] = 1;
 schedule->consumers_in_task[5] = 1;
 schedule->consumers_in_task[6] = 1;
 schedule->consumers_in_task[7] = 1;
 schedule->consumers_in_task[8] = 1;
 schedule->consumers_in_task[9] = 1;
 schedule->consumers_in_task[10] = 1;
 schedule->consumers_in_task[11] = 1;
 schedule->consumers_in_task[12] = 1;
 schedule->consumers_in_task[13] = 1;
 schedule->consumers_in_task[14] = 1;

 schedule->producers_in_task = malloc(sizeof(size_t) * schedule->task_number);
 schedule->producers_in_task[0] = 2;
 schedule->producers_in_task[1] = 2;
 schedule->producers_in_task[2] = 2;
 schedule->producers_in_task[3] = 2;
 schedule->producers_in_task[4] = 2;
 schedule->producers_in_task[5] = 2;
 schedule->producers_in_task[6] = 2;
 schedule->producers_in_task[7] = 0;
 schedule->producers_in_task[8] = 0;
 schedule->producers_in_task[9] = 0;
 schedule->producers_in_task[10] = 0;
 schedule->producers_in_task[11] = 0;
 schedule->producers_in_task[12] = 0;
 schedule->producers_in_task[13] = 0;
 schedule->producers_in_task[14] = 0;

 schedule->remote_consumers_in_task = malloc(sizeof(size_t) * schedule->task_number);
 schedule->remote_consumers_in_task[0] = 0;
 schedule->remote_consumers_in_task[1] = 0;
 schedule->remote_consumers_in_task[2] = 0;
 schedule->remote_consumers_in_task[3] = 0;
 schedule->remote_consumers_in_task[4] = 0;
 schedule->remote_consumers_in_task[5] = 0;
 schedule->remote_consumers_in_task[6] = 0;
 schedule->remote_consumers_in_task[7] = 0;
 schedule->remote_consumers_in_task[8] = 0;
 schedule->remote_consumers_in_task[9] = 0;
 schedule->remote_consumers_in_task[10] = 0;
 schedule->remote_consumers_in_task[11] = 0;
 schedule->remote_consumers_in_task[12] = 0;
 schedule->remote_consumers_in_task[13] = 0;
 schedule->remote_consumers_in_task[14] = 0;

 schedule->remote_producers_in_task = malloc(sizeof(size_t) * schedule->task_number);
 schedule->remote_producers_in_task[0] = 0;
 schedule->remote_producers_in_task[1] = 0;
 schedule->remote_producers_in_task[2] = 0;
 schedule->remote_producers_in_task[3] = 0;
 schedule->remote_producers_in_task[4] = 0;
 schedule->remote_producers_in_task[5] = 0;
 schedule->remote_producers_in_task[6] = 0;
 schedule->remote_producers_in_task[7] = 0;
 schedule->remote_producers_in_task[8] = 0;
 schedule->remote_producers_in_task[9] = 0;
 schedule->remote_producers_in_task[10] = 0;
 schedule->remote_producers_in_task[11] = 0;
 schedule->remote_producers_in_task[12] = 0;
 schedule->remote_producers_in_task[13] = 0;
 schedule->remote_producers_in_task[14] = 0;

 schedule->producers_id = malloc(sizeof(size_t*) * schedule->task_number);
 schedule->producers_id[0] = malloc(sizeof(size_t) * 2);
 schedule->producers_id[0][0] = 2;
 schedule->producers_id[0][1] = 3;
 schedule->producers_id[1] = malloc(sizeof(size_t) * 2);
 schedule->producers_id[1][0] = 4;
 schedule->producers_id[1][1] = 5;
 schedule->producers_id[2] = malloc(sizeof(size_t) * 2);
 schedule->producers_id[2][0] = 7;
 schedule->producers_id[2][1] = 6;
 schedule->producers_id[3] = malloc(sizeof(size_t) * 2);
 schedule->producers_id[3][0] = 9;
 schedule->producers_id[3][1] = 8;
 schedule->producers_id[4] = malloc(sizeof(size_t) * 2);
 schedule->producers_id[4][0] = 11;
 schedule->producers_id[4][1] = 10;
 schedule->producers_id[5] = malloc(sizeof(size_t) * 2);
 schedule->producers_id[5][0] = 12;
 schedule->producers_id[5][1] = 13;
 schedule->producers_id[6] = malloc(sizeof(size_t) * 2);
 schedule->producers_id[6][0] = 15;
 schedule->producers_id[6][1] = 14;
 schedule->producers_id[7] = ((void *)0);
 schedule->producers_id[8] = ((void *)0);
 schedule->producers_id[9] = ((void *)0);
 schedule->producers_id[10] = ((void *)0);
 schedule->producers_id[11] = ((void *)0);
 schedule->producers_id[12] = ((void *)0);
 schedule->producers_id[13] = ((void *)0);
 schedule->producers_id[14] = ((void *)0);

 schedule->consumers_id = malloc(sizeof(size_t*) * schedule->task_number);
 schedule->consumers_id[0] = ((void *)0);
 schedule->consumers_id[1] = malloc(sizeof(size_t) * 1);
 schedule->consumers_id[1][0] = 1;
 schedule->consumers_id[2] = malloc(sizeof(size_t) * 1);
 schedule->consumers_id[2][0] = 1;
 schedule->consumers_id[3] = malloc(sizeof(size_t) * 1);
 schedule->consumers_id[3][0] = 2;
 schedule->consumers_id[4] = malloc(sizeof(size_t) * 1);
 schedule->consumers_id[4][0] = 2;
 schedule->consumers_id[5] = malloc(sizeof(size_t) * 1);
 schedule->consumers_id[5][0] = 3;
 schedule->consumers_id[6] = malloc(sizeof(size_t) * 1);
 schedule->consumers_id[6][0] = 3;
 schedule->consumers_id[7] = malloc(sizeof(size_t) * 1);
 schedule->consumers_id[7][0] = 4;
 schedule->consumers_id[8] = malloc(sizeof(size_t) * 1);
 schedule->consumers_id[8][0] = 4;
 schedule->consumers_id[9] = malloc(sizeof(size_t) * 1);
 schedule->consumers_id[9][0] = 5;
 schedule->consumers_id[10] = malloc(sizeof(size_t) * 1);
 schedule->consumers_id[10][0] = 5;
 schedule->consumers_id[11] = malloc(sizeof(size_t) * 1);
 schedule->consumers_id[11][0] = 6;
 schedule->consumers_id[12] = malloc(sizeof(size_t) * 1);
 schedule->consumers_id[12][0] = 6;
 schedule->consumers_id[13] = malloc(sizeof(size_t) * 1);
 schedule->consumers_id[13][0] = 7;
 schedule->consumers_id[14] = malloc(sizeof(size_t) * 1);
 schedule->consumers_id[14][0] = 7;

 schedule->schedule = malloc(sizeof(drake_schedule_task_t*) * schedule->core_number);
 schedule->schedule[0] = malloc(sizeof(drake_schedule_task_t) * 15);
 schedule->schedule[0][0].id = 1;
 schedule->schedule[0][0].start_time = 0;
 schedule->schedule[0][0].frequency = 1;
 schedule->schedule[0][1].id = 3;
 schedule->schedule[0][1].start_time = 32;
 schedule->schedule[0][1].frequency = 1;
 schedule->schedule[0][2].id = 2;
 schedule->schedule[0][2].start_time = 48;
 schedule->schedule[0][2].frequency = 1;
 schedule->schedule[0][3].id = 7;
 schedule->schedule[0][3].start_time = 64;
 schedule->schedule[0][3].frequency = 1;
 schedule->schedule[0][4].id = 6;
 schedule->schedule[0][4].start_time = 72;
 schedule->schedule[0][4].frequency = 1;
 schedule->schedule[0][5].id = 5;
 schedule->schedule[0][5].start_time = 80;
 schedule->schedule[0][5].frequency = 1;
 schedule->schedule[0][6].id = 4;
 schedule->schedule[0][6].start_time = 88;
 schedule->schedule[0][6].frequency = 1;
 schedule->schedule[0][7].id = 13;
 schedule->schedule[0][7].start_time = 96;
 schedule->schedule[0][7].frequency = 1;
 schedule->schedule[0][8].id = 12;
 schedule->schedule[0][8].start_time = 100;
 schedule->schedule[0][8].frequency = 1;
 schedule->schedule[0][9].id = 11;
 schedule->schedule[0][9].start_time = 104;
 schedule->schedule[0][9].frequency = 1;
 schedule->schedule[0][10].id = 10;
 schedule->schedule[0][10].start_time = 108;
 schedule->schedule[0][10].frequency = 1;
 schedule->schedule[0][11].id = 9;
 schedule->schedule[0][11].start_time = 112;
 schedule->schedule[0][11].frequency = 1;
 schedule->schedule[0][12].id = 8;
 schedule->schedule[0][12].start_time = 116;
 schedule->schedule[0][12].frequency = 1;
 schedule->schedule[0][13].id = 15;
 schedule->schedule[0][13].start_time = 120;
 schedule->schedule[0][13].frequency = 1;
 schedule->schedule[0][14].id = 14;
 schedule->schedule[0][14].start_time = 122;
 schedule->schedule[0][14].frequency = 1;
}

void drake_schedule_destroy_merge(drake_schedule_t* schedule)
{
 free(schedule->schedule[0]);

 free(schedule->schedule);
 free(schedule->consumers_id[0]);
 free(schedule->consumers_id[1]);
 free(schedule->consumers_id[2]);
 free(schedule->consumers_id[3]);
 free(schedule->consumers_id[4]);
 free(schedule->consumers_id[5]);
 free(schedule->consumers_id[6]);
 free(schedule->consumers_id[7]);
 free(schedule->consumers_id[8]);
 free(schedule->consumers_id[9]);
 free(schedule->consumers_id[10]);
 free(schedule->consumers_id[11]);
 free(schedule->consumers_id[12]);
 free(schedule->consumers_id[13]);
 free(schedule->consumers_id[14]);
 free(schedule->consumers_id);

 free(schedule->producers_id[0]);
 free(schedule->producers_id[1]);
 free(schedule->producers_id[2]);
 free(schedule->producers_id[3]);
 free(schedule->producers_id[4]);
 free(schedule->producers_id[5]);
 free(schedule->producers_id[6]);
 free(schedule->producers_id[7]);
 free(schedule->producers_id[8]);
 free(schedule->producers_id[9]);
 free(schedule->producers_id[10]);
 free(schedule->producers_id[11]);
 free(schedule->producers_id[12]);
 free(schedule->producers_id[13]);
 free(schedule->producers_id[14]);
 free(schedule->producers_id);
 free(schedule->remote_producers_in_task);
 free(schedule->remote_consumers_in_task);
 free(schedule->producers_in_task);
 free(schedule->consumers_in_task);
 free(schedule->producers_in_core);
 free(schedule->consumers_in_core);
 free(schedule->tasks_in_core);
 free(schedule->task_name);
}

void*
drake_function_merge(size_t id, task_status_t status)
{
 switch(id)
 {
  default:

  break;
  case 1:
   switch(status)
   {
    case TASK_INVALID:
     return 0;
    break;
    case TASK_INIT:
     return (void*)&merge_init_task_1;
     break;
      case TASK_START:
       return (void*)&merge_start_task_1;
      break;
      case TASK_RUN:
       return (void*)&merge_run_task_1;
      break;
      case TASK_KILLED:
       return (void*)&merge_kill_task_1;
      break;
      case TASK_ZOMBIE:
       return 0;
      break;
      case TASK_DESTROY:
       return (void*)&merge_destroy_task_1;
      break;
      default:
       return 0;
      break;
     }
    break;
  case 2:
   switch(status)
   {
    case TASK_INVALID:
     return 0;
    break;
    case TASK_INIT:
     return (void*)&merge_init_task_2;
     break;
      case TASK_START:
       return (void*)&merge_start_task_2;
      break;
      case TASK_RUN:
       return (void*)&merge_run_task_2;
      break;
      case TASK_KILLED:
       return (void*)&merge_kill_task_2;
      break;
      case TASK_ZOMBIE:
       return 0;
      break;
      case TASK_DESTROY:
       return (void*)&merge_destroy_task_2;
      break;
      default:
       return 0;
      break;
     }
    break;
  case 3:
   switch(status)
   {
    case TASK_INVALID:
     return 0;
    break;
    case TASK_INIT:
     return (void*)&merge_init_task_3;
     break;
      case TASK_START:
       return (void*)&merge_start_task_3;
      break;
      case TASK_RUN:
       return (void*)&merge_run_task_3;
      break;
      case TASK_KILLED:
       return (void*)&merge_kill_task_3;
      break;
      case TASK_ZOMBIE:
       return 0;
      break;
      case TASK_DESTROY:
       return (void*)&merge_destroy_task_3;
      break;
      default:
       return 0;
      break;
     }
    break;
  case 4:
   switch(status)
   {
    case TASK_INVALID:
     return 0;
    break;
    case TASK_INIT:
     return (void*)&merge_init_task_4;
     break;
      case TASK_START:
       return (void*)&merge_start_task_4;
      break;
      case TASK_RUN:
       return (void*)&merge_run_task_4;
      break;
      case TASK_KILLED:
       return (void*)&merge_kill_task_4;
      break;
      case TASK_ZOMBIE:
       return 0;
      break;
      case TASK_DESTROY:
       return (void*)&merge_destroy_task_4;
      break;
      default:
       return 0;
      break;
     }
    break;
  case 5:
   switch(status)
   {
    case TASK_INVALID:
     return 0;
    break;
    case TASK_INIT:
     return (void*)&merge_init_task_5;
     break;
      case TASK_START:
       return (void*)&merge_start_task_5;
      break;
      case TASK_RUN:
       return (void*)&merge_run_task_5;
      break;
      case TASK_KILLED:
       return (void*)&merge_kill_task_5;
      break;
      case TASK_ZOMBIE:
       return 0;
      break;
      case TASK_DESTROY:
       return (void*)&merge_destroy_task_5;
      break;
      default:
       return 0;
      break;
     }
    break;
  case 6:
   switch(status)
   {
    case TASK_INVALID:
     return 0;
    break;
    case TASK_INIT:
     return (void*)&merge_init_task_6;
     break;
      case TASK_START:
       return (void*)&merge_start_task_6;
      break;
      case TASK_RUN:
       return (void*)&merge_run_task_6;
      break;
      case TASK_KILLED:
       return (void*)&merge_kill_task_6;
      break;
      case TASK_ZOMBIE:
       return 0;
      break;
      case TASK_DESTROY:
       return (void*)&merge_destroy_task_6;
      break;
      default:
       return 0;
      break;
     }
    break;
  case 7:
   switch(status)
   {
    case TASK_INVALID:
     return 0;
    break;
    case TASK_INIT:
     return (void*)&merge_init_task_7;
     break;
      case TASK_START:
       return (void*)&merge_start_task_7;
      break;
      case TASK_RUN:
       return (void*)&merge_run_task_7;
      break;
      case TASK_KILLED:
       return (void*)&merge_kill_task_7;
      break;
      case TASK_ZOMBIE:
       return 0;
      break;
      case TASK_DESTROY:
       return (void*)&merge_destroy_task_7;
      break;
      default:
       return 0;
      break;
     }
    break;
  case 8:
   switch(status)
   {
    case TASK_INVALID:
     return 0;
    break;
    case TASK_INIT:
     return (void*)&merge_init_task_8;
     break;
      case TASK_START:
       return (void*)&merge_start_task_8;
      break;
      case TASK_RUN:
       return (void*)&merge_run_task_8;
      break;
      case TASK_KILLED:
       return (void*)&merge_kill_task_8;
      break;
      case TASK_ZOMBIE:
       return 0;
      break;
      case TASK_DESTROY:
       return (void*)&merge_destroy_task_8;
      break;
      default:
       return 0;
      break;
     }
    break;
  case 9:
   switch(status)
   {
    case TASK_INVALID:
     return 0;
    break;
    case TASK_INIT:
     return (void*)&merge_init_task_9;
     break;
      case TASK_START:
       return (void*)&merge_start_task_9;
      break;
      case TASK_RUN:
       return (void*)&merge_run_task_9;
      break;
      case TASK_KILLED:
       return (void*)&merge_kill_task_9;
      break;
      case TASK_ZOMBIE:
       return 0;
      break;
      case TASK_DESTROY:
       return (void*)&merge_destroy_task_9;
      break;
      default:
       return 0;
      break;
     }
    break;
  case 10:
   switch(status)
   {
    case TASK_INVALID:
     return 0;
    break;
    case TASK_INIT:
     return (void*)&merge_init_task_10;
     break;
      case TASK_START:
       return (void*)&merge_start_task_10;
      break;
      case TASK_RUN:
       return (void*)&merge_run_task_10;
      break;
      case TASK_KILLED:
       return (void*)&merge_kill_task_10;
      break;
      case TASK_ZOMBIE:
       return 0;
      break;
      case TASK_DESTROY:
       return (void*)&merge_destroy_task_10;
      break;
      default:
       return 0;
      break;
     }
    break;
  case 11:
   switch(status)
   {
    case TASK_INVALID:
     return 0;
    break;
    case TASK_INIT:
     return (void*)&merge_init_task_11;
     break;
      case TASK_START:
       return (void*)&merge_start_task_11;
      break;
      case TASK_RUN:
       return (void*)&merge_run_task_11;
      break;
      case TASK_KILLED:
       return (void*)&merge_kill_task_11;
      break;
      case TASK_ZOMBIE:
       return 0;
      break;
      case TASK_DESTROY:
       return (void*)&merge_destroy_task_11;
      break;
      default:
       return 0;
      break;
     }
    break;
  case 12:
   switch(status)
   {
    case TASK_INVALID:
     return 0;
    break;
    case TASK_INIT:
     return (void*)&merge_init_task_12;
     break;
      case TASK_START:
       return (void*)&merge_start_task_12;
      break;
      case TASK_RUN:
       return (void*)&merge_run_task_12;
      break;
      case TASK_KILLED:
       return (void*)&merge_kill_task_12;
      break;
      case TASK_ZOMBIE:
       return 0;
      break;
      case TASK_DESTROY:
       return (void*)&merge_destroy_task_12;
      break;
      default:
       return 0;
      break;
     }
    break;
  case 13:
   switch(status)
   {
    case TASK_INVALID:
     return 0;
    break;
    case TASK_INIT:
     return (void*)&merge_init_task_13;
     break;
      case TASK_START:
       return (void*)&merge_start_task_13;
      break;
      case TASK_RUN:
       return (void*)&merge_run_task_13;
      break;
      case TASK_KILLED:
       return (void*)&merge_kill_task_13;
      break;
      case TASK_ZOMBIE:
       return 0;
      break;
      case TASK_DESTROY:
       return (void*)&merge_destroy_task_13;
      break;
      default:
       return 0;
      break;
     }
    break;
  case 14:
   switch(status)
   {
    case TASK_INVALID:
     return 0;
    break;
    case TASK_INIT:
     return (void*)&merge_init_task_14;
     break;
      case TASK_START:
       return (void*)&merge_start_task_14;
      break;
      case TASK_RUN:
       return (void*)&merge_run_task_14;
      break;
      case TASK_KILLED:
       return (void*)&merge_kill_task_14;
      break;
      case TASK_ZOMBIE:
       return 0;
      break;
      case TASK_DESTROY:
       return (void*)&merge_destroy_task_14;
      break;
      default:
       return 0;
      break;
     }
    break;
  case 15:
   switch(status)
   {
    case TASK_INVALID:
     return 0;
    break;
    case TASK_INIT:
     return (void*)&merge_init_task_15;
     break;
      case TASK_START:
       return (void*)&merge_start_task_15;
      break;
      case TASK_RUN:
       return (void*)&merge_run_task_15;
      break;
      case TASK_KILLED:
       return (void*)&merge_kill_task_15;
      break;
      case TASK_ZOMBIE:
       return 0;
      break;
      case TASK_DESTROY:
       return (void*)&merge_destroy_task_15;
      break;
      default:
       return 0;
      break;
     }
    break;

 }

 return 0;
}
