/**
 * Copyright (c) 2012-2019 Qualcomm Technologies, Inc.
 * All Rights Reserved.
 * Confidential and Proprietary - Qualcomm Technologies, Inc
 */


#ifndef RPCMEM_H
#define RPCMEM_H

#ifndef AEESTDDEF_H
#define AEESTDDEF_H
/*
=======================================================================

FILE:         AEEStdDef.h

DESCRIPTION:  definition of basic types, constants, 
                 preprocessor macros

=======================================================================
        Copyright 2005,2007,2016 Qualcomm Technologies Incorporated.
               All Rights Reserved.
            QUALCOMM Proprietary and Confidential
=======================================================================
*/

#if defined(COMDEF_H) /* guards against a known re-definer */
#define _BOOLEAN_DEFINED
#define _UINT32_DEFINED
#define _UINT16_DEFINED
#define _UINT8_DEFINED
#define _INT32_DEFINED
#define _INT16_DEFINED
#define _INT8_DEFINED
#define _UINT64_DEFINED
#define _INT64_DEFINED
#define _BYTE_DEFINED
#endif /* #if !defined(COMDEF_H) */

/* -----------------------------------------------------------------------
** Standard Types
** ----------------------------------------------------------------------- */

/* The following definitions are the same accross platforms.  This first
** group are the sanctioned types.
*/
#include <stddef.h>
#include <stdint.h>

#ifndef _BOOLEAN_DEFINED
typedef  uint8_t      boolean;     /* Boolean value type. */
#define _BOOLEAN_DEFINED
#endif

#ifndef _UINT32_DEFINED
typedef  uint32_t     uint32;      /* Unsigned 32 bit value */
#define _UINT32_DEFINED
#endif

#ifndef _UINT16_DEFINED
typedef  uint16_t     uint16;      /* Unsigned 16 bit value */
#define _UINT16_DEFINED
#endif

#ifndef _UINT8_DEFINED
typedef  uint8_t      uint8;       /* Unsigned 8  bit value */
#define _UINT8_DEFINED
#endif

#ifndef _INT32_DEFINED
typedef  int32_t      int32;       /* Signed 32 bit value */
#define _INT32_DEFINED
#endif

#ifndef _INT16_DEFINED
typedef  int16_t      int16;       /* Signed 16 bit value */
#define _INT16_DEFINED
#endif

#ifndef _INT8_DEFINED
typedef  int8_t       int8;        /* Signed 8  bit value */
#define _INT8_DEFINED
#endif

#ifndef _UINT64_DEFINED
typedef  uint64_t     uint64;      /* Unsigned 64 bit value */
#define _UINT64_DEFINED
#endif

#ifndef _INT64_DEFINED
typedef  int64_t      int64;       /* Signed 64 bit value */
#define _INT64_DEFINED
#endif

#ifndef _BYTE_DEFINED
typedef  uint8_t      byte;        /* byte type */
#define  _BYTE_DEFINED
#endif


#ifndef _AEEUID_DEFINED
typedef uint32             AEEUID;
#define _AEEUID_DEFINED
#endif

#ifndef _AEEIID_DEFINED
typedef uint32             AEEIID;
#define _AEEIID_DEFINED
#endif

#ifndef _AEECLSID_DEFINED
typedef uint32             AEECLSID;
#define _AEECLSID_DEFINED
#endif

#ifndef _AEEPRIVID_DEFINED
typedef uint32             AEEPRIVID;
#define _AEEPRIVID_DEFINED
#endif

#ifndef _AECHAR_DEFINED
typedef uint16             AECHAR;
#define _AECHAR_DEFINED
#endif

#ifndef _AEERESULT_DEFINED
typedef int                AEEResult;
#define _AEERESULT_DEFINED
#endif

/* -----------------------------------------------------------------------
** Function Calling Conventions
** ----------------------------------------------------------------------- */

#ifndef CDECL
#ifdef _MSC_VER
#define CDECL __cdecl
#else
#define CDECL
#endif /* _MSC_VER */
#endif /* CDECL */

/* -----------------------------------------------------------------------
** Constants
** ----------------------------------------------------------------------- */

#ifndef TRUE
#define TRUE   1   /* Boolean true value. */
#endif

#ifndef FALSE
#define FALSE  0   /* Boolean false value. */
#endif

#ifndef NULL
#define NULL  0
#endif

#ifndef MIN_INT8
#define MIN_INT8 -128
#endif
#ifndef MIN_INT16
#define MIN_INT16 -32768
#endif
#ifndef MIN_INT32
#define MIN_INT32 (~0x7fffffff)   /* -2147483648 is unsigned */
#endif
#ifndef MIN_INT64
#define MIN_INT64 (~0x7fffffffffffffffLL) /* -9223372036854775808 is unsigned */
#endif

#ifndef MAX_INT8
#define MAX_INT8 127
#endif
#ifndef MAX_INT16
#define MAX_INT16 32767
#endif
#ifndef MAX_INT32
#define MAX_INT32 2147483647
#endif
#ifndef MAX_INT64
#define MAX_INT64 9223372036854775807LL
#endif

#ifndef MAX_UINT8
#define MAX_UINT8 255
#endif
#ifndef MAX_UINT16
#define MAX_UINT16 65535
#endif
#ifndef MAX_UINT32
#define MAX_UINT32 4294967295u
#endif
#ifndef MAX_UINT64
#define MAX_UINT64 18446744073709551615uLL
#endif

#ifndef MIN_AECHAR
#define MIN_AECHAR 0
#endif
#ifndef MAX_AECHAR
#define MAX_AECHAR 65535
#endif


/* -----------------------------------------------------------------------
** Preprocessor helpers
** ----------------------------------------------------------------------- */
#define __STR__(x) #x
#define __TOSTR__(x) __STR__(x)
#define __FILE_LINE__ __FILE__ ":" __TOSTR__(__LINE__)

/* -----------------------------------------------------------------------
** Types for code generated from IDL
** ----------------------------------------------------------------------- */

#ifndef __QIDL_WCHAR_T_DEFINED__
#define __QIDL_WCHAR_T_DEFINED__
typedef uint16 _wchar_t;
#endif

/* __STRING_OBJECT__ will be deprecated in the future */
#if !defined(__QIDL_STRING_OBJECT_DEFINED__) && !defined(__STRING_OBJECT__)
#define __QIDL_STRING_OBJECT_DEFINED__
#define __STRING_OBJECT__
typedef struct _cstring_s {
   char* data;
   int dataLen;
   int dataLenReq;
} _cstring_t;

typedef struct _wstring_s {
   _wchar_t* data;
   int dataLen;
   int dataLenReq;
} _wstring_t;
#endif /* __QIDL_STRING_OBJECT_DEFINED__ */

/*
=======================================================================
  DATA STRUCTURES DOCUMENTATION
=======================================================================

boolean

Description:
    This type is used to express boolean values (TRUE or FALSE).

Definition:
    typedef  unsigned char boolean

=======================================================================

uint32

Description:
    This is a 32-bit unsigned integer.

Definition:
    typedef  unsigned long int  uint32

=======================================================================

uint16

Description:
    This is a 16-bit unsigned integer.

Definition:
    typedef  unsigned short     uint16

=======================================================================

uint8

Description:
    This is an 8-bit unsigned integer.

Definition:
    typedef  unsigned char      uint8

=======================================================================

int32

Description:
    This is a 32-bit signed integer.

Definition:
    typedef  signed long int    int32

=======================================================================

int16

Description:
    This is a 16-bit signed integer.

Definition:
    typedef  signed short       int16

=======================================================================

int8

Description:
    This is an 8-bit signed integer.

Definition:
    typedef  signed char        int8

=======================================================================

uint64

Description:
    This is a 64-bit unsigned integer.

Definition:
    typedef  unsigned __int64   uint64

=======================================================================

int64

Description:
    This is a 64-bit signed integer.

Definition:
    typedef  __int64            int64

=======================================================================

byte

Description:
    This is a byte.

Definition:
    typedef  unsigned char     byte

=======================================================================

AEEUID

Description:
   This is a BREW unique ID.  Used to express unique types, interfaces, classes
     groups and privileges.  The BREW ClassID Generator generates 
     unique IDs that can be used anywhere you need a new AEEIID, AEECLSID,
     or AEEPRIVID.

Definition:
    typedef uint32             AEEUID

=======================================================================

AEEIID

Description:
   This is an interface ID type, used to denote a BREW interface. It is a special case
     of AEEUID.

Definition:
    typedef uint32             AEEIID

=======================================================================

AEECLSID

Description:
   This is a classe ID type, used to denote a BREW class. It is a special case
     of AEEUID.

Definition:
    typedef uint32             AEECLSID

=======================================================================

AEEPRIVID

Description:
   This is a privilege ID type, used to express a privilege.  It is a special case
     of AEEUID.

Definition:
    typedef uint32             AEEPRIVID

=======================================================================

AECHAR

Description:
   This is a 16-bit character type.

Definition:
   typedef uint16             AECHAR

=======================================================================

AEEResult

Description:
   This is the standard result type.

Definition:
   typedef int                AEEResult

=======================================================================

_wchar_t

Description:
   This is a 16-bit character type corresponding to the IDL 'wchar'
   type.

Definition:
   typedef uint16             _wchar_t

=======================================================================

_cstring_t

Description:
   This structure is used to represent an IDL string when used inside a
   sequence or union.

Definition:
   typedef struct _cstring_s {
      char* data;
      int dataLen;
      int dataLenReq;
   } _cstring_t;

Members:
   data       : A pointer to the NULL-terminated string.
   dataLen    : The size, in chars, of the buffer pointed to by 'data',
                including the NULL terminator.  This member is only used
                when the structure is part of an rout or inrout
                parameter, but must be supplied by the caller as an
                input in these cases.
   dataLenReq : The size that would have been required to store the
                entire result string.  This member is only used when the
                structure is part of an rout or inrout parameter, when
                it is an output value set by the callee.  The length of
                the returned string (including the NULL terminator)
                after a call is the minimum of dataLen and dataLenReq.

=======================================================================

_wstring_t

Description:
   This structure is used to represent an IDL wstring when used inside a
   sequence or union.

Definition:
   typedef struct _wstring_s {
      _wchar_t* data;
      int dataLen;
      int dataLenReq;
   } _wstring_t;

Members:
   data       : A pointer to the NULL-terminated wide string.
   dataLen    : The size, in 16-bit characters, of the buffer pointed to
                by 'data', including the NULL terminator.  This member
                is only used when the structure is part of an rout or
                inrout parameter, but must be supplied by the caller as
                an input in these cases.
   dataLenReq : The number of 16-bit characters that would have been
                required to store the entire result string.  This member
                is only used when the structure is part of an rout or
                inrout parameter, when it is an output value set by the
                callee.  The length of the returned wstring (including
                the NULL terminator) after a call is the minimum of
                dataLen and dataLenReq.

=======================================================================
CONSTANTS DOCUMENTATION
=======================================================================

TRUE  

Description:
   TRUE is the boolean "true."

Definition:

   #define TRUE    1

=======================================================================

FALSE 

Description:
   FALSE is the boolean "false."

Definition:

   #define FALSE   0

=======================================================================

NULL 

Description:
   NULL is the null value, usually used to test a pointer.

Definition:

   #define NULL   0


=======================================================================

MIN_INT8

Description:
  MIN_INT8 is the minimum signed 8-bit integer value.

Definition:

   #define MIN_INT8  -128

=======================================================================

MIN_INT16

Description:
  MIN_INT16 is the minimum signed 16-bit integer value

Definition:

   #define MIN_INT16  -32768


=======================================================================

MIN_INT32

Description:
   MIN_INT32 is the minimum signed 32-bit integer value.

Definition:

   #define MIN_INT32 (~0x7fffffff)

Comments:
    We use (~0x7fffffff), because -2147483648 is treated as unsigned by 
    compilers

=======================================================================

MIN_INT64

Description:
   MIN_INT64 is the minimum signed 64-bit integer value.

Definition:

   #define MIN_INT64 (~0x7fffffffffffffffll)

Comments:
    We use (~0x7fffffffffffffffll), because -9223372036854775808 is 
       treated as unsigned by compilers

=======================================================================

MAX_INT8

Description:
   MAX_INT8 is the maximum signed 8-bit integer value

Definition:

   #define MAX_INT8  127

=======================================================================

MAX_INT16

Description:
   MAX_INT16 is the maximum signed 16-bit integer value.

Definition:

   #define MAX_INT16  32767

=======================================================================

MAX_INT32

Description:
   MAX_INT32 is the maximum signed 32-bit integer value.

Definition:

   #define MAX_INT32  2147483647

=======================================================================

MAX_INT64

Description:
   MAX_INT64 is the maximum signed 64-bit integer value.

Definition:

   #define MAX_INT64  9223372036854775807ll

=======================================================================

MAX_UINT8

Description:
   MAX_UINT8 is the maximum unsigned 8-bit integer value.

Definition:

   #define MAX_UINT8  255

=======================================================================

MAX_UINT16

Description:
   MAX_UINT16 is the maximum unsigned 16-bit integer value.

Definition:

   #define MAX_UINT16  65535

=======================================================================

MAX_UINT32

Description:
   MAX_UINT32 is the maximum unsigned 32-bit integer value.

Definition:

   #define MAX_UINT32  4294967295u


=======================================================================

MAX_UINT64

Description:
   MAX_UINT64 is the maximum unsigned 64-bit integer value.

Definition:

   #define MAX_UINT64  18446744073709551615ull

=======================================================================

MIN_AECHAR

Description:
  MIN_AECHAR is the minimum AECHAR value.

Definition:

   #define MIN_AECHAR  0

=======================================================================

MAX_AECHAR

Description:
  MAX_AECHAR is the maximum AECHAR value.

Definition:

   #define MAX_AECHAR  65535

=======================================================================
MACROS DOCUMENTATION
=======================================================================

__STR__()

Description:
   The __STR__() makes a token into a string, used to string-ize things already
      defined.

Definition:

   #define __STR__(x)  #x

Parameters:
    x: token to make into a string

=======================================================================

__TOSTR__()

Description:
   The __TOSTR__() makes a token's value into a string, used to string-ize things 
    already defined, used with __STR__.

Definition:

   #define __TOSTR__(x)  __STR__(x)

Parameters:
    x: token to evaluate and string-ize

Evaluation Value:
    the token's replacement as a string

=======================================================================

__FILE_LINE__

Description:
   The compiler's __FILE__ (a string) and __LINE__ (an integer) are pasted
     together as a single string with a ":" between.

Definition:

   #define __FILE_LINE__ __FILE__ ":"  __TOSTR__(__LINE__)

Evaluation Value:
   __FILE__":""__LINE__"

=======================================================================
*/

#endif /* #ifndef AEESTDDEF_H */



/**
 * RPCMEM_DEFAULT_HEAP
 * Dynamicaly select the heap to use.  This should be ok for most usecases.
 */
#define RPCMEM_DEFAULT_HEAP -1


/**
 * RPCMEM_DEFAULT_FLAGS should allocate memory with the same properties
 * as the ION_FLAG_CACHED flag
 */
#ifdef ION_FLAG_CACHED
#define RPCMEM_DEFAULT_FLAGS ION_FLAG_CACHED
#else
#define RPCMEM_DEFAULT_FLAGS 1
#endif

/**
 * RPCMEM HEAP IDs
 * SYSTEM HEAP:
 *   - non-contiguous physical memory
 *   - for sub-systems with SMMU
 *   - recommended for HVX/CDSPs
 * CONTIG HEAP:
 *   - Contiguous physical memory
 *   - limited memory
 *   - for sub-systems without SMMU (ex. sDSP and mDSP)
 */
#define RPCMEM_HEAP_ID_SYSTEM         (25)
#define RPCMEM_HEAP_ID_CONTIG         (22)
#define RPCMEM_HEAP_ID_SECURE         (9)
#define RPCMEM_HEAP_ID_SYSTEM_CONTIG  (21)

/**
 * RPCMEM_FLAG_UNCACHED
 * ION_FLAG_CACHED should be defined as 1
 */
#define RPCMEM_FLAG_UNCACHED 0
#define RPCMEM_FLAG_CACHED RPCMEM_DEFAULT_FLAGS

/**
 * examples:
 *
 * heap 22, uncached, 1kb
 *    rpcmem_alloc(22, 0, 1024);
 *    rpcmem_alloc(22, RPCMEM_FLAG_UNCACHED, 1024);
 *
 * heap 21, cached, 2kb
 *    rpcmem_alloc(21, RPCMEM_FLAG_CACHED, 2048);
 *    #include <ion.h>
 *    rpcmem_alloc(21, ION_FLAG_CACHED, 2048);
 *
 * just give me the defaults, 2kb
 *    rpcmem_alloc(RPCMEM_DEFAULT_HEAP, RPCMEM_DEFAULT_FLAGS, 2048);
 *    rpcmem_alloc_def(2048);
 *
 * give me the default flags, but from heap 18, 4kb
 *    rpcmem_alloc(18, RPCMEM_DEFAULT_FLAGS, 4096);
 *
 */
#define ION_SECURE_FLAGS    ((1 << 31) | (1 << 19))
#ifdef __cplusplus
extern "C" {
#endif

/**
 * call once to initialize the library
 * Note: should not call this if rpcmem is linked from libadsprpc.so
 * /libcdsprpc.so/libmdsprpc.so/libsdsprpc.so
 */
extern void rpcmem_init(void);

/**
 * call once for cleanup
 * Note: should not call this if rpcmem is linked from libadsprpc.so
 * /libcdsprpc.so/libmdsprpc.so/libsdsprpc.so
 */
extern void rpcmem_deinit(void);

/**
 * Allocate via ION a buffer of size
 * @heapid, the heap id to use
 * @flags, ion flags to use to when allocating
 * @size, the buffer size to allocate
 * @retval, 0 on failure, pointer to buffer on success
 *
 * For example:
 *    buf = rpcmem_alloc(RPCMEM_DEFAULT_HEAP, RPCMEM_DEFAULT_FLAGS, size);
 */

extern void* rpcmem_alloc(int heapid, uint32 flags, int size);

/**
 * allocate with default settings
 */
 #if !defined(WINNT) && !defined (_WIN32_WINNT)
__attribute__((unused))
#endif
static __inline void* rpcmem_alloc_def(int size) {
   return rpcmem_alloc(RPCMEM_DEFAULT_HEAP, RPCMEM_DEFAULT_FLAGS, size);
}

/**
 * free buffer, ignores invalid buffers
 */
extern void rpcmem_free(void* po);

/**
 * returns associated fd
 */
extern int rpcmem_to_fd(void* po);

int is_kernel_alloc_supported(int dev, int domain);

/*
 * API to allocate ION memory for internal purposes
 * Returns NULL if allocation fails
 *
 */
void* rpcmem_alloc_internal(int heapid, uint32 flags, int size);

/*
 * API to free internally allocated ION memory
 *
 */
void rpcmem_free_internal(void* po);

/*
 * API to get fd of internally allocated ION buffer
 * Returns valid fd on success and -1 on failure
 *
 */
int rpcmem_to_fd_internal(void *po);

#ifdef __cplusplus
}
#endif

#define RPCMEM_HEAP_DEFAULT     0x80000000
#define RPCMEM_HEAP_NOREG       0x40000000
#define RPCMEM_HEAP_UNCACHED    0x20000000
#define RPCMEM_HEAP_NOVA        0x10000000
#define RPCMEM_HEAP_NONCOHERENT 0x08000000

#endif /* RPCMEM_H */

