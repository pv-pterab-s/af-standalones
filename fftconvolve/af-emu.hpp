#include <CL/sycl.hpp>
// #include <arrayfire.h>
#include <stdlib.h>

#define ONEAPI_DEBUG_FINISH(S) ;

#define divup(a, b) (((a) + (b)-1) / (b))

typedef unsigned char uchar;

typedef long long dim_t;

typedef struct {
    dim_t dims[4];
    dim_t strides[4];
    dim_t offset;
} KParam;

template<typename T>
struct Param {
  sycl::buffer<T> *data;
  KParam info;
};

sycl::queue getQueue() {
    return sycl::queue();
}

template<typename T>
Param<T> create_param(size_t dims0, size_t dims1 = 1, size_t dims2 = 1, size_t dims3 = 1) {
    Param<T> out;
    out.info.dims[0] = dims0;
    out.info.dims[1] = dims1;
    out.info.dims[2] = 1;
    out.info.dims[3] = 1;
    out.info.strides[0] = 1;
    out.info.strides[1] = out.info.strides[0] * out.info.dims[0];
    out.info.strides[2] = out.info.strides[1] * out.info.dims[1];
    out.info.strides[3] = out.info.strides[2] * out.info.dims[2];
    out.info.offset = 0;
    out.data = new sycl::buffer<T, 1>{sycl::range{dims0 * dims1}};
    return out;
}

using cdouble = std::complex<double>;
using cfloat  = std::complex<float>;

// TODO AF_BATCH_UNSUPPORTED is not required and shouldn't happen
//      Code changes are required to handle all cases properly
//      and this enum value should be removed.
typedef enum {
    AF_BATCH_UNSUPPORTED = -1, /* invalid inputs */
    AF_BATCH_NONE,             /* one signal, one filter   */
    AF_BATCH_LHS,              /* many signal, one filter  */
    AF_BATCH_RHS,              /* one signal, many filter  */
    AF_BATCH_SAME,             /* signal and filter have same batch size */
    AF_BATCH_DIFF,             /* signal and filter have different batch size */
} AF_BATCH_KIND;

unsigned nextpow2(unsigned x) {
    x = x - 1U;
    x = x | (x >> 1U);
    x = x | (x >> 2U);
    x = x | (x >> 4U);
    x = x | (x >> 8U);
    x = x | (x >> 16U);
    return x + 1U;
}
