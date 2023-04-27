#include "af-emu.hpp"
#include "msg.hpp"
#include "io.hpp"

#include "fftconvolve_common.hpp"
#include "fftconvolve_pack.hpp"
#include "fftconvolve_pad.hpp"
#include "fftconvolve_multiply.hpp"
#include "fftconvolve_reorder.hpp"

int main(int argc, char **argv) {
    Param<float> out; Param<cfloat> packed; Param<float> signal; Param<float> filter; int rank; AF_BATCH_KIND kind; bool expand;
    // OPEN_R("/home/gpryor/af-standalones/fftconvolve_pack/fftconvolve_pack.raw");
    // READ(packed); READ(signal); READ(filter); READ(rank); READ(kind);

    // printf("before packDataHelper\n");
    // arrayfire::oneapi::kernel::packDataHelper<cfloat, float>(packed, signal, filter, rank, kind);
    // printf("after packDataHelper\n");
    // arrayfire::oneapi::kernel::padDataHelper<cfloat, float>(packed, signal, filter, rank, kind);
    // printf("after padDataHelper\n");

    // OPEN_R("/home/gpryor/af-standalones/fftconvolve_pack/fftconvolve_multiply.raw");
    // READ(packed); READ(signal); READ(filter); READ(rank); READ(kind);

    // arrayfire::oneapi::kernel::complexMultiplyHelper<cfloat, float>(packed, signal, filter, rank, kind);

    OPEN_R("/home/gpryor/af-standalones/fftconvolve_pack/fftconvolve_reorder.raw");
    READ(out); READ(packed); READ(signal); READ(filter); READ(rank); READ(kind); READ(expand);

    // const sycl::host_accessor<cfloat, 1> out_data(*packed.data);
    // for (unsigned i = 0; i < packed.info.dims[0]; i++) {
    //   for (unsigned j = 0; j < packed.info.dims[1]; j++) {
    //     const unsigned idx =
    //       i * packed.info.strides[0] +
    //       j * packed.info.strides[1];
    //     printf("(%f,%f) ", out_data[idx].real(), out_data[idx].imag());
    //   }
    //   printf("\n");
    // }

    arrayfire::oneapi::kernel::reorderOutputHelper<float, cfloat>(out, packed, signal, filter, rank, kind, expand);
    M(out);

    return 0;
}
