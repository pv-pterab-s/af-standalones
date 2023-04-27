#pragma once

#include <iostream>

template <typename T> void write(std::fstream& outstream, T& out) {
    const size_t bytes = sizeof(out);
    outstream.write((const char *)&out, bytes);
}
template <typename T> void write(std::fstream& outstream, const Param<T>& out) {
    for (int i = 0; i < 4; i++) write(outstream, out.info.dims[i]);
    for (int i = 0; i < 4; i++) write(outstream, out.info.strides[i]);
    write(outstream, out.info.offset);

    const sycl::host_accessor<T> out_data(*out.data);
    const int nelems = out.info.dims[0] * out.info.dims[1] * out.info.dims[2] * out.info.dims[3];
    for (int i = 0; i < nelems; i++) write(outstream, out_data[i]);
}
template <typename T> void write(std::fstream& outstream, Param<T>& out) {
    write(outstream, (const Param<T>&) out);
}
template <typename T> void write(std::fstream& outstream, sycl::buffer<T, 2>& out) {
    Param<T> outParam;
    auto dims = out.get_range();
    outParam.info.dims[0] = dims[0];
    outParam.info.dims[1] = dims[1];
    outParam.info.dims[2] = 1;
    outParam.info.dims[3] = 1;
    outParam.info.strides[0] = 1;
    outParam.info.strides[1] = outParam.info.strides[0] * outParam.info.dims[0];
    outParam.info.strides[2] = outParam.info.strides[1] * outParam.info.dims[1];
    outParam.info.strides[3] = outParam.info.strides[2] * outParam.info.dims[2];
    outParam.info.offset = 0;
    outParam.data = new sycl::buffer<T, 1>{sycl::range{dims[0] * dims[1]}};
    {
      const sycl::host_accessor<T, 2> outAcc(out);
      const sycl::host_accessor<T, 1> outBufferAcc(*outParam.data);
      int idx = 0;
      for (int j = 0; j < dims[1]; j++)
        for (int i = 0; i < dims[0]; i++)
          outBufferAcc[idx++] = outAcc[i][j];
    }
    write(outstream, outParam);
}
template <typename T> void write(std::fstream& outstream, sycl::buffer<T, 1>& out) {
    Param<T> outParam;
    auto dims = out.get_range();
    outParam.info.dims[0] = dims[0];
    outParam.info.dims[1] = 1;
    outParam.info.dims[2] = 1;
    outParam.info.dims[3] = 1;
    outParam.info.strides[0] = 1;
    outParam.info.strides[1] = outParam.info.strides[0] * outParam.info.dims[0];
    outParam.info.strides[2] = outParam.info.strides[1] * outParam.info.dims[1];
    outParam.info.strides[3] = outParam.info.strides[2] * outParam.info.dims[2];
    outParam.info.offset = 0;
    outParam.data = new sycl::buffer<T, 1>{sycl::range{dims[0]}};
    {
      const sycl::host_accessor<T, 1> outAcc(out);
      const sycl::host_accessor<T, 1> outBufferAcc(*outParam.data);
      for (int i = 0; i < dims[0]; i++)
        outBufferAcc[i] = outAcc[i];
    }
    write(outstream, outParam);
}
template <typename T> void read(T &out, std::fstream &instream) {
  size_t bytes = sizeof(T);
  instream.read((char *)&out, bytes);
}
template <typename T> void read(Param<T>& out, std::fstream& instream) {
    for (int i = 0; i < 4; i++) read(out.info.dims[i], instream);
    for (int i = 0; i < 4; i++) read(out.info.strides[i], instream);
    read(out.info.offset, instream);

    const int nelems = out.info.dims[0] * out.info.dims[1] * out.info.dims[2] * out.info.dims[3];
    out.data = new sycl::buffer<T, 1>{sycl::range{nelems}};

    const sycl::host_accessor<T> out_data(*out.data);
    for (int i = 0; i < nelems; i++) read(out_data[i], instream);
}

#define OPEN_W(FN)                                                      \
  std::fstream in_out(FN, std::ios::binary | std::ios::out | std::ios::trunc); \
  if (!in_out.good()) {                                                 \
    printf("could not open file %s\n", FN);                             \
    exit(1);                                                            \
  }
#define OPEN_R(FN)                                              \
  std::fstream in_out(FN, std::ios::binary | std::ios::in);     \
  if (!in_out.good()) {                                         \
    printf("could not open file %s\n", FN);                     \
    exit(1);                                                    \
  }
#define WRITE(VAR)  write(in_out, VAR)
#define READ(VAR)   read(VAR, in_out)
