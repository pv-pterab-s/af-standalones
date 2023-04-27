template <typename T>
void M_(T s, const char *fn_name, const unsigned lineno);
#define P  printf("%s:%d: ", fn_name, lineno);
template <>
void M_(const char *s, const char *fn_name, const unsigned lineno) {
    P; printf("%s\n", s);
}
template <typename T>
void M_(T s, const char *fn_name, const unsigned lineno) {
    P; printf("%.1f\n", (float)s);
}
template <typename T>
sycl::buffer<T, 2> M_(sycl::buffer<T, 2>& out, const char *fn_name, const unsigned lineno) {
    auto dims = out.get_range();
    const sycl::host_accessor<T, 2> out_data(out);
    P; printf("sycl::buffer<T,2> (%d,%d)\n", (unsigned)dims[0], (unsigned)dims[1]);
    for (int i = 0; i < dims[0]; i++) {
      P; printf("    ");
      for (int j = 0; j < dims[1]; j++) {
        printf(" %3.0f", out_data[i][j]);
      }
      printf("\n");
    }
    return out;
}
template <typename T>
void M_(const Param<T> &out, const char *fn_name, const unsigned lineno) {
    auto& dims = out.info.dims; auto& strides = out.info.strides;
    P; printf("Param<T> (%lld,%lld,%lld,%lld)\n", dims[0], dims[1], dims[2], dims[3]);
    const sycl::host_accessor<T, 1> out_data(*out.data);
    for (int i = 0; i < dims[0]; i++) {
      P; printf("    ");
      for (int j = 0; j < dims[1]; j++) {
        printf(" %3.0f", out_data[i + j * strides[0]]);
      }
      printf("\n");
    }
}
#define M(s) M_(s, __FILE__, __LINE__)
