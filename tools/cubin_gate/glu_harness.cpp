// AOT CUBIN GATE — minimal CUDA driver harness.
// Loads a Triton-generated sm_120 cubin, launches glu_kernel on random fp32
// data, writes the output, and prints the GPU kernel time (ms) via CUevents.
//
// Build:  /opt/cuda/bin/nvcc -O2 -arch=sm_120 -o glu_harness glu_harness.cpp -L/usr/lib -lcuda
// Run:    ./glu_harness <cubin> <g.bin> <u.bin> <out.bin> <n> <block> <warps> <iters>
//   n         : number of fp32 elements in g.bin / u.bin
//   block     : Triton BLOCK constexpr (1024)
//   warps     : num_warps used at AOT compile time (4) -> CTA threads = warps*32
//   iters     : timed launches (warmup is 5 extra launches)
#include <cuda.h>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <vector>

#define CUCHECK(x)                                                            \
  do {                                                                        \
    CUresult e_ = (x);                                                        \
    if (e_ != CUDA_SUCCESS) {                                                 \
      const char *s_ = nullptr;                                               \
      cuGetErrorString(e_, &s_);                                              \
      std::fprintf(stderr, "CU error %d (%s) at %s:%d\n", e_,                 \
                   s_ ? s_ : "?", __FILE__, __LINE__);                        \
      std::exit(1);                                                           \
    }                                                                         \
  } while (0)

static std::vector<float> read_floats(const char *path, size_t n) {
  std::ifstream f(path, std::ios::binary);
  if (!f) {
    std::fprintf(stderr, "cannot open %s\n", path);
    std::exit(1);
  }
  std::vector<float> v(n);
  f.read(reinterpret_cast<char *>(v.data()), (std::streamsize)(n * sizeof(float)));
  if ((size_t)f.gcount() != n * sizeof(float)) {
    std::fprintf(stderr, "short read from %s\n", path);
    std::exit(1);
  }
  return v;
}

int main(int argc, char **argv) {
  if (argc != 9) {
    std::fprintf(stderr, "usage: %s cubin g.bin u.bin out.bin n block warps iters\n", argv[0]);
    return 2;
  }
  const char *cubin_path = argv[1];
  const char *out_path = argv[4];
  const long long n = std::atoll(argv[5]);
  const int block = std::atoi(argv[6]);
  const int warps = std::atoi(argv[7]);
  const int iters = std::atoi(argv[8]);

  std::vector<float> hg = read_floats(argv[2], (size_t)n);
  std::vector<float> hu = read_floats(argv[3], (size_t)n);
  std::vector<float> ho((size_t)n, 0.0f);

  // Load the cubin bytes.
  std::ifstream cf(cubin_path, std::ios::binary | std::ios::ate);
  if (!cf) {
    std::fprintf(stderr, "cannot open %s\n", cubin_path);
    return 1;
  }
  const std::streamoff size = cf.tellg();
  cf.seekg(0);
  std::vector<unsigned char> blob((size_t)size);
  cf.read(reinterpret_cast<char *>(blob.data()), size);

  CUCHECK(cuInit(0));
  CUdevice dev;
  CUCHECK(cuDeviceGet(&dev, 0));
  CUcontext ctx;
  CUCHECK(cuDevicePrimaryCtxRetain(&ctx, dev));
  CUCHECK(cuCtxSetCurrent(ctx));
  CUmodule mod;
  CUCHECK(cuModuleLoadData(&mod, blob.data()));
  CUfunction func;
  CUCHECK(cuModuleGetFunction(&func, mod, "glu_kernel"));

  size_t bytes = (size_t)n * sizeof(float);
  CUdeviceptr dG, dU, dOut;
  CUCHECK(cuMemAlloc(&dG, bytes));
  CUCHECK(cuMemAlloc(&dU, bytes));
  CUCHECK(cuMemAlloc(&dOut, bytes));
  CUCHECK(cuMemcpyHtoD(dG, hg.data(), bytes));
  CUCHECK(cuMemcpyHtoD(dU, hu.data(), bytes));

  const unsigned grid = (unsigned)((n + block - 1) / block);
  const unsigned threads = (unsigned)(warps * 32);
  int nn = (int)n;
  // Triton's cubin appends two unused scratch-pointer params (global/profile).
  CUdeviceptr scratch = 0;
  void *args[] = {&dG, &dU, &dOut, &nn, &scratch, &scratch};
  auto launch = [&]() {
    CUCHECK(cuLaunchKernel(func, grid, 1, 1, threads, 1, 1, 0, nullptr, args, nullptr));
  };

  for (int i = 0; i < 5; ++i) launch();  // warm up
  CUevent st, en;
  CUCHECK(cuEventCreate(&st, 0));
  CUCHECK(cuEventCreate(&en, 0));
  CUCHECK(cuEventRecord(st, 0));
  for (int i = 0; i < iters; ++i) launch();
  CUCHECK(cuEventRecord(en, 0));
  CUCHECK(cuEventSynchronize(en));
  float total_ms = 0.0f;
  CUCHECK(cuEventElapsedTime(&total_ms, st, en));

  CUCHECK(cuMemcpyDtoH(ho.data(), dOut, bytes));
  std::ofstream out(out_path, std::ios::binary);
  out.write(reinterpret_cast<const char *>(ho.data()), (std::streamsize)bytes);

  std::printf("grid=%u threads=%u ms_cpp=%.6f\n", grid, threads, total_ms / iters);

  CUCHECK(cuEventDestroy(st));
  CUCHECK(cuEventDestroy(en));
  CUCHECK(cuMemFree(dG));
  CUCHECK(cuMemFree(dU));
  CUCHECK(cuMemFree(dOut));
  CUCHECK(cuModuleUnload(mod));
  CUCHECK(cuDevicePrimaryCtxRelease(dev));
  return 0;
}
