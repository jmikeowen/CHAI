//------------------------------------------------------------------------------
// Copyright (c) 2016-24, Lawrence Livermore National Security, LLC and CHAI
// project contributors. See the CHAI LICENSE file for details.
//
// SPDX-License-Identifier: BSD-3-Clause
//------------------------------------------------------------------------------
#include "chai/ChaiMacros.hpp"
#include "chai/ManagedSharedPtr.hpp"
#include "chai/SharedPtrManager.hpp"
#include "gtest/gtest.h"
#include "umpire/ResourceManager.hpp"

#define GPU_TEST(X, Y)              \
  static void gpu_test_##X##Y();    \
  TEST(X, Y) { gpu_test_##X##Y(); } \
  static void gpu_test_##X##Y()

#include "chai/config.hpp"
#include "chai/ArrayManager.hpp"
#include "chai/ManagedArray.hpp"
#include "chai/managed_ptr.hpp"
#include "chai/ManagedSharedPtr.hpp"

#include "../src/util/forall.hpp"

#include <cstdlib>

//------------------------------------------------------------------------------
// Execute CUDA code with error checking
//------------------------------------------------------------------------------
inline void gpuErrorCheck(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) {
      fprintf(stderr, "[CHAI] GPU Error: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) {
         exit(code);
      }
   }
}

#define GPU_ERROR_CHECK(code) { gpuErrorCheck((code), __FILE__, __LINE__); }

//------------------------------------------------------------------------------
// Macro for executing code on host
//------------------------------------------------------------------------------
#define HOST_EXEC(code) do {                   \
  forall(sequential(), 0, 1, [=] (int i) {     \
    code                                       \
  });                                          \
} while (false)

//------------------------------------------------------------------------------
// Macro for executing code on GPU
//------------------------------------------------------------------------------
#define GPU_EXEC(code) do {                    \
  forall(gpu(), 0, 1, [=] __device__ (int i) { \
    code                                       \
  });                                          \
  GPU_ERROR_CHECK( cudaPeekAtLastError() );    \
  GPU_ERROR_CHECK( cudaDeviceSynchronize() );  \
} while (false)

//------------------------------------------------------------------------------
// Class definitions we'd like to use on both host and device
//------------------------------------------------------------------------------
template<size_t size>
class A {
public:

  CHAI_HOST_DEVICE A()                               { printf("A::A()\n"); this->fill(0); }
  CHAI_HOST_DEVICE virtual ~A()                      { printf("A::~A()\n"); }
  CHAI_HOST_DEVICE void fill(const int x)            { printf("A::fill(%d)\n", x); for (auto i = 0u; i < size; ++i) mstuff[i] = x; }
  CHAI_HOST_DEVICE void print_stuff() const          { printf("A::print_stuff:"); for (auto i = 0u; i < size; ++i) printf(" %d", mstuff[i]); printf("\n"); }
  CHAI_HOST_DEVICE virtual void func(int x)          { printf("A::func(%d)\n", x); for (auto i = 0u; i < size; ++i) mstuff[i] += x; }
  CHAI_HOST_DEVICE int* stuff()                      { printf("A::stuff()\n"); return mstuff; }
  CHAI_HOST_DEVICE const int* stuff() const          { printf("A::stuff() (const)\n"); return mstuff; }
protected:
  int mstuff[size];
};

template<size_t size>
class B: public A<size> {
public:
  using A<size>::mstuff;
  CHAI_HOST_DEVICE B(): A<size>()                    { printf("B::B()\n"); }
  CHAI_HOST_DEVICE virtual ~B()                      { printf("B::~B()\n"); }
  CHAI_HOST_DEVICE virtual void func(int x) override { printf("B::func(%d)\n", x); for (auto i = 0u; i < size; ++i) mstuff[i] -= x; }
};

//------------------------------------------------------------------------------
// Serialization methods for packing/unpacking objects
//------------------------------------------------------------------------------
// Define a type we'll use for buffers of binary data
using Buffer = chai::ManagedArray<uint8_t>;

template<size_t size>
CHAI_HOST
void pack_host(const A<size>& a, Buffer& buf) {
  const auto binsize = sizeof(int) * size;
  auto* astuffptr = reinterpret_cast<const uint8_t*>(a.stuff());
  buf.allocate(binsize);
  memcpy(buf.data(), astuffptr, binsize);
}

template<size_t size>
CHAI_DEVICE
void pack_device(const A<size>& a, Buffer const& buf) {
  const auto binsize = sizeof(int) * size;
  assert(buf.size() == binsize);
  memcpy(buf.data(), a.stuff(), binsize);
}

template<size_t size>
CHAI_HOST_DEVICE
void unpack(A<size>& a, Buffer const& buf) {
  const auto binsize = sizeof(int) * size;
  assert(buf.size() == binsize);
  auto* astuffptr = reinterpret_cast<uint8_t*>(a.stuff());
  memcpy(astuffptr, buf.data(), binsize);
}

//------------------------------------------------------------------------------
// Make an object on the device
//------------------------------------------------------------------------------
template<typename T>
T*
constructOnDevice() {
  printf("constructOnDevice sizeof(T): %d\n", sizeof(T));
  T* Tptr;
  GPU_ERROR_CHECK(cudaMalloc((void**) &Tptr, sizeof(T)));
  GPU_EXEC( new(Tptr) T(); );
  return Tptr;
}

//------------------------------------------------------------------------------
// The test!
//------------------------------------------------------------------------------
GPU_TEST(managed_ptr, polymorphic_type_test) {

  printf("\n--------------------------------------------------------------------------------\n");
  printf("Allocating objects on host\n");
  A<20u> ahost;
  B<10u> bhost;
  printf("Host initial object states:\n");
  ahost.print_stuff();
  bhost.print_stuff();

  printf("\n--------------------------------------------------------------------------------\n");
  printf("Allocate objects on the device\n");
  A<20u>* agpuPtr = constructOnDevice<A<20u>>();
  B<10u>* bgpuPtr = constructOnDevice<B<10u>>();
  printf("GPU initial object states:\n");
  GPU_EXEC( 
           agpuPtr->print_stuff();
           bgpuPtr->print_stuff();
            );

  printf("\n--------------------------------------------------------------------------------\n");
  printf("Alter the objects on host\n");
  ahost.func(10);
  bhost.func(5);
  printf("After modification on host:\n");
  ahost.print_stuff();
  bhost.print_stuff();

  printf("\n--------------------------------------------------------------------------------\n");
  printf("Move object states from host -> GPU\n");
  Buffer abuf, bbuf;
  pack_host(ahost, abuf);
  pack_host(bhost, bbuf);
  printf("abuf.size(): %d\n", abuf.size());
  printf("bbuf.size(): %d\n", bbuf.size());
  GPU_EXEC(
           unpack(*agpuPtr, abuf);
           unpack(*bgpuPtr, bbuf);
           agpuPtr->print_stuff();
           bgpuPtr->print_stuff();
           );

  printf("\n--------------------------------------------------------------------------------\n");
  printf("Alter the objects on device\n");
  GPU_EXEC(
           agpuPtr->func(100);
           bgpuPtr->func(50);
           );
  printf("After modification on device:\n");
  GPU_EXEC(
           agpuPtr->print_stuff();
           bgpuPtr->print_stuff();
           );

  printf("\n--------------------------------------------------------------------------------\n");
  printf("Move object states from GPU -> host\n");
  GPU_EXEC(
           pack_device(*agpuPtr, abuf);
           pack_device(*bgpuPtr, bbuf);
           );
  unpack(ahost, abuf);
  unpack(bhost, bbuf);
  ahost.print_stuff();
  bhost.print_stuff();

  printf("\n--------------------------------------------------------------------------------\n");
  printf("Free memory.  This is the sort of thing I don't want to have to worry about...\n");
  GPU_EXEC(
           agpuPtr->~A();
           bgpuPtr->~B();
          );
  cudaFree(agpuPtr);
  cudaFree(bgpuPtr);
}
