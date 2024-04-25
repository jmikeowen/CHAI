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
// Class definitions we'd like to use on both host and device
//------------------------------------------------------------------------------
template<size_t msize>
class A {
public:
  CHAI_HOST_DEVICE A()                               { printf("A::A()\n"); this->fill(0); }
  CHAI_HOST_DEVICE virtual ~A()                      { printf("A::~A()\n"); }
  CHAI_HOST_DEVICE void fill(const int x)            { printf("A::fill(%d)\n", x); for (auto i = 0u; i < msize; ++i) mstuff[i] = x; }
  CHAI_HOST_DEVICE void print_stuff() const          { printf("A::print_stuff:"); for (auto i = 0u; i < msize; ++i) printf(" %d", mstuff[i]); printf("\n"); }
  CHAI_HOST_DEVICE virtual void func(int x)          { printf("A::func(%d)\n", x); for (auto i = 0u; i < msize; ++i) mstuff[i] += x; }
protected:
  int mstuff[msize];
};

template<size_t msize>
class B: public A<msize> {
public:
  using A<msize>::mstuff;
  CHAI_HOST_DEVICE B(): A<msize>()                   { printf("B::B()\n"); }
  CHAI_HOST_DEVICE virtual ~B()                      { printf("B::~B()\n"); }
  CHAI_HOST_DEVICE virtual void func(int x) override { printf("B::func(%d)\n", x); for (auto i = 0u; i < msize; ++i) mstuff[i] -= x; }
};

//------------------------------------------------------------------------------
// Make an object on the device
//------------------------------------------------------------------------------
template<typename T>
T*
constructOnDevice() {
  printf("constructOnDevice sizeof(T): %d\n", sizeof(T));
  unsigned char* memory;
  cudaMalloc((void**) &memory, sizeof(T));
  T* Tptr = reinterpret_cast<T*>(memory);
  forall(gpu(), 0, 1, [=] __device__ (int i) { new(Tptr) T(); });
  return Tptr;
}

//------------------------------------------------------------------------------
// The test!
//------------------------------------------------------------------------------
GPU_TEST(managed_ptr, polymorphic_type_test) {

  // Allocate objects on the host
  A<20u> ahost;
  B<20u> bhost;
  printf("Host initial object states:\n");
  ahost.print_stuff();
  bhost.print_stuff();
  printf("\n--------------------------------------------------------------------------------\n");

  // Allocate objects on the device
  A<20u>* agpuPtr = constructOnDevice<A<20u>>();
  B<20u>* bgpuPtr = constructOnDevice<B<20u>>();
  printf("GPU initial object states:\n");
  agpuPtr->print_stuff();
  bgpuPtr->print_stuff();

  // Alter the objects on host
  ahost.func(10);
  bhost.func(5);
  printf("After modification on host:\n");
  ahost.print_stuff();
  bhost.print_stuff();
}
