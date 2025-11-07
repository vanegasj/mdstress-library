#include <cassert>
#include <iostream>
#include <random>
#include <cmath>
#include "mdstress/mds_stressgrid.h"

// add this helper back
static inline void header(const char* name) {
  std::cout << "\n=== " << name << " ===\n";
}

// Test 1: diffarray3 basic geometry
static void test_diffarray3() {
  header("diffarray3");

  mds::state_t s{};

  // Initialize a 3x3 cubic box of length 10
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j)
      s.box[i][j] = 0.0;
  s.box[0][0] = 10.0;
  s.box[1][1] = 10.0;
  s.box[2][2] = 10.0;

  s.periodic[0] = s.periodic[1] = s.periodic[2] = false;

  mds::array3_mds a = {1.0, 2.0, 3.0};
  mds::array3_mds b = {4.0, 5.0, 6.0};
  mds::array3_mds out = {0.0, 0.0, 0.0};

  mds::diffarray3(b, a, out, s.box, s.periodic);

  assert(std::abs(out[0] - 3.0) < 1e-12 &&
         std::abs(out[1] - 3.0) < 1e-12 &&
         std::abs(out[2] - 3.0) < 1e-12);

  std::cout << "ok\n";
}

// call the test (optional, but removes the “defined but not used” warning)
int main() {
  test_diffarray3();
  return 0;
}
