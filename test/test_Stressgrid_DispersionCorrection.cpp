#include <cassert>
#include <cstring>
#include <cstdio>

#include "mds_stressgrid.h"

namespace mds {
	
class StressGridTestFixture {
public: 
    static void test_DispersionCorrection() {
    	mds::StressGrid grid;

	matrix3_ext box = {{5.0, 0.0, 0.0},
		           {0.0, 5.0, 0.0},
			   {0.0, 0.0, 5.0}};
	grid.SetBox(box, 1);
	grid.SetForceDecomposition(mds_ccfd);
	grid.SetFileName("test_output");
	grid.SetMaxThreads(1);
	grid.Init();
	
	// Test 1: verify dispcor is applied when nodispcor = false
        grid.settings.nodispcor = false;
    	
	real_mds original_values[3];
	original_values[0] = grid.alloc.current_grid[0][0][0];
	original_values[1] = grid.alloc.current_grid[0][1][1];
        original_values[2] = grid.alloc.current_grid[0][2][2];

	printf("   Orignal grid[0] diags: [%f, %f, %f]\n",
		original_values[0], original_values[1], original_values[2]);

	real_ext shift = 5.0;
	grid.DispersionCorrection(shift);

	// validating that shift was applied
	for (int i = 0; i < grid.state.nCells; i++) {
	    real_mds expected = original_values[0] + realval_mds(shift);
	    assert(fabs(grid.alloc.current_grid[i][0][0] - expected) < 1e-6 &&
                    "Diagonal [0][0] should be shifted");

	    expected = original_values[1] + realval_mds(shift);
	    assert(fabs(grid.alloc.current_grid[i][1][1] - expected) < 1e-6 &&
		      "Diagonal [1][1] should be shifted");
           
	    expected = original_values[2] + realval_mds(shift);
	    assert(fabs(grid.alloc.current_grid[i][2][2] - expected) < 1e-6 &&
                    "Diagonal [2][2] should be shifted");
	}

	printf("Shift of %f applied to all %lu cells\n", shift, grid.state.nCells);
    }	
};
} // namespace mds

int main() {
    mds::StressGridTestFixture::test_DispersionCorrection();
    printf("All tests completed successfully!");
    return 0;
}
