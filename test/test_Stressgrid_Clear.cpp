#include <cassert>
#include <cstring>
#include "mds_stressgrid.h"

namespace mds {

class StressGridTestFixture {
    friend class StressGrid;

public:
    static void test_Clear() {
	mds::StressGrid grid;
	
//	StressGrid::Alloc& alloc = grid.alloc;
//	StressGrid::State& state = grid.state;

//  grid.SetNumberOfGridCellsX(5);
//	grid.SetNumberOfGridCellsY(5);
//	grid.SetNumberOfGridCellsZ(5);
//	grid.SetSpacing(1.0f);

	matrix3_ext box = {{5.0, 0.0, 0.0},
	                   {0.0, 5.0, 0.0},
	                   {0.0, 0.0, 5.0}};
	grid.SetBox(box, 1);
	grid.SetForceDecomposition(mds_ccfd);
	grid.SetFileName("test_output");
	grid.SetMaxThreads(1);
	grid.Init();

	grid.state.nframes = 100;
	grid.state.avg_boxvol = 125.0;
	grid.state.var_boxvol = 10.5;

	void* old_sum_grid = grid.alloc.sum_grid;
	void* old_avg_grid = grid.alloc.avg_grid;

	//assert(old_sum_grid != nullptr && "Pointers should be allocated");
	//assert(old_avg_grid != nullptr && "Pointers should be allocation");

	grid.Clear();

	assert(grid.alloc.sum_grid == nullptr);
	assert(grid.alloc.avg_grid == nullptr);
	assert(grid.alloc.sum_grid_volcovar == nullptr);
	assert(grid.alloc.sum_grid_elcovar == nullptr);
	assert(grid.alloc.sum_grid_elkin == nullptr);
	assert(grid.alloc.sum_grid_elborn == nullptr);
	assert(grid.alloc.current_grid == nullptr);
	assert(grid.alloc.current_grid_elborn == nullptr);
	assert(grid.alloc.current_grid_elkin == nullptr);

	assert(grid.state.nframes == 0);
	assert(grid.state.avg_boxvol == 0.0);
	assert(grid.state.var_boxvol == 0.0);
	assert(!grid.settings.initialized);

	printf("test_StressGrid_Clear.cpp: Friend access Clear() tests passed!\n");
    }
};

} // namespace

int main() {
    mds::StressGridTestFixture::test_Clear();
    return 0;
}

