#ifndef mds_custress_h
#define mds_custress_h

#include "mds_defines.h"

// used to allocate space on device, initialize it and deallocate it
void custress_init(size_t nbatches, size_t ncells, int nx, int ny, int nz);
void custress_set_periodic(bool x, bool y, bool z, bool enforce);
void custress_update_box_spacings(const mds::dmatrix box, const mds::dmatrix invbox, const mds::darray gridsp);
void custress_clear();

// adds a pair interaction to batch process at a later stage (everything else private)
void custress_distribute_pair_interaction(const mds::darray xi, const mds::darray xj, const mds::darray Fij, int batch_id);

// sum grid copies batches any remainder and copies grid from custress to mdstresslib
void custress_sum_grid(mds::dmatrix * current_grid);

// checks if GPU is busy
bool custress_is_ready(int batch_id);

#endif // mds_custress_h
