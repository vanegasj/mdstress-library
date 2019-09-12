#pragma once

#include "mds_defines.h"

void custress_init(long ncells, mds::barray periodic);
void custress_clear();

void custress_update_box_spacings(mds::dmatrix box, mds::dmatrix invbox, mds::darray gridsp, mds::iarray nxyz);
void custress_process_batch(int batch_index, mds::batcharrays * h_batch);
