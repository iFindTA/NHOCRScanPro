//
//  cvm_ios.h
//  NHOpenCVPro
//
//  Created by hu jiaju on 15/8/29.
//  Copyright (c) 2015年 hu jiaju. All rights reserved.
//

#include "cvm_olm.h"

// Skews input image from 4 dmz_points to the given dmz_rect.
// Results are rendered to output IplImage, which should already be created.
// from_points should have the following ordering: top-left, top-right, bottom-left, bottom-right
void ios_gpu_unwarp(IplImage *input,const cvm_point from_points[4], IplImage *output);

void *mz_create(void);
void mz_destroy(void *mz);
void mz_prepare_for_backgrounding(void *mz);