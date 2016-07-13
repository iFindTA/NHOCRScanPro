//
//  cvm_prep.h
//  NHOpenCVPro
//
//  Created by hu jiaju on 15/8/31.
//  Copyright (c) 2015年 hu jiaju. All rights reserved.
//

#ifndef NHOpenCVPro_cvm_prep_h
#define NHOpenCVPro_cvm_prep_h

#include <opencv2/opencv.hpp>

#if defined (WIN32) || defined (_WIN32)
#include <objbase.h>
#endif

#include <stdlib.h>
#include <stdio.h>

#if defined (WIN32) || defined (_WIN32)
#include <io.h>
#elif defined (linux) || defined (__linux__)

#endif

#include <iostream>
#include <fstream>
#include <assert.h>
#include <algorithm>
#include <cstdlib>
#include <time.h>
#include <math.h>

using namespace std;
using namespace cv;
#define projectpath "."

#endif
