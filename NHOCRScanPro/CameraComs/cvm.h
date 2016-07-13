//
//  cvm.h
//  NHOpenCVPro
//
//  Created by hu jiaju on 15/8/29.
//  Copyright (c) 2015年 hu jiaju. All rights reserved.
//

#ifndef __NHOpenCVPro__cvm__
#define __NHOpenCVPro__cvm__

#include "cvm_olm.h"
#include <opencv2/core/types_c.h>

using namespace std;
using namespace cv;

namespace opencvm {
    
    typedef enum {
        CardTypeID,
        CardTypeBank
    }CardType;
    
    class Cvm{
    
    public:
        Cvm();
        
        float cvm_focus_score(IplImage *image, bool use_full_image);
        float cvm_brightness_score(IplImage *image, bool use_full_image);
        void cvm_deinterleave_uint8_c2(IplImage *interleaved, IplImage **channel1, IplImage **channel2);
        bool cvm_detect_edges(IplImage *y_sample, IplImage *cb_sample, IplImage *cr_sample,FrameOrientation orientation, cvm_edges *found_edges, cvm_corner_points *corner_points);
        void cvm_transform_card( IplImage *sample, cvm_corner_points corner_points, FrameOrientation orientation, bool upsample, IplImage **transformed);
        IplImage *segementImg(IplImage *src, CardType type,float x, float y, float w,float h, float t);
        
    private:
        
    };
}

#endif /* defined(__NHOpenCVPro__cvm__) */
