//
//  cvm.cpp
//  NHOpenCVPro
//
//  Created by hu jiaju on 15/8/29.
//  Copyright (c) 2015年 hu jiaju. All rights reserved.
//

#include "cvm.h"
//#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <string>
#include <fstream>
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/core/core_c.h>
#include <opencv2/core/types_c.h>
#include <opencv2/imgproc/types_c.h>
#include <opencv2/imgproc/imgproc_c.h>
#include <opencv2/highgui/highgui_c.h>
#include <opencv2/imgproc/imgproc.hpp>

#define cvm_likely(x) __builtin_expect(!!(x),1)
#define cvm_unlikely(x) __builtin_expect(!!(x),0)

#ifdef _ARM_ARCH_7
#define DMZ_HAS_NEON_COMPILETIME 1
#else
#define DMZ_HAS_NEON_COMPILETIME 0
#endif

namespace opencvm {
    
#define kHoughGradientAngleThreshold 10
    
#define kHoughThresholdLengthDivisor 6 // larger value --> accept more lines as lines
    
#define kHorizontalAngle ((float)(CV_PI / 2.0f))
#define kVerticalAngle ((float)CV_PI)
#define kMaxAngleDeviationAllowed ((float)(5.0f * (CV_PI / 180.0f)))
    
#define kVerticalPercentSlop 0.03f
#define kHorizontalPercentSlop 0.03f
    
#define kSmallCharacterWidth 9
#define kSmallCharacterHeight 15
    
#define kTrimmedCharacterImageWidth 11
#define kTrimmedCharacterImageHeight 16
    
#define kMinimumExpiryStripCharacters 5
#define kMinimumNameStripCharacters 5
    
    typedef struct {
        CvRect top;
        CvRect bottom;
        CvRect left;
        CvRect right;
    } DetectionBoxes;
    
    enum {
        LineOrientationVertical = 0,
        LineOrientationHorizontal = 1,
    };
    typedef uint8_t LineOrientation;
    
    typedef struct CvLinePolar {
        float rho;
        float angle;
        bool is_null;
    } CvLinePolar;
    
    Cvm::Cvm(){
        
    }
    
#pragma mark -- Function Declary --
    
    
    
#pragma mark -- Hough Method --
    
#define TO_RADIANS(in_degrees) (CV_PI * (in_degrees) / 180.0f)
    
    CvLinePolar llcv_hough(const CvArr *src_image, IplImage *dx, IplImage *dy, float rho, float theta, int threshold, float theta_min, float theta_max, bool vertical, float gradient_angle_threshold) {
        CvMat img_stub, *img = (CvMat*)src_image;
        img = cvGetMat(img, &img_stub);
        
        CvMat dx_stub, *dx_mat = (CvMat*)dx;
        dx_mat = cvGetMat(dx_mat, &dx_stub);
        
        CvMat dy_stub, *dy_mat = (CvMat*)dy;
        dy_mat = cvGetMat(dy_mat, &dy_stub);
        
        if(!CV_IS_MASK_ARR(img)) {
            CV_Error(CV_StsBadArg, "The source image must be 8-bit, single-channel");
        }
        
        if(rho <= 0 || theta <= 0 || threshold <= 0) {
            CV_Error(CV_StsOutOfRange, "rho, theta and threshold must be positive");
        }
        
        if(theta_max < theta_min + theta) {
            CV_Error(CV_StsBadArg, "theta + theta_min (param1) must be <= theta_max (param2)");
        }
        
        cv::AutoBuffer<int> _accum;
        cv::AutoBuffer<int> _tabSin, _tabCos;
        
        const uchar* image;
        int step, width, height;
        int numangle, numrho;
        float ang;
        int r, n;
        int i, j;
        float irho = 1 / rho;
        float scale;
        
        CV_Assert( CV_IS_MAT(img) && CV_MAT_TYPE(img->type) == CV_8UC1 );
        
        image = img->data.ptr;
        step = img->step;
        width = img->cols;
        height = img->rows;
        
        const uint8_t *dx_mat_ptr = (uint8_t *)(dx_mat->data.ptr);
        int dx_step = dx_mat->step;
        const uint8_t *dy_mat_ptr = (uint8_t *)(dy_mat->data.ptr);
        int dy_step = dy_mat->step;
        
        numangle = cvRound((theta_max - theta_min) / theta);
        numrho = cvRound(((width + height) * 2 + 1) / rho);
        
        _accum.allocate((numangle+2) * (numrho+2));
        _tabSin.allocate(numangle);
        _tabCos.allocate(numangle);
        int *accum = _accum;
        int *tabSin = _tabSin, *tabCos = _tabCos;
        
        memset(accum, 0, sizeof(accum[0]) * (numangle + 2) * (numrho + 2));
        
#define FIXED_POINT_EXPONENT 10
#define FIXED_POINT_MULTIPLIER (1 << FIXED_POINT_EXPONENT)
        
        for(ang = theta_min, n = 0; n < numangle; ang += theta, n++) {
            tabSin[n] = (int)floorf(FIXED_POINT_MULTIPLIER * sinf(ang) * irho);
            tabCos[n] = (int)floorf(FIXED_POINT_MULTIPLIER * cosf(ang) * irho);
        }
        
        float slope_bound_a, slope_bound_b;
        if(vertical) {
            slope_bound_a = tanf((float)TO_RADIANS(180 - gradient_angle_threshold));
            slope_bound_b = tanf((float)TO_RADIANS(180 + gradient_angle_threshold));
        } else {
            slope_bound_a = tanf((float)TO_RADIANS(90 - gradient_angle_threshold));
            slope_bound_b = tanf((float)TO_RADIANS(90 + gradient_angle_threshold));
        }
        
        // stage 1. fill accumulator
        for(i = 0; i < height; i++) {
            int16_t *dx_row_ptr = (int16_t *)(dx_mat_ptr + i * dx_step);
            int16_t *dy_row_ptr = (int16_t *)(dy_mat_ptr + i * dy_step);
            for(j = 0; j < width; j++) {
                if(image[i * step + j] != 0) {
                    int16_t del_x = dx_row_ptr[j];
                    int16_t del_y = dy_row_ptr[j];
                    
                    bool use_pixel = false;
                    
                    if(cvm_likely(del_x != 0)) { // avoid div by 0
                        float slope = (float)del_y / (float)del_x;
                        if(vertical) {
                            if(slope >= slope_bound_a && slope <= slope_bound_b) {
                                use_pixel = true;
                            }
                        } else {
                            if(slope >= slope_bound_a || slope <= slope_bound_b) {
                                use_pixel = true;
                            }
                        }
                    } else {
                        use_pixel = !vertical;
                    }
                    
                    if(use_pixel) {
                        for(n = 0; n < numangle; n++) {
                            r = (j * tabCos[n] + i * tabSin[n]) >> FIXED_POINT_EXPONENT;
                            r += (numrho - 1) / 2;
                            accum[(n+1) * (numrho+2) + r+1]++;
                        }
                    }
                }
            }
        }
        
        // stage 2. find maximum
        // TODO: NEON implementation of max/argmax to use here
        int maxVal = 0;
        int maxBase = 0;
        for( r = 0; r < numrho; r++ ) {
            for( n = 0; n < numangle; n++ ) {
                int base = (n + 1) * (numrho + 2) + r + 1;
                int accumVal = accum[base];
                if(accumVal > maxVal) {
                    maxVal = accumVal;
                    maxBase = base;
                }
            }
        }
        
        
        // stage 3. if local maximum is above threshold, add it
        CvLinePolar line;
        line.rho = 0.0f;
        line.angle = 0.0f;
        line.is_null = true;
        
        if(maxVal > threshold) {
            scale = 1.0f / (numrho + 2);
            int idx = maxBase;
            int n = cvFloor(idx * scale) - 1;
            int r = idx - (n + 1) * (numrho + 2) - 1;
            line.rho = (r - (numrho - 1) * 0.5f) * rho;
            line.angle = n * theta + theta_min;
            line.is_null = false;
        }
        return line;
    }
    
#pragma mark -- Canny Method --
    
    void llcv_canny7_precomputed_sobel(IplImage *srcarr, IplImage *dstarr, IplImage *sobel_dx, IplImage *sobel_dy, double low_thresh, double high_thresh) {
        cv::AutoBuffer<char> buffer;
        std::vector<uchar*> stack;
        uchar **stack_top = 0, **stack_bottom = 0;
        
        CvMat srcstub, *src = cvGetMat( srcarr, &srcstub );
        CvMat dststub, *dst = cvGetMat( dstarr, &dststub );
        CvSize size;
        int low, high;
        int* mag_buf[3];
        uchar* map;
        ptrdiff_t mapstep;
        int maxsize;
        int i, j;
        CvMat mag_row;
        
        if( CV_MAT_TYPE( src->type ) != CV_8UC1 ||
           CV_MAT_TYPE( dst->type ) != CV_8UC1 )
            CV_Error( CV_StsUnsupportedFormat, "" );
        
        if( !CV_ARE_SIZES_EQ( src, dst ))
            CV_Error( CV_StsUnmatchedSizes, "" );
        
        if( low_thresh > high_thresh )
        {
            double t;
            CV_SWAP( low_thresh, high_thresh, t );
        }
        ///size = cvGetMatSize( src );
        size = cvGetSize( src );
        
        CvMat *dx;
        CvMat *dy;
        
        CvMat dx_stub, dy_stub;
        dx = cvGetMat(sobel_dx, &dx_stub);
        dy = cvGetMat(sobel_dy, &dy_stub);
        
        low = cvFloor(low_thresh);
        high = cvFloor(high_thresh);
        
        buffer.allocate( (size.width+2)*(size.height+2) + (size.width+2)*3*sizeof(int) );
        
        mag_buf[0] = (int*)(char*)buffer;
        mag_buf[1] = mag_buf[0] + size.width + 2;
        mag_buf[2] = mag_buf[1] + size.width + 2;
        map = (uchar*)(mag_buf[2] + size.width + 2);
        mapstep = size.width + 2;
        
        maxsize = MAX( 1 << 10, size.width*size.height/10 );
        stack.resize( maxsize );
        stack_top = stack_bottom = &stack[0];
        
        memset( mag_buf[0], 0, (size.width+2)*sizeof(int) );
        memset( map, 1, mapstep );
        memset( map + mapstep*(size.height + 1), 1, mapstep );
        
        /* sector numbers
         (Top-Left Origin)
         
         1   2   3
          *  *  *
           * * *
         0*******0
           * * *
          *  *  *
         3   2   1
         */
        
#define CANNY_PUSH(d)    *(d) = (uchar)2, *stack_top++ = (d)
#define CANNY_POP(d)     (d) = *--stack_top
        
        mag_row = cvMat( 1, size.width, CV_32F );
        
        // calculate magnitude and angle of gradient, perform non-maxima supression.
        // fill the map with one of the following values:
        //   0 - the pixel might belong to an edge
        //   1 - the pixel can not belong to an edge
        //   2 - the pixel does belong to an edge
        for( i = 0; i <= size.height; i++ )
        {
            int* _mag = mag_buf[(i > 0) + 1] + 1;
            const short* _dx = (short*)(dx->data.ptr + dx->step*i);
            const short* _dy = (short*)(dy->data.ptr + dy->step*i);
            uchar* _map;
            int64_t x, y;
            ptrdiff_t magstep1, magstep2;
            int prev_flag = 0;
            
            if( i < size.height )
            {
                _mag[-1] = _mag[size.width] = 0;
                
                /*
                 // TODO: Needs dmz neon protection
                 // TODO: Test and enable this code, if we can get enough other performance benefits from NEON elsewhere
                 // in this function to make it worth having a dedicated NEON or assembly version.
                 #define kVectorSize 8
                 uint16_t scalar_cols = size.width % kVectorSize;
                 uint16_t vector_chunks = size.width / kVectorSize;
                 uint16_t vector_cols = vector_chunks * kVectorSize;
                 
                 for(uint16_t vector_index = 0; vector_index < vector_chunks; vector_index++) {
                 uint16_t col_index = vector_index * kVectorSize;
                 
                 int16x8_t dx_q = vld1q_s16(_dx + col_index);
                 int16x8_t dy_q = vld1q_s16(_dy + col_index);
                 
                 int16x4_t dx_d0 = vget_low_s16(dx_q);
                 int16x4_t dx_d1 = vget_high_s16(dx_q);
                 int16x4_t dy_d0 = vget_low_s16(dy_q);
                 int16x4_t dy_d1 = vget_high_s16(dy_q);
                 
                 int32x4_t dx_wq0 = vmovl_s16(dx_d0);
                 int32x4_t dx_wq1 = vmovl_s16(dx_d1);
                 int32x4_t dy_wq0 = vmovl_s16(dy_d0);
                 int32x4_t dy_wq1 = vmovl_s16(dy_d1);
                 
                 int32x4_t abs_q0 = vaddq_s32(vabsq_s32(dx_wq0), vabsq_s32(dy_wq0));
                 int32x4_t abs_q1 = vaddq_s32(vabsq_s32(dx_wq1), vabsq_s32(dy_wq1));
                 
                 vst1q_s32(_mag + col_index, abs_q0);
                 vst1q_s32(_mag + col_index + (kVectorSize / 2), abs_q1);
                 }
                 
                 for(uint16_t scalar_index = 0; scalar_index < scalar_cols; scalar_index++) {
                 uint16_t col_index = scalar_index + vector_cols;
                 _mag[col_index] = abs(_dx[col_index]) + abs(_dy[col_index]);
                 }
                 #undef kVectorSize 8
                 */
                
                for( j = 0; j < size.width; j++ )
                    _mag[j] = abs(_dx[j]) + abs(_dy[j]);
            }
            else
                memset( _mag-1, 0, (size.width + 2)*sizeof(int) );
            
            // at the very beginning we do not have a complete ring
            // buffer of 3 magnitude rows for non-maxima suppression
            if( i == 0 )
                continue;
            
            _map = map + mapstep*i + 1;
            _map[-1] = _map[size.width] = 1;
            
            _mag = mag_buf[1] + 1; // take the central row
            _dx = (short*)(dx->data.ptr + dx->step*(i-1));
            _dy = (short*)(dy->data.ptr + dy->step*(i-1));
            
            magstep1 = mag_buf[2] - mag_buf[1];
            magstep2 = mag_buf[0] - mag_buf[1];
            
            if( (stack_top - stack_bottom) + size.width > maxsize )
            {
                int sz = (int)(stack_top - stack_bottom);
                maxsize = MAX( maxsize * 3/2, maxsize + 8 );
                stack.resize(maxsize);
                stack_bottom = &stack[0];
                stack_top = stack_bottom + sz;
            }
            
            for( j = 0; j < size.width; j++ )
            {
#define CANNY_SHIFT 15L
                // 0.4142135623730950488016887242097 == tan(22.5 degrees)
#define TG22  ((int)(0.4142135623730950488016887242097*(1<<CANNY_SHIFT) + 0.5))
                
                x = _dx[j];
                y = _dy[j];
                int s = (x ^ y) < 0 ? -1 : 1;
                int m = _mag[j];
                
                x = llabs(x);
                y = llabs(y);
                if( m > low )
                {
                    int64_t tg22x = x * TG22;
                    int64_t tg67x = tg22x + ((x + x) << CANNY_SHIFT);
                    
                    y <<= CANNY_SHIFT;
                    
                    if( y < tg22x )
                    {
                        if( m > _mag[j-1] && m >= _mag[j+1] )
                        {
                            if( m > high && !prev_flag && _map[j-mapstep] != 2 )
                            {
                                CANNY_PUSH( _map + j );
                                prev_flag = 1;
                            }
                            else
                                _map[j] = (uchar)0;
                            continue;
                        }
                    }
                    else if( y > tg67x )
                    {
                        if( m > _mag[j+magstep2] && m >= _mag[j+magstep1] )
                        {
                            if( m > high && !prev_flag && _map[j-mapstep] != 2 )
                            {
                                CANNY_PUSH( _map + j );
                                prev_flag = 1;
                            }
                            else
                                _map[j] = (uchar)0;
                            continue;
                        }
                    }
                    else
                    {
                        if( m > _mag[j+magstep2-s] && m > _mag[j+magstep1+s] )
                        {
                            if( m > high && !prev_flag && _map[j-mapstep] != 2 )
                            {
                                CANNY_PUSH( _map + j );
                                prev_flag = 1;
                            }
                            else
                                _map[j] = (uchar)0;
                            continue;
                        }
                    }
                }
                prev_flag = 0;
                _map[j] = (uchar)1;
            }
            
            // scroll the ring buffer
            _mag = mag_buf[0];
            mag_buf[0] = mag_buf[1];
            mag_buf[1] = mag_buf[2];
            mag_buf[2] = _mag;
        }
        
        // now track the edges (hysteresis thresholding)
        while( stack_top > stack_bottom )
        {
            uchar* m;
            if( (stack_top - stack_bottom) + 8 > maxsize )
            {
                int sz = (int)(stack_top - stack_bottom);
                maxsize = MAX( maxsize * 3/2, maxsize + 8 );
                stack.resize(maxsize);
                stack_bottom = &stack[0];
                stack_top = stack_bottom + sz;
            }
            
            CANNY_POP(m);
            
            if( !m[-1] )
                CANNY_PUSH( m - 1 );
            if( !m[1] )
                CANNY_PUSH( m + 1 );
            if( !m[-mapstep-1] )
                CANNY_PUSH( m - mapstep - 1 );
            if( !m[-mapstep] )
                CANNY_PUSH( m - mapstep );
            if( !m[-mapstep+1] )
                CANNY_PUSH( m - mapstep + 1 );
            if( !m[mapstep-1] )
                CANNY_PUSH( m + mapstep - 1 );
            if( !m[mapstep] )
                CANNY_PUSH( m + mapstep );
            if( !m[mapstep+1] )
                CANNY_PUSH( m + mapstep + 1 );
        }
        
        // the final pass, form the final image
        for( i = 0; i < size.height; i++ )
        {
            const uchar* _map = map + mapstep*(i+1) + 1;
            uchar* _dst = dst->data.ptr + dst->step*i;
            
            for( j = 0; j < size.width; j++ )
                _dst[j] = (uchar)-(_map[j] >> 1);
        }
    }
    
    double sum_abs_magnitude_c(IplImage *image) {
        IplImage *image_abs = cvCreateImage(cvGetSize(image), image->depth, image->nChannels);
        cvAbs(image, image_abs);
        CvScalar sum = cvSum(image_abs);
        cvReleaseImage(&image_abs);
        return sum.val[0];
    }
    
    double sum_abs_magnitude(IplImage *image) {
        return sum_abs_magnitude_c(image);
    }
    
    void llcv_adaptive_canny7_precomputed_sobel(IplImage *src, IplImage *dst, IplImage *dx, IplImage *dy) {
        CvSize src_size = cvGetSize(src);
        // We can use either sum_abs_magnitude (|dx| + |dy|) or sum_magnitude (sqrt(dx^2 + dy^2)) here. They yield
        // comparable results, and sum_abs_magnitude is marginally faster to compute, and can be made faster
        // still than our current implementation.
        double mean = (sum_abs_magnitude(dx) + sum_abs_magnitude(dy)) / (src_size.width * src_size.height);
        // double mean = sum_magnitude(dx, dy) / (src_size.width * src_size.height);
        
        double low_threshold = mean;
        double high_threshold = 3.0f * low_threshold;
        
        llcv_canny7_precomputed_sobel(src, dst, dx, dy, low_threshold, high_threshold);
    }
    
#pragma mark focus score
    
#pragma mark -- Slobe Method --
    
    void llcv_sobel7_c(IplImage *src, IplImage *dst, bool dx, bool dy) {
        cvSobel(src, dst, !!dx, !!dy, 7); // !! to ensure 0/1-ness of dx, dy
    }
    
    void llcv_sobel7(IplImage *src, IplImage *dst, IplImage *scratch, bool dx, bool dy) {
        assert(src != NULL);
        assert(dst != NULL);
        assert(dx ^ dy);
        assert(src->nChannels == 1);
        assert(src->depth == IPL_DEPTH_8U);
        assert(dst->nChannels == 1);
        assert(dst->depth == IPL_DEPTH_16S);
        
        llcv_sobel7_c(src, dst, dx, dy);
    }
    
    
    
    
#pragma mark llcv_sobel3_dx_dy_c_neon
    
    // For reference, the sobel3_dx_dy kernel is:
    //  1,  0, -1
    //  0,  0,  0
    // -1,  0,  1
    void llcv_sobel3_dx_dy_c_neon(IplImage *src, IplImage *dst) {
#define kSobel3VectorSize 8
        
        CvSize src_size = cvGetSize(src);
        assert(src_size.width > kSobel3VectorSize);
        
        uint8_t *src_data_origin = (uint8_t *)llcv_get_data_origin(src);
        uint16_t src_width_step = (uint16_t)src->widthStep;
        
        uint8_t *dst_data_origin = (uint8_t *)llcv_get_data_origin(dst);
        uint16_t dst_width_step = (uint16_t)dst->widthStep;
        
        uint16_t last_col_index = (uint16_t)(src_size.width - 1);
        //bool can_use_neon = dmz_has_neon_runtime();
        bool can_use_neon = false;
        
        for(uint16_t row_index = 0; row_index < src_size.height; row_index++) {
            uint16_t row1_index = row_index == 0 ? 0 : row_index - 1;
            uint16_t last_row = (uint16_t)(src_size.height - 1);
            uint16_t row2_index = row_index == last_row ? last_row : row_index + 1;
            
            const uint8_t *src_row1_origin = src_data_origin + row1_index * src_width_step;
            const uint8_t *src_row2_origin = src_data_origin + row2_index * src_width_step;
            
            int16_t *dst_row_origin = (int16_t *)(dst_data_origin + row_index * dst_width_step);
            
            uint16_t col_index = 0;
            while(col_index < src_size.width) {
                bool is_first_col = col_index == 0;
                bool is_last_col = col_index == last_col_index;
                bool can_process_next_chunk_as_vector = col_index + kSobel3VectorSize < last_col_index;
                if(is_first_col || is_last_col || !can_use_neon || !can_process_next_chunk_as_vector) {
                    // scalar step
                    int16_t sum;
                    if(cvm_unlikely(is_first_col)) {
                        sum = src_row1_origin[col_index] - src_row1_origin[col_index + 1] - src_row2_origin[col_index] + src_row2_origin[col_index + 1];
                    } else if(cvm_unlikely(is_last_col)) {
                        sum = src_row1_origin[col_index - 1] - src_row1_origin[col_index] - src_row2_origin[col_index - 1] + src_row2_origin[col_index];
                    } else {
                        sum = src_row1_origin[col_index - 1] - src_row1_origin[col_index + 1] - src_row2_origin[col_index - 1] + src_row2_origin[col_index + 1];
                    }
                    
                    // write result
                    dst_row_origin[col_index] = sum;
                    
                    col_index++;
                } else {
                    // vector step
#if DMZ_HAS_NEON_COMPILETIME
                    uint8x8_t tl = vld1_u8(src_row1_origin + col_index - 1);
                    uint8x8_t tr = vld1_u8(src_row1_origin + col_index + 1);
                    uint8x8_t bl = vld1_u8(src_row2_origin + col_index - 1);
                    uint8x8_t br = vld1_u8(src_row2_origin + col_index + 1);
                    int16x8_t tl_s16 = vreinterpretq_s16_u16(vmovl_u8(tl));
                    int16x8_t tr_s16 = vreinterpretq_s16_u16(vmovl_u8(tr));
                    int16x8_t bl_s16 = vreinterpretq_s16_u16(vmovl_u8(bl));
                    int16x8_t br_s16 = vreinterpretq_s16_u16(vmovl_u8(br));
                    int16x8_t sums = vaddq_s16(vsubq_s16(tl_s16, tr_s16), vsubq_s16(br_s16, bl_s16));
                    dst_row_origin[col_index + 0] = vgetq_lane_s16(sums, 0);
                    dst_row_origin[col_index + 1] = vgetq_lane_s16(sums, 1);
                    dst_row_origin[col_index + 2] = vgetq_lane_s16(sums, 2);
                    dst_row_origin[col_index + 3] = vgetq_lane_s16(sums, 3);
                    dst_row_origin[col_index + 4] = vgetq_lane_s16(sums, 4);
                    dst_row_origin[col_index + 5] = vgetq_lane_s16(sums, 5);
                    dst_row_origin[col_index + 6] = vgetq_lane_s16(sums, 6);
                    dst_row_origin[col_index + 7] = vgetq_lane_s16(sums, 7);
                    col_index += kSobel3VectorSize;
#endif
                }
            }
        }
        
#undef kSobel3VectorSize
    }
    
    void llcv_sobel3_dx_dy(IplImage *src, IplImage *dst) {
        assert(src->nChannels == 1);
        assert(src->depth == IPL_DEPTH_8U);
        
        assert(dst->nChannels == 1);
        assert(dst->depth == IPL_DEPTH_16S);
        
        CvSize src_size = cvGetSize(src);
        CvSize dst_size = cvGetSize(dst);
#pragma unused(src_size, dst_size) // work around broken compiler warnings
        
        assert(dst_size.width == src_size.width);
        assert(dst_size.height == src_size.height);
        
        llcv_sobel3_dx_dy_c_neon(src, dst);
    }
    
    // For reference, the scharr3_dx kernel is:
    //  -3,  0,  +3                                                    |  +3 |
    // -10,  0, +10  =  [-1, 0, +1] applied to each pixel, followed by | +10 |
    //  -3,  0,  +3                                                    |  +3 |
    //
    // Note that this function actually returns the ABSOLUTE VALUE of each Scharr score.
    void llcv_scharr3_dx_abs_c_neon(IplImage *src, IplImage *dst) {
#define kScharr3VectorSize 8
        
        CvSize src_size = cvGetSize(src);
        assert(src_size.width > kScharr3VectorSize);
        
        uint8_t *src_data_origin = (uint8_t *)llcv_get_data_origin(src);
        uint16_t src_width_step = (uint16_t)src->widthStep;
        
        uint8_t *dst_data_origin = (uint8_t *)llcv_get_data_origin(dst);
        uint16_t dst_width_step = (uint16_t)dst->widthStep;
#if DMZ_HAS_NEON_COMPILETIME
        uint16_t dst_width_step_in_int16s = (uint16_t)(dst->widthStep / 2);
#endif
        
        uint16_t last_col_index = (uint16_t)(src_size.width - 1);
        //bool can_use_neon = dmz_has_neon_runtime();
        bool can_use_neon = false;
        int16_t intermediate[src_size.width][src_size.height];  // note: intermediate[col][row]
        
        for(uint16_t row_index = 0; row_index < src_size.height; row_index++) {
            const uint8_t *src_row_origin = src_data_origin + row_index * src_width_step;
            uint16_t col_index = 0;
            while(col_index <= last_col_index) {
                uint16_t col_left_index = col_index == 0 ? 0 : col_index - 1;
                uint16_t col_right_index = col_index == last_col_index ? last_col_index : col_index + 1;
                bool can_process_next_chunk_as_vector = col_index + kScharr3VectorSize - 1 <= last_col_index;
                if (!can_use_neon || !can_process_next_chunk_as_vector) {
                    // scalar step
                    intermediate[col_index][row_index] = (int16_t)abs(src_row_origin[col_right_index] - src_row_origin[col_left_index]);
                    col_index++;
                }
                else {
                    // vector step
#if DMZ_HAS_NEON_COMPILETIME
                    uint8x8_t tl = vld1_u8(src_row_origin + col_left_index);
                    uint8x8_t tr = vld1_u8(src_row_origin + col_right_index);
                    int16x8_t tl_s16 = vreinterpretq_s16_u16(vmovl_u8(tl));
                    int16x8_t tr_s16 = vreinterpretq_s16_u16(vmovl_u8(tr));
                    int16x8_t sums = vabdq_s16(tr_s16, tl_s16);
                    intermediate[col_index + 0][row_index] = vgetq_lane_s16(sums, 0);
                    intermediate[col_index + 1][row_index] = vgetq_lane_s16(sums, 1);
                    intermediate[col_index + 2][row_index] = vgetq_lane_s16(sums, 2);
                    intermediate[col_index + 3][row_index] = vgetq_lane_s16(sums, 3);
                    intermediate[col_index + 4][row_index] = vgetq_lane_s16(sums, 4);
                    intermediate[col_index + 5][row_index] = vgetq_lane_s16(sums, 5);
                    intermediate[col_index + 6][row_index] = vgetq_lane_s16(sums, 6);
                    intermediate[col_index + 7][row_index] = vgetq_lane_s16(sums, 7);
                    col_index += kScharr3VectorSize;
#endif
                }
            }
        }
        
        uint16_t last_row_index = (uint16_t)(src_size.height - 1);
        
        for(uint16_t col_index = 0; col_index < src_size.width; col_index++) {
            uint16_t row_index = 0;
            while(row_index <= last_row_index) {
                int16_t *dst_row_origin = (int16_t *)(dst_data_origin + row_index * dst_width_step);
                uint16_t row_top_index = row_index == 0 ? 0 : row_index - 1;
                uint16_t row_bot_index = row_index == last_row_index ? last_row_index : row_index + 1;
                bool can_process_next_chunk_as_vector = row_index + kScharr3VectorSize - 1 <= last_row_index;
                if (!can_use_neon || !can_process_next_chunk_as_vector) {
                    // scalar step
                    dst_row_origin[col_index] = 3 * (intermediate[col_index][row_top_index] + intermediate[col_index][row_bot_index]) + 10 * intermediate[col_index][row_index];
                    row_index++;
                }
                else {
                    // vector step
#if DMZ_HAS_NEON_COMPILETIME
                    int16x8_t qt = vld1q_s16(intermediate[col_index] + row_top_index);
                    int16x8_t qm = vld1q_s16(intermediate[col_index] + row_index);
                    int16x8_t qb = vld1q_s16(intermediate[col_index] + row_bot_index);
                    int16x8_t sums = vaddq_s16(qt, qb);
                    sums = vmulq_n_s16(sums, 3);
                    sums = vmlaq_n_s16(sums, qm, 10);
                    dst_row_origin[col_index] = vgetq_lane_s16(sums, 0);
                    dst_row_origin += dst_width_step_in_int16s;
                    dst_row_origin[col_index] = vgetq_lane_s16(sums, 1);
                    dst_row_origin += dst_width_step_in_int16s;
                    dst_row_origin[col_index] = vgetq_lane_s16(sums, 2);
                    dst_row_origin += dst_width_step_in_int16s;
                    dst_row_origin[col_index] = vgetq_lane_s16(sums, 3);
                    dst_row_origin += dst_width_step_in_int16s;
                    dst_row_origin[col_index] = vgetq_lane_s16(sums, 4);
                    dst_row_origin += dst_width_step_in_int16s;
                    dst_row_origin[col_index] = vgetq_lane_s16(sums, 5);
                    dst_row_origin += dst_width_step_in_int16s;
                    dst_row_origin[col_index] = vgetq_lane_s16(sums, 6);
                    dst_row_origin += dst_width_step_in_int16s;
                    dst_row_origin[col_index] = vgetq_lane_s16(sums, 7);
                    row_index += kScharr3VectorSize;
#endif
                }
            }
        }
        
#undef kScharr3VectorSize
    }
    
    float llcv_stddev_of_abs_c(IplImage *image) {
        cvAbs(image, image);
        CvScalar stddev;
        cvAvgSdv(image, NULL, &stddev, NULL);
        return (float)stddev.val[0];
    }
    
    float llcv_stddev_of_abs(IplImage *image) {
        assert(image->depth == IPL_DEPTH_16S);
        assert(image->nChannels == 1);
        
        return llcv_stddev_of_abs_c(image);
    }
    
    float cvm_focus_score_for_image(IplImage *image) {
        assert(image->nChannels == 1);
        assert(image->depth == IPL_DEPTH_8U);
        
        CvSize image_size = cvGetSize(image);
        IplImage *sobel_image = cvCreateImage(image_size, IPL_DEPTH_16S, 1);
        
        llcv_sobel3_dx_dy(image, sobel_image);
        
        float stddev = llcv_stddev_of_abs(sobel_image);
        cvReleaseImage(&sobel_image);
        return stddev;
    }
    
    CvRect cvm_card_rect_for_screen(CvSize standardCardSize, CvSize standardScreenSize, CvSize actualScreenSize) {
        if (standardCardSize.width == 0 || standardCardSize.height == 0 ||
            standardScreenSize.width == 0 || standardScreenSize.height == 0 ||
            actualScreenSize.width == 0 || actualScreenSize.height == 0) {
            return cvRect(0, 0, 0, 0);
        }
        
        CvRect actualCardRect;
        
        if (actualScreenSize.width == standardScreenSize.width && actualScreenSize.height == standardScreenSize.height) {
            actualCardRect.width = standardCardSize.width;
            actualCardRect.height = standardCardSize.height;
        }
        else {
            float screenWidthRatio = ((float)actualScreenSize.width) / ((float)standardScreenSize.width);
            float screenHeightRatio = ((float)actualScreenSize.height) / ((float)standardScreenSize.height);
            float screenRatio = MIN(screenWidthRatio, screenHeightRatio);
            
            actualCardRect.width = (int)(standardCardSize.width * screenRatio);
            actualCardRect.height = (int)(standardCardSize.height * screenRatio);
        }
        
        actualCardRect.x = (actualScreenSize.width - actualCardRect.width) / 2;
        actualCardRect.y = (actualScreenSize.height - actualCardRect.height) / 2;
        
        return actualCardRect;
    }
    
    void cvm_set_roi_for_scoring(IplImage *image, bool use_full_image) {
        // Usually we calculate the focus score only on the center 1/9th of the credit card
        // in the image (assume it is centered), for performance reasons
        CvSize focus_size;
        if (use_full_image) {
            focus_size = cvSize(kCreditCardTargetWidth, kCreditCardTargetHeight);
        }
        else {
            focus_size = cvSize(kCreditCardTargetWidth / 3, kCreditCardTargetHeight / 3);
        }
        
        CvRect focus_rect = cvm_card_rect_for_screen(focus_size,
                                                     cvSize(kLandscapeSampleWidth, kLandscapeSampleHeight),
                                                     cvGetSize(image));
        
        cvSetImageROI(image, focus_rect);
    }
    
    float Cvm::cvm_focus_score(IplImage *image, bool use_full_image){
        cvm_set_roi_for_scoring(image, use_full_image);
        float focus_score = cvm_focus_score_for_image(image);
        cvResetImageROI(image);
        return focus_score;
    }
    
    float cvm_brightness_score_for_image(IplImage *image) {
        assert(image->nChannels == 1);
        assert(image->depth == IPL_DEPTH_8U);
        
        // could Neon and/or GPU this; however, this call to cvAvg apparently has NO effect on FPS (iPhone 4S)
        CvScalar mean = cvAvg(image, NULL);
        return (float)mean.val[0];
    }
    
    float Cvm::cvm_brightness_score(IplImage *image, bool use_full_image) {
        cvm_set_roi_for_scoring(image, use_full_image);
        float focus_score = cvm_brightness_score_for_image(image);
        cvResetImageROI(image);
        return focus_score;
    }
    
#pragma mark -- 
    
    void llcv_split_u8_c(IplImage *interleaved, IplImage *channel1, IplImage *channel2) {
        cvSplit(interleaved, channel1, channel2, NULL, NULL);
    }
    
    void llcv_split_u8(IplImage *interleaved, IplImage *channel1, IplImage *channel2) {
        assert(interleaved->nChannels == 2);
        assert(channel1->nChannels == 1);
        assert(channel2->nChannels == 1);
        
        assert(interleaved->depth == IPL_DEPTH_8U);
        assert(channel1->depth == IPL_DEPTH_8U);
        assert(channel2->depth == IPL_DEPTH_8U);
        
        llcv_split_u8_c(interleaved, channel1, channel2);
    }
    
#define MAX5(a, b, c, d, e) MAX(a, MAX(b, MAX(c, MAX(d, e))))
#define MIN5(a, b, c, d, e) MIN(a, MIN(b, MIN(c, MIN(d, e))))
    void llcv_morph_grad3_2d_cross_u8_c_neon(IplImage *src, IplImage *dst) {
#define kMorphGrad3Cross2DVectorSize 16
        
        CvSize src_size = cvGetSize(src);
        assert(src_size.width > kMorphGrad3Cross2DVectorSize);
        
        uint8_t *src_data_origin = (uint8_t *)llcv_get_data_origin(src);
        uint16_t src_width_step = (uint16_t)src->widthStep;
        
        uint8_t *dst_data_origin = (uint8_t *)llcv_get_data_origin(dst);
        uint16_t dst_width_step = (uint16_t)dst->widthStep;
        bool can_use_neon = false;
        
        for(uint16_t row_index = 0; row_index < src_size.height; row_index++) {
            uint16_t row1_index = row_index == 0 ? row_index : row_index - 1;
            uint16_t row2_index = row_index;
            uint16_t row3_index = row_index == src_size.height - 1 ? row_index : row_index + 1;
            
            const uint8_t *src_row1_origin = src_data_origin + row1_index * src_width_step;
            const uint8_t *src_row2_origin = src_data_origin + row2_index * src_width_step;
            const uint8_t *src_row3_origin = src_data_origin + row3_index * src_width_step;
            
            uint8_t *dst_row_origin = dst_data_origin + row_index * dst_width_step;
            
            uint16_t col_index = 0;
            while(col_index < src_size.width) {
                bool is_first_col = col_index == 0;
                uint16_t last_col_index = (uint16_t)(src_size.width - 1);
                bool is_last_col = col_index == last_col_index;
                bool can_process_next_chunk_as_vector = col_index + kMorphGrad3Cross2DVectorSize < last_col_index;
                if(is_first_col || is_last_col || !can_use_neon || !can_process_next_chunk_as_vector) {
                    // scalar step
                    uint16_t col1_index = is_first_col ? col_index : col_index - 1;
                    uint16_t col2_index = col_index;
                    uint16_t col3_index = is_last_col ? col_index : col_index + 1;
                    
                    uint8_t grad =
                    MAX5(src_row1_origin[col2_index], src_row2_origin[col1_index], src_row2_origin[col2_index], src_row2_origin[col3_index], src_row3_origin[col2_index]) -
                    MIN5(src_row1_origin[col2_index], src_row2_origin[col1_index], src_row2_origin[col2_index], src_row2_origin[col3_index], src_row3_origin[col2_index]);
                    
                    // write result
                    dst_row_origin[col_index] = grad;
                    
                    col_index++;
                } else {
                    // vector step
#if DMZ_HAS_NEON_COMPILETIME
                    // north, east, center, west, south
                    uint8x16_t n = vld1q_u8(src_row1_origin + col_index);
                    uint8x16_t w = vld1q_u8(src_row2_origin + col_index - 1);
                    uint8x16_t c = vld1q_u8(src_row2_origin + col_index);
                    uint8x16_t e = vld1q_u8(src_row2_origin + col_index + 1);
                    uint8x16_t s = vld1q_u8(src_row3_origin + col_index);
                    uint8x16_t max_vec = vmaxq_u8(n, vmaxq_u8(w, vmaxq_u8(c, vmaxq_u8(e, s))));
                    uint8x16_t min_vec = vminq_u8(n, vminq_u8(w, vminq_u8(c, vminq_u8(e, s))));
                    uint8x16_t grad_vec = vsubq_u8(max_vec, min_vec);
                    dst_row_origin[col_index +  0] = vgetq_lane_u8(grad_vec,  0);
                    dst_row_origin[col_index +  1] = vgetq_lane_u8(grad_vec,  1);
                    dst_row_origin[col_index +  2] = vgetq_lane_u8(grad_vec,  2);
                    dst_row_origin[col_index +  3] = vgetq_lane_u8(grad_vec,  3);
                    dst_row_origin[col_index +  4] = vgetq_lane_u8(grad_vec,  4);
                    dst_row_origin[col_index +  5] = vgetq_lane_u8(grad_vec,  5);
                    dst_row_origin[col_index +  6] = vgetq_lane_u8(grad_vec,  6);
                    dst_row_origin[col_index +  7] = vgetq_lane_u8(grad_vec,  7);
                    dst_row_origin[col_index +  8] = vgetq_lane_u8(grad_vec,  8);
                    dst_row_origin[col_index +  9] = vgetq_lane_u8(grad_vec,  9);
                    dst_row_origin[col_index + 10] = vgetq_lane_u8(grad_vec, 10);
                    dst_row_origin[col_index + 11] = vgetq_lane_u8(grad_vec, 11);
                    dst_row_origin[col_index + 12] = vgetq_lane_u8(grad_vec, 12);
                    dst_row_origin[col_index + 13] = vgetq_lane_u8(grad_vec, 13);
                    dst_row_origin[col_index + 14] = vgetq_lane_u8(grad_vec, 14);
                    dst_row_origin[col_index + 15] = vgetq_lane_u8(grad_vec, 15);
                    col_index += kMorphGrad3Cross2DVectorSize;
#endif
                }
            }
        }
#undef kMorphGrad3Cross2DVectorSize
    }
    
#define  CV_CAST_8U(t)  (uchar)(!((t) & ~255) ? (t) : (t) > 0 ? 255 : 0)
    // This implementation copied directly from OpenCV's cvEqualizeHist, as
    // part of an effort to remove dependencies on libopencv_imgproc.a.
    void llcv_equalize_hist(const IplImage *srcimg, IplImage *dstimg) {
        CvMat sstub, *src = cvGetMat(srcimg, &sstub);
        CvMat dstub, *dst = cvGetMat(dstimg, &dstub);
        
        CV_Assert( CV_ARE_SIZES_EQ(src, dst) && CV_ARE_TYPES_EQ(src, dst) &&
                  CV_MAT_TYPE(src->type) == CV_8UC1 );
        CvSize size = cvGetSize(src);
        if( CV_IS_MAT_CONT(src->type & dst->type) )
        {
            size.width *= size.height;
            size.height = 1;
        }
        int x, y;
        const int hist_sz = 256;
        int hist[hist_sz];
        memset(hist, 0, sizeof(hist));
        
        for( y = 0; y < size.height; y++ )
        {
            const uchar* sptr = src->data.ptr + src->step*y;
            for( x = 0; x < size.width; x++ )
                hist[sptr[x]]++;
        }
        
        float scale = 255.f/(size.width*size.height);
        int sum = 0;
        uchar lut[hist_sz+1];
        
        for( int i = 0; i < hist_sz; i++ )
        {
            sum += hist[i];
            int val = cvRound(sum*scale);
            lut[i] = CV_CAST_8U(val);
        }
        
        lut[0] = 0;
        for( y = 0; y < size.height; y++ )
        {
            const uchar* sptr = src->data.ptr + src->step*y;
            uchar* dptr = dst->data.ptr + dst->step*y;
            for( x = 0; x < size.width; x++ )
                dptr[x] = lut[sptr[x]];
        }
    }
    
    void Cvm::cvm_deinterleave_uint8_c2(IplImage *interleaved, IplImage **channel1, IplImage **channel2) {
        CvSize image_size = cvGetSize(interleaved);
        
        *channel1 = cvCreateImage(image_size, IPL_DEPTH_8U, 1);
        *channel2 = cvCreateImage(image_size, IPL_DEPTH_8U, 1);
        
        llcv_split_u8(interleaved, *channel1, *channel2);
    }
    
#pragma mark -- Wrapper --
    
#pragma mark -- Detect Edges --
    
    static inline CvRect cvInsetRect(CvRect originalRect, int horizontalInset, int verticalInset) {
        return cvRect(originalRect.x + horizontalInset,
                      originalRect.y + verticalInset,
                      originalRect.width - 2 * horizontalInset,
                      originalRect.height - 2 * verticalInset);
    }
    
#pragma mark: detection_boxes_for_sample
    DetectionBoxes detection_boxes_for_sample(IplImage *sample, FrameOrientation orientation) {
        CvSize size = cvGetSize(sample);
        //dmz_trace_log("detection_boxes_for_sample sized %ix%i with orientation:%i", size.width, size.height, orientation);
        int absolute_inset_vert, absolute_slop_vert, absolute_inset_horiz, absolute_slop_horiz;
        
        // Regardless of the dimensions of the incoming image (640x480, 1280x720, etc),
        // we do everything based on the central 4:3 rectangle (which for 640x480 is the entire image).
        int width = (size.height * 4) / 3;
        int leftMargin = (size.width - width) / 2;
        size.width = width;
        
        switch(orientation) {
            case FrameOrientationPortrait:
                /* no break */
            case FrameOrientationPortraitUpsideDown:
                absolute_inset_vert = (int)roundf(kPortraitHorizontalPercentInset * size.height);
                absolute_slop_vert = (int)roundf(kHorizontalPercentSlop * size.height);
                absolute_inset_horiz = (int)roundf(kPortraitVerticalPercentInset * size.width);
                absolute_slop_horiz = (int)roundf(kVerticalPercentSlop * size.width);
                break;
            case FrameOrientationLandscapeLeft:
                /* no break */
            case FrameOrientationLandscapeRight:
                absolute_inset_vert = (int)roundf(kLandscapeVerticalPercentInset * size.height);
                absolute_slop_vert = (int)roundf(kHorizontalPercentSlop * size.height);
                absolute_inset_horiz = (int)roundf(kLandscapeHorizontalPercentInset * size.width);
                absolute_slop_horiz = (int)roundf(kVerticalPercentSlop * size.width);
                break;
            default:
                absolute_inset_vert = 0;
                absolute_slop_vert = 0;
                absolute_inset_horiz = 0;
                absolute_slop_horiz = 0;
                break;
        }
        
        CvRect image_rect = cvRect(leftMargin, 0, size.width - 1, size.height - 1);
        CvRect outerSlopRect = cvInsetRect(image_rect,
                                           absolute_inset_horiz - absolute_slop_horiz,
                                           absolute_inset_vert - absolute_slop_vert);
        CvRect innerSlopRect = cvInsetRect(image_rect,
                                           absolute_inset_horiz + absolute_slop_horiz,
                                           absolute_inset_vert + absolute_slop_vert);
        
        DetectionBoxes boxes;
        
        boxes.top = cvRect(innerSlopRect.x, outerSlopRect.y,
                           innerSlopRect.width, 2 * absolute_slop_vert);
        //dmz_trace_log("boxes.top: {x:%i y:%i w:%i h:%i}", boxes.top.x, boxes.top.y, boxes.top.width, boxes.top.height);
        boxes.bottom = cvRect(innerSlopRect.x, innerSlopRect.y + innerSlopRect.height,
                              innerSlopRect.width, 2 * absolute_slop_vert);
        //dmz_trace_log("boxes.bottom: {x:%i y:%i w:%i h:%i}", boxes.bottom.x, boxes.bottom.y, boxes.bottom.width, boxes.bottom.height);
        
        boxes.left = cvRect(outerSlopRect.x, innerSlopRect.y,
                            2 * absolute_slop_horiz, innerSlopRect.height);
        //dmz_trace_log("boxes.left: {x:%i y:%i w:%i h:%i}", boxes.left.x, boxes.left.y, boxes.left.width, boxes.left.height);
        
        boxes.right = cvRect(innerSlopRect.x + innerSlopRect.width, innerSlopRect.y,
                             2 * absolute_slop_horiz, innerSlopRect.height);
        //dmz_trace_log("boxes.right: {x:%i y:%i w:%i h:%i}", boxes.right.x, boxes.right.y, boxes.right.width, boxes.right.height);
        
        return boxes;
    }
    
    bool is_parametric_line_none(ParametricLine line_to_test) {
        return ((line_to_test).theta == FLT_MAX);
    }
    
    bool parametricIntersect(ParametricLine line1, ParametricLine line2, float *x, float *y) {
        if(is_parametric_line_none(line1) || is_parametric_line_none(line2)) {
            return false;
        }
        
        Eigen::Matrix2f t;
        Eigen::Vector2f r;
        t << cosf(line1.theta), sinf(line1.theta), cosf(line2.theta), sinf(line2.theta);
        r << line1.rho, line2.rho;
        
        if(t.determinant() < 1e-10) {
            return false;
        }
        
        Eigen::Vector2f intersection = t.inverse() * r;
        *x = intersection(0);
        *y = intersection(1);
        return true;
    }
    
    static inline ParametricLine ParametricLineNone() {
        ParametricLine l;
        l.rho = FLT_MAX;
        l.theta = FLT_MAX;
        return l;
    }
    
    ParametricLine lineByShiftingOrigin(ParametricLine oldLine, int xOffset, int yOffset) {
        ParametricLine newLine;
        newLine.theta = oldLine.theta;
        double offsetAngle = xOffset == 0 ? CV_PI / 2.0f : atan((float)yOffset / (float)xOffset);
        double deltaAngle = oldLine.theta - offsetAngle + CV_PI / 2.0f; // because we're working with the line *normal* to theta
        double offsetMagnitude = sqrt(xOffset * xOffset + yOffset * yOffset);
        double delta_rho = offsetMagnitude * cos(CV_PI / 2 - deltaAngle);
        newLine.rho = (float)(oldLine.rho + delta_rho);
        return newLine;
    }
    
#pragma mark: best_line_for_sample
    ParametricLine best_line_for_sample(IplImage *image, LineOrientation expectedOrientation) {
        bool expected_vertical = expectedOrientation == LineOrientationVertical;
        
        // Calculate dx and dy derivatives; they'll be reused a lot throughout
        CvSize image_size = cvGetSize(image);
        assert(image_size.width > 0 && image_size.height > 0);
        //dmz_trace_log("looking for best line in %ix%i patch with orientation:%i", image_size.width, image_size.height, expectedOrientation);
        IplImage *sobel_scratch = cvCreateImage(cvSize(image_size.height, image_size.width), IPL_DEPTH_16S, 1);
        assert(sobel_scratch != NULL);
        IplImage *dx = cvCreateImage(image_size, IPL_DEPTH_16S, 1);
        assert(dx != NULL);
        IplImage *dy = cvCreateImage(image_size, IPL_DEPTH_16S, 1);
        assert(dy != NULL);
        llcv_sobel7(image, dx, sobel_scratch, 1, 0);
        llcv_sobel7(image, dy, sobel_scratch, 0, 1);
        cvReleaseImage(&sobel_scratch);
        
        // Calculate the canny image
        IplImage *canny_image = cvCreateImage(image_size, IPL_DEPTH_8U, 1);
        llcv_adaptive_canny7_precomputed_sobel(image, canny_image, dx, dy);
        
        // Calculate the hough transform, throwing away edge components with the wrong gradient angles
        int hough_accumulator_threshold = MAX(image_size.width, image_size.height) / kHoughThresholdLengthDivisor;
        float base_angle = expected_vertical ? kVerticalAngle : kHorizontalAngle;
        float theta_min = base_angle - kMaxAngleDeviationAllowed;
        float theta_max = base_angle + kMaxAngleDeviationAllowed;
        
        CvLinePolar best_line = llcv_hough(canny_image,
                                           dx, dy,
                                           1, // rho resolution
                                           (float)CV_PI / 180.0f, // theta resolution
                                           hough_accumulator_threshold,
                                           theta_min,
                                           theta_max,
                                           expected_vertical,
                                           kHoughGradientAngleThreshold);
        
        ParametricLine ret = ParametricLineNone();
        if(!best_line.is_null) {
            ret.rho = best_line.rho;
            ret.theta = best_line.angle;
        }
        
        cvReleaseImage(&dx);
        cvReleaseImage(&dy);
        cvReleaseImage(&canny_image);
        return ret;
    }
    
#define kNumColorPlanes 3
    
#pragma mark: find_line_in_detection_rects
    void find_line_in_detection_rects(IplImage **samples, float *rho_multiplier, CvRect *detection_rects, cvm_found_edge *found_edge, LineOrientation line_orientation) {
        assert(detection_rects != NULL);
        assert(found_edge != NULL);
        assert(samples != NULL);
        //dmz_trace_log("inputs to find_line_in_detection_rects are valid");
        for(int i = 0; i < kNumColorPlanes && !found_edge->found; i++) {
            IplImage *image = samples[i];
            assert(image != NULL);
#if DMZ_TRACE
            CvSize imageSize = cvGetSize(image);
            dmz_trace_log("sample %i has size %ix%i", i, imageSize.width, imageSize.height);
            CvRect r = detection_rects[i];
            dmz_trace_log("detection_rect {x:%i y:%i w:%i h:%i}", r.x, r.y, r.width, r.height);
#endif
            cvSetImageROI(image, detection_rects[i]);
            ParametricLine local_edge = best_line_for_sample(image, line_orientation);
            //dmz_trace_log("local_edge - {rho:%f theta:%f}", local_edge.rho, local_edge.theta);
            cvResetImageROI(image);
            found_edge->location = lineByShiftingOrigin(local_edge, detection_rects[i].x, detection_rects[i].y);
            found_edge->location.rho *= rho_multiplier[i];
            found_edge->found = !is_parametric_line_none(found_edge->location);
        }
        //dmz_trace_log("resulting edge - {found:%i ...}", found_edge->found);
    }
    
#pragma mark: dmz_found_all_edges
    bool dmz_found_all_edges(cvm_edges found_edges) {
        return (found_edges.top.found && found_edges.bottom.found && found_edges.left.found && found_edges.right.found);
    }
    
    bool Cvm::cvm_detect_edges(IplImage *y_sample, IplImage *cb_sample, IplImage *cr_sample,FrameOrientation orientation, cvm_edges *found_edges, cvm_corner_points *corner_points) {
        assert(y_sample != NULL);
        assert(cb_sample != NULL);
        assert(cr_sample != NULL);
        assert(found_edges != NULL);
        assert(corner_points != NULL);
        
        //dmz_trace_log("dmz_detect_edges");
        
        IplImage *samples[kNumColorPlanes] = {y_sample, cb_sample, cr_sample};
        DetectionBoxes boxes[kNumColorPlanes];
        float rho_multiplier[kNumColorPlanes] = {1.0f, 2.0f, 2.0f}; // cb and cr are half the size of Y
        
        for(int i = 0; i < kNumColorPlanes; i++) {
            boxes[i] = detection_boxes_for_sample(samples[i], orientation);
        }
        
        //dmz_trace_log("got boxes, looking for lines...");
        
        found_edges->top.found = 0;
        found_edges->bottom.found = 0;
        found_edges->left.found = 0;
        found_edges->right.found = 0;
        
        CvRect detection_rects[kNumColorPlanes];
        
        for(uint8_t i = 0; i < kNumColorPlanes; i++) {
            detection_rects[i] = boxes[i].top;
        }
        find_line_in_detection_rects(samples, rho_multiplier, detection_rects, &found_edges->top, LineOrientationHorizontal);
        //dmz_trace_log("dmz top edge? %i", found_edges->top.found);
        
        for(uint8_t i = 0; i < kNumColorPlanes; i++) {
            detection_rects[i] = boxes[i].bottom;
        }
        find_line_in_detection_rects(samples, rho_multiplier, detection_rects, &found_edges->bottom, LineOrientationHorizontal);
        //dmz_trace_log("dmz bottom edge? %i", found_edges->bottom.found);
        
        for(uint8_t i = 0; i < kNumColorPlanes; i++) {
            detection_rects[i] = boxes[i].left;
        }
        find_line_in_detection_rects(samples, rho_multiplier, detection_rects, &found_edges->left, LineOrientationVertical);
        //dmz_trace_log("dmz left edge? %i", found_edges->left.found);
        
        for(uint8_t i = 0; i < kNumColorPlanes; i++) {
            detection_rects[i] = boxes[i].right;
        }
        find_line_in_detection_rects(samples, rho_multiplier, detection_rects, &found_edges->right, LineOrientationVertical);
        //dmz_trace_log("dmz right edge? %i", found_edges->right.found);
        
        // Find corner intersections
        bool found_all_corners = true;
        if(dmz_found_all_edges(*found_edges)) {
            bool tl_intersects = parametricIntersect(found_edges->top.location, found_edges->left.location, &corner_points->top_left.x, &corner_points->top_left.y);
            bool bl_intersects = parametricIntersect(found_edges->bottom.location, found_edges->left.location, &corner_points->bottom_left.x, &corner_points->bottom_left.y);
            bool tr_intersects = parametricIntersect(found_edges->top.location, found_edges->right.location, &corner_points->top_right.x, &corner_points->top_right.y);
            bool br_intersects = parametricIntersect(found_edges->bottom.location, found_edges->right.location, &corner_points->bottom_right.x, &corner_points->bottom_right.y);
            int all_intersect = tl_intersects && bl_intersects && tr_intersects && br_intersects;
            if(!all_intersect) {
                // never seen this happen, but best to be safe
                found_all_corners = false;
            }
        } else {
            found_all_corners = false;
        }
        
        return found_all_corners;
    }
    
#pragma mark transform
    
    bool llcv_warp_auto_upsamples() {
        return true;
    }
    
    void Cvm::cvm_transform_card( IplImage *sample, cvm_corner_points corner_points, FrameOrientation orientation, bool upsample, IplImage **transformed) {
        
        cvm_point src_points[4];
        switch(orientation) {
            case FrameOrientationPortrait:
                src_points[0] = corner_points.bottom_left;
                src_points[1] = corner_points.top_left;
                src_points[2] = corner_points.bottom_right;
                src_points[3] = corner_points.top_right;
                break;
            case FrameOrientationLandscapeLeft:
                src_points[0] = corner_points.bottom_right;
                src_points[1] = corner_points.bottom_left;
                src_points[2] = corner_points.top_right;
                src_points[3] = corner_points.top_left;
                break;
            case FrameOrientationLandscapeRight: // this is the canonical one
                src_points[0] = corner_points.top_left;
                src_points[1] = corner_points.top_right;
                src_points[2] = corner_points.bottom_left;
                src_points[3] = corner_points.bottom_right;
                break;
            case FrameOrientationPortraitUpsideDown:
                src_points[0] = corner_points.top_right;
                src_points[1] = corner_points.bottom_right;
                src_points[2] = corner_points.top_left;
                src_points[3] = corner_points.bottom_left;
                break;
        }
        
        if(upsample) {
            if(!llcv_warp_auto_upsamples()) {
                // upsample source_points, since CbCr are half size.
                for(unsigned int i = 0; i < sizeof(cvm_corner_points) / sizeof(cvm_point); i++) {
                    src_points[i].x /= 2.0f;
                    src_points[i].y /= 2.0f;
                }
            }
        }
        
        // Destination rectangle is the same as the size of the image
        cvm_rect dst_rect = cvm_create_rect(0, 0, kCreditCardTargetWidth - 1, kCreditCardTargetHeight - 1);
        
        int nChannels = sample->nChannels;
        
        // Some environments (Android) may prefer to dictate where the result image is stored.
        if (*transformed == NULL) {
            *transformed = cvCreateImage(cvSize(kCreditCardTargetWidth, kCreditCardTargetHeight), sample->depth, nChannels);
        }
        ///仿射变换
        /**
         * sample:       src IplImage
         * src_points:   src points
         * dst_rect:     dst rect size
         * transformed:  dst IplImage
         */
        llcv_unwarp(sample, src_points, dst_rect, *transformed);
        
        /*自己尝试实现仿射变换
        int num=4;int scale = 2;
        CvPoint *validAreaPoints=new CvPoint[num];
        //CvPoint *innerRectPoints = new CvPoint[num];
        for(int j=0;j<num;j++)
        {
            validAreaPoints[j].x=src_points[j].x*scale;
            validAreaPoints[j].y=src_points[j].y*scale;
        }
        //sortPoints( validAreaPoints);
        
        validAreaPoints[0].x = validAreaPoints[0].x*1.05;
        validAreaPoints[0].y = validAreaPoints[0].y*1.05;
        validAreaPoints[1].x = validAreaPoints[1].x*1.05;
        validAreaPoints[1].y = validAreaPoints[1].y*0.98;;
        validAreaPoints[2].x = validAreaPoints[2].x*0.95;
        validAreaPoints[2].y = validAreaPoints[2].y*0.98;
        validAreaPoints[3].x = validAreaPoints[3].x*0.95;
        validAreaPoints[3].y = validAreaPoints[3].y*1.05;
        
        IplImage* mask = cvCreateImage(cvGetSize(sample),IPL_DEPTH_8U,1);
        cvSetZero(mask);//mask多边形区域填充为以为图像相乘做准备
        cvFillPoly(mask,&validAreaPoints,&num,1,CV_RGB(1,1,1));
        IplImage* maskimg = cvCreateImage( cvGetSize(sample), sample->depth, sample->nChannels );
        cvSetZero(maskimg);
        cvCopy(sample, maskimg, mask); //ResetImageROI must be done on timg;
        
        CvPoint2D32f srcTri[4],dstTri[4];
        for (int i = 0; i < 4; i++)
        {
            dstTri[i].x = validAreaPoints[i].x;
            dstTri[i].y = validAreaPoints[i].y;
        }
        
        CvSize rsz;
        rsz.width=856*2;
        rsz.height=540*2;
        srcTri[0].x = 0;
        srcTri[0].y = 0;
        srcTri[1].x = 0;  //缩小一个像素
        srcTri[1].y = rsz.height;
        srcTri[2].x = rsz.width;  //bot right
        srcTri[2].y = rsz.height;
        srcTri[3].x = rsz.width;
        srcTri[3].y = 0;
        
        IplImage *resultImage = cvCreateImage( rsz, sample->depth, sample->nChannels );
        cvSetZero(resultImage);
        CvMat* warp_mat = cvCreateMat( 3, 3, CV_32FC1 );
        resultImage ->origin = maskimg ->origin;
        cvGetPerspectiveTransform(  dstTri,srcTri, warp_mat );  //由三对点计算仿射变换
        cvWarpPerspective( maskimg,resultImage,warp_mat,CV_INTER_LINEAR,cvScalarAll(0) );  //对图像做仿射变换
        cvReleaseMat( &warp_mat );
        if (*transformed == NULL) {
            *transformed = resultImage;
        }
        
        delete []validAreaPoints;
        cvReleaseImage( &maskimg);
        cvReleaseImage( &mask);
        //*/
    }
    
#pragma mark -- Segement Image --
    
    void GetImagePixel(IplImage * TargetImage,int x,int y,float& pixelvalue,int channel){
        if (channel<0 || channel>=TargetImage->nChannels ||y<0 || y>=TargetImage->height ||x<0 || x>=TargetImage->width){
            return ;
        }
        pixelvalue=((TargetImage->imageData + TargetImage->widthStep*y))[x*TargetImage->nChannels+channel];
        
        if (pixelvalue<0){
            pixelvalue+=256;
        }
        
        return ;
    }
    
    
    void SetImagePixel(IplImage * TargetImage,int x,int y,float pixelvalue,int channel) {
        
        if (channel<0 || channel>=TargetImage->nChannels ||y<0 || y>=TargetImage->height ||x<0 || x>=TargetImage->width){
            return ;
        }
        
        ((TargetImage->imageData + TargetImage->widthStep*y))[x*TargetImage->nChannels+channel]=pixelvalue;
        return ;
    }
    
    
    void RunEqualizeHistogram(IplImage* pInputImage,IplImage*pOutputImage) {
        if (!pInputImage||!pOutputImage) {
            return ;
        }
        int i=0;
        IplImage *pImageChannel[4] = { 0, 0, 0, 0 };
        for( i = 0; i < pInputImage->nChannels; i++ ){
            pImageChannel[i] = cvCreateImage( cvGetSize(pInputImage), pInputImage->depth, 1 );
        }
        
        // separate each channel
        cvSplit( pInputImage, pImageChannel[0], pImageChannel[1], pImageChannel[2], pImageChannel[3] );
        
        for( i = 0; i < pInputImage->nChannels; i++ ){
            // histogram equalization
            cvEqualizeHist( pImageChannel[i], pImageChannel[i] );
        }
        // integer each channel
        cvMerge( pImageChannel[0], pImageChannel[1], pImageChannel[2], pImageChannel[3], pOutputImage);
        
        for( i = 0; i < pInputImage->nChannels; i++ ){
            if ( pImageChannel[i] ){
                cvReleaseImage( &pImageChannel[i] );
                pImageChannel[i] = 0;
            }
        }
        
        return ;
        
    }
    
    //该函数是从初始设定的数字ROI中，提取出精确的数字区域上边界
    int  ExtractNumbers(IplImage *pNumberROI) {
        
        // 从ROI图像最底部进行投影，从而精确确定数字ROI区域
        int nRealTop=0;// 用来保存图像数字的真实高度
        bool bLine=false;
        float nRowSum;
        int nIter=0;   //表示发现的线行数
        for (int j=pNumberROI->height;j<0;j--){
            nRowSum=0;
            // 取图像上的点进行X方向的投影（投影结果进行了归一化处理）
            for (int i=0;i<pNumberROI->width;i++){
                float fPixelVal=0;
                GetImagePixel(pNumberROI,i,j,fPixelVal,0);
                nRowSum+=(fPixelVal/255);
                
            }
            //投影结束
            //// 找到了一行文字，默认为一行文字至少要有6个前景点
            if (nRowSum<pNumberROI->width-5&&bLine==false){
                nIter++;
                bLine=true;
            }
             //找到了空白行，将表示是否找到文字的flag设为0，，默认为一行文字至少要有6个前景点
            if (nRowSum>pNumberROI->width-5&&bLine){
                bLine=false;
                //已经找到数字的上边缘，记录此时行坐标,RealTop 在真实上标界基础上增加10 pix
                if (nIter==1){
                    nRealTop=fmax(j-10,0);
                }
            }
        }
        
        return nRealTop;
    }
    
    void CloneImage(IplImage *psrc,IplImage *pdst,int nLeft, int nTop) {
        
        if (!psrc||!pdst){
            return ;
        }
        if ((pdst->height+nTop)>psrc->height||(pdst->width+nLeft)>psrc->width||pdst->depth>psrc->depth||pdst->nChannels>psrc->nChannels||pdst->nSize>psrc->nSize){
            return ;
        }
        if (nLeft<0||nLeft>=psrc->width){
            return ;
        }
        if (nTop<0||nTop>=psrc->height){
            return ;
        }
        if (psrc->nChannels!=pdst->nChannels||psrc->depth!=pdst->depth){
            return ;
        }
        
        IplImage * ptempImage=cvCreateImage(cvGetSize(psrc),psrc->depth,psrc->nChannels);
        
        int i, j, k;
        int height = psrc->height;
        int width = psrc->width;
        int channels = psrc->nChannels;
        int step = psrc->widthStep;
        uchar *psrcimageData = (uchar *)psrc->imageData;
        uchar *pdstimageData = (uchar *)ptempImage->imageData;
        for(i = 0; i < height; i++)
        {
            for(j = 0; j < width; j++)
            {
                for(k = 0; k < channels; k++)
                {
                    pdstimageData[i*step + j*channels + k]=psrcimageData[i*step + j*channels + k];
                }
            }
        }
        
        cvZero( pdst );
        cvSetImageROI( ptempImage, cvRect(nLeft,nTop,pdst->width,pdst->height));
        cvAdd( ptempImage, pdst, pdst, NULL );
        cvReleaseImage(&ptempImage);
        
        return ;
    }
    
    void SubBackGround(IplImage *pBinaryImage){
        
        if (!pBinaryImage){
            return ;
        }
        int nchannel=pBinaryImage->nChannels;
        int nWidth=pBinaryImage->width;
        int nHeight=pBinaryImage->height;
        if (nchannel!=1)
        {
            
            return ;
        }
        int nSeedX=1;
        int nSeedY=1;
        
        
        // 8-neighborhood directions
        int nDx[]={-1,0,1,-1,1,-1,0,1};
        int nDy[]={-1,-1,-1,0,0,1,1,1};
        // 8-neighborhood directions control flag
        int k =0;
        
        
        // define stack for storing the region coordinate and process flag
        int * pnGrowQueX ;
        int * pnGrowQueY ;
        int * pnProcessFlag;
        pnGrowQueX = new int [nWidth*nHeight];
        pnGrowQueY = new int [nWidth*nHeight];
        pnProcessFlag=new int[nWidth*nHeight];
        
        for (int i=0;i<nWidth*nHeight;i++)
        {
            pnProcessFlag[i]=0;
        }
        
        // define the start flag and end flag for the region stack
        // if nStart>nEnd, represent region is empty
        // if nStart=nEnd, represent only one point in the stack
        int nStart;
        int nEnd ;
        nStart = 0 ;
        nEnd = 0 ;
        pnGrowQueX[nEnd] = nSeedX;
        pnGrowQueY[nEnd] = nSeedY;
        
        
        // current processed pixel
        int nCurrX =nSeedX;
        int nCurrY =nSeedY;
        
        // represent one of current pixel's 8-neighborhood
        int xx=0;
        int yy=0;
        
        
        while (nStart<=nEnd)
        {
            
            
            nCurrX = pnGrowQueX[nStart];
            nCurrY = pnGrowQueY[nStart];
            
            float nPixelValue=0;
            
            GetImagePixel(pBinaryImage,nCurrX,nCurrY,nPixelValue,0);
            
            
            // check the current pixel's 8-neighborhood
            for (k=0; k<8; k++)
            {
                
                xx = nCurrX+nDx[k] ;
                yy = nCurrY+nDy[k];
                float nComparedPixelValue=0;
                GetImagePixel(pBinaryImage,xx,yy,nComparedPixelValue,0);
                if ( (xx < nWidth) && (xx>=0) && (yy>=0) && (yy<nHeight) && pnProcessFlag[yy*nWidth+xx]==0)
                {
                    //if the pixel is in image
                    //if the pixel is processed
                    //if the pixel satisfy region growing condition
                    
                    if (abs(nPixelValue-nComparedPixelValue)<10)
                    {
                        nEnd++;
                        // push (xx，yy) in stack
                        pnGrowQueX[nEnd] = xx;
                        pnGrowQueY[nEnd] = yy;
                        pnProcessFlag[yy*nWidth+xx] = 1;
                        SetImagePixel(pBinaryImage,xx,yy,255,0);
                        
                    }
                    
                }
                
            }
            nStart++;
        }
        
        delete []pnGrowQueX;
        delete []pnGrowQueY;
        delete []pnProcessFlag;
        
        return ;
    }
    
    IplImage *subBack(IplImage *src){
        //2.估计图像背景
        IplImage*tmp = cvCreateImage( cvGetSize(src), src->depth, src->nChannels);
        IplImage*src_back = cvCreateImage( cvGetSize(src), src->depth, src->nChannels);
        //创建结构元素
        IplConvKernel*element = cvCreateStructuringElementEx( 4, 4, 1, 1, CV_SHAPE_ELLIPSE, 0);
        //用该结构对源图象进行数学形态学的开操作后，估计背景亮度
        cvErode( src, tmp, element, 10);
        cvDilate( tmp, src_back, element, 10);
        cvReleaseImage(&tmp);
        cvReleaseStructuringElement(&element);
        
        //3.从源图象中减区背景图像
        IplImage*dst_gray = cvCreateImage( cvGetSize(src), src->depth, src->nChannels);
        cvSub( src, src_back, dst_gray, 0);
        cvReleaseImage(&src_back);
        //4.使用阈值操作将图像转换为二值图像
        //cvThreshold( dst_gray, dst_gray ,200, 255, CV_THRESH_BINARY ); //取阈值为50把图像转为二值图像
        
        return dst_gray;
    }
    
    IplImage *Cvm::segementImg(IplImage *src, CardType type, float x, float y, float w,float h, float t) {
        
        // 从模版图像上分割身份证号码，并储存在pNumberROI
        IplImage* pNumberROI=cvCreateImage(cvSize(src->width*w,src->height*h),src->depth,src->nChannels);
        CloneImage(src,pNumberROI,x*src->width,y*src->height);
        ///查看返回的是否是号码区域 结果:正确
        //return pNumberROI;
        
        if (type == CardTypeID) {
            
            ///处理身份证
            
            //精确定位数字ROI上边界
            int nTop=ExtractNumbers(pNumberROI);
            if (nTop>0){
                printf("精确定位上边距:%d 重新调整图像\n",nTop);
                IplImage *pTemp=cvCreateImage(cvSize(pNumberROI->width,pNumberROI->height-nTop),pNumberROI->depth,pNumberROI->nChannels);
                CloneImage(pNumberROI,pTemp,0,nTop);
                cvReleaseImage(&pNumberROI);
                pNumberROI=cvCreateImage(cvGetSize(pTemp),pTemp->depth,pTemp->nChannels);
                CloneImage(pTemp,pNumberROI,0,0);
                cvReleaseImage(&pTemp);
            }
            //分割完毕
            
            ///
            IplImage* pBinaryImage=NULL;
            //cvReleaseImage(&pBinaryImage);
            pBinaryImage=cvCreateImage(cvGetSize(pNumberROI),pNumberROI->depth,1);
            RunEqualizeHistogram(pNumberROI,pNumberROI);
            ///查看直方图 结果：正确
            //return pNumberROI;
            
            // set general threshold on image
            for (int i=0;i<pNumberROI->width;i++)
            {
                for (int j=0;j<pNumberROI->height;j++)
                {
                    float fPixR=0.0;
                    float fPixG=0.0;
                    float fPixB=0.0;
                    GetImagePixel(pNumberROI,i,j,fPixR,0);
                    GetImagePixel(pNumberROI,i,j,fPixG,1);
                    GetImagePixel(pNumberROI,i,j,fPixB,2);
                    if (fPixR<t&&fPixG<t&&fPixB<t)
                    {
                        SetImagePixel(pBinaryImage,i,j,0,0);
                    }
                    else
                    {
                        SetImagePixel(pBinaryImage,i,j,255,0);
                    }
                }
            }
            
            ///查看阈值化图像 结果：正确
            //return pBinaryImage;
            
            //滤波处理 简单、高斯、中值滤波
            cvSmooth(pBinaryImage,pBinaryImage,CV_GAUSSIAN,3,pBinaryImage->nChannels,0,0);
            //局部去噪，保证能够精确获得每个字符位置
            //cvDilate(pBinaryImage,pBinaryImage);
            //查看去噪声结果 结果：不好
            //return pBinaryImage;
            
            
            // 简单对图像进行开操作，去除斑点噪声
            int filterSize = 7;
            IplConvKernel *kernal = cvCreateStructuringElementEx(filterSize, filterSize, (filterSize - 1)*0.5, (filterSize - 1)*0.5, CV_SHAPE_RECT, NULL);
            //cvMorphologyEx(pBinaryImage,pBinaryImage,NULL,kernal,CV_MOP_GRADIENT);
            //cvMorphologyEx(pBinaryImage,pBinaryImage,NULL,kernal,CV_MOP_DILATE);
            //cvMorphologyEx(pBinaryImage,pBinaryImage,NULL,kernal,CV_MOP_ERODE);
            cvMorphologyEx(pBinaryImage,pBinaryImage,NULL,kernal,CV_MOP_BLACKHAT);
            cvReleaseStructuringElement(&kernal);
            cvReleaseImage(&pNumberROI);
            
            return pBinaryImage;
        }else if (type == CardTypeBank){
            
            ///处理银行卡
            
            SubBackGround(pNumberROI);//不明显
            
            
            IplConvKernel *kernel = cvCreateStructuringElementEx(3, 3, 1, 1, CV_SHAPE_CROSS, NULL);
            cvMorphologyEx(pNumberROI, pNumberROI, NULL, kernel, CV_MOP_GRADIENT, 1);
            cvReleaseStructuringElement(&kernel);
            llcv_equalize_hist(pNumberROI, pNumberROI);
            
            pNumberROI = subBack(pNumberROI);
            
            IplImage *grad_x = cvCreateImage(cvGetSize(pNumberROI), pNumberROI->depth, pNumberROI->nChannels);
            IplImage *grad_y = cvCreateImage(cvGetSize(pNumberROI), pNumberROI->depth, pNumberROI->nChannels);
            IplImage *abs_grad_x = cvCreateImage(cvGetSize(pNumberROI), pNumberROI->depth, pNumberROI->nChannels);
            IplImage *abs_grad_y = cvCreateImage(cvGetSize(pNumberROI), pNumberROI->depth, pNumberROI->nChannels);
            
            cvSobel(pNumberROI, grad_x, 1, 0, 7);
            cvConvertScaleAbs(grad_x, abs_grad_x);
            
            cvSobel(pNumberROI, grad_y, 0, 1, 7);
            cvConvertScaleAbs(grad_y, abs_grad_y);
            
            //cvAddWeighted(abs_grad_x, 1, abs_grad_y, 0, 0, pNumberROI);
            
            return pNumberROI;
            
            
            //cvConvertScale(pNumberROI, pNumberROI, 1.0f / 255.0f, 0);
            //cvNormalize(pNumberROI, pNumberROI, 0.0f, 1.0f, CV_MINMAX, NULL);
            return pNumberROI;
            
            int m_otsu = otsu(pNumberROI);//35
            printf("m_otsu:%d\n",m_otsu);
            //m_otsu = 35;
            cvThreshold(pNumberROI, pNumberROI, m_otsu, 255, CV_THRESH_BINARY);
        }
        return pNumberROI;
    }
}