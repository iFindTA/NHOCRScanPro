//
//  cvm_olm.cpp
//  NHOpenCVPro
//
//  Created by hu jiaju on 15/8/29.
//  Copyright (c) 2015年 hu jiaju. All rights reserved.
//

#include "cvm_olm.h"
#include "cvm_ios.h"

cvm_point cvm_create_point(float x, float y) {
    cvm_point d;
    d.x = x;
    d.y = y;
    return d;
}

cvm_rect cvm_create_rect(float x, float y, float w, float h) {
    cvm_rect r;
    r.x = x, r.y = y, r.w = w, r.h = h;
    return r;
}

void cvm_rect_get_points(cvm_rect rect, cvm_point points[4]) {
    points[0] = cvm_create_point(rect.x, rect.y);
    points[1] = cvm_create_point(rect.x + rect.w, rect.y);
    points[2] = cvm_create_point(rect.x, rect.y + rect.h);
    points[3] = cvm_create_point(rect.x + rect.w, rect.y + rect.h);
}

cvm_context *dmz_context_create(void) {
    cvm_context *dmz = (cvm_context *) calloc(1, sizeof(cvm_context));
    dmz->mz = mz_create();
    return dmz;
}

void dmz_context_destroy(cvm_context *dmz) {
    mz_destroy(dmz->mz);
    free(dmz);
}

void dmz_prepare_for_backgrounding(cvm_context *dmz) {
    mz_prepare_for_backgrounding(dmz->mz);
}

cvm_rect cvm_guide_frame(FrameOrientation orientation, float preview_width, float preview_height) {
    cvm_rect guide;
    float inset_w;
    float inset_h;
    
    switch(orientation) {
        case FrameOrientationPortrait:
            /* no break */
        case FrameOrientationPortraitUpsideDown:
            inset_w = kPortraitHorizontalPercentInset * preview_width;
            inset_h = kPortraitVerticalPercentInset * preview_height;
            break;
        case FrameOrientationLandscapeLeft:
            /* no break */
        case FrameOrientationLandscapeRight:
            inset_w = kLandscapeVerticalPercentInset * preview_width;
            inset_h = kLandscapeHorizontalPercentInset * preview_height;
            break;
        default:
            inset_w = 0.0f;
            inset_h = 0.0f;
            break;
    }
    
    guide.x = inset_w;
    guide.y = inset_h;
    guide.w = preview_width - 2.0f * inset_w;
    guide.h = preview_height - 2.0f * inset_h;
    
    return guide;
}

FrameOrientation cvm_opposite_orientation(FrameOrientation orientation) {
    switch(orientation) {
        case FrameOrientationPortrait:
            return FrameOrientationPortraitUpsideDown;
        case FrameOrientationPortraitUpsideDown:
            return FrameOrientationPortrait;
        case FrameOrientationLandscapeRight:
            return FrameOrientationLandscapeLeft;
        case FrameOrientationLandscapeLeft:
            return FrameOrientationLandscapeRight;
        default:
            return FrameOrientationPortrait;
    }
}

void llcv_unwarp(IplImage *input, const cvm_point source_points[4], const cvm_rect to_rect, IplImage *output) {
    
    /* if dmz_use_gles_warp() has changed from above, then we've encountered an error and are falling back to the old way.*/
    
    //  dmz_point source_points[4], dest_points[4];
    ios_gpu_unwarp(input, source_points, output);
    
    /**
     *
     // Old-fashioned openCV
     float matrix[16];
     cvm_point dest_points[4];
     cvm_rect_get_points(to_rect, dest_points);
     
     // Calculate row-major matrix
     llcv_calc_persp_transform(matrix, 16, true, source_points, dest_points);
     CvMat *cv_persp_mat = cvCreateMat(3, 3, CV_32FC1);
     for (int r = 0; r < 3; r++) {
     for (int c = 0; c < 3; c++) {
     CV_MAT_ELEM(*cv_persp_mat, float, r, c) = matrix[3 * r + c];
     }
     }
     cvWarpPerspective(input, output, cv_persp_mat, CV_INTER_LINEAR + CV_WARP_FILL_OUTLIERS, cvScalarAll(0));
     cvReleaseMat(&cv_persp_mat);
     */
}

void llcv_calc_persp_transform(float *matrixData, int matrixDataSize, bool rowMajor, const cvm_point sourcePoints[], const cvm_point destPoints[]) {
    
    // Set up matrices a and b so we can solve for x from ax = b
    // See http://xenia.media.mit.edu/~cwren/interpolator/ for a
    // good explanation of the basic math behind this.
    
    typedef Eigen::Matrix<float, 8, 8> Matrix8x8;
    typedef Eigen::Matrix<float, 8, 1> Matrix8x1;
    
    Matrix8x8 a;
    Matrix8x1 b;
    
    for(int i = 0; i < 4; i++) {
        a(i, 0) = sourcePoints[i].x;
        a(i, 1) = sourcePoints[i].y;
        a(i, 2) = 1;
        a(i, 3) = 0;
        a(i, 4) = 0;
        a(i, 5) = 0;
        a(i, 6) = -sourcePoints[i].x * destPoints[i].x;
        a(i, 7) = -sourcePoints[i].y * destPoints[i].x;
        
        a(i + 4, 0) = 0;
        a(i + 4, 1) = 0;
        a(i + 4, 2) = 0;
        a(i + 4, 3) = sourcePoints[i].x;
        a(i + 4, 4) = sourcePoints[i].y;
        a(i + 4, 5) = 1;
        a(i + 4, 6) = -sourcePoints[i].x * destPoints[i].y;
        a(i + 4, 7) = -sourcePoints[i].y * destPoints[i].y;
        
        b(i, 0) = destPoints[i].x;
        b(i + 4, 0) = destPoints[i].y;
    }
    
    // Solving ax = b for x, we get the values needed for our perspective
    // matrix. Table of options on the eigen site at
    // /dox/TutorialLinearAlgebra.html#TutorialLinAlgBasicSolve
    //
    // We use householderQr because it places no restrictions on matrix A,
    // is moderately fast, and seems to be sufficiently accurate.
    //
    // partialPivLu() seems to work as well, but I am wary of it because I
    // am unsure of A is invertible. According to the documenation and basic
    // performance testing, they are both roughly equivalent in speed.
    //
    // - @burnto
    
    Matrix8x1 x = a.householderQr().solve(b);
    
    // Initialize matrixData
    for (int i = 0; i < matrixDataSize; i++) {
        matrixData[i] = 0.0f;
    }
    int matrixSize = (matrixDataSize >= 16) ? 4 : 3;
    
    // Initialize a 4x4 eigen matrix. We may not use the final
    // column/row, but that's ok.
    Eigen::Matrix4f perspMatrix = Eigen::Matrix4f::Zero();
    
    // Assign a, b, d, e, and i
    perspMatrix(0, 0) = x(0, 0); // a
    perspMatrix(0, 1) = x(1, 0); // b
    perspMatrix(1, 0) = x(3, 0); // d
    perspMatrix(1, 1) = x(4, 0); // e
    perspMatrix(2, 2) = 1.0f;    // i
    
    // For 4x4 matrix used for 3D transform, we want to assign
    // c, f, g, and h to the fourth col and row.
    // So we use an offset for thes values
    int o = matrixSize - 3; // 0 or 1
    perspMatrix(0, 2 + o) = x(2, 0); // c
    perspMatrix(1, 2 + o) = x(5, 0); // f
    perspMatrix(2 + o, 0) = x(6, 0); // g
    perspMatrix(2 + o, 1) = x(7, 0); // h
    perspMatrix(2 + o, 2 + o) = 1.0f; // i
    
    // Assign perspective matrix to our matrixData buffer,
    // swapping row versus column if needed, and taking care not to
    // overflow if user didn't provide a large enough matrixDataSize.
    for(int c = 0; c < matrixSize; c++) {
        for(int r = 0; r < matrixSize; r++) {
            int index = rowMajor ? (c + r * matrixSize) : (r + c * matrixSize);
            if (index < matrixDataSize) {
                matrixData[index] = perspMatrix(r, c);
            }
        }
    }
    // TODO - instead of copying final values into matrixData return array, do one of:
    // (a) assign directly into matrixData, or
    // (b) use Eigen::Mat so that assignment goes straight into underlying matrixData
}

void cvm_deinterleave_RGBA_to_R(uint8_t *source, uint8_t *dest, int size) {
#if DMZ_HAS_NEON_COMPILETIME
    assert(size >= 16); // required for the vectorized handling of leftover_bytes; also, a reasonable expectation!
    
    for (int offset = 0; offset + 15 < size; offset += 16) {
        uint8x16x4_t r1 = vld4q_u8(&source[offset * 4]);
        vst1q_u8(&dest[offset], r1.val[0]);
    }
    
    // use "overlapping" to process the remaining bytes
    // See http://community.arm.com/groups/processors/blog/2010/05/10/coding-for-neon--part-2-dealing-with-leftovers
    if (size % 16 > 0) {
        int offset = size - 16;
        uint8x16x4_t r1 = vld4q_u8(&source[offset * 4]);
        vst1q_u8(&dest[offset], r1.val[0]);
    }
#endif
    for (int offset = 0; offset + 7 < size; offset += 8) {
        int bufferOffset = offset * 4;
        dest[offset] = source[bufferOffset];
        dest[offset + 1] = source[bufferOffset + (1 * 4)];
        dest[offset + 2] = source[bufferOffset + (2 * 4)];
        dest[offset + 3] = source[bufferOffset + (3 * 4)];
        dest[offset + 4] = source[bufferOffset + (4 * 4)];
        dest[offset + 5] = source[bufferOffset + (5 * 4)];
        dest[offset + 6] = source[bufferOffset + (6 * 4)];
        dest[offset + 7] = source[bufferOffset + (7 * 4)];
    }
    
    int leftover_bytes = size % 8; // each RGBA pixel is 4 bytes, so can assume size % 4 == 0
    if (leftover_bytes > 0) {
        for (int offset = size - leftover_bytes; offset < size; offset += 4) {
            int bufferOffset = offset * 4;
            dest[offset] = source[bufferOffset];
            dest[offset + 1] = source[bufferOffset + (1 * 4)];
            dest[offset + 2] = source[bufferOffset + (2 * 4)];
            dest[offset + 3] = source[bufferOffset + (3 * 4)];
        }
    }
}

uint8_t llcv_get_pixel_step(IplImage *image) {
    uint8_t pixel_step = 0;
    if (IPL_DEPTH_8S == image->depth || IPL_DEPTH_8U == image->depth) {
        pixel_step = sizeof(uint8_t);
    }else if (IPL_DEPTH_16S == image->depth || IPL_DEPTH_16U == image->depth){
        pixel_step = sizeof(uint16_t);
    }else if (IPL_DEPTH_32F == image->depth || IPL_DEPTH_32S == image->depth){
        pixel_step = sizeof(uint32_t);
    }else if (IPL_DEPTH_64F == image->depth){
        pixel_step = sizeof(uint64_t);
    }
    return pixel_step;
}

void* llcv_get_data_origin(IplImage *image) {
    uint8_t pixel_step = llcv_get_pixel_step(image);
    uint8_t *data_origin = (uint8_t *)image->imageData;
    if(NULL != image->roi) {
        data_origin += image->roi->yOffset * image->widthStep + image->roi->xOffset * pixel_step;
    }
    return data_origin;
}

void llcv_YCbCr2RGB_u8_c(IplImage *y, IplImage *cb, IplImage *cr, IplImage *dst) {
    // Could vectorize this, but the math gets ugly, and we only do it once, and really, it's fast enough.
#define DESCALE_14(x) ((x + (1 << 13)) >> 14)
#define SATURATED_BYTE(x) (uint8_t)((x < 0) ? 0 : ((x > 255) ? 255 : x))
    
    bool addAlpha = (dst->nChannels == 4);
    
    CvSize src_size = cvGetSize(y);
    
    uint8_t *y_data_origin = (uint8_t *)llcv_get_data_origin(y);
    uint16_t y_width_step = (uint16_t)y->widthStep;
    
    uint8_t *cb_data_origin = (uint8_t *)llcv_get_data_origin(cb);
    uint16_t cb_width_step = (uint16_t)cb->widthStep;
    
    uint8_t *cr_data_origin = (uint8_t *)llcv_get_data_origin(cr);
    uint16_t cr_width_step = (uint16_t)cr->widthStep;
    
    uint8_t *dst_data_origin = (uint8_t *)llcv_get_data_origin(dst);
    uint16_t dst_width_step = (uint16_t)dst->widthStep;
    
    for(uint16_t row_index = 0; row_index < src_size.height; row_index++) {
        const uint8_t *y_row_origin = y_data_origin + row_index * y_width_step;
        const uint8_t *cb_row_origin = cb_data_origin + row_index * cb_width_step;
        const uint8_t *cr_row_origin = cr_data_origin + row_index * cr_width_step;
        
        uint8_t *dst_row_origin = dst_data_origin + row_index * dst_width_step;
        
        uint16_t col_index = 0;
        while(col_index < src_size.width) {
            uint8_t pix_y = y_row_origin[col_index];
            uint8_t pix_cb = cb_row_origin[col_index];
            uint8_t pix_cr = cr_row_origin[col_index];
            int8_t sCb = pix_cb - 128;
            int8_t sCr = pix_cr - 128;
            int32_t pix_b = pix_y + DESCALE_14(sCb * 29049);
            int32_t pix_g = pix_y + DESCALE_14(sCb * -5636 + sCr * -11698);
            int32_t pix_r = pix_y + DESCALE_14(sCr * 22987);
            
            uint16_t col_pixel_pos = (uint16_t)(col_index * dst->nChannels);
            
            // the SATURATED_BYTE macro is necessary to ensure that we're really only writing one
            // byte, and that we stay within it's limits. It appears that the clang (and possibly
            // gcc 4.6 vs 4.4.3) differ in how the shift/cast combo behaves.
            dst_row_origin[col_pixel_pos] = SATURATED_BYTE(pix_r);
            dst_row_origin[col_pixel_pos + 1] = SATURATED_BYTE(pix_g);
            dst_row_origin[col_pixel_pos + 2] = SATURATED_BYTE(pix_b);
            
            if (addAlpha) {
                dst_row_origin[col_pixel_pos + 3] = 0xff; // make an opaque image
            }
            
            col_index++;
        }
    }
}

void llcv_YCbCr2RGB_u8(IplImage *y, IplImage *cb, IplImage *cr, IplImage *dst) {
    
    assert(y->nChannels == 1);
    assert(cb->nChannels == 1);
    assert(cr->nChannels == 1);
    assert(dst->nChannels == 3 || dst->nChannels == 4);
    
    assert(y->depth == IPL_DEPTH_8U);
    assert(cb->depth == IPL_DEPTH_8U);
    assert(cr->depth == IPL_DEPTH_8U);
    assert(dst->depth == IPL_DEPTH_8U);
    
    llcv_YCbCr2RGB_u8_c(y, cb, cr, dst);
    
}

void cvm_YCbCr_to_RGB(IplImage *y, IplImage *cb, IplImage *cr, IplImage **rgb) {
    if (*rgb == NULL) {
        *rgb = cvCreateImage(cvGetSize(y), y->depth, 3);
    }
    llcv_YCbCr2RGB_u8(y, cb, cr, *rgb);
}

int otsu(IplImage *image) {
    
    assert(NULL != image);
    
    int width = image->width;
    int height = image->height;
    int x=0,y=0;
    int pixelCount[256];
    float pixelPro[256];
    int i, j, pixelSum = width * height, threshold = 0;
    
    uchar* data = (uchar*)image->imageData;
    
    //初始化
    for(i = 0; i < 256; i++)
    {
        pixelCount[i] = 0;
        pixelPro[i] = 0;
    }
    
    //统计灰度级中每个像素在整幅图像中的个数
    for(i = y; i < height; i++)
    {
        for(j = x;j <width;j++)
        {
            pixelCount[data[i * image->widthStep + j]]++;
        }
    }
    
    
    //计算每个像素在整幅图像中的比例
    for(i = 0; i < 256; i++)
    {
        pixelPro[i] = (float)(pixelCount[i]) / (float)(pixelSum);
    }
    
    //经典ostu算法,得到前景和背景的分割
    //遍历灰度级[0,255],计算出方差最大的灰度值,为最佳阈值
    float w0, w1, u0tmp, u1tmp, u0, u1, u,deltaTmp, deltaMax = 0;
    for(i = 0; i < 256; i++)
    {
        w0 = w1 = u0tmp = u1tmp = u0 = u1 = u = deltaTmp = 0;
        
        for(j = 0; j < 256; j++)
        {
            if(j <= i) //背景部分
            {
                //以i为阈值分类，第一类总的概率
                w0 += pixelPro[j];
                u0tmp += j * pixelPro[j];
            }
            else       //前景部分
            {
                //以i为阈值分类，第二类总的概率
                w1 += pixelPro[j];
                u1tmp += j * pixelPro[j];
            }
        }
        
        u0 = u0tmp / w0;		//第一类的平均灰度
        u1 = u1tmp / w1;		//第二类的平均灰度
        u = u0tmp + u1tmp;		//整幅图像的平均灰度
        //计算类间方差
        deltaTmp = w0 * (u0 - u)*(u0 - u) + w1 * (u1 - u)*(u1 - u);
        //找出最大类间方差以及对应的阈值
        if(deltaTmp > deltaMax)
        {
            deltaMax = deltaTmp;
            threshold = i;
        }
    }
    //返回最佳阈值;
    return threshold;
}

int ThresholdOtsu(cv::Mat mat){
    int height=mat.rows;
    int width=mat.cols;
    
    //histogram
    float histogram[256]={0};
    for(int i=0;i<height;i++) {
        
        for(int j=0;j<width;j++)
        {
            
            unsigned char p=(unsigned char)((mat.data[i*mat.step[0]+j]));
            histogram[p]++;
        }
    }
    //normalize histogram
    int size=height*width;
    for(int i=0;i<256;i++) {
        histogram[i]=histogram[i]/size;
    }
    
    //average pixel value
    float avgValue=0;
    for(int i=0;i<256;i++) {
        avgValue+=i*histogram[i];
    }
    
    int thresholdV = 30;
    float maxVariance=0;
    float w=0,u=0;
    for(int i=0;i<256;i++) {
        w+=histogram[i];
        u+=i*histogram[i];
        
        float t=avgValue*w-u;
        float variance=t*t/(w*(1-w));
        if(variance>maxVariance) {
            maxVariance=variance;
            thresholdV=i;
        }
    }
    
    return thresholdV;
}

//! 将Rect按位置从左到右进行排序
int sortRect(const vector<CvRect>& vecRect, vector<CvRect>& out) {
    vector<int> orderIndex;
    vector<int> xpositions;
    
    for (int i = 0; i < vecRect.size(); i++)
    {
        orderIndex.push_back(i);
        xpositions.push_back(vecRect[i].x);
    }
    
    float min = xpositions[0];
    int minIdx = 0;
    for (int i = 0; i< xpositions.size(); i++)
    {
        min = xpositions[i];
        minIdx = i;
        for (int j = i; j<xpositions.size(); j++)
        {
            if (xpositions[j]<min){
                min = xpositions[j];
                minIdx = j;
            }
        }
        int aux_i = orderIndex[i];
        int aux_min = orderIndex[minIdx];
        orderIndex[i] = aux_min;
        orderIndex[minIdx] = aux_i;
        
        float aux_xi = xpositions[i];
        float aux_xmin = xpositions[minIdx];
        xpositions[i] = aux_xmin;
        xpositions[minIdx] = aux_xi;
    }
    
    for (int i = 0; i<orderIndex.size(); i++)
    {
        out.push_back(vecRect[orderIndex[i]]);
    }
    
    return 0;
}