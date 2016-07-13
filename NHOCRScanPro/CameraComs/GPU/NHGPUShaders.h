//
//  NHGPUShaders.h
//  NHOpenCVPro
//
//  Created by hu jiaju on 15/8/29.
//  Copyright (c) 2015年 hu jiaju. All rights reserved.
//
#pragma once
#ifndef NHOpenCVPro_NHGPUShaders_h
#define NHOpenCVPro_NHGPUShaders_h

// via GPUImage's GPUImageFilter.h
#define STRINGIZE(x) #x
#define STRINGIZE2(x) STRINGIZE(x)
#define SHADER_STRING(text) @ STRINGIZE2(text)

#ifdef __cplusplus
#import <opencv2/opencv.hpp>
#endif

#endif
