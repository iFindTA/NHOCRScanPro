//
//  cvm_scan.h
//  NHOpenCVPro
//
//  Created by hu jiaju on 15/8/31.
//  Copyright (c) 2015年 hu jiaju. All rights reserved.
//

#ifndef __NHOpenCVPro__cvm_scan__
#define __NHOpenCVPro__cvm_scan__

#include <stdio.h>
#include "cvm_olm.h"
#include <opencv2/ml/ml.hpp>
#include <opencv2/imgproc/imgproc_c.h>

using namespace cv;

namespace opencv_scan {
    class CvScan{
        
    public:
        CvScan();
        
        //! 装载ANN模型
        void LoadANNModel();
        void LoadANNModel(string s);
        //! 装载SVM模型
        void LoadSVMModel();
        void LoadSVMModel(string s);
        //! 设置与读取模型路径
        inline void setModelPath(string path){	m_path = path;	}
        inline string getModelPath() { return m_path;	}
        bool getBusyState(){return isBusy;};
        
        //字符分割
        void charsSegement(IplImage *src, vector<Mat> &vector);
        //字符分割
        void charsMatSegement(Mat input, vector<Mat> &vector);
        void charsImgSegement(IplImage *src, vector<IplImage*> &vector);
        
        //! 字符分类
        int classify(Mat f);
        int classify(Mat, bool);
        //! 字符鉴别
        string charsIdentify(Mat input);
        string charsIdentify(Mat input, bool isChinese);
        
        void train(Mat traindata, Mat classes, int nlayers);
    private:
        //！使用的ANN模型
        CvANN_MLP ann;
        
        //! 模型存储路径
        string m_path;
        //! 特征尺寸
        int m_predictSize;
        
        bool isBusy;
        //CvSVM svm;
        
        bool trained;
        
    };
}

#endif /* defined(__NHOpenCVPro__cvm_scan__) */
