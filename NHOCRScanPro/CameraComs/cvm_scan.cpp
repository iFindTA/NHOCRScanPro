//
//  cvm_scan.cpp
//  NHOpenCVPro
//
//  Created by hu jiaju on 15/8/31.
//  Copyright (c) 2015年 hu jiaju. All rights reserved.
//

#include "cvm_scan.h"
#include <opencv2/core/core_c.h>
#include <opencv2/core/version.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <unistd.h>
#define HORIZONTAL    1
#define VERTICAL    0

#define kSmallCharacterWidth 9
#define kSmallCharacterHeight 15

#define kTrimmedCharacterImageWidth 11
#define kTrimmedCharacterImageHeight 16


namespace opencv_scan {
    
    //目标字符
    const char strCharacters[] = {'0','1','2','3','4','5','6','7','8','9','A','B', 'C', 'D', 'E','F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P','Q', 'R', 'S', 'T','U', 'V', 'W', 'X', 'Y', 'Z'};
    const int numCharacter = 34; /* 11个字符 10个数字＋1个X */
    //const int numFilesChars[]={50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50};
    
    const int numAll = numCharacter+31; /* 34+31=65 */
    
    
    //!preprocessChar所用常量
    static const int CHAR_SIZE = 20;
    
    cv::Mat features(cv::Mat in, int sizeData);
    
    CvScan:: CvScan(){
        
        isBusy = false;
        m_predictSize = 10;
        m_path = "NHOpenCVPro/NHScanSDK/model/ann.xml";
        //LoadANNModel();
        
        cout << "当前OpenCV Version:" << CV_VERSION << endl;
    }
    
    void CvScan::LoadANNModel(){
        //ann.clear();
        ann.load(m_path.c_str(), "ann");
    }
    
    void CvScan::LoadANNModel(string s){
        
        ann.clear();
        //ann.load(s.c_str(), "ann");
        //return;
        
        FileStorage fs;
        fs.open(s, FileStorage::READ);
        if (fs.isOpened()){
            cout<<"File is opened:"<<s<<endl;
        }
        
        Mat TrainingData;
        Mat Classes;
        fs["TrainingDataF15"]>>TrainingData;
        fs["classes"]>>Classes;
        fs.release();
        //cout << trainingDataF5 << endl;
        train(TrainingData, Classes, 10);
        //*/
    }
    
    void CvScan::LoadSVMModel(){
        //svm.clear();
        //svm.load(m_path.c_str(), "svm");
    }
    
    void CvScan::LoadSVMModel(std::string s){
        //svm.clear();
        //svm.load(s.c_str(), "svm");
    }
    
    //训练分类器
    void CvScan::train(Mat traindata, Mat classes, int nlayers){
        
        printf(" ANN network start trainning!\n");
        
        ///训练时间较短
        //int ar[]={256,numCharacter,nlayers};
        Mat layers(1, 3, CV_32SC1);//CV_32FC1 浮点型
        layers.at<int>(0) = traindata.cols;
        layers.at<int>(1) = nlayers;
        layers.at<int>(2) = numCharacter;
        
        ann.create(layers, CvANN_MLP::SIGMOID_SYM,1,1);
        
        // 创建一个矩阵，其中存放n个训练数据，并将其分为m类
        Mat trainClasses;
        trainClasses.create(traindata.rows, numCharacter, CV_32FC1);
        for (int i = 0; i < trainClasses.rows; i++){
            for (int k = 0; k < trainClasses.cols; k++){
                if (k == classes.at<int>(i))
                    trainClasses.at<float>(i, k) = 1;
                else
                    trainClasses.at<float>(i, k) = 0;
            }
        }
        
        // Set up BPNetwork's parameters
        CvANN_MLP_TrainParams params;//迭代次数5000次
        //params.term_crit = cvTermCriteria( CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, \
                                          5000, 0.01 );
        params.train_method = CvANN_MLP_TrainParams::BACKPROP;
        params.bp_dw_scale = 0.1;
        params.bp_moment_scale = 0.1;
        
        //RPROP 时需要
        //Mat weights(1, traindata.rows, CV_32FC1, Scalar::all(1));
        // 分类器学习
        ann.train(traindata, trainClasses, Mat(),Mat(),params);
        trained = true;
        
        // 存储MPL
        //ann.save("mpl.xml");
        
        printf("ANN network over trained!\n");
        
        const CvMat *mat=ann.get_layer_sizes();
        
    }
    
    int CvScan::classify(cv::Mat f, bool isChinses){
        int result = -1;
        cv::Mat output(1, numCharacter, CV_32FC1);
        ann.predict(f, output);//使用ann对字符做判断
        
        //对中文字符的判断
        if (!isChinses){
            result = 0;
            float maxVal = -2;
            for(int j = 0; j < numCharacter; j++){
                float val = output.at<float>(j);
                //cout << "j:" << j << "val:"<< val << endl;
                if (val > maxVal){
                    //求得中文字符权重最大的那个，也就是通过ann认为最可能的字符
                    maxVal = val;
                    result = j;
                }
            }
        }else{//对数字和英文字母的判断
            result = numCharacter;
            float maxVal = -2;
            for(int j = numCharacter; j < numAll; j++){
                float val = output.at<float>(j);
                //cout << "j:" << j << "val:"<< val << endl;
                if (val > maxVal){
                    maxVal = val;
                    result = j;
                }
            }
            
        }
        return result;
    }
    
    //! 字符预处理
    cv::Mat preprocessChar(cv::Mat in){
        //Remap image
        int h = in.rows;
        int w = in.cols;
        int charSize = CHAR_SIZE;	//统一每个字符的大小
        cv::Mat transformMat = cv::Mat::eye(2, 3, CV_32F);
        int m = MAX(w, h);
        transformMat.at<float>(0, 2) = m / 2 - w / 2;
        transformMat.at<float>(1, 2) = m / 2 - h / 2;
        
        cv::Mat warpImage(m, m, in.type());
        warpAffine(in, warpImage, transformMat, warpImage.size(), cv::INTER_LINEAR, cv::BORDER_CONSTANT, cvScalar(0));
        
        cv::Mat out;
        resize(warpImage, out, cvSize(charSize, charSize));
        
        return out;
    }
    
    int CvScan::classify(cv::Mat f) {
        int result = -1;
        cv::Mat output(1, numCharacter, CV_32FC1);
        ann.predict(f, output);//使用ann对字符做判断
        
        float maxVal = -2;
        for(int j = 0; j < numCharacter; j++){
            float val = output.at<float>(j);
            //cout << "j:" << j << "val:"<< val << endl;
            if (val > maxVal){
                //求得中文字符权重最大的那个，也就是通过ann认为最可能的字符
                maxVal = val;
                result = j;
            }
        }
        
        for(int j = numCharacter; j < numAll; j++){
            float val = output.at<float>(j);
            //cout << "j:" << j << "val:"<< val << endl;
            if (val > maxVal){
                maxVal = val;
                result = j;
            }
        }
        
        return result;
    }
    
#pragma mark -- Scan And Recognize --
    
    //! 字符尺寸验证
    bool verifyMatCharSizes(cv::Mat r){
        //Char sizes 45x90
        float aspect = 45.0f / 90.0f;
        float charAspect = (float)r.cols / (float)r.rows;
        float error = 0.7;
        float minHeight = 10;
        float maxHeight = 35;
        //We have a different aspect ratio for number 1, and it can be ~0.2
        float minAspect = 0.05;
        float maxAspect = aspect + aspect*error;
        //area of pixels
        float area = countNonZero(r);
        //bb area
        float bbArea = r.cols*r.rows;
        //% of pixel in area
        float percPixels = area / bbArea;
        
        if (percPixels <= 1 && charAspect > minAspect && charAspect < maxAspect && r.rows >= minHeight && r.rows < maxHeight)
            return true;
        else
            return false;
    }
    
    //! 字符尺寸验证
    bool verifyImgCharSizes(IplImage *r){
        //Char sizes 45x90
        float aspect = 45.0f / 90.0f;
        float charAspect = (float)r->width / (float)r->height;
        float error = 0.7;
        float minHeight = 10;
        float maxHeight = 35;
        float minWidth = 8;
        float maxWidth = 16;
        //We have a different aspect ratio for number 1, and it can be ~0.2
        float minAspect = 0.05;
        float maxAspect = aspect + aspect*error;
        //area of pixels
        //float area = cvContourArea(r);
        float area = cvCountNonZero(r);
        //bb area
        float bbArea = r->width*r->height;
        //% of pixel in area
        float percPixels = area / bbArea;
        
        if (percPixels < 1 && charAspect > minAspect && charAspect < maxAspect && r->height >= minHeight && r->height < maxHeight && r->width>minWidth && r->width < maxWidth)
            return true;
        else
            return false;
    }
    
    // ！获取垂直和水平方向直方图
    cv::Mat ProjectedHistogram(cv::Mat src, int t){
        /*
        cv::Mat image1;
        IplImage* image2;
        image2 = cvCreateImage(cvSize(image1.cols,image1.rows),8,3);
        IplImage ipltemp=image1;
        cvCopy(&ipltemp,image2);
         
         cv::Mat src = cv::cvarrToMat(img);
        //*/
        
        int sz = (t) ? src.rows : src.cols;
        cv::Mat mhist = cv::Mat::zeros(1, sz, CV_32F);
        
        for (int j = 0; j<sz; j++){
            cv::Mat data = (t) ? src.row(j) : src.col(j);
            
            mhist.at<float>(j) =countNonZero(data);	//统计这一行或一列中，非零元素的个数，并保存到mhist中
        }
        
        //Normalize histogram
        double min, max;
        minMaxLoc(mhist, &min, &max);
        
        if (max>0)
            mhist.convertTo(mhist, -1, 1.0f / max, 0);//用mhist直方图中的最大值，归一化直方图
        //IplImage *dst;
        //IplImage tmp = mhist;
        //cvCopy(&tmp, dst);
        return mhist;
    }
    
    void CvScan::charsMatSegement(Mat input, vector<Mat> &vector){
        
        if (!input.data)
            return ;
        
        int w = input.cols;
        int h = input.rows;
        
        Mat tmp = input(Rect(w*0.1,h*0.1,w*0.8,h*0.8));
        int threadHoldV = ThresholdOtsu(tmp);
        
        Mat img_threshold = input.clone();
        threshold(input, img_threshold,threadHoldV, 255, CV_THRESH_BINARY);
        Mat img_contours;
        img_threshold.copyTo(img_contours);
    }
    
    void CvScan::charsSegement( IplImage *src, vector<Mat> &vector){
        
        if (src == NULL) {
            return;
        }
        
        IplImage *pimg = cvCreateImage(cvSize(src->width*1.1, src->height*1.1), src->depth, src->nChannels);
        /*
        int m_otsu = otsu(pimg);
        printf("m_otsu:%d\n",m_otsu);
        cvReleaseImage(&pimg);
        pimg = cvCreateImage(cvGetSize(src), src->depth, src->nChannels);
        cvThreshold(src, pimg, m_otsu, 255, CV_THRESH_BINARY);
        //cvZero(pimg);
        //*/
        
        //IplImage imgHSV = *cvCreateImage(cvGetSize(pimg), 8, 1);
        //cv::Mat matImg(&imgHSV,true);
        
        cv::Mat img_contours(pimg->width,pimg->height,CV_8UC1,Scalar::all(0));
        img_contours.data = (uchar *)pimg->imageData;
        
        
        std::vector< std::vector< CvPoint> > contours;
        
        findContours(img_contours,
                     contours, // a vector of contours
                     CV_RETR_EXTERNAL, // retrieve the external contours
                     CV_CHAIN_APPROX_NONE); // all pixels of each contours
        //Start to iterate to each contour founded
        std::vector<std::vector<CvPoint>>::iterator itc = contours.begin();
        std::vector<CvRect> vecRect;
        
        //Remove patch that are no inside limits of aspect ratio and area.
        //将不符合特定尺寸的图块排除出去
        while (itc != contours.end()){
            Rect mr = boundingRect(cv::Mat(*itc));
            cv::Mat auxRoi(img_contours, mr);
            if (verifyMatCharSizes(auxRoi))
                vecRect.push_back(mr);
            
            ++itc;
        }
        
        if (vecRect.size() == 0)
            return ;
        std::vector<CvRect> sortedRect;
        ////对符合尺寸的图块按照从左到右进行排序
        sortRect(vecRect, sortedRect);
        
        for (int i = 0; i < sortedRect.size(); i++){
            CvRect mr = sortedRect[i];
            cv::Mat auxRoi(img_contours, mr);
            
            auxRoi = preprocessChar(auxRoi);
            vector.push_back(auxRoi);
        }
        
        /* 另一个方法
        
        //*/
        return ;
    }
    
    void CvScan::charsImgSegement(IplImage *src, vector<IplImage*> &vector) {
        
        if (src == NULL) {
            return;
        }
        IplImage *pimg = cvCreateImage(cvSize(src->width*1.1, src->height*1.1), src->depth, src->nChannels);
        //*
        int m_otsu = otsu(pimg);
        printf("m_otsu:%d\n",m_otsu);
        cvReleaseImage(&pimg);
        cvZero(pimg);
        pimg = cvCreateImage(cvGetSize(src), src->depth, src->nChannels);
        cvThreshold(src, pimg, m_otsu, 255, CV_THRESH_BINARY);
        //查看 ret:right
        //vector.push_back(pimg);
        //return;
        //*/
        
        std::vector<CvRect> contours;
        CvSeq* contour;
        CvMemStorage *storage = cvCreateMemStorage(0);
        CvContourScanner scanner= cvStartFindContours(pimg,storage,sizeof(CvContour),CV_RETR_CCOMP,CV_CHAIN_APPROX_SIMPLE,cvPoint(0,0));
        //开始遍历轮廓树
        CvRect rect;
        double tmparea = 0.0;double indexArea = 0.0;double minarea = 5*5;double pixels = pimg->width*pimg->height;
        int i = 0;uchar *pp;IplImage *pdst;
        while ((contour = cvFindNextContour(scanner))) {
            tmparea = fabs(cvContourArea(contour));
            indexArea = fabs(cvContourArea(contour)/pixels);
            rect = cvBoundingRect(contour,0);
            
//            if (indexArea < 0.02 || indexArea >= 1 || tmparea < minarea) {
//                //不符合条件 删除区域
//                cvSubstituteContour(scanner, NULL);
//            }else{
//                contours.push_back(rect);
//            }
            //*
            if (tmparea<minarea){
                //当连通区域的中心点为白色时，而且面积较小则用黑色进行填充
                pp=(uchar*)(pimg->imageData+pimg->widthStep*(rect.y+rect.height/2)+rect.x+rect.width/2);
                if (pp[0]==255){
                    for (int y=rect.y;y<rect.y+rect.height;y++){
                        for (int x=rect.x;x<rect.x+rect.width;x++){
                            pp=(uchar*)(pimg->imageData+pimg->widthStep*y+x);
                            if(pp[0]==255){
                                pp[0]=0;
                            }
                        }
                    }
                }
            }else{
                contours.push_back(rect);
            };
            //*/
        }
        cvEndFindContours(&scanner);
        int size = (int)contours.size();
        if (size <= 0) {
            return;
        }
        printf("检测出的矩形个数:%d\n",size);
        
        std::vector<CvRect> sortedRect;
        ////对符合尺寸的图块按照从左到右进行排序
        sortRect(contours, sortedRect);
        for (i = 0; i < sortedRect.size(); i++) {
            
            //printf("找到的rect:%d-%d-%d-%d\n",sortedRect[i].x,sortedRect[i].y,sortedRect[i].width,sortedRect[i].height);
            pdst = cvCreateImage(cvSize(sortedRect[i].width,sortedRect[i].height), IPL_DEPTH_8U, 1);
            cvSetImageROI(pimg, sortedRect[i]);
            //cvAdd(pimg, pdst, pdst, NULL);
            cvCopy(pimg, pdst, NULL);
            //cvReleaseImage(&pdst);
            cvResetImageROI(pimg);
            if (verifyImgCharSizes(pdst)) {
                IplImage *dst = cvCreateImage(cvSize(kTrimmedCharacterImageWidth, kTrimmedCharacterImageHeight), pdst->depth, pdst->nChannels);
                cvResize(pdst, dst, CV_INTER_LINEAR);
                vector.push_back(dst);
                cvReleaseImage(&pdst);
            }
        }
        //printf("共找到%d个字符块\n",i);
    }
    
    cv::Mat features(cv::Mat in, int sizeData){
        
        cv::Mat vhist = ProjectedHistogram(in, VERTICAL);
        cv::Mat hhist = ProjectedHistogram(in, HORIZONTAL);
        //resize(src,lowData,cvSize(sizeData, sizeData));
        
        cv::Mat lowData;
        cv::resize(in, lowData, cvSize(sizeData, sizeData));
        
        //Last 10 is the number of moments components
        int numCols=vhist.cols+hhist.cols+lowData.cols*lowData.cols;
        
        cv::Mat out=cv::Mat::zeros(1,numCols,CV_32F);
        //Asign values to feature,ANN的样本特征为水平、垂直直方图和低分辨率图像所组成的矢量
        int j=0;
        for(int i=0; i<vhist.cols; i++)
        {
            out.at<float>(j)=vhist.at<float>(i);
            j++;
        }
        for(int i=0; i<hhist.cols; i++)
        {
            out.at<float>(j)=hhist.at<float>(i);
            j++;
        }
        for(int x=0; x<lowData.cols; x++)
        {
            for(int y=0; y<lowData.rows; y++){
                out.at<float>(j)=(float)lowData.at<unsigned char>(x,y);
                j++;
            }
        }
        //if(DEBUG)
        //	cout << out << "\n===========================================\n";
        return out;
        
    }
    
    std::string CvScan::charsIdentify(cv::Mat input) {
        isBusy = true;
        input = preprocessChar(input);
        cv::Mat f = features(input, m_predictSize);
        std::string result = "";
        int index = classify(f);//使用ann来判别那个字符
        
        if (index >= numCharacter){
            //std::string s = strChinese[index - numCharacter];
            //std::string province = m_map[s];
            //return s;
        }
        else
        {
            char s =  strCharacters[index];
            char szBuf[216];
            sprintf(szBuf,"%c",s);
            return szBuf;
        }
        isBusy = false;
        return "";
    }
    
    //输入当个字符Mat,生成字符的string
    std::string CvScan::charsIdentify(cv::Mat input, bool isChinese){
        
        isBusy = true;
        input = preprocessChar(input);
        cv::Mat f = features(input, m_predictSize);
        std::string result = "";
        int index = classify(f, isChinese);//使用ann来判别那个字符
        
        if (!isChinese){
            result = result + strCharacters[index];
        }else{
            //std::string s = strChinese[index - numCharacter];
            //std::string province = m_map[s];
            //result = province + result;
        }
        
        isBusy = false;
        return result;
    }
}