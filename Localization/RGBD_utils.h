#ifndef RGBD_H
#define RGBD_H

#include <iostream>
#include <fstream>
#include <string>
#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgcodecs.hpp"
#include <opencv2/imgproc/imgproc.hpp>
#include "cudaSift.h"
#ifdef CPURANSAC
#include "estimatRTCPU.h"
#endif
using namespace std;

////////////////////////////////////////////////////////////////////////////////

typedef struct cameraModel {
  float cx,cy,fx,fy;
  float Kdepth[3][3];
  float R_d2c[3][3];
  float T_d2c[3];
} cameraModel;



////////////////////////////////////////////////////////////////////////////////
cv::Mat GetDepthData(const string &file_name);
cv::Mat depth2XYZcamera(cameraModel cam_K,const cv::Mat depth,const float Scale);
void WritePlyFile(const char* plyfile, const cv::Mat pointCloud, const int sample);
void WritePlyFile(const char* plyfile, const cv::Mat pointCloud, const cv::Mat color, const int sample);
void writeMatToFile(const cv::Mat& m, const char* filename);
cv::Mat transformPointCloud(cv::Mat pointsCloud,float T[12]);
void ransacfitRt(const cv::Mat refCoord, const cv::Mat movCoord, float* rigidtransform, 
                 int* numMatches,int numLoops, float thresh);

#endif
