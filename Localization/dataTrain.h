#ifndef DataTrain_H
#define DataTrain_H

#include "RGBD_utils.h"
#include "cudaSift.h"
#include <sys/stat.h>

/********************************** Helper functions **********************************/
void readTraindata(const string dataRoot,const string sequenceName,
                   vector<string> &color_list,vector<string>& depth_list,
                   float* &extrinsic,int* numofframe, cameraModel &cam_K);

int getSift3dPoints(SiftData siftData1,const cv::Mat pointCloud_l,const int imgw);

SiftData computeSift(cv::Mat inputImage);
inline bool exists(const std::string& name) {
  struct stat buffer;   
  return (stat (name.c_str(), &buffer) == 0); 
}
/********************************** Class defination **********************************/
class DataTrain {
  public:
  vector<string> color_list;
  vector<string> depth_list;
  float* extrinsic;
  int numofframe;
  int totalNumofSift;
  cameraModel cameraModel;
  string pathTodata;
  SiftData siftDataTrain;
  int* siftFrameIDs;
  int* numofSiftPerframe;
  // BOWs
  SiftData siftDataCenter;
  int numofcenters;
  cv::Mat BOWfeatureTrain;
  /************ memember functions ***************/
  DataTrain(const string dataRoot,const string sequenceName,const string Test);
  DataTrain(const string,const string); 
  ~DataTrain();
  void preComputeSift(const string filename = "preComputeSift");
  void outputPly(const string filename, int numofsample);
  void loadComputedSift(const string filename = "preComputeSift");
  void outputKeyPoint(const string filename);
  
  void TrainBOW();
  void preComputeBOW();
  void loadBOW();
  cv::Mat findNearFrameBOW(const cv::Mat &BOWTest);
};
#endif

