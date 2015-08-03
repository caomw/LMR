#ifndef CUDASIFT_H
#define CUDASIFT_H

#include "cudaImage.h"

typedef struct {
  float xpos;
  float ypos;   
  float scale;
  float sharpness;
  float edgeness;
  float orientation;
  float score;
  float ambiguity;
  int match;// k = sift1[j].match;  -> sift2[k]
  float match_xpos;//sift1[p1].match_xpos = sift2[maxIndex[ty*16]].xpos;
  float match_ypos;//sift1[p1].match_ypos = sift2[maxIndex[ty*16]].ypos;
  float match_error;
  float point3d[4]; // 3d point loc 
  float data[128];
  int valid;
  //float point3d[3];
} SiftPoint;

typedef struct {
  int numPts;         // Number of available Sift points
  int maxPts;         // Number of allocated Sift points
  SiftPoint *h_data;  // Host (CPU) data
  SiftPoint *d_data;  // Device (GPU) data
} SiftData;

void InitCuda();
double ExtractSift(SiftData &siftData, CudaImage &img, int numOctaves, double initBlur, float thresh, float lowestScale = 0.0f, float subsampling = 1.0f);
void InitSiftData(SiftData &data, int num = 1024, bool host = false, bool dev = true);
void FreeSiftData(SiftData &data);
void PrintSiftData(SiftData &data);
double MatchSiftData(SiftData &data1, SiftData &data2);
double findRigidTransform(SiftData &data,  float *T,  int *numMatches, int numLoops = 1000, float thresh = 0.05f);
void gpuRASACfindRT(const float * h_coord, int* h_randPts,float * bestRT,int * inlier_count,
                    int numLoops,int numValid,float thresh2);

#endif
