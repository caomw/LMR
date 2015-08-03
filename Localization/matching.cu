#include "cudaSift.h"
#include "cudautils.h"

//================= Device matching functions =====================//

__global__ void MatchSiftPoints(SiftPoint *sift1, SiftPoint *sift2, float *corrData, int numPts1, int numPts2)
{
  __shared__ float siftPoint[128];
  __shared__ float sums[16*16];
  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  const int p1 = blockIdx.x;
  const int p2 = blockIdx.y*16 + ty;
  const float *ptr1 = sift1[p1].data;
  const float *ptr2 = sift2[p2].data;
  const int i = ty*16 + tx;
  if (ty<8)
    siftPoint[i] = ptr1[i];
  __syncthreads();
  float sum = 0.0f;
  if (p2<numPts2)
    for (int j=0;j<8;j++)
      sum += siftPoint[16*j+tx] * ptr2[16*j+tx];
  sums[i] = sum;
  __syncthreads();
  if (tx<8)
    sums[i] += sums[i+8];
  __syncthreads();
  if (tx<4)
    sums[i] += sums[i+4];
  __syncthreads();
  if (tx<2)
    sums[i] += sums[i+2];
  __syncthreads();
  if (tx<1)
    sums[i] += sums[i+1];
  __syncthreads();
  if (ty==0) {
    corrData[p1*gridDim.y*16 + blockIdx.y*16 + tx] = sums[16*tx];
    //printf("corr = %.2f\n", sums[16*tx]);
  }
  __syncthreads();
}


__global__ void FindMaxCorr(float *corrData, SiftPoint *sift1, SiftPoint *sift2, int numPts1, int corrWidth, int siftSize)
{
  __shared__ float maxScore[16*16];
  __shared__ float maxScor2[16*16];
  __shared__ int maxIndex[16*16];
  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  const int idx = ty*16 + tx;
  int p1 = blockIdx.x*16 + threadIdx.y;
  p1 = (p1>=numPts1 ? numPts1-1 : p1);
  maxScore[idx] = -1.0f;
  maxScor2[idx] = -1.0f;
  maxIndex[idx] = -1;
  __syncthreads();
  float *corrs = &corrData[p1*corrWidth];
  for (int i=tx;i<corrWidth;i+=16) {
    float val = corrs[i];
    if (val>maxScore[idx]) {
      maxScor2[idx] = maxScore[idx];
      maxScore[idx] = val;
      maxIndex[idx] = i;
    } else if (val>maxScor2[idx])
      maxScor2[idx] = val;
  }
  //if (p1==1)
  //  printf("tx = %d, score = %.2f, scor2 = %.2f, index = %d\n", 
  //	   tx, maxScore[idx], maxScor2[idx], maxIndex[idx]);
  __syncthreads();
  for (int len=8;len>0;len/=2) {
    if (tx<8) {
      float val = maxScore[idx+len];
      int i = maxIndex[idx+len];
      if (val>maxScore[idx]) {
	maxScor2[idx] = maxScore[idx];
	maxScore[idx] = val;
	maxIndex[idx] = i;
      } else if (val>maxScor2[idx])
	maxScor2[idx] = val;
      float va2 = maxScor2[idx+len];
      if (va2>maxScor2[idx])
	maxScor2[idx] = va2;
    }
    __syncthreads();
    //if (p1==1 && tx<len) 
    //  printf("tx = %d, score = %.2f, scor2 = %.2f, index = %d\n", 
    //	     tx, maxScore[idx], maxScor2[idx], maxIndex[idx]);
  }
  if (tx==6)
    sift1[p1].score = maxScore[ty*16];
  if (tx==7)
    sift1[p1].ambiguity = maxScor2[ty*16] / (maxScore[ty*16] + 1e-6);
  if (tx==8)
    sift1[p1].match = maxIndex[ty*16];
  if (tx==9)
    sift1[p1].match_xpos = sift2[maxIndex[ty*16]].xpos;
  if (tx==10)
    sift1[p1].match_ypos = sift2[maxIndex[ty*16]].ypos;
  __syncthreads();
  //if (tx==0)
  //  printf("index = %d/%d, score = %.2f, ambiguity = %.2f, match = %d\n", 
  //	p1, numPts1, sift1[p1].score, sift1[p1].ambiguity, sift1[p1].match);
}




double MatchSiftData(SiftData &data1, SiftData &data2)
{
  if (data1.d_data==NULL || data2.d_data==NULL)
    return 0.0f;
  TimerGPU timer(0);

  int numPts1 = data1.numPts;
  int numPts2 = data2.numPts;
//printf("numPts: %d,%d\n",numPts1,numPts2);
  SiftPoint *sift1 = data1.d_data;
  SiftPoint *sift2 = data2.d_data;
  
  float *d_corrData; 
  int corrWidth = iDivUp(numPts2, 16)*16;
  int corrSize = sizeof(float)*numPts1*corrWidth;
  safeCall(cudaMalloc((void **)&d_corrData, corrSize));
  dim3 blocks(numPts1, iDivUp(numPts2, 16));
  dim3 threads(16, 16); // each block: 1 points x 16 points
  MatchSiftPoints<<<blocks, threads>>>(sift1, sift2, d_corrData, numPts1, numPts2);
  safeCall(cudaThreadSynchronize());
  checkMsg("MatchSiftPoints() execution failed\n");
  dim3 blocksMax(iDivUp(numPts1, 16));
  dim3 threadsMax(16, 16);
  FindMaxCorr<<<blocksMax, threadsMax>>>(d_corrData, sift1, sift2, numPts1, corrWidth, sizeof(SiftPoint));
  safeCall(cudaThreadSynchronize());
  checkMsg("FindMaxCorr() execution failed\n");
  safeCall(cudaFree(d_corrData));
  if (data1.h_data!=NULL) {
    float *h_ptr = &data1.h_data[0].score;
    float *d_ptr = &data1.d_data[0].score;
    safeCall(cudaMemcpy2D(h_ptr, sizeof(SiftPoint), d_ptr, sizeof(SiftPoint), 5*sizeof(float), data1.numPts, cudaMemcpyDeviceToHost));
  }

  double gpuTime = timer.read();
  #ifdef VERBOSE
    printf("MatchSiftData time =          %.2f ms\n", gpuTime);
  #endif
  return gpuTime;
}		 

