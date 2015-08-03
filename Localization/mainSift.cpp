//********************************************************//
// CUDA SIFT extractor by Marten Björkman aka Celebrandil //
//              celle @ nada.kth.se                       //
//********************************************************//  

//#include <iomanip>
#include "cudaImage.h"
#include "cudaSift.h"
#include "RGBD_utils.h"
#include "dataTrain.h"
#include <ctime>




//int ImproveHomography(SiftData &data, float *homography, int numLoops, float minScore, float maxAmbiguity, float thresh);
cv::Mat PrintMatchData(SiftData &siftData1, SiftData &siftData2, cv::Mat limg, cv::Mat rimg)
{
  int numPts = siftData1.numPts;
  SiftPoint *sift1 = siftData1.h_data;
  cv::Mat im3(limg.size().height, limg.size().width+rimg.size().width, CV_32FC1);
  cv::Mat left(im3, cv::Rect(0, 0, limg.size().width, limg.size().height));
  limg.copyTo(left);
  cv::Mat right(im3, cv::Rect(limg.size().width, 0, rimg.size().width, rimg.size().height));
  rimg.copyTo(right);
  
  int w = limg.size().width+rimg.size().width;
  for (int j=0;j<numPts;j++) { 
    if (sift1[j].valid==1) {
      float dx = sift1[j].match_xpos+limg.size().width - sift1[j].xpos;
      float dy = sift1[j].match_ypos - sift1[j].ypos;
      int len = (int)(fabs(dx)>fabs(dy) ? fabs(dx) : fabs(dy));
      for (int l=0;l<len;l++) {
            int x = (int)(sift1[j].xpos + dx*l/len);
            int y = (int)(sift1[j].ypos + dy*l/len);
            im3.at<float>(y,x) = 255.0f;

        } 
    }
  }
  //std::cout << std::setprecision(6);
  return im3;
}
void PrintMatchSiftData(SiftData &siftData1, const char* filename,int imgw){
  ofstream fout(filename);
  if(!fout)
    {
        cout<<"File Not Opened"<<endl;  return;
    }
    SiftPoint *sift1 = siftData1.h_data;
    for(int i=0; i<siftData1.numPts; i++)
    {
      if (sift1[i].valid){
        int ind  = ((int)sift1[i].xpos+(int)sift1[i].ypos*imgw);
        int ind2 = ((int)sift1[i].match_xpos+(int)sift1[i].match_ypos*imgw);

        fout<<sift1[i].xpos<<"\t"<<sift1[i].ypos<<"\t";
        fout<<sift1[i].match_xpos<<"\t"<<sift1[i].match_ypos<<"\t";
        fout<<ind<<"\t"<<ind2<<"\t";
        fout<<endl;
      }
        
    }

    fout.close();

 }


int getSift3dPointsMatch(const SiftData siftDataTest,const SiftData siftDataTrain, cv::Mat* refCoord, cv::Mat* movCoord,int imgw,
                         float minScore = 0.75f, float maxAmbiguity = 0.95f)
{
    int numTomatchedsift =0;
    SiftPoint *sift1 = siftDataTest.h_data;
    for (int i = 0;i<siftDataTest.numPts;i++){
        int ind  = ((int)sift1[i].xpos+(int)sift1[i].ypos*imgw);
        int matchInd = sift1[i].match;
        //int ind2 = ((int)sift1[i].match_xpos+(int)sift1[i].match_ypos*imgw);
        
        if (sift1[i].score>minScore&&sift1[i].valid>0)
        {    
            cv::Mat onerow = cv::Mat(1, 3, CV_32FC1, siftDataTrain.h_data[matchInd].point3d);
            refCoord->push_back(onerow);
            onerow.at<float>(0,0) = sift1[i].point3d[0];
            onerow.at<float>(0,1) = sift1[i].point3d[1];
            onerow.at<float>(0,2) = sift1[i].point3d[2];
            movCoord->push_back(onerow);
            //std::cout<<"pointCloudTest"<<pointCloudTest.at<float>(ind,0)<<","<<pointCloudTest.at<float>(ind,1)<<","<<pointCloudTest.at<float>(ind,2)<<","<<std::endl;
            //std::cout<<"movCoord"<<movCoord->at<float>(numTomatchedsift,0)<<","<<movCoord->at<float>(numTomatchedsift,1)<<","<<movCoord->at<float>(numTomatchedsift,2)<<","<<std::endl;
            //std::cout<<"refCoord"<<refCoord->at<float>(numTomatchedsift,0)<<","<<refCoord->at<float>(numTomatchedsift,1)<<","<<refCoord->at<float>(numTomatchedsift,2)<<","<<std::endl;
            sift1[i].valid =1;
            numTomatchedsift++;
        }
        else{
            sift1[i].valid =-1;
        }
    } 
    //std::cout<<"lengthrefCoord"<<refCoord->size().height<<std::endl;
    return numTomatchedsift;
}


cv::Mat ComputeBOWfea(SiftData siftDataCenter,SiftData siftDataTest){
  MatchSiftData(siftDataTest,siftDataCenter);
  cv::Mat BOWfea = cv::Mat::zeros(1,siftDataCenter.numPts,cv::DataType<float>::type);
  int validnum=0;
  for (int i =0;i<siftDataTest.numPts;i++){
    if (siftDataTest.h_data[i].valid>0){
       BOWfea.at<float>(0,siftDataTest.h_data[i].match) = BOWfea.at<float>(0,siftDataTest.h_data[i].match)+1;
       validnum++;
    }
    
  }
  //BOWfea = BOWfea/(siftDataTest.numPts+1);
  //cv::Mat BOWTest_tranpo;
  //cout<<BOWTest_tranpo<<endl;
  BOWfea = BOWfea/(validnum+1);
  return BOWfea;
}



///////////////////////////////////////////////////////////////////////////////
// Main program
///////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv) 
{     
  
  InitCuda();
  // Read images using OpenCV
  string dataRoot ="/Users/shuran/Documents/";
  string sequenceName ="third_floor_tearoom/";
  DataTrain dataTrain(dataRoot,sequenceName);
  if (!exists(dataTrain.pathTodata+"preComputeSift")||!exists((dataTrain.pathTodata+"preComputeSiftNum"))){
      dataTrain.preComputeSift();
  }
  dataTrain.loadComputedSift();
  dataTrain.preComputeBOW();
  //dataTrain.outputKeyPoint("../result/keypoint.ply");
 
  std::cout<<"------------------FINISH INITIALING------------------"<<std::endl;
  
  //start to align new view 
  string rimg_file = "/Users/shuran/Documents/third_floor_tearoom/data/spot_0001/step_0108/device1/image/0024569-1427763731008041.tif";
  string rdepth_file ="/Users/shuran/Documents/third_floor_tearoom/data/spot_0001/step_0108/device1/depthMedian/0014611-1427763730547306.tif";
  //string rimg_file = dataTrain.color_list[0];
  //string rdepth_file = dataTrain.depth_list[0];
  cv::Mat rdepth = GetDepthData(rdepth_file);

  cv::Mat rimg = cv::imread(rimg_file, 0);  
  cv::Mat pointCloud_Test = depth2XYZcamera(dataTrain.cameraModel,rdepth,1);
  SiftData siftDataTrain = dataTrain.siftDataTrain;
  SiftData siftDataTest = computeSift(rimg);
  getSift3dPoints(siftDataTest,pointCloud_Test,rdepth.size().width);
  
  // Find the colosest frames using BOWs
  clock_t startTime = clock();
  cv::Mat BOWTest = ComputeBOWfea(dataTrain.siftDataCenter,siftDataTest);
  cv::Mat  dst = dataTrain.findNearFrameBOW(BOWTest);
  clock_t endTime = clock();
  double timeInmSeconds = (endTime-startTime)*1000 / (double) CLOCKS_PER_SEC;
  printf("ComputeBOWfea + findNearFrameBOW =  %.2f ms\n", timeInmSeconds);

  // TODO: construct siftDataToMatch
  // DEVICETODEVICE copy 
  for (int i =0;i<3;i++){
      cout<<dst.at<int>(i)<<":"<<dataTrain.color_list[dst.at<int>(i)]<<endl;

  }
  
  // Match Sift features 
  MatchSiftData(siftDataTest,siftDataTrain);

  // get the coordinate of matched points
  cv::Mat refCoord(0,3,cv::DataType<float>::type);
  cv::Mat movCoord(0,3,cv::DataType<float>::type);
  int imgw = rimg.size().width;
  int imgh = rimg.size().height;
  int numTomatchedsift =  getSift3dPointsMatch(siftDataTest,siftDataTrain,&refCoord,&movCoord,imgw);

  std::cout << "numTomatchedsift:" <<  numTomatchedsift << std::endl;  

  int numMatches[1];
  float rigidtrans[12];
  int numLoops = 1280;
  
  ransacfitRt(refCoord,movCoord,rigidtrans, numMatches,numLoops,0.1);


  cv::Mat pointCloud_Test_t = transformPointCloud(pointCloud_Test,rigidtrans);
  cv::Mat color = cv::imread(rimg_file);
  WritePlyFile("../result/mov.ply",pointCloud_Test,color);
  WritePlyFile("../result/mov_afteralign.ply",pointCloud_Test_t,color);

  FreeSiftData(siftDataTest);
  return 1;

  //clock_t startTime = clock();
  //WritePlyFile("../result/refCoord.ply", refCoord);
  //WritePlyFile("../result/movCoord.ply", movCoord);
  //clock_t endTime = clock();
  //double timeInSeconds = (endTime-startTime)*1000 / (double) CLOCKS_PER_SEC;
  //printf("ransac CPU time =         %.2f ms\n", timeInSeconds);


  /*
  limg.convertTo(limg, CV_32FC1);
  rimg.convertTo(rimg, CV_32FC1);
  cv::Mat imRresult = PrintMatchData(siftDataTest, siftDataTrain,rimg,limg);
  printf("write image\n");
  cv::imwrite("../result/imRresult_beforeransac.jpg", imRresult);
  imRresult.release();  
  color = cv::imread(limg_file);
  pointCloud_Train = transformPointCloud(pointCloud_Train,&dataTrain.extrinsic[12*frame_i]);
  WritePlyFile("../result/ref.ply",pointCloud_Train,color);
  */

  
  // Print out and store summary data

  //std::cout << "Number of original features: " <<  siftData1.numPts << " " << siftData2.numPts << std::endl;
  //std::cout << "Number of matching features: " << numFit << " " << numMatches << " " << 100.0f*numMatches/std::min(siftData1.numPts, siftData2.numPts) << "%" << std::endl;
  
  // Free Sift data from device
}

/*
int getMatch3dPoints(const SiftData siftData1,const cv::Mat pointCloud_l, const cv::Mat pointCloud_r,
                    cv::Mat* refCoord, cv::Mat* movCoord,int imgw,int imgh,
                    float minScore = 0.75f, float maxAmbiguity = 0.95f)
{
    int numTomatchedsift =0;
    SiftPoint *sift1 = siftData1.h_data;
    for (int i = 0;i<siftData1.numPts;i++){
        int ind  = ((int)sift1[i].xpos+(int)sift1[i].ypos*imgw);
        int ind2 = ((int)sift1[i].match_xpos+(int)sift1[i].match_ypos*imgw);
        
        if (sift1[i].ambiguity<maxAmbiguity&&sift1[i].score>minScore&&pointCloud_l.at<float>(ind,2)>0&&pointCloud_r.at<float>(ind2,2)>0)
        {    

            //std::cout <<sift1[i].xpos<<"\t"<<sift1[i].ypos<<"\t"<<ind<<"\t"<<imgw<<std::endl;
            // std::cout<<pointCloud_l.at<float>(ind,2)<<","<<pointCloud_r.at<float>(ind2,2)<<","<<sift1[i].ambiguity<<","<<sift1[i].score<<std::endl;  
            refCoord->push_back(pointCloud_l.row(ind));
            movCoord->push_back(pointCloud_r.row(ind2));
            sift1[i].valid =1;
            numTomatchedsift++;
        }
    } 
    return numTomatchedsift;
}
*/

