#include "dataTrain.h"
#include <cudautils.h>
#include "opencv2/features2d.hpp"



DataTrain::DataTrain(const string dataRoot,const string sequenceName){
  pathTodata = dataRoot+"/"+sequenceName+"/";
  siftFrameIDs =NULL;
  numofSiftPerframe =NULL;
  readTraindata(dataRoot,sequenceName,color_list,depth_list,extrinsic,&numofframe,cameraModel);
  this->numofcenters=200;
  this->BOWfeatureTrain= cv::Mat::zeros(numofframe,numofcenters,cv::DataType<float>::type);
  
}
DataTrain::~DataTrain(){
    FreeSiftData(siftDataTrain);
    FreeSiftData(siftDataCenter);
    if (extrinsic!=NULL){
       free(extrinsic);
    }
    if (siftFrameIDs!=NULL){
       free(siftFrameIDs);
    } 
    if (numofSiftPerframe!=NULL){
       free(numofSiftPerframe);
    } 
    BOWfeatureTrain.release();
};
void DataTrain::outputKeyPoint(const string filename){
    FILE *fp = fopen(filename.c_str(),"w");
    fprintf(fp, "ply\n");
    fprintf(fp, "format binary_little_endian 1.0\n");
    fprintf(fp, "element vertex %d\n", totalNumofSift);
    fprintf(fp, "property float x\n");
    fprintf(fp, "property float y\n");
    fprintf(fp, "property float z\n");
    fprintf(fp, "end_header\n");
    for (int i =0;i<totalNumofSift;i++){
        fwrite(siftDataTrain.h_data[i].point3d, sizeof(float), 3, fp);
        
    }
    fclose(fp);
}
void DataTrain::TrainBOW(){
    //Construct BOWKMeansTrainer
    //To store all the descriptors that are extracted from all the images.
    cv::Mat featuresUnclustered;
    int sample =10;
    int i =0;
    while (i<siftDataTrain.numPts){
         cv::Mat descriptor = cv::Mat(1, 128, CV_32FC1, siftDataTrain.h_data[i].data);
         featuresUnclustered.push_back(descriptor);  
         i=i+sample;
    }
    printf("Start clustering with %d of sift\n",featuresUnclustered.size().height);
    //define Term Criteria
    cv::TermCriteria tc(CV_TERMCRIT_ITER,100,0.001);

    //retries number
    int retries=1;
    //necessary flags
    int flags=cv::KMEANS_PP_CENTERS;
    //Create the BoW (or BoF) trainer   
    cv::BOWKMeansTrainer bowTrainer(this->numofcenters,tc,retries,flags);

    //cluster the feature vectors
    cv::Mat dictionary=bowTrainer.cluster(featuresUnclustered);   
    printf("End clustering with %d of sift\n",featuresUnclustered.size().height);
    
    //store the vocabulary
    cv::FileStorage fs(this->pathTodata+"dictionary.yml", cv::FileStorage::WRITE);
    fs << "vocabulary" << dictionary;
    fs.release();

}


void DataTrain::preComputeBOW(){
    if (!exists(this->pathTodata+"dictionary.yml")){
      this->TrainBOW();
    }
    cv::Mat dictionary;
    cv::FileStorage fs(this->pathTodata+"dictionary.yml", cv::FileStorage::READ);
    fs["vocabulary"] >> dictionary;
    fs.release();  
    InitSiftData(siftDataCenter, numofcenters, true, true);
    siftDataCenter.numPts = numofcenters;
    for (int c_id =0; c_id < numofcenters; c_id++)
    {
      //float* ptr = dictionary.ptr<float>(c_id);
      //for (int i =0;i<128;i++){
      memcpy(siftDataCenter.h_data[c_id].data,dictionary.ptr<float>(c_id),sizeof(float)*128);
      //}
      
    }
    safeCall(cudaMemcpy(siftDataCenter.d_data, siftDataCenter.h_data, 
             sizeof(SiftPoint)*siftDataCenter.numPts, cudaMemcpyHostToDevice));
  

    if (0){//(exists(this->pathTodata+"BOWTrain")){
       printf("loading BOWTrain ...\n");
       FILE * pFile = fopen((this->pathTodata+"BOWTrain").c_str(),"wb");
       fread(BOWfeatureTrain.data, sizeof(float),numofframe*numofcenters,pFile);
       fclose(pFile);
    }
    else{
      printf("computing BOWTrain ...\n");
      MatchSiftData(this->siftDataTrain,this->siftDataCenter);
      for (int i =0;i<siftDataTrain.numPts;i++){
          BOWfeatureTrain.at<float>(siftFrameIDs[i],this->siftDataTrain.h_data[i].match) =
          BOWfeatureTrain.at<float>(siftFrameIDs[i],this->siftDataTrain.h_data[i].match)+1;
      }
      cv::Mat BOWfeatureTrain_summed;
      cv::reduce(BOWfeatureTrain, BOWfeatureTrain_summed, 1, CV_REDUCE_SUM, cv::DataType<float>::type);
/*
FILE * ss = fopen("siftFrameIDs.bin","wb");
fwrite(siftFrameIDs, sizeof(int),totalNumofSift,ss);
fclose(ss);      
writeMatToFile(BOWfeatureTrain, "BOWfeatureTrain.txt");
writeMatToFile(BOWfeatureTrain_summed, "BOWfeatureTrain_summed.txt");
*/
//cout<<BOWfeatureTrain.row(0)<<endl;
      //cout <<"BOWfeatureTrain_summed"<<BOWfeatureTrain_summed.size().height<<","<<BOWfeatureTrain_summed.size().width<<endl;
      for (int i =0;i<BOWfeatureTrain.size().height;i++){
          for (int j = 0;j<BOWfeatureTrain.size().width;j++){
            BOWfeatureTrain.at<float>(i,j) = BOWfeatureTrain.at<float>(i,j)/(BOWfeatureTrain_summed.at<float>(i)+1);
          }
          
      }
      FILE * pFile = fopen((this->pathTodata+"BOWTrain").c_str(),"wb");
      fwrite(BOWfeatureTrain.data, sizeof(float),numofframe*numofcenters,pFile);
      fclose(pFile);
    }
}
cv::Mat DataTrain::findNearFrameBOW(const cv::Mat &BOWTest){
        cv::Mat distance = cv::Mat(this->numofframe,1,cv::DataType<float>::type);
        for (int i =0;i<this->numofframe;i++){
            cv::Mat subtrans = (this->BOWfeatureTrain.row(i) - BOWTest);
            float sum = 0;
            for (int j = 0;j<this->numofcenters;j++){
               sum += subtrans.at<float>(j) *subtrans.at<float>(j);
            }
           distance.at<float>(i) = sum;
        }
        
       cv::Mat dst;
       cv::sortIdx(distance, dst, CV_SORT_EVERY_COLUMN + CV_SORT_ASCENDING);
       return dst;
}

void DataTrain::outputPly(const string filename, int numofsample=10000){
  int numofPtsperImage = floor(numofsample/numofframe);
  numofsample = numofPtsperImage*numofframe;
  for (int frame_id =0; frame_id < this->numofframe; frame_id++){
    cv::Mat limg = cv::imread(this->color_list[frame_id]);
    cv::Mat ldepth = GetDepthData(this->depth_list[frame_id]);
    cv::Mat pointCloud_l = depth2XYZcamera(this->cameraModel,ldepth,1);
    pointCloud_l = transformPointCloud(pointCloud_l,&(this->extrinsic[12*frame_id]));
    char buffer [50];
    sprintf (buffer, "%s%d.ply", filename.c_str(), frame_id);
    WritePlyFile(buffer, pointCloud_l, limg);
  }
}

void DataTrain::preComputeSift(const string filename)
{
  // for each Train image compute and save sift
  this->totalNumofSift = 0;
  FILE * pFile = fopen((this->pathTodata+filename).c_str(),"wb");
  for (int frame_id =0; frame_id < this->numofframe; frame_id++){
    // get sift

    cv::Mat limg = cv::imread(this->color_list[frame_id], 0);
    SiftData siftData1 = computeSift(limg);
    cv::Mat ldepth = GetDepthData(this->depth_list[frame_id]);
    // get points and rotate the point to world cordinate
    cv::Mat pointCloud_l = depth2XYZcamera(this->cameraModel,ldepth,1);
    pointCloud_l = transformPointCloud(pointCloud_l,&(this->extrinsic[12*frame_id]));
    int numofvalid = getSift3dPoints(siftData1,pointCloud_l,ldepth.size().width);
//cout<<this->color_list[frame_id]<<endl;
    cout<<"numofvalid:"<<numofvalid<<endl;

    SiftPoint *sift1 = siftData1.h_data;
    for (int i = 0;i<siftData1.numPts;i++){
      if ( sift1[i].valid >0){
        fwrite(&sift1[i].xpos,sizeof(float),1,pFile);
        fwrite(&sift1[i].ypos,sizeof(float),1,pFile);
        fwrite(&frame_id,sizeof(int),1,pFile);
        fwrite(sift1[i].data,sizeof(float),128,pFile);
        fwrite(sift1[i].point3d,sizeof(float),3,pFile);
        this->totalNumofSift++;
      }
    }
    FreeSiftData(siftData1);
  }

  fclose(pFile);
  FILE * pFile2 = fopen((this->pathTodata+filename+"Num").c_str(),"wb");
  fwrite(&this->totalNumofSift,sizeof(int),1,pFile2);
  fclose(pFile2);
  printf("totalNumofSift: %d\n",this->totalNumofSift);
}

void DataTrain::loadComputedSift(const string filename)
{
  FILE * pFile2 = fopen((this->pathTodata+filename+"Num").c_str(),"rb");
  fread(&totalNumofSift,sizeof(int),1,pFile2);
  
  InitSiftData(siftDataTrain, totalNumofSift, true, true);
  siftDataTrain.maxPts = siftDataTrain.numPts = totalNumofSift;  
  this->siftFrameIDs = (int *)malloc(sizeof(int)*totalNumofSift);

  this->numofSiftPerframe = (int *)malloc(sizeof(int)*this->numofframe);
  for (int i = 1;i<this->numofframe;i++){
      numofSiftPerframe[i]=0;
  }
  // load all sift of 
  FILE * pFile = fopen((this->pathTodata+filename).c_str(),"rb");
  for (int sift_id =0; sift_id < this->totalNumofSift; sift_id++)
  {
     fread(&siftDataTrain.h_data[sift_id].xpos,sizeof(float),1,pFile);
     fread(&siftDataTrain.h_data[sift_id].ypos,sizeof(float),1,pFile);
     fread(&siftFrameIDs[sift_id], sizeof(int), 1, pFile);
     fread(siftDataTrain.h_data[sift_id].data,sizeof(float),128,pFile);
     fread(siftDataTrain.h_data[sift_id].point3d,sizeof(float),3,pFile);
     numofSiftPerframe[siftFrameIDs[sift_id]] = numofSiftPerframe[siftFrameIDs[sift_id]]+1;
  }
  
  

  safeCall(cudaMemcpy(siftDataTrain.d_data, siftDataTrain.h_data, sizeof(SiftPoint)*siftDataTrain.numPts, cudaMemcpyHostToDevice));
  
  printf("totalNumofSift: %d size:%lu\n ",siftDataTrain.numPts,sizeof(SiftPoint)*siftDataTrain.numPts);
  return;
}


SiftData computeSift(cv::Mat inputImage){
  //InitCuda();
  inputImage.convertTo(inputImage, CV_32FC1);
  unsigned int w = inputImage.cols;
  unsigned int h = inputImage.rows;
  //std::cout << "Image size = (" << w << "," << h << ")" << std::endl;
  cv::GaussianBlur(inputImage, inputImage, cv::Size(5,5), 1.0);
  
  //std::cout << "Initializing data..." << std::endl;
  CudaImage img1;
  img1.Allocate(w, h, iAlignUp(w, 128), false, NULL, (float*)inputImage.data);
  //(int w, int h, int p, bool host, float *devmem, float *hostmem) 
  img1.Download();
  SiftData siftData1;
  InitSiftData(siftData1, 2048, true, true);
  float initBlur = 0.0f;
  float thresh = 4.0f; 
  double timesift1 = ExtractSift(siftData1, img1, 5, initBlur, thresh, 0.0f);
  std::cout << "Extract sift time: " <<  timesift1 <<"ms"<< std::endl;
  std::cout << "Number of original features: " <<  siftData1.numPts << std::endl;
  return siftData1;
}


void readTraindata(const string dataRoot,const string sequenceName,
                   vector<string> &color_list,vector<string>& depth_list,
                   float* &extrinsic,int* numofframe, cameraModel &cam_K)
{
    // color
    string listfile_color = dataRoot+sequenceName+"colorTrain.txt";
    string line;
    ifstream file_color (listfile_color);
    if (file_color.is_open()){
      getline(file_color,line);
      *numofframe = atoi( line.c_str() );
      while(getline(file_color,line)){
           color_list.push_back(line);
      }
      file_color.close();
    }
    else cout << "Unable to open file: "<< listfile_color;
    
    
    //depth
    string listfile_depth = dataRoot+sequenceName+"depthTrain.txt";
    ifstream  myfile(listfile_depth);
    if (myfile.is_open()){
      getline(myfile,line);
      while(getline(myfile,line)){
            depth_list.push_back(line);
      }
      myfile.close();
    }
    else cout << "Unable to open file: "<< listfile_color;

    // extrinsic
    string extrinsic_file = dataRoot+sequenceName+"extrinsics.txt";
    fstream ex_myfile(extrinsic_file);
    if (ex_myfile.is_open()){
      extrinsic = (float*) malloc (sizeof(float)*numofframe[0]*12);
      for (int i=0;i<12*numofframe[0];i++){
           ex_myfile>>extrinsic[i];
      }
      ex_myfile.close();
    }

    string cam_file = dataRoot+sequenceName+"/intrinsics.txt";
    ifstream  cam_myfile(cam_file);
    float tmp;
    if (cam_myfile.is_open()){
        cam_myfile>>cam_K.fx;
        cam_myfile>>tmp;
        cam_myfile>>cam_K.cx;
        cam_myfile>>tmp;
        cam_myfile>>cam_K.fy;
        cam_myfile>>cam_K.cy;
        cam_myfile.close();
        //cout << cam_K.fx <<","<< cam_K.fy <<","<<cam_K.cx<<","<<cam_K.cy<<endl;
    }
    else cout << "Unable to open file: "<< cam_file;
    return;
}

int getSift3dPoints(SiftData siftData1,const cv::Mat pointCloud_l,const int imgw)
{
    int numTomatchedsift =0;
    SiftPoint *sift1 = siftData1.h_data;
    for (int i = 0;i<siftData1.numPts;i++){
        int ind  = ((int)sift1[i].xpos+(int)sift1[i].ypos*imgw);
        const float *ptr = (float*)pointCloud_l.ptr(ind);  

        if (ptr[3]>0.0001) {  
            numTomatchedsift++;
            sift1[i].valid =1;
            sift1[i].point3d[0] = ptr[0];
            sift1[i].point3d[1] = ptr[1];
            sift1[i].point3d[2] = ptr[2];
        }
        else{
            sift1[i].valid =-1;
        }
    } 
    return numTomatchedsift;
}