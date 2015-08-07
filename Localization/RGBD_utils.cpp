#include "RGBD_utils.h"
////////////////////////////////////////////////////////////////////////////////
cv::Mat GetDepthData(const string &file_name) {
  cv::Mat img = cv::imread(file_name,cv::IMREAD_ANYDEPTH);//CV_16UC1
  cv::Mat depth(img.size().height, img.size().width, cv::DataType<float>::type);
  for (int i = 0; i < img.size().height; ++i) {
    for (int j = 0; j < img.size().width; ++j) {
      unsigned short s = img.at<ushort>(i,j);
      s = (s << 13 | s >> 3);
      float f = (float)s/1000.0;
      depth.at<float>(i,j) = f;
    }
  }
  return depth;
}

void WritePlyFileSeq(const char* plyfile,float* extrinsic,
                     vector<string> *color_list,vector<string> *depth_list){
}

//XYZcam = depth2XYZcamera(data.K, Scale*depthRead(data.depth{frameID},data));
cv::Mat depth2XYZcamera(cameraModel cam_K,const cv::Mat depth,const float Scale){
	cv::Mat pointsCloud(depth.size().height*depth.size().width,4,cv::DataType<float>::type);
	for (int v = 0; v < depth.size().height; ++v) {
	    for (int u = 0; u < depth.size().width; ++u) {
	      float iz = depth.at<float>(v,u);
	      float ix = iz * (u - cam_K.cx) / cam_K.fx;
	      float iy = iz * (v - cam_K.cy) / cam_K.fy;
	      pointsCloud.at<float>(v*depth.size().width+u,0) =ix;
	      pointsCloud.at<float>(v*depth.size().width+u,1) =iy;
	      pointsCloud.at<float>(v*depth.size().width+u,2) =iz;
        pointsCloud.at<float>(v*depth.size().width+u,3) =(float)((iz>0.0001)-0.5);
	    }
	  }
	return pointsCloud;
}

cv::Mat transformPointCloud(cv::Mat pointsCloud,float T[12]){
	cv::Mat pointsCloud_out(pointsCloud.size().height,4,cv::DataType<float>::type);
//for (int i =0; i <3;i++){
//  printf("%f,%f,%f,%f\n",T[i*4+0],T[i*4+1],T[i*4+2],T[i*4+3]);
//}
	for (int v = 0; v < pointsCloud.size().height; ++v) 
  {
		float ix = pointsCloud.at<float>(v,0);
		float iy = pointsCloud.at<float>(v,1);
		float iz = pointsCloud.at<float>(v,2);

		pointsCloud_out.at<float>(v,0) = T[0] * ix + T[1] * iy + T[2] * iz + T[3];
		pointsCloud_out.at<float>(v,1) = T[4] * ix + T[5] * iy + T[6] * iz + T[7];
		pointsCloud_out.at<float>(v,2) = T[8] * ix + T[9] * iy + T[10] * iz + T[11];
    pointsCloud_out.at<float>(v,3) = pointsCloud.at<float>(v,3);
	}
	return pointsCloud_out;
}


void WritePlyFile(const char* plyfile, const cv::Mat pointCloud,const int sample=2){
  FILE *fp = fopen(plyfile,"w");
  int pointCount =0;
  for (int v = 0; v < pointCloud.size().height; v += sample) {
    if (pointCloud.at<float>(v,3)>0.0001){
      pointCount++;
    }
  }

  cout << "Write PLY" << endl;

  fprintf(fp, "ply\n");
  fprintf(fp, "format binary_little_endian 1.0\n");
  fprintf(fp, "element vertex %d\n", pointCount);
  fprintf(fp, "property float x\n");
  fprintf(fp, "property float y\n");
  fprintf(fp, "property float z\n");
  fprintf(fp, "end_header\n");
  for (int v = 0; v < pointCloud.size().height; v += sample) {
      if (pointCloud.at<float>(v,3)>0.0001){
         fwrite(&pointCloud.at<float>(v,0), sizeof(float), 1, fp);
         fwrite(&pointCloud.at<float>(v,1), sizeof(float), 1, fp);
         fwrite(&pointCloud.at<float>(v,2), sizeof(float), 1, fp);      
      }
  }
  fclose(fp);
}



//outputPly(PLYfilename, cameraRtC2W, data,Scale)
void WritePlyFile(const char* plyfile, const cv::Mat pointCloud, const cv::Mat color,const int sample=2){
	FILE *fp = fopen(plyfile,"w");
	int pointCount =0;
	for (int v = 0; v < pointCloud.size().height; ++v) {
		if (pointCloud.at<float>(v,3)>0.0001){
			pointCount++;
		}
	}

	cout << "Write PLY" << endl;

	fprintf(fp, "ply\n");
	fprintf(fp, "format binary_little_endian 1.0\n");
	fprintf(fp, "element vertex %d\n", pointCount);
	fprintf(fp, "property float x\n");
	fprintf(fp, "property float y\n");
	fprintf(fp, "property float z\n");
	fprintf(fp, "property uchar red\n");
	fprintf(fp, "property uchar green\n");
	fprintf(fp, "property uchar blue\n");
	fprintf(fp, "end_header\n");
	for (int v = 0; v < pointCloud.size().height; ++v) {
			if (pointCloud.at<float>(v,3)>0.0001){
			   fwrite(&pointCloud.at<float>(v,0), sizeof(float), 1, fp);
			   fwrite(&pointCloud.at<float>(v,1), sizeof(float), 1, fp);
			   fwrite(&pointCloud.at<float>(v,2), sizeof(float), 1, fp);			
				 int i= (int)v/color.size().width;
				 int j= (int)v%color.size().width;
				 fwrite(&color.at<cv::Vec3b>(i,j)[2], sizeof(uchar), 1, fp);
				 fwrite(&color.at<cv::Vec3b>(i,j)[1], sizeof(uchar), 1, fp);
				 fwrite(&color.at<cv::Vec3b>(i,j)[0], sizeof(uchar), 1, fp);
			}
	}
	fclose(fp);
}

void writeMatToFile(const cv::Mat& m, const char* filename)
{
    ofstream fout(filename);

    if(!fout)
    {
        cout<<"File Not Opened"<<endl;  return;
    }

    for(int i=0; i<m.rows; i++)
    {
        for(int j=0; j<m.cols; j++)
        {
            fout<<m.at<float>(i,j)<<" ";
        }
        fout<<endl;
    }

    fout.close();
}



void ransacfitRt(const cv::Mat refCoord, const cv::Mat movCoord, float* rigidtransform, 
                 int* numMatches,int numLoops, float thresh)
{
    cv::Mat coord(refCoord.size().height, refCoord.size().width+movCoord.size().width, CV_32FC1);
    cv::Mat left(coord, cv::Rect(0, 0, refCoord.size().width, refCoord.size().height));
    refCoord.copyTo(left);
    cv::Mat right(coord, cv::Rect(refCoord.size().width, 0, movCoord.size().width, movCoord.size().height));
    movCoord.copyTo(right);
    float * h_coord = (float*)coord.data;
//writeMatToFile(refCoord, "refCoord.txt"); 
//writeMatToFile(movCoord, "movCoord.txt"); 
//writeMatToFile(coord, "h_coord.txt"); 

    int numValid = refCoord.size().height;
    int randSize = 4*sizeof(int)*numLoops;
    int* h_randPts = (int*)malloc(randSize);

    // generate random samples for each loop 
    for (int i=0;i<numLoops;i++) {
        int p1 = rand() % numValid;
        int p2 = rand() % numValid;
        int p3 = rand() % numValid;
        while (p2==p1) p2 = rand() % numValid;
        while (p3==p1 || p3==p2) p3 = rand() % numValid;
        h_randPts[i+0*numLoops] = p1;
        h_randPts[i+1*numLoops] = p2;
        h_randPts[i+2*numLoops] = p3;
    }
    
    
    int h_count =-1;
    float thresh2 = thresh*thresh;
    
    #ifdef CPURANSAC
    float h_RT[12]; 
    int maxIndex = -1;
    int maxCount = -1;
    for(int idx= 0;idx<numLoops;idx++){
    	
      estimateRigidTransform(h_coord, h_randPts, idx, numLoops, h_RT);  
      TestRigidTransform(h_coord, h_RT, &h_count, numValid, thresh2);
      
      if (h_count>maxCount){
          maxCount = h_count;
          for (int i = 0;i<12;i++){
            rigidtransform[i] = h_RT[i];
          }
      }
    }

    numMatches[0] = maxCount;
    
    #endif

    // gpu ransac;
    gpuRASACfindRT(h_coord,h_randPts,rigidtransform,numMatches,numLoops,numValid,thresh2);
    #ifdef VERBOSE
      printf("numofMatch = %d \n",numMatches[0]);
    #endif
    free(h_randPts);

    return;
}



/*
int main(){
	string depthfile = "/Users/shuran/Documents/SUNrgbd_v2_data/2015-07-14T20.08.19.564/depth/0000000939-2015-07-14T20.08.19.569.png";
	string colorfile = "/Users/shuran/Documents/SUNrgbd_v2_data/2015-07-14T20.08.19.564/color/0000000939-2015-07-14T20.08.19.569.jpg";

	cv::Mat depth = GetDepthData(depthfile);
	cv::Mat color = cv::imread(colorfile);
	//writeMatToFile(depth,"test.txt");
	_cameraModel cam_K;
	cam_K.cx=316.9164;
	cam_K.cy=244.3097;
	cam_K.fx=573;
	cam_K.fy=568;


	cv::Mat pointCloud = depth2XYZcamera(cam_K,depth,1);
	WritePlyFile("test.ply", pointCloud, color);


    return 0;
}
*/
