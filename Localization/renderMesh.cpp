/*
% install lOSMesa
% compile
% mex WarpMesh.cpp -lGLU -lOSMesa
% or
% mex WarpMesh.cpp -lGLU -lOSMesa -I/media/Data/usr/Mesa-9.1.2/include
% on mac:
% mex WarpMesh.cpp -lGLU -lOSMesa -I/opt/X11/include/ -L/opt/X11/lib/
*/
// g++  renderMesh.cpp RGBD_utils.cpp -lGLU -lOSMesa -I/opt/X11/include/ -L/opt/X11/lib/

#include <string.h>
//#include <GL/glu.h>
//#include <GL/osmesa.h>
#include <GL/osmesa.h>
#include <GL/glu.h>
#include "RGBD_utils.h"
#include "renderMesh.h"

#define Square(x) ((x)*(x))

unsigned int uchar2uint(unsigned char* in){
  unsigned int out = (((unsigned int)(in[0])) << 16) + (((unsigned int)(in[1])) << 8) + ((unsigned int)(in[2]));
  return out;
}


void uint2uchar(unsigned int in, unsigned char* out){
  out[0] = (in & 0x00ff0000) >> 16;
  out[1] = (in & 0x0000ff00) >> 8;
  out[2] =  in & 0x000000ff;
  
  //printf("%d=>[%d,%d,%d]=>%d\n",in,out[0],out[1],out[2], uchar2uint(out));
}

Mesh::Mesh(string filename){
     FILE * pFile = fopen ( filename.c_str() , "rb" );
     fread(&(this->numVertex),sizeof(int),1,pFile);
     fread(&(this->numFace),sizeof(int),1,pFile);
//cout<<numVertex<<endl;
//cout<<numFace<<endl;

     this->vertex = (float*)malloc(numVertex*3*sizeof(float));
     this->annolabel = (unsigned int*)malloc(numVertex*sizeof(unsigned int));
     this->faces = (unsigned int*)malloc(3*numFace*sizeof(unsigned int));

     fread(this->vertex,sizeof(float),3*numVertex,pFile);
     fread(this->annolabel,sizeof(unsigned int),numVertex,pFile);
     fread(this->faces,sizeof(unsigned int),3*numFace,pFile);
//cout<<vertex[3*(numVertex-1)]<<","<<vertex[3*(numVertex-1)+1]<<","<<vertex[3*(numVertex-1)+2]<<endl;
//cout<<annolabel[0]<<","<<annolabel[1]<<","<<annolabel[2]<<endl;
//cout<<faces[3*327934]<<","<<faces[3*327934+1]<<","<<faces[3*327934+2]<<endl;
     fclose(pFile);
}
Mesh::~Mesh(){
  free(this->vertex);
  free(this->annolabel);
  free(this->faces);
}

// Input: 
//     arg0: 3x4 Projection matrix, 
//     arg1: image width, 
//     arg2: image height, 
//     arg3: width*height*4 double matrix, 
// Output: you will need to transpose the result in Matlab manually. see the demo

void Mesh::renderMesh(const float* projection, int m_width, int m_height, 
                      unsigned int* result, float* depth_render) {
  //printf("renderMesh\n"); 

  float m_near = 0.3;
  float m_far = 1e8;
  int m_level = 0;
  
  double dis_threshold_square = Square(0.2);
  
  //printf("output size:\nm_width=%d\nm_height=%d\n", m_width,m_height);


  // Step 1: setup off-screen binding 
  OSMesaContext ctx;
  ctx = OSMesaCreateContextExt(OSMESA_BGR, 32, 0, 0, NULL ); // strange hack not sure why it is not OSMESA_RGB
  unsigned char * pbuffer = new unsigned char [3 * m_width * m_height];
  // Bind the buffer to the context and make it current
  if (!OSMesaMakeCurrent(ctx, (void*)pbuffer, GL_UNSIGNED_BYTE, m_width, m_height)) {
     printf("OSMesaMakeCurrent failed!: ");
     return;
  }
  OSMesaPixelStore(OSMESA_Y_UP, 0);

  
  // Step 2: Setup basic OpenGL setting
  glEnable(GL_DEPTH_TEST);
  glDisable(GL_LIGHTING);
  glEnable(GL_CULL_FACE);
  glCullFace(GL_BACK);
  glPolygonMode(GL_FRONT, GL_FILL);
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
  //glClearColor(m_clearColor[0], m_clearColor[1], m_clearColor[2], 1.0f); // this line seems useless
  glViewport(0, 0, m_width, m_height);

  // Step 3: Set projection matrices
  float scale = (0x0001) << m_level;
  float final_matrix[16];

  // new way: faster way by reuse computation and symbolic derive. See sym_derive.m to check the math.
  float inv_width_scale  = 1.0/(m_width*scale);
  float inv_height_scale = 1.0/(m_height*scale);
  float inv_width_scale_1 =inv_width_scale - 1.0;
  float inv_height_scale_1_s = -(inv_height_scale - 1.0);
  float inv_width_scale_2 = inv_width_scale*2.0;
  float inv_height_scale_2_s = -inv_height_scale*2.0;
  float m_far_a_m_near = m_far + m_near;
  float m_far_s_m_near = m_far - m_near;
  float m_far_d_m_near = m_far_a_m_near/m_far_s_m_near;
  final_matrix[ 0]= projection[2+0*3]*inv_width_scale_1 + projection[0+0*3]*inv_width_scale_2;
  final_matrix[ 1]= projection[2+0*3]*inv_height_scale_1_s + projection[1+0*3]*inv_height_scale_2_s;
  final_matrix[ 2]= projection[2+0*3]*m_far_d_m_near;
  final_matrix[ 3]= projection[2+0*3];
  final_matrix[ 4]= projection[2+1*3]*inv_width_scale_1 + projection[0+1*3]*inv_width_scale_2;
  final_matrix[ 5]= projection[2+1*3]*inv_height_scale_1_s + projection[1+1*3]*inv_height_scale_2_s; 
  final_matrix[ 6]= projection[2+1*3]*m_far_d_m_near;    
  final_matrix[ 7]= projection[2+1*3];
  final_matrix[ 8]= projection[2+2*3]*inv_width_scale_1 + projection[0+2*3]*inv_width_scale_2; 
  final_matrix[ 9]= projection[2+2*3]*inv_height_scale_1_s + projection[1+2*3]*inv_height_scale_2_s;
  final_matrix[10]= projection[2+2*3]*m_far_d_m_near;
  final_matrix[11]= projection[2+2*3];
  final_matrix[12]= projection[2+3*3]*inv_width_scale_1 + projection[0+3*3]*inv_width_scale_2;
  final_matrix[13]= projection[2+3*3]*inv_height_scale_1_s + projection[1+3*3]*inv_height_scale_2_s;  
  final_matrix[14]= projection[2+3*3]*m_far_d_m_near - (2*m_far*m_near)/m_far_s_m_near;
  final_matrix[15]= projection[2+3*3];
  
  // matrix is ready. use it
  glMatrixMode(GL_PROJECTION);
  glLoadMatrixf(final_matrix);
  glMatrixMode(GL_MODELVIEW);
  glLoadIdentity();

  // Step 3: render the mesh with encoded color from their ID
  unsigned char colorBytes[3];
  
  for (int face_id =0; face_id<this->numFace; face_id++){

      unsigned int l00 = annolabel[faces[face_id*3+0]];
      unsigned int l01 = annolabel[faces[face_id*3+1]];
      unsigned int l10 = annolabel[faces[face_id*3+2]];
      float* v0 = &vertex[3*faces[face_id*3+0]+0]; 
      float* v1 = &vertex[3*faces[face_id*3+1]+0]; 
      float* v2 = &vertex[3*faces[face_id*3+2]+0]; 
      if (l00==0){
          l00 = (l01==0?l10:l01);
      }
      if (l00!=l10&&l10==l01){
          l00 = l10;
      }

      if(l00!=0){
        uint2uchar(l00,colorBytes);
        glColor3ubv(colorBytes);
        glBegin(GL_TRIANGLES);
        glVertex3f(v0[0],v0[1],v0[2]);
        glVertex3f(v1[0],v1[1],v1[2]);
        glVertex3f(v2[0],v2[1],v2[2]);
        glEnd(); 
      }
  }

  glFinish(); // done rendering
  
 
  // Step 5: convert the result from color to interger array  
  unsigned char * pbufferCur = pbuffer;
  for (int j =0;j < m_height;j++){
    for (int i =0;i < m_width;i++){
        result[(m_width-i-1)*m_height+j] = uchar2uint(pbufferCur);
        pbufferCur += 3;
        /*
        if (result[(m_width-i-1)*m_height+j]!=0){
          printf("%d=[%d,%d,%d]\n",result[i],pbufferCur[0],pbufferCur[1],pbufferCur[2]);
        }
        */
    }
  }
  GLint outWidth, outHeight, bitPerDepth;
  unsigned int* pDepthBuffer;
  OSMesaGetDepthBuffer(ctx, &outWidth, &outHeight, &bitPerDepth, (void**)&pDepthBuffer);
  //printf("w = %d, h = %d, bitPerDepth = %d\n", outWidth, outHeight, bitPerDepth);
  //plhs[1] = mxCreateNumericMatrix((int)outWidth, (int)outHeight, mxUINT32_CLASS, mxREAL);
  
  // do the conversion
  float z_far = 10;
  for (int j =0;j < m_height;j++){
    for (int i =0;i < m_width;i++){
        float depth_value = (float)pDepthBuffer[j*m_width+i]/4294967296.0;
        depth_value = m_near/(1-depth_value);
        if (depth_value<m_near|depth_value>z_far){
            depth_value =0;
        }
        depth_render[(m_width-i-1)*m_height+(m_height-j-1)] = (float)depth_value;
    }
  }
  
  OSMesaDestroyContext(ctx);
  delete [] pbuffer;

}
/*
int main(){
  Mesh meshTrain = Mesh("/Users/shuran/Documents/third_floor_tearoom/MeshTrain.bin");
  float P[12] = {570.3422,0,0,
                 0,570.3422,0,
                 320.0000,240.0000,1.0000,
                 0,0,0};

  int m_width = 640;
  int m_height = 480;
  float* depth_render = (float *) malloc(sizeof(float)*m_width * m_height);
  unsigned int* result = (unsigned int* ) malloc(sizeof(unsigned int)*m_width * m_height);


  meshTrain.renderMesh(P,m_width,m_height,result,depth_render);
  FILE* fp = fopen("debug_debpth.bin","wb");
  fwrite(depth_render,sizeof(float),640*480,fp);
  fclose(fp);

  fp = fopen("debug_label.bin","wb");
  fwrite(result,sizeof(unsigned int),640*480,fp);
  fclose(fp);

  return 1;
}
*/