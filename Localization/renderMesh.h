class Mesh{
      public:
      float* vertex;
      unsigned int* annolabel;
      unsigned int* faces;
      int numVertex;
      int numFace;
      /**********Member functions**************/
      Mesh(string filename);
      ~Mesh();
      void renderMesh(const float* projection, int m_width, int m_height, 
                      unsigned int* result, float* depth_render);
};
