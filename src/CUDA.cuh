#define BLOCK_SIZE 16

void initCuda(float **data, float **data_g, int size);

void kernel(float *data, int size);

void runKernel(float *data_g, int size);

void copyDeviceToHost(float *data, float *data_g, int size);
