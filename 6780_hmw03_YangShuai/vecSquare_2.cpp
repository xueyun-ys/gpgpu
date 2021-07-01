#include <stdio.h>
#include <string.h>
#include <CL/cl.h>

#define N 40
#define BLOCK_SIZE 8

// void createMat()
// {
//
// }


char* loadProgSource(const char* filename, const char* preamble, size_t *sz) {
  FILE* fptr = NULL;
  size_t szSource, szPreamble, howmany;
  char* sourceString;

  // Open the OpenCL source code file
  fptr = fopen(filename, "r");
  szPreamble = strlen(preamble);

  // Get the length of the source code
  fseek(fptr, 0, SEEK_END);
  szSource = ftell(fptr);
  fseek(fptr, 0, SEEK_SET);

  // Allocate a buffer for the source code string and read it in
  sourceString = (char *) calloc(szSource + szPreamble+1, sizeof(char));
  howmany = fread((sourceString) + szPreamble, szSource, 1, fptr);
  fclose(fptr);
  *sz = szSource + szPreamble;
  sourceString[szSource + szPreamble] = '\0';
  return sourceString;
}

int main(void) {
  cl_platform_id platform_id;
  cl_uint num_of_platforms = 0;
  cl_uint num_of_devices = 0;
  cl_device_id device_id;
  cl_context_properties properties[3];
  cl_int err;
  cl_context context;
  cl_command_queue command_queue;
  char *kernelSource;
  size_t kernelSize;
  cl_program program;
  cl_kernel kernel;
  cl_mem input, input2, output;
  size_t global[2];
  size_t local[2];

  cl_float *inputMatrix1;
  cl_float *inputMatrix2;
  cl_float *results;
  cl_uint width = N;

  const int Ndim = N;
  const int Mdim = N;
  //const int Pdim = N;

  //createMat();
  int x,y;
  int data = 0;
  inputMatrix1 = (cl_float *)malloc(sizeof(cl_float)*width*width);
  inputMatrix2 = (cl_float *)malloc(sizeof(cl_float)*width*width);
  results = (cl_float *) malloc(sizeof(cl_float)*width*width);

  for(y=0; y<width; y++)
  {
    for(x=0; x<width; x++)
    {
      inputMatrix1[y*width+x]=data;
      inputMatrix2[y * width +x]=data;
      results[y *width +x]=0;
      data++;
    }
  }

  // Print the Matrix
  // printf("input1: ");
  // for (int j = 0; j < N*N; j++) {
  //   printf("%f ", inputMatrix1[j]);
  // }
  // printf("\n");
  // printf("input2: ");
  // for (int j = 0; j < N*N; j++) {
  //   printf("%f ", inputMatrix2[j]);
  // }
  // printf("\n");

//==============================================================================
  int i;
  // Retrives a list of platforms available
  if (clGetPlatformIDs(1, &platform_id, &num_of_platforms) != CL_SUCCESS) {
    printf("Unable to get platform_id\n");
    return 1;
  }

  // Get a supported GPU device
  if (clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_GPU, 1, &device_id,
     &num_of_devices) != CL_SUCCESS) {
     printf("Unable to get device_id\n");
     return 1;
  }

  // Context properties list (must be terminated with 0)
  properties[0] = CL_CONTEXT_PLATFORM;
  properties[1] = (cl_context_properties) platform_id;
  properties[2] = 0;

  // Create a context with the GPU device
  context = clCreateContext(properties, 1, &device_id, NULL, NULL, &err);

  // Create a command queue using the context and device
  command_queue = clCreateCommandQueue(context, device_id, CL_QUEUE_PROFILING_ENABLE, &err);

  // Load kernel file, prepend static info, and return total kernel size
  kernelSource = loadProgSource("vecSquare_2.cl", "", &kernelSize);

  // Create a program from the kernel source code
  program = clCreateProgramWithSource(context, 1, (const char **)
            &kernelSource, NULL, &err);

  // Compile the program
  if (clBuildProgram(program, 0, NULL, NULL, NULL, NULL) != CL_SUCCESS) {
     printf("Error building program\n");
     return 1;
  }

  // Specify which kernel from the program to execute
  kernel = clCreateKernel(program, "Mat_multi", &err);

  // Create buffers for the input and output
  input = clCreateBuffer(context, CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR,
          sizeof(float) * N*N, inputMatrix1, NULL);
  input2 = clCreateBuffer(context, CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR,
          sizeof(float) * N*N, inputMatrix2, NULL);
  output = clCreateBuffer(context, CL_MEM_READ_WRITE|CL_MEM_COPY_HOST_PTR,
          sizeof(float) * N*N, results, NULL);

  // Load data into the input buffer
  clEnqueueWriteBuffer(command_queue, input, CL_TRUE, 0,
                       sizeof(float) * N, inputMatrix1, 0, NULL, NULL);
  clEnqueueWriteBuffer(command_queue, input2, CL_TRUE, 0,
                       sizeof(float) * N, inputMatrix2, 0, NULL, NULL);

  // Set the argument list for the kernel command
  clSetKernelArg(kernel, 0, sizeof(cl_mem), &input);
  clSetKernelArg(kernel, 1, sizeof(cl_mem), &input2);
  clSetKernelArg(kernel, 2, sizeof(cl_mem), &output);
  clSetKernelArg(kernel, 3, sizeof(int), &Ndim);
  clSetKernelArg(kernel, 4, sizeof(int), &Mdim);
  //clSetKernelArg(kernel, 5, sizeof(cl_mem), &output);
  //clSetKernelArg(kernel, 3, sizeof(int), N);
  global[0] = (size_t)width;
  global[1] = (size_t)width;
  local[0] = (size_t)BLOCK_SIZE;
  local[1] = (size_t)BLOCK_SIZE;

  cl_event prof_event;
  // cl_ulong ev_start_time = (cl_ulong)0;
  // cl_ulong ev_end_time = (cl_ulong)0;
  double run_time;

  // Enqueue the kernel command for execution
  err = clEnqueueNDRangeKernel(command_queue, kernel, 2, NULL, global,
                               local, 0, NULL, &prof_event);
  clFinish(command_queue);
  err = clWaitForEvents(1, &prof_event);
  cl_ulong start_time, end_time;
  size_t return_bytes;
  err = clGetEventProfilingInfo(prof_event, CL_PROFILING_COMMAND_START,
                 sizeof(cl_ulong), &start_time, &return_bytes);
  err = clGetEventProfilingInfo(prof_event, CL_PROFILING_COMMAND_END,
                     sizeof(cl_ulong), &end_time, &return_bytes);
  run_time = (double)(end_time - start_time);

  // Copy the results from out of the output buffer
  clEnqueueReadBuffer(command_queue, output, CL_TRUE, 0,
                      sizeof(float) * N*N, results, 0, NULL, NULL);

  // Print the results
  printf("output: ");
  for (i = 0; i < N*N; i++) {
    printf("%f ", results[i]);
  }
  printf("time: ");
  printf("%f ", run_time);
  printf("\n");

  // Cleanup (release OpenCL resources)
  clReleaseContext(context);
  clReleaseCommandQueue(command_queue);
  clReleaseProgram(program);
  clReleaseKernel(kernel);
  clReleaseMemObject(input);
  clReleaseMemObject(input2);
  clReleaseMemObject(output);

  return 0;
}
