
#include "xcl2.hpp"
#include <vector>
#include"cnpy.h"
#include "layerdef.h"
#include <unistd.h>
#include <stdio.h>
#include <limits.h>
#include <iostream>
#include <string>
#include <fstream>
#include <cstdlib>
#include <math.h>

using std::vector;
float* filter;
float* filter_proper;
float* h_bias;

int layer_count = 0;
char PrintCurrentDirectory(){
	 char cwd[PATH_MAX];
	   if (getcwd(cwd, sizeof(cwd)) != NULL) {
		   printf("===================================\n");
		   printf("Current working dir: %s\n", cwd);
		   printf("===================================\n");

	   } else {
		   perror("getcwd() error");
		   return 1;
	   }
}

void GetInputImage(std::string fname, float* im1, float* im2, float* im3)
{
	cnpy::NpyArray _cnpyBuff;
    _cnpyBuff =cnpy::npy_load(fname);
    //get shape for index 0: (example)
    vector<unsigned int> __shape1(_cnpyBuff.shape.begin(),_cnpyBuff.shape.end());

    /*
    //print shape
    printf("Shape of Tensor1:");
    for(unsigned int i :__shape1){
    	printf("%ux",i);
    }printf("\n");
    */
    //accessing tensor data:
    float *imd = _cnpyBuff.data<float>();
    /*
    for(unsigned int i=0; i<__shape1[0]*__shape1[1]*__shape1[2]; i++){
    	printf("input[%d]=%f\n",i,imd[i]);
    }
    printf("\n=========================================\n");
    */
    int i,j;
    for(i=0,j=0; i<HEIGHT_0*WIDTH_0; i++,j+=3){
        im1[i] = imd[j];
        im2[i] = imd[j+1];
        im3[i] = imd[j+2];
     }
}


void GetWeightsFromNPY(std::string fname)
{

	cnpy::NpyArray _cnpyBuff;
    _cnpyBuff =cnpy::npy_load(fname);

    //get shape for index 0: (example)
    vector<unsigned int> __shape1(_cnpyBuff.shape.begin(),_cnpyBuff.shape.end());
/*
    //print shape
    printf("Shape of Tensor1:");
    for(unsigned int i :__shape1){
    	printf("%ux",i);
    }printf("\n");
*/
    //accessing tensor data:
    filter = _cnpyBuff.data<float>();
/*
    for(unsigned int i=0; i<__shape1[0]*__shape1[1]*__shape1[2]*__shape1[3]; i++){
    	printf("weight[%d]=%f\n",i,filter[i]);
    }
    printf("\n=========================================\n");
*/
}


void GetBiasFromNPY( std::string fname, float* bias, int size)
{
	cnpy::NpyArray _cnpyBuff;
    _cnpyBuff=cnpy::npy_load(fname);

    //get shape for index 0: (example)
    vector<unsigned int> __shape1(_cnpyBuff.shape.begin(),_cnpyBuff.shape.end());

    //print shape
    printf("Shape of Tensor1:");
    for(unsigned int i :__shape1){
    	printf("%ux",i);
    }printf("\n");

    //accessing tensor data:
    //bias = _cnpyBuff.data<float>();
    memcpy(&(*bias), &(*(_cnpyBuff.data<float>())), size*sizeof(float));
    /*
   for(unsigned int i=0; i<__shape1[0]; i++){
    	printf("Bias[%d]=%f\n",i,bias[i]);
    }
    printf("\n=========================================\n");
*/
}


void ArrangeWightsforDepth(float* ip, float* op, int fsize)
{
    int nof,ele_per_filter,i=0;
    for (nof=0; nof<fsize; nof++)
    {
        for(ele_per_filter=0;ele_per_filter<9;ele_per_filter++,i++)
        {
            op[i]=ip[0+(ele_per_filter*(fsize))+nof];
        }
    }
}

void ArrangeWeightsforStd(float* ip, float* op, int fsize)
{
	int i,j,k,idx=0;
	for (i = 0; i < fsize; i++) {
		for (j = 0; j < 3; j++) {
			for (k = 0; k < 9; k++,idx++) {
				op[idx] = ip[(k*3)+j+(i*27)];
				//printf("%d\t", op[idx]);
			}
			//printf("\n");
		}
	}
}


void ConvStandard (float* opfm , cl::Context &context, cl::CommandQueue &q, cl::Kernel &ConvStd) {


	cl_int err;
	float* image_r = (float*) malloc(HEIGHT_0 * WIDTH_0 * sizeof(float)); //R channel
	float* image_g = (float*) malloc(HEIGHT_0 * WIDTH_0 * sizeof(float)); //G channel
	float* image_b = (float*) malloc(HEIGHT_0 * WIDTH_0 * sizeof(float)); //B channel

	//Bias
	float* h_bias;
	h_bias= (float*) malloc(IP_FM_1 * sizeof(float));

	GetBiasFromNPY("../../bias_std.npy", h_bias, 32);
	GetInputImage("../../input_image.npy",image_r,image_g,image_b);
    GetWeightsFromNPY("../../weight_std.npy");
    ArrangeWeightsforStd(filter, filter_proper, IP_FM_1*FDIM*FDIM*FDIM);

	//Create buffer for device
    cl::Buffer d_image_r(context,CL_MEM_READ_ONLY, HEIGHT_0*WIDTH_0*sizeof(float) ,nullptr);
    cl::Buffer d_image_b(context,CL_MEM_READ_ONLY, HEIGHT_0*WIDTH_0*sizeof(float) ,nullptr);
    cl::Buffer d_image_g(context,CL_MEM_READ_ONLY, HEIGHT_0*WIDTH_0*sizeof(float) ,nullptr);
    cl::Buffer d_output (context,CL_MEM_WRITE_ONLY, HEIGHT_1*WIDTH_1*IP_FM_1*sizeof(float) ,nullptr);
    cl::Buffer d_filter (context,CL_MEM_READ_ONLY, IP_FM_1*FDIM*FDIM*FDIM*sizeof(float) ,nullptr);
    cl::Buffer d_bias   (context,CL_MEM_READ_ONLY, IP_FM_1*sizeof(float) ,nullptr);


    //cl::Event blocking_call_event;
    q.enqueueWriteBuffer(d_image_r,CL_TRUE,0,HEIGHT_0*WIDTH_0*sizeof(float),image_r,nullptr ,nullptr);
    q.enqueueWriteBuffer(d_image_g,CL_TRUE,0,HEIGHT_0*WIDTH_0*sizeof(float),image_g,nullptr ,nullptr);
    q.enqueueWriteBuffer(d_image_b,CL_TRUE,0,HEIGHT_0*WIDTH_0*sizeof(float),image_b,nullptr ,nullptr);
    q.enqueueWriteBuffer(d_filter ,CL_TRUE,0,IP_FM_1*FDIM*FDIM*FDIM*sizeof(float),filter_proper,nullptr,nullptr);
    q.enqueueWriteBuffer(d_bias   ,CL_TRUE,0,IP_FM_1*sizeof(float),h_bias,nullptr,nullptr);

	int rows = HEIGHT_0;
	int cols = WIDTH_0;
	int filtersize = FDIM;
	int no_fm_0 = OP_FM_0;
	int stride = 2;

    OCL_CHECK(err, err=ConvStd.setArg(0,d_output));
    OCL_CHECK(err, err=ConvStd.setArg(1,d_image_r));
    OCL_CHECK(err, err=ConvStd.setArg(2,d_image_g));
    OCL_CHECK(err, err=ConvStd.setArg(3,d_image_b));
    OCL_CHECK(err, err=ConvStd.setArg(4,d_filter));
    OCL_CHECK(err, err=ConvStd.setArg(5,d_bias));
    OCL_CHECK(err, err=ConvStd.setArg(6,rows));
    OCL_CHECK(err, err=ConvStd.setArg(7,cols));
    OCL_CHECK(err, err=ConvStd.setArg(8,filtersize));
    OCL_CHECK(err, err=ConvStd.setArg(9,stride));
    OCL_CHECK(err, err=ConvStd.setArg(10,no_fm_0));
/*
	size_t local[2], global[2];
	local[0] = 16;
	local[1] = 16;
	global[0] = 112;
	global[1] = 112;
   // cl::Event myevent;
    *
    *
    */
    q.enqueueNDRangeKernel(ConvStd,cl::NullRange,cl::NDRange(112,112),cl::NullRange);

    q.enqueueReadBuffer(d_output ,CL_TRUE, 0, IP_FM_1*(HEIGHT_1)*(WIDTH_1)*sizeof(float), opfm, nullptr, nullptr);

    q.finish();

	free(image_r);
	free(image_g);
	free(image_b);

}



void ConvDepthwise(float* ipfm, float* opfm, cl::string fileName_bias,
				   cl::string fileName_filter, int iph, int ipw, int oph, int opw, int ip_fsize,
				   int op_fsize, int stride, cl::Context &context, cl::CommandQueue &q, cl::Kernel &DepthKrnl) {
	cl_int err;
	//kernelExecTimeNs = 0;
	//int i,j,k;
	/*Bias*/
	float* h_bias;
	h_bias = (float*)malloc(sizeof(int) * op_fsize);

    //getBias(h_bias,fileName_bias,op_fsize);
	GetBiasFromNPY(fileName_bias, h_bias ,op_fsize);
/*
	for(int i=0; i<30 ;i++)
	{
		printf("H_bias[%d]=%f\n", i , h_bias[i]);

	};
	*/
	//Get filter values
	GetWeightsFromNPY(fileName_filter);
	ArrangeWightsforDepth(filter, filter_proper, op_fsize);
	//arrangWeightsDepthwise(filter, filter_proper, op_fsize);

	// for(i = 0; i < op_fsize*9; i++)
	// 	printf("%d\t", filter_proper[i]);
	// printf("\n");

	//Create buffer for device

    cl::Buffer d_filter (context,CL_MEM_READ_ONLY, op_fsize*FDIM*FDIM*sizeof(float) ,nullptr);
    cl::Buffer d_bias   (context,CL_MEM_READ_ONLY, op_fsize*sizeof(float) ,nullptr);
    cl::Buffer d_output (context,CL_MEM_WRITE_ONLY, oph*opw*op_fsize*sizeof(float) ,nullptr);
    cl::Buffer d_input  (context,CL_MEM_READ_ONLY, iph*ipw*ip_fsize*sizeof(float) ,nullptr);

    q.enqueueWriteBuffer(d_filter,CL_TRUE,0,op_fsize*FDIM*FDIM*sizeof(float),filter_proper,nullptr ,nullptr);
    q.enqueueWriteBuffer(d_bias  ,CL_TRUE,0,op_fsize*sizeof(float),h_bias,nullptr,nullptr);
    q.enqueueWriteBuffer(d_input ,CL_TRUE,0,iph*ipw*ip_fsize*sizeof(float),ipfm,nullptr ,nullptr);
/*
	err = clEnqueueWriteBuffer(commands, d_input, CL_TRUE, 0, iph*ipw*ip_fsize*sizeof(unsigned char), ipfm, 0, NULL, NULL);
	err |= clEnqueueWriteBuffer(commands, d_filter, CL_TRUE, 0, op_fsize*FDIM*FDIM*sizeof(unsigned char), filter_proper, 0, NULL, NULL);
	err |= clEnqueueWriteBuffer(commands, d_bias, CL_TRUE, 0, op_fsize*sizeof(int), h_bias, 0, NULL, NULL);
*/
	int rows = iph;
	int cols = ipw;
	int filtersize = FDIM;

    OCL_CHECK(err, err=DepthKrnl.setArg(0,d_output));
    OCL_CHECK(err, err=DepthKrnl.setArg(1,d_input));
    OCL_CHECK(err, err=DepthKrnl.setArg(2,d_filter));
    OCL_CHECK(err, err=DepthKrnl.setArg(3,d_bias));
    OCL_CHECK(err, err=DepthKrnl.setArg(4,rows));
    OCL_CHECK(err, err=DepthKrnl.setArg(5,cols));
    OCL_CHECK(err, err=DepthKrnl.setArg(6,filtersize));
    OCL_CHECK(err, err=DepthKrnl.setArg(7,stride));
    OCL_CHECK(err, err=DepthKrnl.setArg(8,op_fsize));

    q.enqueueNDRangeKernel(DepthKrnl,cl::NullRange,cl::NDRange(opw,oph),cl::NullRange);
    q.enqueueReadBuffer(d_output ,CL_TRUE, 0, op_fsize*oph*opw*sizeof(float), opfm, nullptr, nullptr);
    q.finish();

/*
	err = clSetKernelArg(depthwise_conv, 0, sizeof(cl_mem), (void *)&d_output);
	err |= clSetKernelArg(depthwise_conv, 1, sizeof(cl_mem), (void *)&d_input);
	err |= clSetKernelArg(depthwise_conv, 2, sizeof(cl_mem), (void *)&d_filter);
	err |= clSetKernelArg(depthwise_conv, 3, sizeof(cl_mem), (void *)&d_bias);
	err |= clSetKernelArg(depthwise_conv, 4, sizeof(int), (void *)&rows);
	err |= clSetKernelArg(depthwise_conv, 5, sizeof(int), (void *)&cols);
	err |= clSetKernelArg(depthwise_conv, 6, sizeof(int), (void *)&filtersize);
	err |= clSetKernelArg(depthwise_conv, 7, sizeof(int), (void *)&stride);
	err |= clSetKernelArg(depthwise_conv, 8, sizeof(int), (void *)&op_fsize);

	size_t localWorkSize[2], globalWorkSize[2];
	localWorkSize[0] = 7;
	localWorkSize[1] = 7;
	globalWorkSize[0] = opw;
	globalWorkSize[1] = oph;
	*/
	//err = clEnqueueNDRangeKernel(commands, depthwise_conv, 2, NULL, globalWorkSize, localWorkSize, 0, NULL, &myevent);

	//clWaitForEvents(1,&myevent);
	//clFinish(commands);
	//clGetEventProfilingInfo(myevent,CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, NULL);
	//clGetEventProfilingInfo(myevent,CL_PROFILING_COMMAND_END,sizeof(cl_ulong), &end, NULL);
	//kernelExecTimeNs += end - start;

	//err = clEnqueueReadBuffer(commands, d_output, CL_TRUE, 0, op_fsize*oph*opw*sizeof(unsigned char), opfm, 0, NULL, NULL);

	//printf("Kernel Execution time for Layer %d: %f\n", layer_count, kernelExecTimeNs/1000000000);
	//printf("Data for Layer %d\n", layer_count);
	// unsigned char* output_proper = (unsigned char*) malloc(oph * opw * op_fsize * sizeof(unsigned char));
	//arrangOutput(opfm, output_proper, op_fsize*FDIM*FDIM, op_fsize*oph*opw);
	// FILE *write_ptr;

	// write_ptr = fopen("depth1.npy","wb");  // w for write, b for binary
	// fwrite(output_proper,oph * opw * op_fsize,1,write_ptr);

	// for (k = 0; k < op_fsize; k++){
	// 	printf("Layer No: %d\n", k);
	// 	for (j = 110; j < 112; j++){
	// 		for(i = 0; i < 112; i++){
	// 			printf("%d\t", opfm[(j*opw+i) + (k*oph*opw)]);
	// 		}
	// 		printf("\n");
	// 	}
    // printf("\n");
	// }
}



void ConvPointwise(float* ipfm, float* opfm, std::string fileName_bias,
				   std::string fileName_filter, int iph, int ipw, int oph, int opw, int ip_fsize,
				   int op_fsize, cl::Context &context, cl::CommandQueue &q, cl::Kernel &PointKrnl) {

	//kernelExecTimeNs = 0;
	//int i,j,k;

	cl_int err;
	/*Bias*/
	float* h_bias;
    h_bias = (float*)malloc(sizeof(float) * op_fsize);
    GetBiasFromNPY (fileName_bias, h_bias,op_fsize);
    GetWeightsFromNPY(fileName_filter);

/*
    for(int u=0; u<74;u++)
 {

	printf("h_bias[%d]=%f\n",u,h_bias[u]);
 }
*/
	//Create buffer for device
    cl::Buffer d_filter (context,CL_MEM_READ_ONLY, ip_fsize*op_fsize*FDIM_P*sizeof(float) ,nullptr);
    cl::Buffer d_bias   (context,CL_MEM_READ_ONLY, op_fsize*sizeof(float) ,nullptr);
    cl::Buffer d_output (context,CL_MEM_WRITE_ONLY, oph*opw*op_fsize*sizeof(float) ,nullptr);
    cl::Buffer d_input  (context,CL_MEM_READ_ONLY, iph*ipw*ip_fsize*sizeof(float) ,nullptr);

    q.enqueueWriteBuffer(d_input,  CL_TRUE, 0, iph*ipw*ip_fsize*sizeof(float),ipfm,nullptr ,nullptr);
    q.enqueueWriteBuffer(d_filter, CL_TRUE, 0, ip_fsize*op_fsize*FDIM_P*FDIM_P*sizeof(float),filter,nullptr ,nullptr);
    q.enqueueWriteBuffer(d_bias,   CL_TRUE, 0, op_fsize*sizeof(float),h_bias,nullptr ,nullptr);

	/*
	d_input = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, iph*ipw*ip_fsize*sizeof(unsigned char), ipfm, &err);
	d_output = clCreateBuffer(context, CL_MEM_WRITE_ONLY, oph*opw*op_fsize*sizeof(unsigned char), NULL, &err);
	d_filter = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, ip_fsize*op_fsize*FDIM_P*sizeof(unsigned char), filter, &err);
	d_bias = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, op_fsize*sizeof(int), h_bias, &err);

	err = clEnqueueWriteBuffer(commands, d_input, CL_TRUE, 0, iph*ipw*ip_fsize*sizeof(unsigned char), ipfm, 0, NULL, NULL);
	err |= clEnqueueWriteBuffer(commands, d_filter, CL_TRUE, 0, ip_fsize*op_fsize*FDIM_P*FDIM_P*sizeof(unsigned char), filter, 0, NULL, NULL);
	err |= clEnqueueWriteBuffer(commands, d_bias, CL_TRUE, 0, op_fsize*sizeof(int), h_bias, 0, NULL, NULL);
*/
	int rows = iph;
	int cols = ipw;
	int filtersize = ip_fsize;

	OCL_CHECK(err, err=PointKrnl.setArg(0,d_output));
	OCL_CHECK(err, err=PointKrnl.setArg(1,d_input));
	OCL_CHECK(err, err=PointKrnl.setArg(2,d_filter));
	OCL_CHECK(err, err=PointKrnl.setArg(3,d_bias));
	OCL_CHECK(err, err=PointKrnl.setArg(4,rows));
	OCL_CHECK(err, err=PointKrnl.setArg(5,cols));
	OCL_CHECK(err, err=PointKrnl.setArg(6,filtersize));
	OCL_CHECK(err, err=PointKrnl.setArg(7,op_fsize));


/*
	err = clSetKernelArg(pointwise_conv, 0, sizeof(cl_mem), (void *)&d_output);
	err |= clSetKernelArg(pointwise_conv, 1, sizeof(cl_mem), (void *)&d_input);
	err |= clSetKernelArg(pointwise_conv, 2, sizeof(cl_mem), (void *)&d_filter);
	err |= clSetKernelArg(pointwise_conv, 3, sizeof(cl_mem), (void *)&d_bias);
	err |= clSetKernelArg(pointwise_conv, 4, sizeof(int), (void *)&rows);
	err |= clSetKernelArg(pointwise_conv, 5, sizeof(int), (void *)&cols);
	err |= clSetKernelArg(pointwise_conv, 6, sizeof(int), (void *)&filtersize);
	err |= clSetKernelArg(pointwise_conv, 7, sizeof(int), (void *)&op_fsize);
	err |= clSetKernelArg(pointwise_conv, 8, sizeof(int), (void *)&Q);
	err |= clSetKernelArg(pointwise_conv, 9, sizeof(float), (void *)&Sbias);
	err |= clSetKernelArg(pointwise_conv, 10, sizeof(unsigned char), (void *)&Z2);
	err |= clSetKernelArg(pointwise_conv, 11, sizeof(int), (void *)&right_shift);

	size_t localWorkSize[2], globalWorkSize[2];
	localWorkSize[0] = 7;
	localWorkSize[1] = 7;
	globalWorkSize[0] = opw;
	globalWorkSize[1] = oph;
	*/
	//err = clEnqueueNDRangeKernel(commands, pointwise_conv, 2, NULL, globalWorkSize, localWorkSize, 0, NULL, &myevent);
    q.enqueueNDRangeKernel(PointKrnl,cl::NullRange,cl::NDRange(opw,oph),cl::NullRange);
    q.enqueueReadBuffer(d_output ,CL_TRUE, 0, op_fsize*oph*opw*sizeof(float), opfm, nullptr, nullptr);
    q.finish();

    //
	//clWaitForEvents(1,&myevent);
	//clFinish(commands);
	//clGetEventProfilingInfo(myevent,CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, NULL);
	//clGetEventProfilingInfo(myevent,CL_PROFILING_COMMAND_END,sizeof(cl_ulong), &end, NULL);
	//kernelExecTimeNs += end - start;
	//err = clEnqueueReadBuffer(commands, d_output, CL_TRUE, 0, op_fsize*oph*opw*sizeof(unsigned char), opfm, 0, NULL, NULL);

	// unsigned char* output_proper = (unsigned char*) malloc(oph * opw * op_fsize * sizeof(unsigned char));
	// arrangOutput(opfm, output_proper, ip_fsize*op_fsize*FDIM_P*FDIM_P, op_fsize*oph*opw);
	// FILE *write_ptr;

	// write_ptr = fopen("point.npy","wb");  // w for write, b for binary
	// fwrite(output_proper,oph * opw * op_fsize,1,write_ptr);

	//Get kernel execution time
	//printf("Kernel Execution time for Layer %d: %f\n", layer_count, kernelExecTimeNs/1000000000);

	// printf("Data for Layer %d\n", layer_count);

	// for (k = 0; k < op_fsize; k++){
	// 	printf("Layer No.: %d\n",k);
	// 	for (j = 100; j < 112; j++){
	// 		for(i = 100; i < 112; i++){
	// 			printf("%d\t", opfm[(j*opw+i) + (k*oph*opw)]);
	// 		}
	// 		printf("\n");
	// 	}
    // 	printf("\n");
	// }
}


void ConvAvgPool(float* ipfm, float* opfm,
				   int iph, int ipw, int oph, int opw, int ip_fsize,
				   int op_fsize,cl::Context &context, cl::CommandQueue &q, cl::Kernel &AvgPool) {

	//kernelExecTimeNs = 0;
	//int i,j,k;

	cl_int err;
	//Create buffer for device
	//d_input = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, iph*ipw*ip_fsize*sizeof(unsigned char), ipfm, &err);
	//d_output = clCreateBuffer(context, CL_MEM_WRITE_ONLY, oph*opw*op_fsize*sizeof(unsigned char), NULL, &err);

    cl::Buffer d_input  (context,CL_MEM_READ_ONLY, iph*ipw*ip_fsize*sizeof(float) ,nullptr);
    cl::Buffer d_output (context,CL_MEM_WRITE_ONLY, oph*opw*op_fsize*sizeof(float) ,nullptr);

    q.enqueueWriteBuffer(d_input,  CL_TRUE, 0, iph*ipw*ip_fsize*sizeof(float),ipfm,nullptr ,nullptr);
	//err = clEnqueueWriteBuffer(commands, d_input, CL_TRUE, 0, iph*ipw*ip_fsize*sizeof(unsigned char), ipfm, 0, NULL, NULL);

	int rows = iph;
	int cols = ipw;

	OCL_CHECK(err, err=AvgPool.setArg(0,d_output));
	OCL_CHECK(err, err=AvgPool.setArg(1,d_input));
	OCL_CHECK(err, err=AvgPool.setArg(2,rows));
	OCL_CHECK(err, err=AvgPool.setArg(3,cols));
    /*/
	err = clSetKernelArg(avgPool, 0, sizeof(cl_mem), (void *)&d_output);
	err |= clSetKernelArg(avgPool, 1, sizeof(cl_mem), (void *)&d_input);
	err |= clSetKernelArg(avgPool, 2, sizeof(int), (void *)&rows);
	err |= clSetKernelArg(avgPool, 3, sizeof(int), (void *)&cols);

	size_t localWorkSize[1], globalWorkSize[1];
	localWorkSize[0] = 16;
	globalWorkSize[0] = op_fsize;
	//err = clEnqueueNDRangeKernel(commands, avgPool, 1, NULL, globalWorkSize, localWorkSize, 0, NULL, &myevent);

	clWaitForEvents(1,&myevent);
	clFinish(commands);
	clGetEventProfilingInfo(myevent,CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, NULL);
	clGetEventProfilingInfo(myevent,CL_PROFILING_COMMAND_END,sizeof(cl_ulong), &end, NULL);
	kernelExecTimeNs += end - start;
	err = clEnqueueReadBuffer(commands, d_output, CL_TRUE, 0, op_fsize*oph*opw*sizeof(unsigned char), opfm, 0, NULL, NULL);
*/
    q.enqueueNDRangeKernel(AvgPool,cl::NullRange,cl::NDRange(op_fsize),cl::NullRange);
    q.enqueueReadBuffer(d_output ,CL_TRUE, 0, op_fsize*oph*opw*sizeof(float), opfm, nullptr, nullptr);
    q.finish();

	//Get kernel execution time
	//printf("Kernel Execution time for Layer %d: %f\n", layer_count, kernelExecTimeNs/1000000000);
	printf("Avg pool Layer\n");
	/*
	for (k = 0; k < 32; k++){
		for (j = 0; j < 5; j++){
			for(i = 0; i < 5; i++){
				printf("%u\t", opfm[(j*112+i) + k]);
			}
			printf("\n");
		}
    	printf("\n");
	}	*/
}

void FullyConectedLayer( float* ipfm, float* opfm, std::string fileName_bias , std::string fileName_filter , int classes , int elements)
{
    int i,j;

	float sum = 0;
	/*Bias*/
	float* h_bias;
    h_bias = (float*) malloc(sizeof(float)*classes);
    GetBiasFromNPY(fileName_bias,h_bias, classes);
    GetWeightsFromNPY(fileName_filter);

    for(i = 0; i < CLASSES; i++)
    {
        for(j = 0; j < ELEMENTS; j++)
        {
            sum += (ipfm[j] * (filter[j + (ELEMENTS * i)]));

        }
		sum = sum + h_bias[i];

		/*
		if (sum <= 0) {
			sum = 0;
		} else if (sum >= 255)
			sum = 255;
*/
		opfm[i] = sum;
		sum = 0;
    }
    printf("Layer 29 Fully Connected Done\n");
}


void Softmax (float* ipfm)
{
    double expo[40], sum, max = 0.0;
	int maxIndex;
    int i;
//	int temp;
	printf("SOFTMAX OP: ");
    for(i = 0; i < CLASSES_SOFTMAX; i++)
    {
        expo[i] = exp(ipfm[i]);
        sum += expo[i];
		//printf("i = %d \t ipfm %d %f\n", i,ipfm[i],expo[i]);
    }

    for(i = 0; i < CLASSES_SOFTMAX; i++)
    {
		expo[i] = expo[i] / sum;
		//printf("%f\t", expo[i]);
    }

	for(i = 0; i < CLASSES_SOFTMAX; i++)
    {
		if ( expo[i] > max){
			max = expo[i];
			maxIndex = i;
		}
    }
    printf("Layer 30 softmax Done\n");
	printf("Prediction - %d\t %f\n", maxIndex , max);
}



/*
 *
 *                                               MAIN    MAIN     MAIN
 *
 *
 */

int main(int argc, char **argv) {


	PrintCurrentDirectory();

	filter = (float*) malloc(FILTER_MAX*FILTER_MAX*FDIM*FDIM*FDIM*sizeof(float));
	filter_proper  = (float*) malloc(FILTER_MAX*FILTER_MAX*FDIM*FDIM*FDIM*sizeof(float));
	float* op_fm_0 = (float*) malloc(IP_FM_1 * HEIGHT_1 * WIDTH_1 * sizeof(float)); //output feature map for layer 0

	cl_int err;
	cl::Event event;

	vector<cl::Device> devices = xcl::get_xil_devices();
    cl::Device device = devices[0];

	std::cout << "Creating Context..." <<std::endl;
    OCL_CHECK(err, cl::Context context (device, NULL, NULL, NULL, &err));
    OCL_CHECK(err, cl::CommandQueue q (context, device, CL_QUEUE_PROFILING_ENABLE, &err));
	OCL_CHECK(err, std::string device_name = device.getInfo<CL_DEVICE_NAME>(&err));
	std::string binaryFile = xcl::find_binary_file(device_name, "kernel");
	cl::Program::Binaries bins = xcl::import_binary_file(binaryFile);
    devices.resize(1);

    OCL_CHECK(err, cl::Program program (context, devices, bins, NULL, &err));
    OCL_CHECK(err, cl::Kernel ConvStd  (program, "convolute", &err));
    OCL_CHECK(err, cl::Kernel DepthKrnl(program, "depthwise", &err));
    OCL_CHECK(err, cl::Kernel PointKrnl(program, "pointwise", &err));
    OCL_CHECK(err, cl::Kernel AvgPool  (program, "avgPool"  , &err));

// This is Layer 0
    ConvStandard(op_fm_0,context,q,ConvStd);

	//Layer 1 Depth-Wise Convolution
	layer_count++;
	float* op_fm_1 = (float*) malloc(IP_FM_2 * HEIGHT_2 * WIDTH_2 * sizeof(float)); //output feature map for layer 1
	ConvDepthwise(op_fm_0, op_fm_1, "../../bias/BConv2d_1_depthwise.npy", "../../weights/Conv2d_1_depthwise.npy", HEIGHT_1, WIDTH_1, HEIGHT_2, WIDTH_2, IP_FM_1, IP_FM_2, 1, context ,q, DepthKrnl);

	//Layer 2 Point-Wise Convolution
	layer_count++;
	float* op_fm_2 = (float*) malloc(IP_FM_3 * HEIGHT_3 * WIDTH_3 * sizeof(float));	//output feature map for layer 2
	ConvPointwise(op_fm_1, op_fm_2, "../../bias/BConv2d_1_pointwise.npy", "../../weights/Conv2d_1_pointwise.npy", HEIGHT_2, WIDTH_2, HEIGHT_3, WIDTH_3, IP_FM_2, IP_FM_3, context, q, PointKrnl);

	//Layer 3 Depth-Wise Convolution

	layer_count++;
	float* op_fm_3 = (float*) malloc(IP_FM_4 * HEIGHT_4 * WIDTH_4 * sizeof(float)); //output feature map for layer 3
	ConvDepthwise(op_fm_2, op_fm_3, "../../bias/BConv2d_2_depthwise.npy", "../../weights/Conv2d_2_depthwise.npy", HEIGHT_3, WIDTH_3, HEIGHT_4, WIDTH_4, IP_FM_3, IP_FM_4, 2,  context ,q, DepthKrnl);

	//Layer 4 Point-Wise Convolution

	layer_count++;
	float* op_fm_4 = (float*) malloc(IP_FM_5 * HEIGHT_5 * WIDTH_5 * sizeof(float));	//output feature map for layer 4
	ConvPointwise(op_fm_3, op_fm_4, "../../bias/BConv2d_2_pointwise.npy", "../../weights/Conv2d_2_pointwise.npy", HEIGHT_4, WIDTH_4, HEIGHT_5, WIDTH_5, IP_FM_4, IP_FM_5,context, q, PointKrnl );

	//Layer 5 Depth-Wise Convolution

	layer_count++;
	float* op_fm_5 = (float*) malloc(IP_FM_6 * HEIGHT_6 * WIDTH_6 * sizeof(float)); //output feature map for layer 5
	ConvDepthwise(op_fm_4, op_fm_5, "../../bias/BConv2d_3_depthwise.npy", "../../weights/Conv2d_3_depthwise.npy", HEIGHT_5, WIDTH_5, HEIGHT_6, WIDTH_6, IP_FM_5, IP_FM_6, 1, context ,q, DepthKrnl);

	//Layer 6 Point-Wise Convolution

	layer_count++;
	float* op_fm_6 = (float*) malloc(IP_FM_7 * HEIGHT_7 * WIDTH_7 * sizeof(float));	//output feature map for layer 6
	ConvPointwise(op_fm_5, op_fm_6, "../../bias/BConv2d_3_pointwise.npy", "../../weights/Conv2d_3_pointwise.npy", HEIGHT_6, WIDTH_6, HEIGHT_7, WIDTH_7, IP_FM_6, IP_FM_7, context, q, PointKrnl);

	//Layer 7 Depth-Wise Convolution

	layer_count++;
	float* op_fm_7 = (float*) malloc(IP_FM_8 * HEIGHT_8 * WIDTH_8 * sizeof(float)); //output feature map for layer 7
	ConvDepthwise(op_fm_6, op_fm_7, "../../bias/BConv2d_4_depthwise.npy", "../../weights/Conv2d_4_depthwise.npy", HEIGHT_7, WIDTH_7, HEIGHT_8, WIDTH_8, IP_FM_7, IP_FM_8, 2, context ,q, DepthKrnl);

	//Layer 8 Point-Wise Convolution

	layer_count++;
	float* op_fm_8 = (float*) malloc(IP_FM_9 * HEIGHT_9 * WIDTH_9 * sizeof(float));	//output feature map for layer 8
	ConvPointwise(op_fm_7, op_fm_8, "../../bias/BConv2d_4_pointwise.npy", "../../weights/Conv2d_4_pointwise.npy", HEIGHT_8, WIDTH_8, HEIGHT_9, WIDTH_9, IP_FM_8, IP_FM_9, context, q, PointKrnl);

	//Layer 9 Depth-Wise Convolution

	layer_count++;
	float* op_fm_9 = (float*) malloc(IP_FM_10 * HEIGHT_10 * WIDTH_10 * sizeof(float)); //output feature map for layer 9
	ConvDepthwise(op_fm_8, op_fm_9, "../../bias/BConv2d_5_depthwise.npy", "../../weights/Conv2d_5_depthwise.npy", HEIGHT_9, WIDTH_9, HEIGHT_10, WIDTH_10, IP_FM_9, IP_FM_10, 1, context ,q, DepthKrnl );

	//Layer 10 Point-Wise Convolution

	layer_count++;
	float* op_fm_10 = (float*) malloc(IP_FM_11 * HEIGHT_11 * WIDTH_11 * sizeof(float));	//output feature map for layer 10
	ConvPointwise(op_fm_9, op_fm_10, "../../bias/BConv2d_5_pointwise.npy", "../../weights/Conv2d_5_pointwise.npy", HEIGHT_10, WIDTH_10, HEIGHT_11, WIDTH_11, IP_FM_10, IP_FM_11, context, q, PointKrnl);

	//Layer 11 Depth-Wise Convolution

	layer_count++;
	float* op_fm_11 = (float*) malloc(IP_FM_12 * HEIGHT_12 * WIDTH_12 * sizeof(float)); //output feature map for layer 11
	ConvDepthwise(op_fm_10, op_fm_11, "../../bias/BConv2d_6_depthwise.npy", "../../weights/Conv2d_6_depthwise.npy", HEIGHT_11, WIDTH_11, HEIGHT_12, WIDTH_12, IP_FM_11, IP_FM_12, 2, context ,q, DepthKrnl);

	//Layer 12 Point-Wise Convolution

	layer_count++;
	float* op_fm_12 = (float*) malloc(IP_FM_13 * HEIGHT_13 * WIDTH_13 * sizeof(float));	//output feature map for layer 12
	ConvPointwise(op_fm_11, op_fm_12, "../../bias/BConv2d_6_pointwise.npy", "../../weights/Conv2d_6_pointwise.npy", HEIGHT_12, WIDTH_12, HEIGHT_13, WIDTH_13, IP_FM_12, IP_FM_13, context, q, PointKrnl);

	//Layer 13 Depth-Wise Convolution

	layer_count++;
	float* op_fm_13 = (float*) malloc(IP_FM_14 * HEIGHT_14 * WIDTH_14 * sizeof(float)); //output feature map for layer 13
	ConvDepthwise(op_fm_12, op_fm_13, "../../bias/BConv2d_7_depthwise.npy", "../../weights/Conv2d_7_depthwise.npy", HEIGHT_13, WIDTH_13, HEIGHT_14, WIDTH_14, IP_FM_13, IP_FM_14, 1,  context ,q, DepthKrnl);

	//Layer 14 Point-Wise Convolution

	layer_count++;
	float* op_fm_14 = (float*) malloc(IP_FM_15 * HEIGHT_15 * WIDTH_15 * sizeof(float));	//output feature map for layer 14
	ConvPointwise(op_fm_13, op_fm_14, "../../bias/BConv2d_7_pointwise.npy", "../../weights/Conv2d_7_pointwise.npy", HEIGHT_14, WIDTH_14, HEIGHT_15, WIDTH_15, IP_FM_14, IP_FM_15, context, q, PointKrnl);

	//Layer 15 Depth-Wise Convolution

	layer_count++;
	float* op_fm_15 = (float*) malloc(IP_FM_16 * HEIGHT_16 * WIDTH_16 * sizeof(float)); //output feature map for layer 15
	ConvDepthwise(op_fm_14, op_fm_15, "../../bias/BConv2d_8_depthwise.npy", "../../weights/Conv2d_8_depthwise.npy", HEIGHT_15, WIDTH_15, HEIGHT_16, WIDTH_16, IP_FM_15, IP_FM_16, 1, context ,q, DepthKrnl );

	//Layer 16 Point-Wise Convolution

	layer_count++;
	float* op_fm_16 = (float*) malloc(IP_FM_17 * HEIGHT_17 * WIDTH_17 * sizeof(float));	//output feature map for layer 16
	ConvPointwise(op_fm_15, op_fm_16, "../../bias/BConv2d_8_pointwise.npy", "../../weights/Conv2d_8_pointwise.npy", HEIGHT_16, WIDTH_16, HEIGHT_17, WIDTH_17, IP_FM_16, IP_FM_17, context, q, PointKrnl);

	//Layer 17 Depth-Wise Convolution

	layer_count++;
	float* op_fm_17 = (float*) malloc(IP_FM_18 * HEIGHT_18 * WIDTH_18 * sizeof(float)); //output feature map for layer 17
	ConvDepthwise(op_fm_16, op_fm_17, "../../bias/BConv2d_9_depthwise.npy", "../../weights/Conv2d_9_depthwise.npy", HEIGHT_17, WIDTH_17, HEIGHT_18, WIDTH_18, IP_FM_17, IP_FM_18, 1,  context ,q, DepthKrnl);

	//Layer 18 Point-Wise Convolution

	layer_count++;
	float* op_fm_18 = (float*) malloc(IP_FM_19 * HEIGHT_19 * WIDTH_19 * sizeof(float));	//output feature map for layer 18
	ConvPointwise(op_fm_17, op_fm_18, "../../bias/BConv2d_9_pointwise.npy", "../../weights/Conv2d_9_pointwise.npy", HEIGHT_18, WIDTH_18, HEIGHT_19, WIDTH_19, IP_FM_18, IP_FM_19,context, q, PointKrnl );

	//Layer 19 Depth-Wise Convolution

	layer_count++;
	float* op_fm_19 = (float*) malloc(IP_FM_20 * HEIGHT_20 * WIDTH_20 * sizeof(float)); //output feature map for layer 19
	ConvDepthwise(op_fm_18, op_fm_19, "../../bias/BConv2d_10_depthwise.npy", "../../weights/Conv2d_10_depthwise.npy", HEIGHT_19, WIDTH_19, HEIGHT_20, WIDTH_20, IP_FM_19, IP_FM_20, 1, context ,q, DepthKrnl );

	//Layer 20 Point-Wise Convolution

	layer_count++;
	float* op_fm_20 = (float*) malloc(IP_FM_21 * HEIGHT_21 * WIDTH_21 * sizeof(float));	//output feature map for layer 20
	ConvPointwise(op_fm_19, op_fm_20, "../../bias/BConv2d_10_pointwise.npy", "../../weights/Conv2d_10_pointwise.npy", HEIGHT_20, WIDTH_20, HEIGHT_21, WIDTH_21, IP_FM_20, IP_FM_21, context, q, PointKrnl);

	//Layer 21 Depth-Wise Convolution

	layer_count++;
	float* op_fm_21 = (float*) malloc(IP_FM_22 * HEIGHT_22 * WIDTH_22 * sizeof(float)); //output feature map for layer 21
	ConvDepthwise(op_fm_20, op_fm_21, "../../bias/BConv2d_11_depthwise.npy", "../../weights/Conv2d_11_depthwise.npy", HEIGHT_21, WIDTH_21, HEIGHT_22, WIDTH_22, IP_FM_21, IP_FM_22, 1, context ,q, DepthKrnl );

	//Layer 22 Point-Wise Convolution

	layer_count++;
	float* op_fm_22 = (float*) malloc(IP_FM_23 * HEIGHT_23 * WIDTH_23 * sizeof(float));	//output feature map for layer 22
	ConvPointwise(op_fm_21, op_fm_22, "../../bias/BConv2d_11_pointwise.npy", "../../weights/Conv2d_11_pointwise.npy", HEIGHT_22, WIDTH_22, HEIGHT_23, WIDTH_23, IP_FM_22, IP_FM_23, context, q, PointKrnl);

	//Layer 23 Depth-Wise Convolution

	layer_count++;
	float* op_fm_23 = (float*) malloc(IP_FM_24 * HEIGHT_24 * WIDTH_24 * sizeof(float)); //output feature map for layer 23
	ConvDepthwise(op_fm_22, op_fm_23, "../../bias/BConv2d_12_depthwise.npy", "../../weights/Conv2d_12_depthwise.npy", HEIGHT_23, WIDTH_23, HEIGHT_24, WIDTH_24, IP_FM_23, IP_FM_24, 2,  context ,q, DepthKrnl);

	//Layer 24 Point-Wise Convolution

	layer_count++;
	float* op_fm_24 = (float*) malloc(IP_FM_25 * HEIGHT_25 * WIDTH_25 * sizeof(float));	//output feature map for layer 24
	ConvPointwise(op_fm_23, op_fm_24, "../../bias/BConv2d_12_pointwise.npy", "../../weights/Conv2d_12_pointwise.npy", HEIGHT_24, WIDTH_24, HEIGHT_25, WIDTH_25, IP_FM_24, IP_FM_25, context, q, PointKrnl);

	//Layer 25 Depth-Wise Convolution

	layer_count++;
	float* op_fm_25 = (float*) malloc(IP_FM_26 * HEIGHT_26 * WIDTH_26 * sizeof(float)); //output feature map for layer 25
	ConvDepthwise(op_fm_24, op_fm_25, "../../bias/BConv2d_13_depthwise.npy", "../../weights/Conv2d_13_depthwise.npy", HEIGHT_25, WIDTH_25, HEIGHT_26, WIDTH_26, IP_FM_25, IP_FM_26, 1,  context ,q, DepthKrnl);


	//Layer 26 Point-Wise Convolution

	layer_count++;
	float* op_fm_26 = (float*) malloc(IP_FM_27 * HEIGHT_27 * WIDTH_27 * sizeof(float));	//output feature map for layer 26
	ConvPointwise(op_fm_25, op_fm_26, "../../bias/BConv2d_13_pointwise.npy", "../../weights/Conv2d_13_pointwise.npy", HEIGHT_26, WIDTH_26, HEIGHT_27, WIDTH_27, IP_FM_26, IP_FM_27, context, q, PointKrnl);


	//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++




	//Layer 27 Average Pool

	layer_count++;
	float* op_fm_27 = (float*) malloc(ELEMENTS * HEIGHT_28 * WIDTH_28 * sizeof(float));	//output feature map for layer 27
	ConvAvgPool(op_fm_26, op_fm_27, HEIGHT_27, WIDTH_27, HEIGHT_28, WIDTH_28, IP_FM_27, ELEMENTS, context, q, AvgPool);


	// printf("Avg pol data\n");
	// for (k = 0; k < ELEMENTS; k++){
	// 		for (j = 0; j < 1; j++){
	// 			for(i = 0; i < 1; i++){
	// 				printf("%d\t", op_fm_27[(j*WIDTH_28+i) + (k*HEIGHT_28*WIDTH_28)]);
	// 			}
	// 			//printf("\n");
	// 		}
	// 	//printf("\n");
	// 	}


	//Layer 28 Fully COnnected

	layer_count++;
	float* op_fm_28 = (float*) malloc(CLASSES_SOFTMAX * HEIGHT_29 * WIDTH_29 * sizeof(float));	//output feature map for layer 28
	FullyConectedLayer(op_fm_27, op_fm_28, "../../bias/BConv2d_fullyconnected.npy", "../../weights/Conv2d_fullyconnected.npy", CLASSES, ELEMENTS);
	// for (k = 0; k < CLASSES; k++){
	// 	for (j = 0; j < 1; j++){
	// 		for(i = 0; i < 1; i++){
	// 			printf("%d\t", op_fm_28[(j*WIDTH_28+i) + (k*HEIGHT_28*WIDTH_28)]);
	// 		}
	// 		//printf("\n");
	// 	}
    // //printf("\n");
	// }

	// Layer 29 Softmax

	layer_count++;
	Softmax(op_fm_28);

    printf("Hala Passed!!!!!!!");
}
