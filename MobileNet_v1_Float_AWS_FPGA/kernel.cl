__kernel void convolute(__global float* output,
						__global float* inp_image_r,
						__global float* inp_image_g,
						__global float* inp_image_b,
						__global float* filter_k,
						__global float* bias,
						int rows, int cols, int filtersize, int stride, int op_size) {


//Printing Input Argumans
/*
	printf("rows=%d\n",rows);
	printf("filtersize=%d\n",filtersize);
	printf("opsize=%d\n",op_size);
	printf("global_id=%d\n",get_global_id(0));
	printf("global_size=%d\n",get_global_size(0));

	for(int p=0; p<10; p++){

		printf("inp_image_r[%d]=%f\n\n",p,inp_image_r[p]);
	}

	for(int p=0; p<32; p++){

		printf("bias[%d]=%f\n",p,bias[p]);
		if(p==31)
		{
			printf("=======================\n");
		};
	}


	for(int p=0; p<30; p++){
		printf("filter_k[%d]=%f\n",p,filter_k[p]);
	}
*/

	int tx = get_global_id(0);
	int ty = get_global_id(1);

	//printf("Global ID is = %d\n",tx);
	if(tx==0&&ty==0){
	printf("Global Size is: %d\n", get_global_size(0));
	}
	int half_filtersize = (filtersize)/2;

	float sum = 0;
	int xindex=0, yindex=0, findex=0, filter_count=0;
	int i,j,l;

	while (filter_count < op_size) {

		int output_shift = (rows / 2) * (cols / 2) * filter_count;

		for(i = 0; i < filtersize; i++){
			yindex = ty * stride + i;
			for(j = 0; j < filtersize; j++,findex++){
				xindex = tx * stride + j;
				if (yindex >= cols || xindex >= rows) {
					sum +=  0 * filter_k[findex];
				}
				else {
					// if ((tx == 0 && ty == 0) && filter_count == 2) {
					// 	printf("Image Index r - y: %d\t x: %d\n",yindex, xindex);
					// 	printf("Image r: %d\t filter index %d \t%d\n",inp_image_r[yindex * get_global_size(0) * stride + xindex], findex, filter_k[findex]);
					// 	printf("Image stracted r: %d\t filter index %d \t%d\n",(inp_image_r[yindex * get_global_size(0) * stride + xindex] - Z1), findex, (filter_k[findex]-Z2));
					// 	printf("Multiplication R %d\n",(inp_image_r[yindex * get_global_size(0) * stride + xindex] - Z1) * (filter_k[findex] - Z2));
					// }
 						sum +=  (inp_image_r[yindex * get_global_size(0)* stride + xindex] ) * (filter_k[findex]);
				}
			}
		}


		for(i = 0; i < filtersize; i++){
			yindex = ty * stride + i;
			for(j = 0; j < filtersize; j++,findex++){
				xindex = tx * stride + j;
				if (yindex >= cols || xindex >= rows) {
					sum +=  0 * filter_k[findex];
				}
				else {
					// if ((tx == 0 && ty == 0) && filter_count == 2) {
					// 	printf("Image Index g - y: %d\t x: %d\n",yindex, xindex);
					// 	printf("Image g: %d\t filter index %d \t%d\n",inp_image_g[yindex * get_global_size(0) * stride + xindex], findex, filter_k[findex]);
					// 	printf("Image substracted g: %d\t filter index %d \t%d\n",(inp_image_g[yindex * get_global_size(0) * stride + xindex] - Z1), findex, (filter_k[findex]-Z2));
					// 	printf("Multiplication G %d\n",(inp_image_g[yindex * get_global_size(0) * stride + xindex] - Z1) * (filter_k[findex] - Z2));
					// }
 					sum +=  (inp_image_g[yindex* get_global_size(0) * stride + xindex] ) * (filter_k[findex]) ;
				}
			}
		}


		for(i = 0; i < filtersize; i++){
			yindex = ty * stride + i;
			for(j = 0; j < filtersize; j++,findex++){
				xindex = tx * stride + j;
				if (yindex >= cols || xindex >= rows) {
					sum +=  0 * filter_k[findex];
				}
				else {
					// if ((tx == 0 && ty == 0) && filter_count == 2) {
					// 	printf("Image Index b - y: %d\t x: %d\n",yindex, xindex);
					// 	printf("Image b: %d\t filter index %d \t%d\n",inp_image_b[yindex * get_global_size(0) * stride + xindex], findex, filter_k[findex]);
					// 	printf("Image substracted b: %d\t filter index %d \t%d\n",(inp_image_b[yindex * get_global_size(0) * stride + xindex] - Z1), findex, (filter_k[findex]-Z2));
					// 	printf("Multiplication B %d\n",(inp_image_b[yindex * get_global_size(0) * stride + xindex] - Z1) * (filter_k[findex] - Z2));
					// }
 					sum +=  (inp_image_b[yindex* get_global_size(0) * stride + xindex]) * (filter_k[findex]);
				}
			}
		}


		// if ((tx == 0 && ty == 0) && filter_count == 2) {
		// 	printf("B Sum: %d\n",(sum));
		// 	printf("Bias %d\n", bias[filter_count]);
		// }

		sum = sum + bias[filter_count];
		// if ((tx == 0 && ty == 0) && filter_count == 2) {
		//	printf("Bias added sum: %d\n",(sum));
	    //}
		//sum = sum * (float)Q / (float)2147483648;
		// if ((tx == 0 && ty == 0) && filter_count == 2) {
		// 	printf("Q: %d\n",Q);
		// 	printf("Q/2^31: %f\n", (float)Q / (float)2147483648);
		// 	printf("sum * Q/2^31: %d\n",(sum));
		// }
		//sum = sum + ((right_shift < 1) ? 0 : (1 << (right_shift - 1)));
		// if ((tx == 0 && ty == 0) && filter_count == 2) {
		// 	printf("Sum + rounding: %d\n",(sum));
		// }
		//sum = sum >> right_shift;
		// if ((tx == 0 && ty == 0) && filter_count == 2) {
		// 	printf("Sum rightshifted: %d\n",(sum));
		// }

		if (sum <= 0) {
			sum = 0;
		} else if (sum >= 6){
			sum = 6;
		}
		// if ((tx == 0 && ty == 0) && filter_count == 2) {
		// 	printf("A Sum: %d\n",(sum));
		// }

		output[(ty* get_global_size(0) + tx) + output_shift] = sum;
		sum = 0;
		filter_count++;
	}

	/*
	if(tx==111 && ty==111)
	{
		for(int s=0; s<112*112*32;s++)
		printf("output_k[%d]=%f\n",s, output[s]);
	}
	*/
}



__kernel void depthwise(__global float* output,
						__global float* inp_image,
						__global float* filter_k,
						__global float* bias,
						int rows, int cols, int filtersize, int stride, int op_size) {

	int tx = get_global_id(0);
	int ty = get_global_id(1);

	int half_filtersize = (filtersize)/2;

	int sum = 0;
	int xindex=0, yindex=0, findex=0, filter_count=0, start, end;
	int i,j,l;

	if (stride == 1) {
		start = -half_filtersize;
		end = half_filtersize;
	} else if (stride == 2) {
		start = 0;
		end = filtersize - 1;
	}

	while (filter_count < op_size) {
		int output_shift = (rows / stride) * (cols / stride) * filter_count;

		for(i = start; i <= end; i++){
			yindex = ty * stride + i;
			for(j = start; j <= end; j++,findex++){
				xindex = tx * stride + j;

				if (stride == 1) {
					if ((yindex < 0 || xindex < 0) || (yindex >= cols || xindex >= rows)) {
						sum +=  0 * filter_k[findex];
					}
					else {
						// if ((tx == 0 && ty == 0) && filter_count == 0) {
						// 	printf("Image data: %d\t filter %d\n",inp_image[yindex * get_global_size(0) * stride + xindex], filter_k[findex]);
						// 	printf("Img data: %d\tfilter index %d\t%d\n",inp_image[(yindex * get_global_size(0) * stride + xindex) + (rows * cols * filter_count)], findex, (filter_k[findex] - Z2));
						// 	printf("Multiplication: %d\n",inp_image[yindex * get_global_size(0) * stride + xindex] * filter_k[findex]);
						// }
						sum +=  inp_image[(yindex * get_global_size(0) * stride + xindex) + (rows * cols * filter_count)] * (filter_k[findex]);
					}
				} else if (stride == 2) {
					if (yindex >= cols || xindex >= rows) {
						sum +=  0 * filter_k[findex];
					}
					else {
						// if ((tx == 0 && ty == 0) && filter_count == 0) {
						// 	printf("Image data: %d\t filter %d\n",inp_image[yindex * get_global_size(0) * stride + xindex], filter_k[findex]);
						// 	printf("Img data: %d\tfilter index %d\t%d\n",inp_image[(yindex * get_global_size(0) * stride + xindex) + (rows * cols * filter_count)], findex, (filter_k[findex] - Z2));
						// 	printf("Multiplication: %d\n",inp_image[(yindex * get_global_size(0) * stride + xindex) + (rows * cols * filter_count)] * (filter_k[findex] - Z2));
						// }
						sum +=  inp_image[(yindex * get_global_size(0) * stride + xindex) + (rows * cols * filter_count)] * (filter_k[findex]);
					}
				}
			}
		}
		// if ((tx == 0 && ty == 0) && filter_count == 0) {
		// 	printf("B Sum: %d\n",(sum));
		// }
		sum = sum + bias[filter_count];
		// if ((tx == 0 && ty == 0) && filter_count == 0) {
		// 	printf("Bias added sum: %d\n",(sum));
		// }

		if (sum <= 0) {
			sum = 0;
		} else if (sum >= 6)
			sum = 6;

		output[(ty * get_global_size(0) + tx) + output_shift] = sum;
		sum = 0;
		filter_count++;
	}
	/*
	if(tx==111 && ty==111)
	{
		for(int e=67000; e<69000 ; e++){
		printf("input_k[%d]=%f\n", e , inp_image[e]);
		}
	}*/
}


__kernel void pointwise(__global float* output,
						__global float* inp_image,
						__global float* filter_k,
						__global float* bias,
						int rows, int cols, int filtersize, int op_size ) {

	int tx = get_global_id(0);
	int ty = get_global_id(1);

	int sum = 0;
	int findex=0, filter_count=0;
	int i,j,l;
	while (filter_count < op_size) {
		int output_shift = rows * cols * filter_count;

		for (i = 0; i < filtersize; i++,findex++) {
			sum += inp_image[(ty * get_global_size(0) + tx) + (rows * cols * i)] * (filter_k[findex]);
		}

		sum = sum + bias[filter_count];
		/*if (tx == 4 && ty == 2) {
			//printf("M: %f\tbias: %f\t\n",M,Sbias);
			//printf("Summ: %d\t\n",(int)((M * sum) + (bias[filter_count] * Sbias)));
			printf(" Point Sum: %d\t\n",sum);
		}*/
		if (sum <= 0) {
			sum = 0;
		} else if (sum >= 6)
			sum = 6;

		output[(ty * get_global_size(0) + tx) + output_shift] =sum;
		sum = 0;
		filter_count++;
	}
	if(tx==111 && ty==111)
	{
		for(int u=0; u<30;u++)
		{
			printf("Bias_k[%d]=%f\n",u,bias[u]);

		}
	}
}

__kernel void avgPool(__global float* output,
					  __global float* inp_image,
					  int rows, int cols ) {

        int tx = get_global_id(0);
        float sum = 0;
        int i;
	    int input_shift = rows * cols;
		for (i = 0; i < rows * cols; i++) {
			sum += inp_image[i + ( tx * input_shift)];
		}
		output[tx] = sum / 49;

}
