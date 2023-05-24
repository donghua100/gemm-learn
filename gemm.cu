#include <ctime>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>

#define NUM_THREADS 256

int init_cuda() {
	int count;
	cudaGetDeviceCount(&count);
	if (count == 0) {
		fprintf(stderr, "There is no device\n");
		return -1;
	}
	printf("There are %d device.\n", count);
	int i;
	for (i = 0; i < count; i++) {
		struct cudaDeviceProp prop;
		if (cudaGetDeviceProperties(&prop, i) == cudaSuccess) {
			if (prop.major >= 1) break;
		}
	}
	if (i == count) {
		fprintf(stderr, "There is no device supporting CUDA 1.x.\n");
		return -1;
	}
	cudaSetDevice(i);
	return 0;
}


void matgen(float *a, int lda, int n) {
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < n; j++) {
			a[i*lda + j] = (float)rand()/RAND_MAX;
		}
	}
}


clock_t matmult(const float *a, int lda, const float *b, int ldb, 
		float *c, int ldc, int n) {
    clock_t start = clock();
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < n; j++) {
			double t = 0;
			for (int k = 0; k < n; k++) {
				t += a[i*lda + k]*b[k*ldb + j];
			}
			c[i*ldc + j] = t;
		}
	}
    clock_t end = clock();
    return end - start;
}

void compare_mat(const float *a, int lda, const float *b, int ldb, int n) {
	float max_err = 0;
	float ave_err = 0;
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < n; j++) {
			if (b[i*ldb + j] != 0) {
				float err = fabs((a[i*lda + j] - b[i*ldb + j])/b[i*ldb + j]);
				if (max_err < err) max_err = err;
				ave_err += err;
			}
		}
	}
	ave_err /= n*n;
	printf("max error: %g, average error: %g\n", max_err, ave_err);
}



__global__ static void matmultCUDA(const float *a, size_t lda, const float *b, size_t ldb,
		float *c, size_t ldc, int n) {
	extern __shared__ float data[];
	const int tid = threadIdx.x;
	const int row = blockIdx.x;

	for (int i = tid; i < n; i += blockDim.x) {
		data[i] = a[row *lda + i];
	}

    __syncthreads();

	for(int j = tid; j < n; j += blockDim.x) {
		float s = 0;
        float cc = 0;
		for (int i = 0; i < n; i++) {
            float y = data[i]*b[i*ldb + j] - cc;
			float t = s + y;
            cc = (t - s) - y;
            s = t;
		}
		c[row*ldc + j] = s;
	}
}


clock_t matMultCUDA(const float *a, int lda,
		const float *b, int ldb, float *c, int ldc, int n) {
	float *ac, *bc, *cc;
	size_t pitch_a, pitch_b, pitch_c;
	clock_t start = clock();
	cudaMallocPitch((void **)&ac, &pitch_a, sizeof(float)*n, n);
	cudaMallocPitch((void **)&bc, &pitch_b, sizeof(float)*n, n);
	cudaMallocPitch((void **)&cc, &pitch_c, sizeof(float)*n, n);

	cudaMemcpy2D(ac, pitch_a, a, sizeof(float)*lda,
			sizeof(float)*n, n, cudaMemcpyHostToDevice);

	cudaMemcpy2D(bc, pitch_b, b, sizeof(float)*ldb,
			sizeof(float)*n, n, cudaMemcpyHostToDevice);

	int blocks = n;
	matmultCUDA<<<blocks, NUM_THREADS, sizeof(float)*n>>>
		(ac, pitch_a/sizeof(float), bc, pitch_b/sizeof(float), cc, pitch_c/sizeof(float), n);

	cudaMemcpy2D(c, sizeof(float)*ldc, cc, pitch_c,
			sizeof(float)*n,n,cudaMemcpyDeviceToHost);

	cudaFree(ac);
	cudaFree(bc);
	cudaFree(cc);

	clock_t end = clock();
	return end - start;

}


int main() {
	if (init_cuda() == 0) {
		printf("CUDA initialized.\n");
	}
	else {
		printf("initialized CUDA fail!\n");
		return -1;
	}
	float *a, *b, *c, *d;
	int n = 1000;
	a = (float *)malloc(sizeof(float)*n*n);
	b = (float *)malloc(sizeof(float)*n*n);
	c = (float *)malloc(sizeof(float)*n*n);
	d = (float *)malloc(sizeof(float)*n*n);

	matgen(a, n, n);
	matgen(b, n, n);

	clock_t gpu_time = matMultCUDA(a, n, b, n, c, n, n);

	double sec = (double)gpu_time/CLOCKS_PER_SEC;
	printf("(GPU)Time used: %.2f sec(%.2lf GFLOPS)\n", sec,
			2.0*n*n*n/(sec*1E9));

	clock_t cpu_time = matmult(a,n,b,n,d,n,n);
    sec = (double)cpu_time/CLOCKS_PER_SEC;
	printf("(CPU)Time used: %.2f sec(%.2lf GFLOPS)\n", sec,
			2.0*n*n*n/(sec*1E9));
	compare_mat(c, n, d, n, n);
	return 0;
}
