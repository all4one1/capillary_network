﻿#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <iostream> 
#include <stdio.h> 
#include <string> 
#include <fstream> 
#include <iomanip> 
#include <sstream> 
#include <cstring> 
#include <cmath>
#include <algorithm> 
#include <ctime>
#include <cuda.h>
#include <vector>

#ifdef _WIN32
#include "windows.h"
#endif


using namespace std;


#define Pi 3.1415926535897932384626433832795
#define pause system("pause");
#define timer timer2 = clock()/ CLOCKS_PER_SEC; 	cout << "time (seconds)= " << (timer2 - timer1) << endl;
#define cudaCheckError() {                                          \
	cudaError_t e = cudaGetLastError();                                \
if (e != cudaSuccess) {\
	printf("Cuda failure %s:%d: '%s'\n", __FILE__, __LINE__, cudaGetErrorString(e));           \
	exit(0); \
}                                                                 \
}

#define cudaCheckErrors(msg) \
    do { \
        cudaError_t __err = cudaGetLastError(); \
        if (__err != cudaSuccess) { \
            fprintf(stderr, "Fatal error: %s (%s at %s:%d)\n", \
                msg, cudaGetErrorString(__err), \
                __FILE__, __LINE__); \
            fprintf(stderr, "*** FAILED - ABORTING\n"); \
            exit(1); \
        } \
    } while (0)



//getting Ek and Vmax
void velocity(unsigned int N, double hx, double hy, double *vx, double *vy, double &Ek, double &Vmax) {
	double V = 0;
	Ek = 0.0; Vmax = 0.0;


	for (unsigned int C = 0; C < N; C++) {
		V = +vx[C] * vx[C] + vy[C] * vy[C];
		Ek += V;
		if (sqrt(V) > Vmax) Vmax = sqrt(V);
	}

	Ek = Ek / 2.0 * hx * hy;
}

double maxval(double* f, unsigned int n)
{
	double max = abs(f[0]);

	for (unsigned int i = 0; i < n; i++) {
		if (abs(f[i])>max)
		{
			max = abs(f[i]);
		}
	}
	return max;
}

double sum(double* f, unsigned int n)
{
	double sum = 0;

	for (unsigned int i = 0; i < n; i++) {
		sum += f[i];
	}
	return sum;
}

//volumetric flow rate
void VFR(double *vx, int *t, unsigned int size, double hy, double &Q_in, double &Q_out, double *C, double &C_average, double &Cv) {
	Q_in = 0; Q_out = 0; C_average = 0; Cv = 0;

	for (unsigned int i = 0; i < size; i++) {
		if (t[i] == 9) {
			Q_in += vx[i];
		}
		if (t[i] == 10)
		{
			Q_out += vx[i];
			C_average += C[i];
			Cv += vx[i] * C[i];
		}
	}
	Q_in = Q_in*hy;
	Q_out = Q_out*hy;
	C_average = C_average*hy/(1.0-hy);
	Cv = Cv*hy;
}


//__device__ double *C, *C0, *ux, *uy, *vx, *vy, *p, *p0, *mu;
//__device__ multi_cross *Md;
__constant__ double hx, hy, tau, Lx, Ly, tau_p;
__constant__ double A, Ca, Gr, Pe, Re, MM;
__constant__ double sinA, cosA, alpha;
__constant__ unsigned int nx, ny, n, offset, border_type;
__constant__ double eps0_d = 1e-5;
__constant__ double pi = 3.1415926535897932384626433832795;
__constant__ int Mx, My, Msize, Moffset, OFFSET;
__constant__ unsigned int iter;
__constant__ unsigned int TOTAL_SIZE;
__constant__ unsigned int nxg, nyg;
__device__ int *n1, *n2, *n3, *n4, *t, *J_back;


__global__ void hello() {

	printf("\n thread x:%i y:%i, information copied from device:\n", threadIdx.x, threadIdx.y);
	printf("A= %f Ca=%f \n", A, Ca);
	printf("Gr= %f Pe=%f \n", Gr, Pe);
	printf("Re= %f M=%f \n", Re, MM);
	printf("hx= %f hy=%f \n", hx, hy);
	printf("tau= %20.16f  \n", tau);
	printf("tau_p= %20.16f  \n", tau_p);
	printf("nx= %i ny=%i  \n", nx, ny);
	printf("Lx= %f Ly=%f \n", Lx, Ly);
	printf("offset= %i  \n", offset);
	printf("sinA= %f cosA=%f \n", sinA, cosA);
	printf("Total number of nodes = %i \n", TOTAL_SIZE);

	printf("\n");
}

__device__ double dx1(unsigned int l, double *f) {
	return 0.5*(f[n3[l]] - f[n1[l]]) / hx;
}
__device__ double dy1(unsigned int l, double *f) {
	return 0.5*(f[n2[l]] - f[n4[l]]) / hy;
}
__device__ double dx2(unsigned int l, double *f) {
	return (f[n3[l]] - 2.0*f[l] + f[n1[l]]) / hx / hx;
}
__device__ double dy2(unsigned int l, double *f) {
	return (f[n2[l]] - 2.0*f[l] + f[n4[l]]) / hy / hy;
}

__device__ double dx1_eq_0_forward(unsigned int l, double *f) {
	return (4.0*f[n3[l]] - f[n3[n3[l]]]) / 3.0;
}
__device__ double dx1_eq_0_back(unsigned int l, double *f) {
	return (4.0*f[n1[l]] - f[n1[n1[l]]]) / 3.0;
}
__device__ double dy1_eq_0_up(unsigned int l, double *f) {
	return (4.0*f[n2[l]] - f[n2[n2[l]]]) / 3.0;
}
__device__ double dy1_eq_0_down(unsigned int l, double *f) {
	return (4.0*f[n4[l]] - f[n4[n4[l]]]) / 3.0;
}

__device__ double dx1_forward(unsigned int l, double *f) {
	return -0.5*(3.0*f[l] - 4.0*f[n3[l]] + f[n3[n3[l]]]) / hx;
}
__device__ double dx1_back(unsigned int l, double *f) {
	return  0.5*(3.0*f[l] - 4.0*f[n1[l]] + f[n1[n1[l]]]) / hx;
}
__device__ double dy1_up(unsigned int l, double *f) {
	return  -0.5*(3.0*f[l] - 4.0*f[n2[l]] + f[n2[l]]) / hy;
}
__device__ double dy1_down(unsigned int l, double *f) {
	return  0.5*(3.0*f[l] - 4.0*f[n4[l]] + f[n4[l]]) / hy;
}

__device__ double dx2_forward(unsigned int l, double *f) {
	return (2.0 * f[l] - 5.0 * f[n3[l]] + 4.0 * f[n3[n3[l]]] - f[n3[n3[n3[l]]]]) / hx / hx;
}
__device__ double dx2_back(unsigned int l, double *f) {
	return (2.0 * f[l] - 5.0 * f[n1[l]] + 4.0 * f[n1[n1[l]]] - f[n1[n1[n1[l]]]]) / hx / hx;
}
__device__ double dy2_up(unsigned int l, double *f) {
	return (2.0 * f[l] - 5.0 * f[n2[l]] + 4.0 * f[n2[n2[l]]] - f[n2[n2[n2[l]]]]) / hy / hy;
}
__device__ double dy2_down(unsigned int l, double *f) {
	return (2.0 * f[l] - 5.0 * f[n4[l]] + 4.0 * f[n4[n4[l]]] - f[n4[n4[n4[l]]]]) / hy / hy;
}

__device__ double r_gamma(unsigned int l)
{
	return (J_back[l] - (J_back[l] / OFFSET)*OFFSET) * hx*cosA +    //cosA*x
		(J_back[l] / OFFSET) * hy*sinA;					   //sinA*y
}
__device__ double x_gamma(unsigned int l) {
	return (J_back[l] - (J_back[l] / OFFSET)*OFFSET) * hx*cosA;    //cosA*x
}
__device__ double y_gamma(unsigned int l) {
	return 	(J_back[l] / OFFSET) * hy*sinA;					   //sinA*y
}

__global__ void chemical_potential(double *mu, double *C)
{
	unsigned int l = threadIdx.x + blockIdx.x*blockDim.x;

	if (l < n)
	{
		switch (t[l])
		{
		case 0: //inner
			mu[l] = -Gr* r_gamma(l) //nu takoe
				+ 2.0 * A * C[l] + 4 * pow(C[l], 3.0) - Ca*(dx2(l, C) + dy2(l, C));
			break;
		case 1: //left rigid
			mu[l] = dx1_eq_0_forward(l, mu);
			break;
		case 2: //upper rigid
			mu[l] = dy1_eq_0_down(l, mu);
			break;
		case 3: //right rigid
			mu[l] = dx1_eq_0_back(l, mu);
			break;
		case 4: //lower rigid
			mu[l] = dy1_eq_0_up(l, mu);
			break;
		case 5: //left upper rigid corner
			mu[l] = 0.5* (dx1_eq_0_forward(l, mu) + dy1_eq_0_down(l, mu));
			break;
		case 6: //right upper rigid corner
			mu[l] = 0.5* (dx1_eq_0_back(l, mu) + dy1_eq_0_down(l, mu));
			break;
		case 7: //right lower rigid corner
			mu[l] = 0.5* (dx1_eq_0_back(l, mu) + dy1_eq_0_up(l, mu));
			break;
		case 8: //left lower rigid corner
			mu[l] = 0.5* (dx1_eq_0_forward(l, mu) + dy1_eq_0_up(l, mu));
			break;
		case 9: //inlet (from left)
			mu[l] = -Ca*dx2_forward(l, mu) - Gr* r_gamma(l);
			break;
		case 10://outlet (to right)
			mu[l] = -Ca*dx2_back(l, mu) - Ca*dy1(l, C) + 2.0 * A * C[l] + 4 * pow(C[l], 3.0) - Gr* r_gamma(l);
			break;
		default:
			break;
		}

	}
}

__global__ void quasi_velocity(double *ux, double *uy, double *vx, double *vy, double *C0, double *mu) {

	unsigned int l = threadIdx.x + blockIdx.x*blockDim.x;


	if (l < n)
	{

		switch (t[l])
		{
		case 0: //inner
				//ux_d
			ux[l] = vx[l]
				+ tau  * (
					-vx[l] * dx1(l, vx) - vy[l] * dy1(l, vx)
					+ (dx2(l, vx) + dy2(l, vx)) / Re
					- C0[l] * dx1(l, mu) / MM
					+ Gr*C0[l] * x_gamma(l)
					);
			//uy_d
			uy[l] = vy[l]
				+ tau  * (
					-vx[l] * dx1(l, vy) - vy[l] * dy1(l, vy)
					+ (dx2(l, vy) + dy2(l, vy)) / Re
					- C0[l] * dy1(l, mu) / MM
					+ Gr*C0[l] * y_gamma(l)
					);
			break;
		case 1: //left rigid
			ux[l] = tau / Re * dx2_forward(l, vx);
			break;
		case 2: //upper rigid
			uy[l] = tau / Re * dy2_down(l, vy);
			break;
		case 3: //right rigid
			ux[l] = tau / Re * dx2_back(l, vx);
			break;
		case 4: //lower rigid
			uy[l] = tau / Re * dy2_up(l, vy);
			break;
		case 5: //left upper rigid corner
			ux[l] = vx[l]
				+ tau  * (
					+(dx2_forward(l, vx) + dy2_down(l, vx)) / Re
					- C0[l] * dx1(l, mu) / MM
					);
			uy[l] = vy[l]
				+ tau  * (
					+(dx2_forward(l, vy) + dy2_down(l, vy)) / Re
					- C0[l] * dy1(l, mu) / MM
					);
			break;
		case 6: //right upper rigid corner
			ux[l] = vx[l]
				+ tau  * (
					+(dx2_back(l, vx) + dy2_down(l, vx)) / Re
					- C0[l] * dx1(l, mu) / MM
					);
			uy[l] = vy[l]
				+ tau  * (
					+(dx2_back(l, vy) + dy2_down(l, vy)) / Re
					- C0[l] * dy1(l, mu) / MM
					);
			break;
		case 7: //right lower rigid corner
			ux[l] = vx[l]
				+ tau  * (
					+(dx2_back(l, vx) + dy2_up(l, vx)) / Re
					- C0[l] * dx1(l, mu) / MM
					);
			uy[l] = vy[l]
				+ tau  * (
					+(dx2_back(l, vy) + dy2_up(l, vy)) / Re
					- C0[l] * dy1(l, mu) / MM
					);
			break;
		case 8: //left lower rigid corner
			ux[l] = vx[l]
				+ tau  * (
					+(dx2_forward(l, vx) + dy2_up(l, vx)) / Re
					- C0[l] * dx1(l, mu) / MM
					);
			uy[l] = vy[l]
				+ tau  * (
					+(dx2_forward(l, vy) + dy2_up(l, vy)) / Re
					- C0[l] * dy1(l, mu) / MM
					);
			break;
		case 9: //inlet (from left)
			ux[l] = vx[l] + tau*(
				-vx[l] * dx1_forward(l, vx) - vy[l] * dy1(l, vx)
				+ (dx2_forward(l, vx) + dy2(l, vx)) / Re
				- C0[l] * dx1_forward(l, mu) / MM
				);

			uy[l] = tau * (
				(dx2_forward(l, vy) + dy2(l, vy)) / Re  //  !быть может, !тут нужно дополнить
				+ C0[l] * dy1(l, mu)
				);
			break;
		case 10: //outlet (to right)
			ux[l] = vx[l] + tau*(
				-vx[l] * dx1_back(l, vx) - vy[l] * dy1(l, vx)
				+ (dx2_back(l, vx) + dy2(l, vx)) / Re
				- C0[l] * dx1_back(l, mu) / MM
				);
			uy[l] = tau * (
				(dx2_back(l, vy) + dy2(l, vy)) / Re
				+ C0[l] * dy1(l, mu)
				);
			break;
		default:
			break;
		}

	}
}

__global__ void concentration(double *C, double *C0, double *vx, double *vy, double *mu) {


	unsigned int l = threadIdx.x + blockIdx.x*blockDim.x;


	if (l < n)
	{

		switch (t[l])
		{
		case 0: //inner
			C[l] = C0[l]
				+ tau * (
					-vx[l] * dx1(l, C0)
					- vy[l] * dy1(l, C0)
					+ (dx2(l, C0) + dy2(l, C0)) / Pe // it should be 'mu' in the general case !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
					);
			break;
		case 1: //left rigid
			C[l] = dx1_eq_0_forward(l, C0);
			break;
		case 2: //upper rigid
			C[l] = dy1_eq_0_down(l, C0);
			break;
		case 3: //right rigid
			C[l] = dx1_eq_0_back(l, C0);
			break;
		case 4: //lower rigid
			C[l] = dy1_eq_0_up(l, C0);
			break;
		case 5: //left upper rigid corner
			C[l] = 0.5* (dx1_eq_0_forward(l, C0) + dy1_eq_0_down(l, C0));
			break;
		case 6: //right upper rigid corner
			C[l] = 0.5* (dx1_eq_0_back(l, C0) + dy1_eq_0_down(l, C0));
			break;
		case 7: //right lower rigid corner
			C[l] = 0.5* (dx1_eq_0_back(l, C0) + dy1_eq_0_up(l, C0));
			break;
		case 8: //left lower rigid corner
			C[l] = 0.5* (dx1_eq_0_forward(l, C0) + dy1_eq_0_up(l, C0));
			break;
		case 9: //inlet (from left)
			C[l] = -0.5;
			break;
		case 10://outlet (to right)
			C[l] = dx1_eq_0_back(l, C0);
			break;
		default:
			break;
		}


	}


}

__global__ void velocity_correction(double *vx, double *vy, double *ux, double *uy, double *p) {

	unsigned int l = threadIdx.x + blockIdx.x*blockDim.x;
	if (l < n)
	{

		switch (t[l])
		{
		case 0: //inner
			vx[l] = ux[l] - tau * dx1(l, p);
			vy[l] = uy[l] - tau * dy1(l, p);
			break;
		case 1: //left rigid
			vx[l] = 0.0; vy[l] = 0.0;
			break;
		case 2: //upper rigid
			vx[l] = 0.0; vy[l] = 0.0;
			break;
		case 3: //right rigid
			vx[l] = 0.0; vy[l] = 0.0;
			break;
		case 4: //lower rigid
			vx[l] = 0.0; vy[l] = 0.0;
			break;
		case 5: //left upper rigid corner
			vx[l] = 0.0; vy[l] = 0.0;
			break;
		case 6: //right upper rigid corner
			vx[l] = 0.0; vy[l] = 0.0;
			break;
		case 7: //right lower rigid corner
			vx[l] = 0.0; vy[l] = 0.0;
			break;
		case 8: //left lower rigid corner
			vx[l] = 0.0; vy[l] = 0.0;
			break;
		case 9: //inlet (from left)
			vx[l] = ux[l] - tau * dx1_forward(l, p);
			vy[l] = uy[l] - tau * dy1(l, p);
			break;
		case 10: //outlet (to right)
			vx[l] = ux[l] - tau * dx1_back(l, p);
			vy[l] = uy[l] - tau * dy1(l, p);
			break;
		default:
			break;
		}
	}
}

__global__ void Poisson(double *p, double *p0, double *ux, double *uy, double *mu, double *C)
{

	unsigned int l = threadIdx.x + blockIdx.x*blockDim.x;
	if (l < n)
	{
		switch (t[l])
		{
		case 0: //inner
			p[l] = p0[l] + tau_p*(
				-(dx1(l, ux) + dy1(l, uy)) / tau
				+ dx2(l, p0) + dy2(l, p0)
				);
			break;
		case 1: //left rigid
			p[l] = dx1_eq_0_forward(l, p0) + ux[l] * 2.0 * hx / tau / 3.0;
			break;
		case 2: //upper rigid
			p[l] = dy1_eq_0_down(l, p0) - uy[l] * 2.0 * hy / tau / 3.0;
			break;
		case 3: //right rigid
			p[l] = dx1_eq_0_back(l, p0) - ux[l] * 2.0 * hx / tau / 3.0;
			break;
		case 4: //lower rigid
			p[l] = dy1_eq_0_up(l, p0) + uy[l] * 2.0 * hy / tau / 3.0;
			break;
		case 5: //left upper rigid corner
			p[l] = 0.5* (dx1_eq_0_forward(l, p0) + ux[l] * 2.0 * hx / tau / 3.0
				+ dy1_eq_0_down(l, p0) - uy[l] * 2.0 * hy / tau / 3.0);
			break;
		case 6: //right upper rigid corner
			p[l] = 0.5* (dx1_eq_0_back(l, p0) - ux[l] * 2.0 * hx / tau / 3.0
				+ dy1_eq_0_down(l, p0) - uy[l] * 2.0 * hy / tau / 3.0);
			break;
		case 7: //right lower rigid corner
			p[l] = 0.5* (dx1_eq_0_back(l, p0) - ux[l] * 2.0 * hx / tau / 3.0
				+ dy1_eq_0_up(l, p0) + uy[l] * 2.0 * hy / tau / 3.0);
			break;
		case 8: //left lower rigid corner
			p[l] = 0.5* (dx1_eq_0_forward(l, p0) + ux[l] * 2.0 * hx / tau / 3.0
				+ dy1_eq_0_up(l, p0) + uy[l] * 2.0 * hy / tau / 3.0);
			break;
		case 9: //inlet (from left)
			p[l] = 8.0 / Re*Lx;
			+(0.5*Ca*pow(dx1_forward(l, C), 2)
				+ -mu[l] * C[l]
				+ A*pow(C[l], 2) + pow(C[l], 4)
				- Gr*C[l] * r_gamma(l)) / MM;
			break;
		case 10://outlet (to right)
			p[l] = 0;
			+(0.5*Ca*pow(dx1_back(l, C), 2)
				+ -mu[l] * C[l]
				+ A*pow(C[l], 2) + pow(C[l], 4)
				- Gr*C[l] * r_gamma(l)) / MM;
			break;
		default:
			break;
		}
	}
}

__global__ void reduction00(double *data, unsigned int n, double* reduced) {
	extern __shared__ double shared[];


	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x*(blockDim.x * 2) + threadIdx.x;

	shared[tid] = (i < n) ? abs(data[i]) : 0;

	if (i + blockDim.x < n)	shared[tid] += abs(data[i + blockDim.x]);


	__syncthreads();


	// do reduction in shared mem
	for (unsigned int s = blockDim.x / 2; s>32; s >>= 1)
	{
		if (tid < s)
		{
			shared[tid] += shared[tid + s];
		}

		__syncthreads();
	}


	if (tid < 32)
	{
		// Fetch final intermediate sum from 2nd warp
		if (blockDim.x >= 64) shared[tid] += shared[tid + 32];
		// Reduce final warp using shuffle
		for (int offset = warpSize / 2; offset > 0; offset /= 2)
		{
			shared[tid] += __shfl_down(shared[tid], offset);
		}
	}



	if (tid == 0) {
		reduced[blockIdx.x] = shared[0];
	}


}

__global__ void reduction0(double *data, unsigned int n, double* reduced) {
	extern __shared__ double shared[];


	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;

	if (i < n) {
		shared[tid] = abs(data[i]);
	}
	else
	{
		shared[tid] = 0.0;
	}

	__syncthreads();


	// do reduction in shared mem
	for (unsigned int s = blockDim.x / 2; s>0; s >>= 1)
	{
		if (tid < s)
		{
			shared[tid] += shared[tid + s];
		}

		__syncthreads();
	}

	if (tid == 0) {
		reduced[blockIdx.x] = shared[0];
	}


}
__global__ void reduction(double *data, unsigned int n, double* reduced) {
	extern  __shared__  double shared[];

	unsigned int tid = threadIdx.x;
	//unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
	unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
	if (i < n) {
		shared[tid] = abs(data[i]);
		//if (i + blockDim.x  < n) shared[tid] += abs(data[i + blockDim.x]);
	}
	else
	{
		shared[tid] = 0.0;
	}

	__syncthreads();

	if (blockDim.x >= 1024) {
		if (tid < 512) { shared[tid] += shared[tid + 512]; } __syncthreads();
	}
	if (blockDim.x >= 512) {
		if (tid < 256) { shared[tid] += shared[tid + 256]; } __syncthreads();
	}
	if (blockDim.x >= 256) {
		if (tid < 128) { shared[tid] += shared[tid + 128]; } __syncthreads();
	}
	if (blockDim.x >= 128) {
		if (tid < 64) { shared[tid] += shared[tid + 64]; } __syncthreads();
	}
	if (tid < 32)
	{
		if (blockDim.x >= 64) shared[tid] += shared[tid + 32];
		if (blockDim.x >= 32) shared[tid] += shared[tid + 16];
		if (blockDim.x >= 16) shared[tid] += shared[tid + 8];
		if (blockDim.x >= 8) shared[tid] += shared[tid + 4];
		if (blockDim.x >= 4) shared[tid] += shared[tid + 2];
		if (blockDim.x >= 2) shared[tid] += shared[tid + 1];
	}




	if (tid == 0) {
		reduced[blockIdx.x] = shared[0];
		//if (blockDim.x==1) *last = shared[0];
	}


}


__global__ void swap_one(double* f_old, double* f_new) {
	unsigned int l = blockIdx.x*blockDim.x + threadIdx.x;
	if (l < n)	f_old[l] = f_new[l];
}
__global__ void swap_3(double* f1_old, double* f1_new, double* f2_old, double* f2_new, double* f3_old, double* f3_new) {
	unsigned int l = blockIdx.x*blockDim.x + threadIdx.x;
	if (l < n)
	{
		f1_old[l] = f1_new[l];
		f2_old[l] = f2_new[l];
		f3_old[l] = f3_new[l];
	}
}

struct cross
{
	int nx[5], ny[5];
	int offset[5];
	int size[5], total_size;
	int idx, idy, id;
	int block[5];

	__host__ __device__ void set_geometry(unsigned int length, unsigned int height)
	{
		nx[0] = height; ny[0] = height;
		nx[1] = length - 1; ny[1] = height;
		nx[2] = height; ny[2] = length - 1;
		nx[3] = length - 1; ny[3] = height;
		nx[4] = height; ny[4] = length - 1;

		total_size = 0;
		for (int i = 0; i < 5; i++)
		{
			size[i] = (nx[i] + 1)*(ny[i] + 1);
			offset[i] = nx[i] + 1;
			total_size += size[i];
		}
	};

	__host__ __device__ void delete_block(int i)
	{
		total_size -= size[i];
		nx[i] = -1;
		ny[i] = -1;
		offset[i] = 0;
		size[i] = 0;
	};

	__host__ __device__ void set_block(int add) {
		block[0] = 0 + add;
		block[1] = block[0] + size[0];
		block[2] = block[1] + size[1];
		block[3] = block[2] + size[2];
		block[4] = block[3] + size[3];
	}

	//~cross() {}

};

struct multi_cross {

	cross *Mcr;
	int *l, *t, *I, *J, *J_back;
	int Mx, My, Msize, Moffset, OFFSET;
	unsigned int iter = 0;
	unsigned int TOTAL_SIZE = 0;
	int *n1, *n2, *n3, *n4;
	unsigned int nxg, nyg;
	double *C0, *C, *p, *p0, *ux, *uy, *vx, *vy, *mu;
	double LX, LY;

	void set_global_size(int input_nx, int input_ny, int input_Mx, int input_My) {

		Mx = input_Mx - 1; My = input_My - 1; Msize = input_Mx*input_My; Moffset = input_Mx;
		//Mcr.resize(Msize, cr);
		Mcr = new cross[Msize];

		for (int i = 0; i < Msize; i++) {
			Mcr[i].set_geometry(input_nx, input_ny);
		}

		for (int i = 0; i <= Mx; i++) {
			for (int j = 0; j <= My; j++) {
				Mcr[i + Moffset*j].id = i + Moffset*j;
				Mcr[i + Moffset*j].idx = i;
				Mcr[i + Moffset*j].idy = j;
				if (j == 0) Mcr[i + Moffset*j].delete_block(4);
				if (j == My) Mcr[i + Moffset*j].delete_block(2);
			}
		}


		for (int i = 0; i < Msize; i++) {
			Mcr[i].set_block(TOTAL_SIZE);
			TOTAL_SIZE += Mcr[i].total_size;
		}

		/*
		C = new double[TOTAL_SIZE];
		C0 = new double[TOTAL_SIZE];
		ux = new double[TOTAL_SIZE];
		uy = new double[TOTAL_SIZE];
		vx = new double[TOTAL_SIZE];
		vy = new double[TOTAL_SIZE];
		p = new double[TOTAL_SIZE];
		p0 = new double[TOTAL_SIZE];
		mu = new double[TOTAL_SIZE];

		for (int i = 0; i < TOTAL_SIZE; i++) {
			C[i] = 0;
			C0[i] = 0;
			p[i] = 0;
			p0[i] = 0;
			ux[i] = 0;
			uy[i] = 0;
			vx[i] = 0;
			vy[i] = 0;
			mu[i] = 0;
		}
		*/
	}

	void set_type() {
		if (Msize == 0) {
			printf("hop hey la la ley, stop it, bro, ya doin it wron' \n");
		}

		l = new int[TOTAL_SIZE];
		t = new int[TOTAL_SIZE];
		for (int i = 0; i < TOTAL_SIZE; i++) {
			l[i] = 0;
			t[i] = 0;
		}

		unsigned int k;
		for (int jm = 0; jm <= My; jm++) {
			for (int im = 0; im <= Mx; im++) {

				k = im + Moffset*jm;
				for (unsigned int q = 0; q < 5; q++)
				{
					if (Mcr[k].size[q] == 0) continue;
					for (int j = 0; j <= Mcr[k].ny[q]; j++) {
						for (int i = 0; i <= Mcr[k].nx[q]; i++) {
							l[iter] = iter;

							if (q == 0) {
								if (i == 0 && j == Mcr[k].ny[q]) t[iter] = 5;
								if (i == Mcr[k].nx[q] && j == Mcr[k].ny[q]) t[iter] = 6;
								if (i == Mcr[k].nx[q] && j == 0) t[iter] = 7;
								if (i == 0 && j == 0) t[iter] = 8;
								if (i == 0 && Mcr[k].size[1] == 0) t[iter] = 1;
								if (i == Mcr[k].nx[q] && Mcr[k].size[3] == 0) t[iter] = 3;
								if (j == 0 && Mcr[k].size[4] == 0) t[iter] = 4;
								if (j == Mcr[k].ny[q] && Mcr[k].size[2] == 0) t[iter] = 2;
							}

							if (q == 2) {
								if (i == 0) t[iter] = 1;
								if (i == Mcr[k].nx[q]) t[iter] = 3;
							}
							if (q == 4) {
								if (i == 0) t[iter] = 1;
								if (i == Mcr[k].nx[q]) t[iter] = 3;
							}


							if (im == 0 && i == 0 && q == 1) t[iter] = 9;
							if (im == Mx && i == Mcr[k].nx[q] && q == 3) t[iter] = 10;

							if (q == 1) {
								if (j == Mcr[k].ny[q]) t[iter] = 2;
								if (j == 0) t[iter] = 4;
							}
							if (q == 3) {
								if (j == Mcr[k].ny[q]) t[iter] = 2;
								if (j == 0) t[iter] = 4;
							}

							iter++;
						}
					}
				}

			}
		}

	}

	void set_neighbor()
	{
		if (Msize == 0 || iter == 0) {
			printf("hop hey la la ley, stop it, bro, ya doin it wron' \n");
		}


		n1 = (int*)malloc(TOTAL_SIZE * sizeof(int));
		n2 = (int*)malloc(TOTAL_SIZE * sizeof(int));
		n3 = (int*)malloc(TOTAL_SIZE * sizeof(int));
		n4 = (int*)malloc(TOTAL_SIZE * sizeof(int));
		for (int i = 0; i < TOTAL_SIZE; i++) {
			n1[i] = -1; n2[i] = -1; n3[i] = -1; n4[i] = -1;
		}

		unsigned int k, it = 0;
		for (int jm = 0; jm <= My; jm++) {
			for (int im = 0; im <= Mx; im++) {

				k = im + Moffset*jm;
				for (unsigned int q = 0; q < 5; q++)
				{
					if (Mcr[k].size[q] == 0) continue;
					for (int j = 0; j <= Mcr[k].ny[q]; j++) {
						for (int i = 0; i <= Mcr[k].nx[q]; i++) {
							//cout << i << " " << j << " " << q  << " " << it<< endl;

							//joint central pore and tubes
							if (q == 0) {

								if (i == 0 && (t[it] == 0 || t[it] == 2 || t[it] == 4 || t[it] == 5 || t[it] == 8)) {
									//n1[it] = it - (i + Mcr[k].offset[0]) + Mcr[k].size[0] + Mcr[k].nx[1] + Mcr[k].offset[1] * j;
									int in = Mcr[k].block[1] + Mcr[k].nx[1] + Mcr[k].offset[1] * j;
									n1[it] = in;
									n3[in] = it;

								}


								if (i == Mcr[k].nx[0] && (t[it] == 0 || t[it] == 2 || t[it] == 4 || t[it] == 6 || t[it] == 7)) {
									int in = Mcr[k].block[3] + Mcr[k].offset[3] * j;
									n3[it] = in;
									n1[in] = it;
								}

								if (j == 0 && (t[it] == 0 || t[it] == 1 || t[it] == 3 || t[it] == 7 || t[it] == 8)) {
									int in = Mcr[k].block[4] + i + Mcr[k].offset[4] * Mcr[k].ny[4];
									n4[it] = in;
									n2[in] = it;
								}

								if (j == Mcr[k].ny[0] && (t[it] == 0 || t[it] == 1 || t[it] == 3 || t[it] == 5 || t[it] == 6)) {
									int in = Mcr[k].block[2] + i;
									n2[it] = in;
									n4[in] = it;
								}

							}



							//inner nodes
							if (i < Mcr[k].nx[q])  n3[it] = it + 1;
							if (i > 0)             n1[it] = it - 1;


							if (j < Mcr[k].ny[q])  n2[it] = it + Mcr[k].offset[q];
							if (j > 0)			   n4[it] = it - Mcr[k].offset[q];




							//borders and inlet/outlet
							if (i == 0 && (t[it] == 1 || t[it] == 9)) {
								n1[it] = it + 2;
								n3[it] = it + 1;
							}

							if (i == Mcr[k].nx[q] && (t[it] == 3 || t[it] == 10)) {
								n1[it] = it - 1;
								n3[it] = it - 2;
							}



							if (t[it] == 4) {
								n2[it] = it + Mcr[k].offset[q];
								n4[it] = it + 2 * Mcr[k].offset[q];
							}
							if (t[it] == 2) {
								n2[it] = it - 2 * Mcr[k].offset[q];
								n4[it] = it - Mcr[k].offset[q];
							}


							//join crosses
							if (q == 3) {
								if (i == Mcr[k].nx[3] && (t[it] == 0 || t[it] == 2 || t[it] == 4) && im < Mx) {
									int in = Mcr[k + 1].block[1] + Mcr[k + 1].offset[1] * j;
									n3[it] = in;
									n1[in] = it;
								}
							}


							if (q == 2) {
								if (j == Mcr[k].ny[2] && (t[it] == 0 || t[it] == 1 || t[it] == 3) && jm < My) {
									int in = Mcr[k + Moffset].block[4] + i;
									n4[in] = it;
									n2[it] = in;
									//printf("n4 = %i n2 = %i\n", n4[in], n2[it]);

								}
							}

							//if (n2[it] == -1) printf("q=%i t=%i i=%i j=%i nx=%i ny=%i  \n", q, t[it], i, j, Mcr[k].nx[q], Mcr[k].ny[q]);
							it++;
						}
					}
				}
			}
		}



	}

	void set_global_id() {
		nxg = 0, nyg = 0;
		for (int im = 0; im <= Mx; im++) 	nxg += Mcr[im].nx[0] + 1 + Mcr[im].nx[1] + 1 + Mcr[im].nx[3] + 1;
		for (int jm = 0; jm <= My; jm++)	nyg += Mcr[jm*Moffset].ny[0] + 1 + Mcr[jm*Moffset].ny[2] + 1 + Mcr[jm*Moffset].ny[4] + 1;
		if (nxg != 0) nxg--; if (nyg != 0) nyg--;


		I = new int[(nxg + 1)*(nyg + 1)];
		J = new int[(nxg + 1)*(nyg + 1)];
		J_back = new int[TOTAL_SIZE];

		OFFSET = nxg + 1;


		for (unsigned int i = 0; i < (nxg + 1)*(nyg + 1); i++) {
			I[i] = 0; J[i] = -1;
		}
		for (int i = 0; i < TOTAL_SIZE; i++) {
			J_back[i] = -1;
		}

		int *shift_x, *shift_y;
		shift_x = new int[Mx + 1];
		shift_y = new int[My + 1];
		shift_x[0] = 0;
		shift_y[0] = 0;

		for (int im = 1; im <= Mx; im++) shift_x[im] = (Mcr[im - 1].nx[0] + 1 + Mcr[im - 1].nx[1] + 1 + Mcr[im - 1].nx[3] + 1 + shift_x[im - 1]);
		for (int jm = 1; jm <= My; jm++) shift_y[jm] = (Mcr[(jm - 1)*Moffset].ny[0] + 1 + Mcr[(jm - 1)*Moffset].ny[2] + 1 + Mcr[(jm - 1)*Moffset].ny[4] + 1 + shift_y[jm - 1]);

		if (Msize == 0 || iter == 0) {
			printf("hop hey la la ley, stop it, bro, ya doin it wron' \n");
		}


		unsigned int k, it = 0, in, ii, jj;
		for (int jm = 0; jm <= My; jm++) {
			for (int im = 0; im <= Mx; im++) {

				k = im + Moffset*jm;
				for (unsigned int q = 0; q < 5; q++)
				{
					if (Mcr[k].size[q] == 0) continue;
					for (int j = 0; j <= Mcr[k].ny[q]; j++) {
						for (int i = 0; i <= Mcr[k].nx[q]; i++) {

							if (q == 1) {
								ii = i + shift_x[im];
								jj = j + shift_y[jm] + (Mcr[k].ny[4] + 1);
							}

							if (q == 0) {
								ii = i + shift_x[im] + Mcr[k].nx[1] + 1;
								jj = j + shift_y[jm] + (Mcr[k].ny[4] + 1);
							}

							if (q == 3) {
								ii = i + shift_x[im] + Mcr[k].nx[1] + 1 + Mcr[k].nx[0] + 1;
								jj = j + shift_y[jm] + (Mcr[k].ny[4] + 1);
							}

							if (q == 2) {
								ii = i + shift_x[im] + Mcr[k].nx[1] + 1;
								jj = j + shift_y[jm] + (Mcr[k].ny[4] + 1) + (Mcr[k].ny[0] + 1);

							}

							if (q == 4) {
								ii = i + shift_x[im] + Mcr[k].nx[1] + 1;
								jj = j + shift_y[jm];
							}

							in = ii + OFFSET*jj;
							I[in] = 1;
							J[in] = it;
							J_back[it] = in;

							it++;
						}
					}
				}
			}
		}


	}

	void write_field(double *f, string file_name, double time, int step) {
#ifdef __linux__ 
		ofstream to_file(("fields/" + file_name + ".dat").c_str());
#endif
#ifdef _WIN32
		ofstream to_file(("fields\\" + file_name + ".dat").c_str());
#endif


		int l, L;
		to_file << time << endl;
		for (int j = 0; j <= nyg; j = j + step) {
			for (int i = 0; i <= nxg; i = i + step) {
				l = i + OFFSET*j; L = J[l];
				//if (J[l] == J[l]) to_file << i << " " << j << " " << f[L] << endl;
				if (I[l] == 1) {
					//to_file << i << " " << j << " " << f[L] << " " << t[L] << " " << L << " " << n1[L] << " " << n2[L] << " " << n3[L] << " " << n4[L] 	<< endl;
					to_file << i << " " << j << " " << f[L] << endl;
				}
				else
				{
					to_file << "skip" << endl;
					//to_file << i << " " << j << " " << NAN << endl;
					//to_file << i << " " << j << " " << 0 << " " << 0 << " " << 0 << " " << 0 << " " << 0 << " " << 0 << " " << 0 	<< endl;
				}

			}
		}
		to_file.close();

	}

	//left to be normal
	void left_normal_in(int first, int last)
	{
		unsigned int k, it = 0;
		for (int jm = 0; jm <= My; jm++) {
			for (int im = 0; im <= Mx; im++) {

				k = im + Moffset*jm;
				for (unsigned int q = 0; q < 5; q++)
				{
					if (Mcr[k].size[q] == 0) continue;
					for (int j = 0; j <= Mcr[k].ny[q]; j++) {
						for (int i = 0; i <= Mcr[k].nx[q]; i++) {

							//if (Mcr[k].idx == 0 && Mcr[k].idy >= in_y)
							if (Mcr[k].idx == 0 && (Mcr[k].idy < first || Mcr[k].idy > last))
								if (t[it] == 9) t[it] = 1;
		



							it++;
						}
					}
				}
			}
		}

	}
	void left_normal_out(int first, int last)
	{
		unsigned int k, it = 0;
		for (int jm = 0; jm <= My; jm++) {
			for (int im = 0; im <= Mx; im++) {

				k = im + Moffset*jm;
				for (unsigned int q = 0; q < 5; q++)
				{
					if (Mcr[k].size[q] == 0) continue;
					for (int j = 0; j <= Mcr[k].ny[q]; j++) {
						for (int i = 0; i <= Mcr[k].nx[q]; i++) {

							//if (Mcr[k].idx == Mx && Mcr[k].idy <= My - out_y)
							if (Mcr[k].idx == Mx && (Mcr[k].idy < first || Mcr[k].idy > last))
								if (t[it] == 10) t[it] = 3;



							it++;
						}
					}
				}
			}
		}

	}

	double permeability() {
		return 0;
	}
	void save(double *vx, double *vy, double *p, double *C, double *mu, unsigned int i_time, unsigned int i_write) {

		ofstream to_file("recovery.dat");
		ofstream to_file2("recovery2.dat");

		to_file << i_time << " " << i_write << endl;
		to_file2 << i_time << " " << i_write << endl;


		for (int i = 0; i < TOTAL_SIZE; i++)
			to_file << vx[i] << " " << vy[i] << " " << p[i] << " " << C[i] << " " << mu[i] << endl;
		for (int i = 0; i < TOTAL_SIZE; i++)
			to_file2 << vx[i] << " " << vy[i] << " " << p[i] << " " << C[i] << " " << mu[i] << endl;




		to_file.close();
		to_file2.close();

	}

	void recover(double *vx, double *vy, double *p, double *C, double *mu, unsigned int &i_time, unsigned int &i_write) {
		ifstream from_file("recovery.dat");

		string str;
		string substr;
		stringstream ss;


		getline(from_file, str);

		ss << str;
		ss >> substr; i_time = atoi(substr.c_str());
		ss >> substr; i_write = atoi(substr.c_str());

		for (int i = 0; i < TOTAL_SIZE; i++) {
			getline(from_file, str);
			ss.str(""); ss.clear();
			ss << str;
			ss >> substr; vx[i] = atof(substr.c_str());
			ss >> substr; vy[i] = atof(substr.c_str());
			ss >> substr; p[i] = atof(substr.c_str());
			ss >> substr; C[i] = atof(substr.c_str());
		}


		from_file.close();
	}



};

struct multi_line {

	int *l, *t, *I, *J, *J_back;
	unsigned int line_N; // number of lines of the whole porous media
	unsigned int tube_N; // number of tubes per line
	unsigned int iter = 0;
	unsigned int TOTAL_SIZE = 0;
	unsigned int OFFSET;
	int *n1, *n2, *n3, *n4;
	unsigned int nxg, nyg;
	double *C0, *C, *p, *p0, *ux, *uy, *vx, *vy, *mu;
	unsigned int x; //length of porous block
	unsigned int y; //width of porous block
	unsigned int z; //width of capillary tube
	unsigned int line_x; //length of line 
	unsigned int line_y; //width of line
	double hx, hy, Lx, Ly;
	unsigned int *gi, *gj;
	unsigned int shiftX = 0, shiftY = 0;
	vector <unsigned int> li, lj;



	void generate_levels(unsigned int x_in, unsigned int y_in, unsigned int z_in, unsigned int N_in, unsigned int tube_in) {
		x = x_in; y = y_in; z = z_in; line_N = N_in; tube_N = tube_in;
		line_x = 2 * (x + z);
		line_y = tube_N*(z + y) - y;
		hy = 1.0 / z; hx = hy; Lx = line_x*hx; Ly = line_y*hy;
		nyg = line_y; nxg = line_N*line_x + x; OFFSET = nxg + 1;

		J = new int[(nxg + 1)*(nyg + 1)];
		I = new int[(nxg + 1)*(nyg + 1)];
		for (int i = 0; i < (nxg + 1)*(nyg + 1); i++) {
			J[i] = -1; I[i] = 0;
		}

		// zero-level
		while (shiftY < line_y) {
			for (int j = 0; j <= z; j++) {
				for (int i = 0; i <= x - 1; i++) {
					//gi[i] = 1; 	gj[j] = 1;
					li.push_back(i + shiftX);
					lj.push_back(j + shiftY);
					iter++;
				}
			}
			shiftY += y + z;
		}
		cout << iter << endl;

		// main levels
		for (int C = 1; C <= line_N; C++)
		{

			//column
			shiftX += x;
			for (int i = 0; i <= z; i++) {
				for (int j = 0; j <= line_y; j++) {
					li.push_back(i + shiftX);
					lj.push_back(j);
					iter++;
				}
			}

			//blocks
			shiftX += z;
			shiftY = (y + z) / 2;
			while (shiftY < line_y) {
				for (int i = 1; i <= x - 1; i++) {
					for (int j = 0; j <= z; j++) {
						li.push_back(i + shiftX);
						lj.push_back(j + shiftY);

						iter++;
					}
				}
				shiftY += y + z;
			}


			//column
			shiftX += x;

			for (int i = 0; i <= z; i++) {
				for (int j = 0; j <= line_y; j++) {
					li.push_back(i + shiftX);
					lj.push_back(j);
					iter++;
				}
			}

			//blocks
			shiftX += z;
			shiftY = 0;
			while (shiftY < line_y) {
				for (int j = 0; j <= z; j++) {
					for (int i = 1; i <= x - 1; i++) {
						//gi[i] = 1; 	gj[j] = 1;
						li.push_back(i + shiftX);
						lj.push_back(j + shiftY);
						iter++;
					}

					if (C == line_N) {
						li.push_back(x + shiftX);
						lj.push_back(j + shiftY);
						iter++;
					}
				}

				shiftY += y + z;
			}

		}


		TOTAL_SIZE = iter;
		J_back = new int[TOTAL_SIZE];
		n1 = new int[TOTAL_SIZE];
		n2 = new int[TOTAL_SIZE];
		n3 = new int[TOTAL_SIZE];
		n4 = new int[TOTAL_SIZE];
		t = new int[TOTAL_SIZE];

		for (int i = 0; i < TOTAL_SIZE; i++) {
			n1[i] = -1;
			n2[i] = -1;
			n3[i] = -1;
			n4[i] = -1;
			t[i] = -1;
		}


		for (int i = 0; i < iter; i++) {
			J_back[i] = li[i] + OFFSET*lj[i];
			J[J_back[i]] = i;
			I[J_back[i]] = 1;
		}










	}

	void set_neighbor() {
		int l, L, l1, l2, l3, l4;


		for (int i = 0; i <= nxg; i++) {
			for (int j = 0; j <= nyg; j++) {
				l = i + OFFSET*j; L = J[l];
				l1 = i - 1 + OFFSET*j;
				l2 = i + OFFSET*j + OFFSET;
				l3 = i + 1 + OFFSET*j;
				l4 = i + OFFSET*j - OFFSET;
				if (I[l] == 1) {

					if (i > 0) if (I[l1] == 1) n1[L] = J[l1];
					if (i < nxg) if (I[l3] == 1) n3[L] = J[l3];
					if (j < nyg) if (I[l2] == 1) n2[L] = J[l2];
					if (j > 0) if (I[l4] == 1) n4[L] = J[l4];


				}

				else {
				}
			}
		}

	}



	void set_type() {
		int l, L, l1, l2, l3, l4;

		//inner
		for (int i = 0; i <= nxg; i++) {
			for (int j = 0; j <= nyg; j++) {
				l = i + OFFSET*j; L = J[l];
				if (I[l] == 1) {
					if (n1[L] != -1 && n2[L] != -1 && n3[L] != -1 && n4[L] != -1)
						t[L] = 0;

				}
			}
		}



		//rigid walls
		for (int i = 0; i <= nxg; i++) {
			for (int j = 0; j <= nyg; j++) {
				l = i + OFFSET*j; L = J[l];
				l1 = i - 1 + OFFSET*j;
				l2 = i + OFFSET*j + OFFSET;
				l3 = i + 1 + OFFSET*j;
				l4 = i + OFFSET*j - OFFSET;
				if (I[l] == 1) {
					if (I[l] == 1) {
						if (n1[L] == -1 && n2[L] != -1 && n3[L] != -1 && n4[L] != -1)
							t[L] = 1;
						if (n1[L] != -1 && n2[L] == -1 && n3[L] != -1 && n4[L] != -1)
							t[L] = 2;
						if (n1[L] != -1 && n2[L] != -1 && n3[L] == -1 && n4[L] != -1)
							t[L] = 3;
						if (n1[L] != -1 && n2[L] != -1 && n3[L] != -1 && n4[L] == -1)
							t[L] = 4;
					}

				}
			}
		}


		//corners
		for (int i = 0; i <= nxg; i++) {
			for (int j = 0; j <= nyg; j++) {
				l = i + OFFSET*j; L = J[l];
				if (I[l] == 1) {
					if (n2[n1[L]] == -1 && n1[L] != -1 && n2[L] != -1)
						t[L] = 5;
					if (n2[n3[L]] == -1 && n3[L] != -1 && n2[L] != -1)
						t[L] = 6;
					if (n3[n4[L]] == -1 && n3[L] != -1 && n4[L] != -1)
						t[L] = 7;
					if (n1[n4[L]] == -1 && n1[L] != -1 && n4[L] != -1)
						t[L] = 8;
				}
			}
		}

		//inlet, outlet
		for (int i = 0; i <= nxg; i = i + nxg) {
			for (int j = 0; j <= nyg; j++) {
				l = i + OFFSET*j; L = J[l];
				if (I[l] == 1) {
					if (i == 0) {
						t[L] = 9;
						if (t[n3[L]] == 2) t[L] = 2;
						if (t[n3[L]] == 4) t[L] = 4;
					}
					if (i == nxg) {
						t[L] = 10;
						if (t[n1[L]] == 2) t[L] = 2;
						if (t[n1[L]] == 4) t[L] = 4;
					}
				}
			}
		}




	}







	void check() {
		int l, L;
		ofstream write("out.txt");
		for (int i = 0; i <= nxg; i++) {
			for (int j = 0; j <= nyg; j++) {
				l = i + OFFSET*j; L = J[l];
				if (I[i + OFFSET*j] == 1)
					write << i << " " << j << " " << 1 << " " << L << " " << t[L] << " " << n1[L] << " " << n2[L] << " " << n3[L] << " " << n4[L] << endl;
				else write << i << " " << j << " " << -1 << " " << -1 << " " << -1 << " " << -1 << " " << -1 << " " << -1 << " " << -1 << endl;
			}
			}
		write.close();


		}
	void write_field(double *f, string file_name, double time, int step) {
#ifdef __linux__ 
		ofstream to_file(("fields/" + file_name + ".dat").c_str());
#endif
#ifdef _WIN32
		ofstream to_file(("fields\\" + file_name + ".dat").c_str());
#endif


		int l, L;
		to_file << time << endl;
		for (int j = 0; j <= nyg; j = j + step) {
			for (int i = 0; i <= nxg; i = i + step) {
				l = i + OFFSET*j; L = J[l];
				//if (J[l] == J[l]) to_file << i << " " << j << " " << f[L] << endl;
				if (I[l] == 1) {
					//to_file << i << " " << j << " " << f[L] << " " << t[L] << " " << L << " " << n1[L] << " " << n2[L] << " " << n3[L] << " " << n4[L] 	<< endl;
					to_file << i << " " << j << " " << f[L] << endl;
				}
				else
				{
					to_file << "skip" << endl;
					//to_file << i << " " << j << " " << NAN << endl;
					//to_file << i << " " << j << " " << 0 << " " << 0 << " " << 0 << " " << 0 << " " << 0 << " " << 0 << " " << 0 	<< endl;
				}

			}
		}
		to_file.close();

	}


	void save(double *vx, double *vy, double *p, double *C, double *mu, unsigned int i_time, unsigned int i_write) {

		ofstream to_file("recovery.dat");
		ofstream to_file2("recovery2.dat");

		to_file << i_time << " " << i_write << endl;
		to_file2 << i_time << " " << i_write << endl;


		for (int i = 0; i < TOTAL_SIZE; i++)
			to_file << vx[i] << " " << vy[i] << " " << p[i] << " " << C[i] << " " << mu[i] << endl;
		for (int i = 0; i < TOTAL_SIZE; i++)
			to_file2 << vx[i] << " " << vy[i] << " " << p[i] << " " << C[i] << " " << mu[i] << endl;




		to_file.close();
		to_file2.close();

	}

	void recover(double *vx, double *vy, double *p, double *C, double *mu, unsigned int &i_time, unsigned int &i_write) {
		ifstream from_file("recovery.dat");

		string str;
		string substr;
		stringstream ss;


		getline(from_file, str);

		ss << str;
		ss >> substr; i_time = atoi(substr.c_str());
		ss >> substr; i_write = atoi(substr.c_str());

		for (int i = 0; i < TOTAL_SIZE; i++) {
			getline(from_file, str);
			ss.str(""); ss.clear();
			ss << str;
			ss >> substr; vx[i] = atof(substr.c_str());
			ss >> substr; vy[i] = atof(substr.c_str());
			ss >> substr; p[i] = atof(substr.c_str());
			ss >> substr; C[i] = atof(substr.c_str());
		}


		from_file.close();
	}



	};




__global__ void stupid_alloc(unsigned int TS) {
	n1 = new int[TS];
	n2 = new int[TS];
	n3 = new int[TS];
	n4 = new int[TS];
	t = new int[TS];
	J_back = new int[TS];
}
__global__ void stupid_swap(int *nn1, int *nn2, int *nn3, int *nn4, int *tt, int* JJ, unsigned int TS) {

	//unsigned int l = blockIdx.x*blockDim.x + threadIdx.x;
	for (unsigned int l = 0; l < TS; l++) {
		//printf("%i \n", l);
		n1[l] = nn1[l];
		n2[l] = nn2[l];
		n3[l] = nn3[l];
		n4[l] = nn4[l];
		t[l] = tt[l];
		J_back[l] = JJ[l];
	}
}
void stupid_step(int *nn1, int *nn2, int *nn3, int *nn4, int *tt, int *JJ, unsigned int TS) {
	/*выделение всего того, но на GPU*/
	unsigned int TSB = TS * sizeof(int);
	int *n1_temp, *n2_temp, *n3_temp, *n4_temp, *t_temp, *J_temp;
	cudaMalloc((void**)&n1_temp, TSB);
	cudaMalloc((void**)&n2_temp, TSB);
	cudaMalloc((void**)&n3_temp, TSB);
	cudaMalloc((void**)&n4_temp, TSB);
	cudaMalloc((void**)&t_temp, TSB);
	cudaMalloc((void**)&J_temp, TSB);

	cudaMemcpy(n1_temp, nn1, TSB, cudaMemcpyHostToDevice);
	cudaMemcpy(n2_temp, nn2, TSB, cudaMemcpyHostToDevice);
	cudaMemcpy(n3_temp, nn3, TSB, cudaMemcpyHostToDevice);
	cudaMemcpy(n4_temp, nn4, TSB, cudaMemcpyHostToDevice);
	cudaMemcpy(t_temp, tt, TSB, cudaMemcpyHostToDevice);
	cudaMemcpy(J_temp, JJ, TSB, cudaMemcpyHostToDevice);
	stupid_alloc << <1, 1 >> > (TS);
	stupid_swap << <1, 1 >> > (n1_temp, n2_temp, n3_temp, n4_temp, t_temp, J_temp, TS); //so
	cudaFree(n1_temp);
	cudaFree(n2_temp);
	cudaFree(n3_temp);
	cudaFree(n4_temp);
	cudaFree(t_temp);
	cudaFree(J_temp);
}


int main() {
	int devID = 0;
	cudaSetDevice(devID);
	cudaDeviceProp deviceProp;
	cudaGetDeviceProperties(&deviceProp, devID);
	printf("\nDevice %d: \"%s\"\n", devID, deviceProp.name);

#ifdef __linux__ 
	system("mkdir -p fields/");
#endif

#ifdef _WIN32
	CreateDirectoryA("fields", NULL);
#endif

	//allocate heap size
	size_t limit = 1024 * 1024 * 1024;
	cudaDeviceSetLimit(cudaLimitMallocHeapSize, limit);
	cudaDeviceGetLimit(&limit, cudaLimitMallocHeapSize);

	double timer1, timer2;
	double pi = 3.1415926535897932384626433832795;
	double eps0 = 1e-5;
	double *C0, *C, *p, *p0, *ux, *uy, *vx, *vy, *mu, *p_true;  //_d - device (GPU) 
	double *C_h, *p_h, *ux_h, *uy_h, *vx_h, *vy_h, *mu_h, *p_true_h;  //_h - host (CPU)
	double *psiav_array, *psiav_array_h, *psiav_d, *psiav_h, psiav0_h, eps_h;		 //  temporal variables
	double hx_h, hy_h, Lx_h, Ly_h, tau_h, tau_p_h, m, psiav, psiav0, eps, alpha_h, sinA_h, cosA_h, A_h, Ca_h, Gr_h, Pe_h, Re_h, MM_h; //parameters 
	double Ek, Ek_old, Vmax, Q_in, Q_out, C_average, Cv;
	unsigned int nx_h, ny_h, Matrix_X, Matrix_Y, iter = 0, niter, nout, nxout, nyout, offset_h, kk, k, mx, my, border, tt, write_i = 0, each = 1;					  //parameters
	double Vxm, Vym, pm, Cm, each_t = 10.0;


	/*
	multi_line M_CROSS;
	M_CROSS.generate_levels(30, 30, 30, 3, 5);
	cout << "approximate memory amount = " << 100 * M_CROSS.TOTAL_SIZE / 1024 / 1024 << " MB" << endl << endl << endl;
	M_CROSS.set_neighbor();
	M_CROSS.set_type();
	pause
	*/


	ny_h = 60;	nx_h = 60;

	hy_h = 1.0 / ny_h;	hx_h = hy_h;

	each_t = 0.1;
	each = 1; //each #th node to write to a file
	Matrix_X = 5;
	Matrix_Y = 5;


	
	multi_cross M_CROSS;

	M_CROSS.set_global_size(nx_h, ny_h, Matrix_X, Matrix_Y);
	cout << "approximate memory amount = " << 100 * M_CROSS.TOTAL_SIZE / 1024 / 1024 << " MB" << endl << endl << endl;
	pause
	M_CROSS.set_type();
	//M_CROSS.left_normal_in((Matrix_Y - 1) / 2, (Matrix_Y - 1) / 2);
	//M_CROSS.left_normal_out((Matrix_Y - 1) / 2, (Matrix_Y - 1) / 2);
	M_CROSS.set_neighbor();
	M_CROSS.set_global_id();
	

	stupid_step(M_CROSS.n1, M_CROSS.n2, M_CROSS.n3, M_CROSS.n4, M_CROSS.t, M_CROSS.J_back, M_CROSS.TOTAL_SIZE);



	cudaCheckError()

	Lx_h = hx_h * (M_CROSS.nxg);
	Ly_h = hy_h * (M_CROSS.nyg);

	cudaDeviceSynchronize();

	



	//1 is 'yes' / true, 0 is 'no' / false
	int picture_switch = 1; //write fields to a file?
	int read_switch = 1; //read to continue or not? 



	tau_h = 0.5e-4;
	tt = round(1.0 / tau_h);

	nxout = 1;
	nyout = 1;


	A_h = -0.5;
	Ca_h = 4 * 1e-4;
	Gr_h = 0.0;
	Pe_h = 1e+3;
	Re_h = 1;
	alpha_h = 90;
	MM_h = 1;
	cosA_h = cos(alpha_h*pi / 180);
	sinA_h = sin(alpha_h*pi / 180);
	tau_p_h = 0.2*hx_h*hx_h;



	ifstream read("inp.dat");
	/*
	//lambda-expression to skip the rest of a line:
	auto skip = [&read]() {while (read.peek() != '\n') read.ignore();  };
	read >> nx; skip();
	read >> ny; skip();
	read >> nxout; skip();
	read >> nyout; skip();
	read >> Lx; skip();
	read >> tau; skip();
	read >> A; skip();
	read >> Ca; skip();
	read >> Gr; skip();
	read >> Pe; skip();
	read >> Re; skip();
	read >> alpha; skip();
	read >> MM; skip();
	read.close();
	*/


	//size setting
	offset_h = nx_h + 1;
	unsigned int size_l = M_CROSS.TOTAL_SIZE; //Number of all nodes/elements 
	if (size_l <= 1024 || size_l >= 1024 * 1024 * 1024) { cout << "data is too small or too large" << endl; return 0; }
	std::cout << "size_l=" << size_l << endl;
	size_t size_b /*size (in) bytes*/ = size_l * sizeof(double); //sizeof(double) = 8 bytes

	size_t thread_x_d /*the dimension of x in a block*/ = 1024;
	size_t threads_per_block = thread_x_d;

	dim3 gridD(ceil((size_l + 0.0) / thread_x_d));
	dim3 blockD(thread_x_d);
	std::cout << "gridD.x=" << gridD.x << endl;
	std::cout << "blockD.x=" << blockD.x << endl;

	//setting for the reduction procedure 
	unsigned long long int *Gp, *Np;
	unsigned int s = 0;

	unsigned int GN = size_l;
	while (true)
	{
		s++;
		GN = ceil(GN / (thread_x_d + 0.0));
		if (GN == 1)  break;
	}
	GN = size_l;
	std::cout << "the number of reduction = " << s << endl;
	Gp = new unsigned long long int[s];
	Np = new unsigned long long int[s];
	for (int i = 0; i < s; i++)
		Gp[i] = GN = ceil(GN / (thread_x_d + 0.0));
	Np[0] = size_l;
	for (int i = 1; i < s; i++)
		Np[i] = Gp[i - 1];
	int last_reduce = pow(2, ceil(log2(Np[s - 1] + 0.0)));
	std::cout << "last reduction = " << last_reduce << endl;
	(s != 1) ? std::cout << "sub array for the Poisson solver = " << Np[1] << endl :
		std::cout << "it shouldn't be here" << endl;
	double *arr[10];


	//allocating memory for arrays on CPU and initializing them 
	{
		C_h = (double*)malloc(size_b); 		mu_h = (double*)malloc(size_b);
		p_h = (double*)malloc(size_b);		p_true_h = (double*)malloc(size_b);
		vx_h = (double*)malloc(size_b);		vy_h = (double*)malloc(size_b);
		psiav_h = (double*)malloc(sizeof(double)); 	psiav_array_h = (double*)malloc(size_b / threads_per_block);
		for (int l = 0; l < size_l; l++) { C_h[l] = 0.5; mu_h[l] = 0; p_h[l] = 0.0; vx_h[l] = 0.0; vy_h[l] = 0.0; }
	}

	//allocating memory for arrays on GPU
	{
		cudaMalloc((void**)&C, size_b); 	cudaMalloc((void**)&C0, size_b);
		cudaMalloc((void**)&p, size_b); 	cudaMalloc((void**)&p0, size_b);
		cudaMalloc((void**)&ux, size_b);	cudaMalloc((void**)&uy, size_b);
		cudaMalloc((void**)&vx, size_b);	cudaMalloc((void**)&vy, size_b);
		cudaMalloc((void**)&mu, size_b);
		(s != 1) ? cudaMalloc((void**)&psiav_array, sizeof(double)*Np[1]) : cudaMalloc((void**)&psiav_array, sizeof(double));
	}

	arr[0] = p;
	for (int i = 1; i <= s; i++)
		arr[i] = psiav_array;




	//ofstream is a class to write data in a file 
	ofstream integrals;
	ofstream k_number;
	read.open("recovery.dat");
	bool file_exists = read.good();
	if (read_switch == 0) file_exists = false;
	if (file_exists == true) { read_switch = 1; 	std::cout << "CONTINUE" << endl; }
	else { read_switch = 0;	iter = 0; std::cout << "from the Start" << endl; }


	if (read_switch == 1) {
		integrals.open("integrals.dat", std::ofstream::app);
		M_CROSS.recover(vx_h, vy_h, p_h, C_h, mu_h, iter, write_i);
	}


	if (read_switch == 0) integrals.open("integrals.dat");



	//copying values from host variables to device ones
	{
		cudaMemcpy(C, C_h, size_b, cudaMemcpyHostToDevice); 	cudaMemcpy(C0, C_h, size_b, cudaMemcpyHostToDevice);
		cudaMemcpy(p0, p_h, size_b, cudaMemcpyHostToDevice); 	cudaMemcpy(p, p_h, size_b, cudaMemcpyHostToDevice);
		cudaMemcpy(ux, vx_h, size_b, cudaMemcpyHostToDevice); 	cudaMemcpy(uy, vy_h, size_b, cudaMemcpyHostToDevice);
		cudaMemcpy(vx, vx_h, size_b, cudaMemcpyHostToDevice); 	cudaMemcpy(vy, vy_h, size_b, cudaMemcpyHostToDevice);
		cudaMemcpy(mu, mu_h, size_b, cudaMemcpyHostToDevice);
	}

	//copying some constant parameters to the fast constant memory
	{
		cudaMemcpyToSymbol(hx, &hx_h, sizeof(double), 0, cudaMemcpyHostToDevice);
		cudaMemcpyToSymbol(hy, &hy_h, sizeof(double), 0, cudaMemcpyHostToDevice);
		cudaMemcpyToSymbol(Lx, &Lx_h, sizeof(double), 0, cudaMemcpyHostToDevice);
		cudaMemcpyToSymbol(Ly, &Ly_h, sizeof(double), 0, cudaMemcpyHostToDevice);
		cudaMemcpyToSymbol(nx, &nx_h, sizeof(unsigned int), 0, cudaMemcpyHostToDevice);
		cudaMemcpyToSymbol(ny, &ny_h, sizeof(unsigned int), 0, cudaMemcpyHostToDevice);
		cudaMemcpyToSymbol(n, &size_l, sizeof(unsigned int), 0, cudaMemcpyHostToDevice);
		cudaMemcpyToSymbol(offset, &offset_h, sizeof(unsigned int), 0, cudaMemcpyHostToDevice);

		cudaMemcpyToSymbol(A, &A_h, sizeof(double), 0, cudaMemcpyHostToDevice);
		cudaMemcpyToSymbol(Ca, &Ca_h, sizeof(double), 0, cudaMemcpyHostToDevice);
		cudaMemcpyToSymbol(Gr, &Gr_h, sizeof(double), 0, cudaMemcpyHostToDevice);
		cudaMemcpyToSymbol(Pe, &Pe_h, sizeof(double), 0, cudaMemcpyHostToDevice);
		cudaMemcpyToSymbol(Re, &Re_h, sizeof(double), 0, cudaMemcpyHostToDevice);
		cudaMemcpyToSymbol(alpha, &alpha_h, sizeof(double), 0, cudaMemcpyHostToDevice);
		cudaMemcpyToSymbol(MM, &MM_h, sizeof(double), 0, cudaMemcpyHostToDevice);

		cudaMemcpyToSymbol(tau, &tau_h, sizeof(double), 0, cudaMemcpyHostToDevice);
		cudaMemcpyToSymbol(tau_p, &tau_p_h, sizeof(double), 0, cudaMemcpyHostToDevice);
		cudaMemcpyToSymbol(sinA, &sinA_h, sizeof(double), 0, cudaMemcpyHostToDevice);
		cudaMemcpyToSymbol(cosA, &cosA_h, sizeof(double), 0, cudaMemcpyHostToDevice);
		cudaMemcpyToSymbol(OFFSET, &M_CROSS.OFFSET, sizeof(int), 0, cudaMemcpyHostToDevice);
//		cudaMemcpyToSymbol(Mx, &M_CROSS.Mx, sizeof(int), 0, cudaMemcpyHostToDevice);
//		cudaMemcpyToSymbol(My, &M_CROSS.My, sizeof(int), 0, cudaMemcpyHostToDevice);
//		cudaMemcpyToSymbol(Msize, &M_CROSS.Msize, sizeof(int), 0, cudaMemcpyHostToDevice);
//		cudaMemcpyToSymbol(Moffset, &M_CROSS.Moffset, sizeof(int), 0, cudaMemcpyHostToDevice);
		cudaMemcpyToSymbol(TOTAL_SIZE, &M_CROSS.TOTAL_SIZE, sizeof(int), 0, cudaMemcpyHostToDevice);
	}
	//just printing parameters from GPU to be confident they are passed correctly 
	hello << <1, 1 >> > ();
	cudaDeviceSynchronize();

	//just time in the double precision format
	double timeq = 0.0;

	m = 0.0; psiav = 0.0; Ek = 0; Ek_old = 0; kk = 1000000;


	pause






		




			//measure real time of calculating 
	timer1 = clock() / CLOCKS_PER_SEC;
	M_CROSS.write_field(C_h, "test", 0, 1);




	


	


	{
#ifdef __linux__ 
		ofstream to_file("fields/param.dat");
#endif
#ifdef _WIN32
		ofstream to_file("fields\\param.dat");
#endif
#define space << " " << 
	to_file << M_CROSS.nxg / each space M_CROSS.nyg / each space hx_h*each space hy_h*each space Lx_h space Ly_h
		space Gr_h space Ca_h space Pe_h space Re_h space A_h space MM_h space alpha_h
		<< endl;
	to_file.close();
	}

	

			// the main time loop of the whole calculation procedure
		while (true) {


			iter = iter + 1; 	timeq = timeq + tau_h;


			//1st step, calculating of time evolutionary parts of velocity (quasi-velocity) and concentration and chemical potential
			{
				//chemical_potential << <gridD, blockD >> > (mu, C);
				quasi_velocity << < gridD, blockD >> > (ux, uy, vx, vy, C0, mu);
				concentration << < gridD, blockD >> > (C, C0, vx, vy, mu);
			}

			
			//2nd step, Poisson equation for pressure 
			{
				eps = 1.0; 		psiav0 = 0.0;		psiav = 0.0;		k = 0;

				while (eps > eps0*psiav0 && k < kk) //take a look
				{




					psiav = 0.0;  k++;
					Poisson << <gridD, blockD >> > (p, p0, ux, uy, mu, C);

					/*old reduction but I am sure it to be working quite well*/
					//reduction00 << < ceil(size_l / 1024.0), 1024, 1024 * sizeof(double) >> > (p, size_l, psiav_array);
					//reduction00 << < 1, last_reduce, last_reduce * sizeof(double) >> > (psiav_array, last_reduce, psiav_array);

					for (int i = 0; i < s; i++)
						reduction00 << < Gp[i], 1024, 1024 * sizeof(double) >> > (arr[i], Np[i], arr[i + 1]);
					swap_one << <gridD, blockD >> > (p0, p);
					cudaMemcpy(&psiav, psiav_array, sizeof(double), cudaMemcpyDeviceToHost);

					eps = abs(psiav - psiav0); 	psiav0 = psiav;



					cudaMemcpy(p_h, p, size_b, cudaMemcpyDeviceToHost);

					if (k % 1000 == 0) {
						cout << "p_iter=" << k << endl; 
					}
				}


			}

			kk = k;
			//pause

			//3rd step, velocity correction and swapping field values
			velocity_correction << <gridD, blockD >> > (vx, vy, ux, uy, p);

			swap_3 << <gridD, blockD >> > (ux, vx, uy, vy, C0, C);




			//4th step, printing resulrs and writing data
			if (iter % (tt / 10) == 0 || iter == 1) {
				cout << setprecision(15) << endl;
				cout << fixed << endl;
				cudaMemcpy(vx_h, vx, size_b, cudaMemcpyDeviceToHost);
				cudaMemcpy(vy_h, vy, size_b, cudaMemcpyDeviceToHost);
				cudaMemcpy(p_h, p, size_b, cudaMemcpyDeviceToHost);
				cudaMemcpy(C_h, C, size_b, cudaMemcpyDeviceToHost);
				velocity(size_l, hx_h, hy_h, vx_h, vy_h, Ek, Vmax);
				VFR(vx_h, M_CROSS.t, size_l, hy_h, Q_in, Q_out, C_h, C_average, Cv);


				timer
				cout << "t= " << tau_h*iter << endl;
				cout << "Vmax= " << Vmax << endl;
				cout << "Ek= " << Ek << endl;
				cout << "dEk= " << abs(Ek - Ek_old) << endl;
				cout << "p_iter=" << k << endl;
				cout << "Q_in=" << Q_in  << endl;
				cout << "Q_out=" << Q_out  << endl;
				cout << "Vx_max=" << maxval(vx_h, size_l) << endl;
				cout << "C_max=" << maxval(C_h, size_l) << endl;

				if (iter == 1)	integrals << "t, Ek, Vmax,  time(min), dEk, Q_in, Q_out, C_average, Q_per_cap, Q_per_width, Cv_per_cap, Cv_per_width" << endl;
				integrals << setprecision(20) << fixed;
				integrals << tau_h*iter << " " << Ek << " " << Vmax << " " << (timer2 - timer1) / 60
					<< " " << abs(Ek - Ek_old) << " " << Q_in << " " << Q_out << " " << C_average / Matrix_Y << " " << Q_out / Matrix_Y << " " << Q_out / Ly_h
					<< " " << Cv / Matrix_Y << " " << Cv / Ly_h
					<< endl;



				Ek_old = Ek;


				//fields writing
				if (iter % ( int(each_t * tt)) == 0 || iter == 1)
				{
					write_i++;
					stringstream ss; string file_name;	ss.str(""); ss.clear();
					ss << write_i;		file_name = ss.str();

					M_CROSS.write_field(C_h, file_name, timeq, each);
				}

				if (iter % (tt) == 0 || iter ==1)
				{
					M_CROSS.save(vx_h, vy_h, p_h, C_h, mu_h, iter, write_i);

				}



				


			} // the end of 4th step


			if (iter*tau_h > 5000) return 0;



		} //the end of the main time loop

}