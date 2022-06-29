#define ThisSoftwareVersion "290622"

#include "cuda_runtime.h"
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
#include <csignal>
#include <time.h>
#ifdef _WIN32
#include "windows.h"
#endif

using namespace std;
using std::cout;
int state;


//__device__ double *C, *C0, *ux, *uy, *vx, *vy, *p, *p0, *mu;
//__device__ multi_cross *Md;
__constant__ double hx, hy, tau, Lx, Ly, tau_p;
__constant__ double A, Ca, Gr, Pe, Re, Gs, MM, dP;
__constant__ double Amp, Omega, vibr_X, vibr_Y, VV;
__constant__ double alpha, sinA, cosA, theta, cosTh, sinTh;
__constant__ unsigned int nx, ny, n, offset, border_type;
__constant__ double eps0_d = 1e-5;
__constant__ double pi = 3.1415926535897932384626433832795;
__constant__ int Mx, My, Msize, Moffset, OFFSET;
__constant__ unsigned int iter;
__constant__ unsigned int TOTAL_SIZE;
__constant__ unsigned int nxg, nyg;
__constant__ unsigned int PHASE;
__constant__ unsigned int PHI_border_left, PHI_border_right, W_BORDER;
__constant__ double PHI_value_left, PHI_value_right;
__device__ int* n1, *n2, *n3, *n4, *t, *J_back;
__device__ double Phi_reference = 0.0;

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
	printf("sinTh= %f cosTh=%f \n", sinTh, cosTh);
	printf("Total number of nodes = %i \n", TOTAL_SIZE);
	printf("P inject factor = %f \n", dP);
	printf("Amp= %f Omega=%f V=%f \n", Amp, Omega, VV);
	printf("vibr_X= %f vibr_Y=%f \n", vibr_X, vibr_Y);
	printf("Vibro border: W = %i, Phi_L = %i, Phi_R = %i, VALUE_L = %f, VALUE_R = %f \n",
		W_BORDER, PHI_border_left, PHI_border_right, PHI_value_left, PHI_value_right);
	if (PHASE == 1) printf("Phase field \n");
	if (PHASE == 0) printf("Single phase flow \n");
	printf("\n");
}



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

#define VarName(Variable) (#Variable)
#define PrintVar(Variable) cout << (#Variable) << " = " << Variable << endl; 
#define defConst(F, type) type F##_h; File.reading<type>(F##_h, #F, 0.0); cudaMemcpyToSymbol(F, &F ## _h, sizeof(type), 0, cudaMemcpyHostToDevice);


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
double MINval(double* f, unsigned int n)
{
	double min = abs(f[0]);

	for (unsigned int i = 0; i < n; i++) {
		if (f[i]<min)
		{
			min = f[i];
		}
	}
	return min;
}
double MAXval(double* f, unsigned int n)
{
	double max = (f[0]);

	for (unsigned int i = 0; i < n; i++) {
		if ((f[i])>max)
		{
			max = (f[i]);
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
	C_average = C_average*hy / (1.0 - hy);
	Cv = Cv*hy;
}


void C_statistics(unsigned int size, double hx, double hy, int *t, double *C, double &C_av, double &C_plus, double &C_minus) {
	C_av = 0; C_plus = 0; C_minus = 0;
	unsigned int n = 0, n2 = 0, n_plus = 0, n2_plus = 0, n_minus = 0, n2_minus = 0;

	for (unsigned int l = 0; l < size; l++) {
		if (t[l] == 0) {
			C_av += C[l];
			n++;
			if (C[l] > 0) {
				C_plus += C[l];
				n_plus++;
			}
			if (C[l] < 0) {
				C_minus += C[l];
				n_minus++;
			}
		}

		else
		{
			C_av += C[l] / 2;
			n2++;
			if (C[l] > 0) {
				C_plus += C[l] / 2;
				n2_plus++;
			}
			if (C[l] < 0) {
				C_minus += C[l] / 2;
				n2_minus++;
			}
		}
	}

	if (n + n2 > 0) C_av /= (n + 0.5*n2);
	if (n_plus + n2_plus > 0) C_plus /= (n_plus + 0.5*n2_plus);
	if (n_minus + n2_minus > 0) C_minus /= (n_minus + 0.5*n2_minus);

}



void reading_parameters(unsigned int &ny_h, unsigned int &nx_h, double &each_t, unsigned int &each, unsigned int &Matrix_X, unsigned int &Matrix_Y, double &tau_h, double &A_h, double &Ca_h, double &Gr_h, double &Pe_h, double &Re_h, double &alpha_h, double &MM_h, double &tecplot, unsigned int &PHASE_h) {

	ifstream read; string str, substr; stringstream ss;
	read.open("inp.dat");

	ss.str(""); ss.clear(); getline(read, str); ss << str; ss >> substr; ny_h = atoi(substr.c_str());
	ss.str(""); ss.clear(); getline(read, str); ss << str; ss >> substr; nx_h = atoi(substr.c_str());
	ss.str(""); ss.clear(); getline(read, str); ss << str; ss >> substr; each_t = atof(substr.c_str());
	ss.str(""); ss.clear(); getline(read, str); ss << str; ss >> substr; each = atoi(substr.c_str());
	ss.str(""); ss.clear(); getline(read, str); ss << str; ss >> substr; Matrix_X = atoi(substr.c_str());
	ss.str(""); ss.clear(); getline(read, str); ss << str; ss >> substr; Matrix_Y = atoi(substr.c_str());
	ss.str(""); ss.clear(); getline(read, str); ss << str; ss >> substr; tau_h = atof(substr.c_str());
	ss.str(""); ss.clear(); getline(read, str); ss << str; ss >> substr; A_h = atof(substr.c_str());
	ss.str(""); ss.clear(); getline(read, str); ss << str; ss >> substr; Ca_h = atof(substr.c_str());
	ss.str(""); ss.clear(); getline(read, str); ss << str; ss >> substr; Gr_h = atof(substr.c_str());
	ss.str(""); ss.clear(); getline(read, str); ss << str; ss >> substr; Pe_h = atof(substr.c_str());
	ss.str(""); ss.clear(); getline(read, str); ss << str; ss >> substr; Re_h = atof(substr.c_str());
	ss.str(""); ss.clear(); getline(read, str); ss << str; ss >> substr; alpha_h = atof(substr.c_str());
	ss.str(""); ss.clear(); getline(read, str); ss << str; ss >> substr; MM_h = atof(substr.c_str());
	ss.str(""); ss.clear(); getline(read, str); ss << str; ss >> substr; tecplot = atof(substr.c_str());
	ss.str(""); ss.clear(); getline(read, str); ss << str; ss >> substr; PHASE_h = atoi(substr.c_str());

	read.close();

}

struct ReadingFile
{
private:
	ifstream read;
	string str, substr, buffer;
	string file_name;
	stringstream ss;
	istringstream iss;
	ostringstream oss;
	int stat, pos;

public:
	ReadingFile(string name)
	{
		file_name = name;
		open_file(file_name);
		stat = 0;
	}
	ReadingFile()
	{
		stat = 0;
	}

	void open_file(string file_name) {
		read.open(file_name.c_str());
		if (read.good()) {
			cout << endl << "the parameter file \"" << file_name << "\" has been read " << endl << endl;
			oss << read.rdbuf();
			buffer = oss.str();
			iss.str(buffer);
		}
		else {
			cout << "the parameter file has been not found, default parameters will be initialized " << endl;
			buffer = "";
			iss.str(buffer);
		}
	}


	template <typename T>
	int reading(T &var, string parameter_name, T def_var, T min = 0, T max = 0) {
		int ret = 0;
		stat = 0;
		transform(parameter_name.begin(), parameter_name.end(), parameter_name.begin(), ::tolower);
		iss.clear();
		iss.seekg(0);

		while (getline(iss, str))
		{
			//substr.clear();
			ss.str("");	ss.clear();	ss << str;	ss >> substr;
			transform(substr.begin(), substr.end(), substr.begin(), ::tolower);
			if (substr == parameter_name) {
				ret = 1;
				pos = (int)ss.tellg();
				while (ss >> substr) {
					if (substr == "=")
					{
						ss >> var;
						stat = 1;
						break;
					}
				}

				if (stat == 0) {
					ss.clear();
					ss.seekg(pos);
					ss >> var;
				}
				break;
			}
		}
		if (iss.fail())
		{
			var = def_var;
		}

		if (min != max && (min + max) != 0) {
			if (var > max || var < min)
			{
				cout << "Warning: \"" + parameter_name + "\" should not be within this range" << endl;
				var = def_var;
			}
		}
		return ret; //return 1 if read
	}

	void reading_string(string &var, string parameter_name, string def_var) {
		stat = 0;
		transform(parameter_name.begin(), parameter_name.end(), parameter_name.begin(), ::tolower);
		iss.clear();
		iss.seekg(0);

		while (getline(iss, str))
		{
			//substr.clear();
			ss.str("");	ss.clear();	ss << str;	ss >> substr;
			transform(substr.begin(), substr.end(), substr.begin(), ::tolower);
			if (substr == parameter_name) {
				pos = (int)ss.tellg();
				while (ss >> substr) {
					if (substr == "=")
					{
						ss >> var;
						stat = 1;
						break;
					}
				}

				if (stat == 0) {
					ss.clear();
					ss.seekg(pos);
					ss >> var;
				}
				break;
			}
		}
		if (iss.fail())
		{
			var = def_var;
		}

	}


};



__device__  double dx1(unsigned int l, double *f) {
	return 0.5*(f[n3[l]] - f[n1[l]]) / hx;
}
__device__  double dy1(unsigned int l, double *f) {
	return 0.5*(f[n2[l]] - f[n4[l]]) / hy;
}
__device__  double dx2(unsigned int l, double *f) {
	return (f[n3[l]] - 2.0*f[l] + f[n1[l]]) / hx / hx;
}
__device__  double dy2(unsigned int l, double *f) {
	return (f[n2[l]] - 2.0*f[l] + f[n4[l]]) / hy / hy;
}

__device__  double dx1_eq_0_forward(unsigned int l, double *f) {
	return (4.0*f[n3[l]] - f[n3[n3[l]]]) / 3.0;
}
__device__  double dx1_eq_0_back(unsigned int l, double *f) {
	return (4.0*f[n1[l]] - f[n1[n1[l]]]) / 3.0;
}
__device__  double dy1_eq_0_up(unsigned int l, double *f) {
	return (4.0*f[n2[l]] - f[n2[n2[l]]]) / 3.0;
}
__device__  double dy1_eq_0_down(unsigned int l, double *f) {
	return (4.0*f[n4[l]] - f[n4[n4[l]]]) / 3.0;
}

__device__  double dx1_forward(unsigned int l, double *f) {
	return -0.5*(3.0*f[l] - 4.0*f[n3[l]] + f[n3[n3[l]]]) / hx;
}
__device__  double dx1_back(unsigned int l, double *f) {
	return  0.5*(3.0*f[l] - 4.0*f[n1[l]] + f[n1[n1[l]]]) / hx;
}
__device__  double dy1_up(unsigned int l, double *f) {
	return  -0.5*(3.0*f[l] - 4.0*f[n2[l]] + f[n2[n2[l]]]) / hy;
}
__device__  double dy1_down(unsigned int l, double *f) {
	return  0.5*(3.0*f[l] - 4.0*f[n4[l]] + f[n4[n4[l]]]) / hy;
}

__device__  double dx2_forward(unsigned int l, double *f) {
	return (2.0 * f[l] - 5.0 * f[n3[l]] + 4.0 * f[n3[n3[l]]] - f[n3[n3[n3[l]]]]) / hx / hx;
}
__device__  double dx2_back(unsigned int l, double *f) {
	return (2.0 * f[l] - 5.0 * f[n1[l]] + 4.0 * f[n1[n1[l]]] - f[n1[n1[n1[l]]]]) / hx / hx;
}
__device__  double dy2_up(unsigned int l, double *f) {
	return (2.0 * f[l] - 5.0 * f[n2[l]] + 4.0 * f[n2[n2[l]]] - f[n2[n2[n2[l]]]]) / hy / hy;
}
__device__  double dy2_down(unsigned int l, double *f) {
	return (2.0 * f[l] - 5.0 * f[n4[l]] + 4.0 * f[n4[n4[l]]] - f[n4[n4[n4[l]]]]) / hy / hy;
}

__device__ double dx2_eq_0_forward(unsigned int l, double* f) {
	return (5.0 * f[n3[l]] - 4.0 * f[n3[n3[l]]] + f[n3[n3[n3[l]]]]) * 0.5;
}
__device__ double dx2_eq_0_back(unsigned int l, double* f) {
	return (5.0 * f[n1[l]] - 4.0 * f[n1[n1[l]]] + f[n1[n1[n1[l]]]]) * 0.5;
}
__device__ double dy2_eq_0_up(unsigned int l, double* f) {
	return (5.0 * f[n2[l]] - 4.0 * f[n2[n2[l]]] + f[n2[n2[n2[l]]]]) * 0.5;
}
__device__ double dy2_eq_0_down(unsigned int l, double* f) {
	return (5.0 * f[n4[l]] - 4.0 * f[n4[n4[l]]] + f[n4[n4[n4[l]]]]) * 0.5;
}



__device__ double dxy1(double *f, int l, int i, int j) {
	int ii = (J_back[l] - (J_back[l] / OFFSET)*OFFSET);
	int jj = (J_back[l] / OFFSET);




	if (i > 0 && i < nx  && j > 0 && j < ny) {
		return (-f[l - 1 + offset] + f[l + 1 + offset] - f[l + 1 - offset] + f[l - 1 - offset]) / hx / hy / 4.0;
	}
	else {
		return 0;
	}
}

__device__ double extrapolate_back(unsigned int l, double *f) {
	return	2.0*f[n3[l]] - f[n3[n3[l]]];
}
__device__ double extrapolate_forward(unsigned int l, double *f) {
	return	2.0*f[n1[l]] - f[n1[n1[l]]];
}
__device__ double extrapolate_down(unsigned int l, double *f) {
	return	2.0*f[n2[l]] - f[n2[n2[l]]];
}
__device__ double extrapolate_up(unsigned int l, double *f) {
	return	2.0*f[n4[l]] - f[n4[n4[l]]];
}

__device__ double VgradF(unsigned int l, double *f, double *vx, double *vy) {
	double val = 0;
	double VR, VL, VU, VD;
	double FR, FL, FU, FD;
	FR = FL = FU = FD = 0;


	VR = (vx[n3[l]] + vx[l])*0.5;
	VL = (vx[l] + vx[n1[l]])*0.5;

	if (VR > 0) FR = f[l];
	else if (VR < 0) FR = f[n3[l]];

	if (VL > 0) FL = f[n1[l]];
	else if (VL < 0) FL = f[l];

	val += (VR*FR - VL*FL) / hx;



	VU = (vy[n2[l]] + vy[l])*0.5;
	VD = (vy[l] + vy[n4[l]])*0.5;

	if (VU > 0) FU = f[l];
	else if (VU < 0) FU = f[n2[l]];

	if (VD > 0) FD = f[n4[l]];
	else if (VD < 0) FD = f[l];

	val += (VU*FU - VD*FD) / hy;



	return val;
}
__device__ double VgradF_forward(unsigned int l, double *f, double *vx, double *vy) {
	double val = 0;
	double VR, VL, VU, VD;
	double FR, FL, FU, FD;
	FR = FL = FU = FD = 0;


	VR = (vx[n3[l]] + vx[l])*0.5;
	VL = (vx[l] + vx[n1[l]])*0.5;

	if (VR > 0) FR = f[l];
	else if (VR < 0) FR = f[n3[l]];

	if (VL > 0) FL = f[n1[l]];
	else if (VL < 0) FL = f[l];

	val += (VR*FR - VL*FL) / hx;



	VU = (vy[n2[l]] + vy[l])*0.5;
	VD = (vy[l] + vy[n4[l]])*0.5;

	if (VU > 0) FU = f[l];
	else if (VU < 0) FU = f[n2[l]];

	if (VD > 0) FD = f[n4[l]];
	else if (VD < 0) FD = f[l];

	val += (VU*FU - VD*FD) / hy;



	return val;
}



/*
#define dx1(l, f)  0.5*(f[n3[l]] - f[n1[l]]) / hx
#define dy1(l, f)  0.5*(f[n2[l]] - f[n4[l]]) / hy

#define dx2(l, f)  (f[n3[l]] - 2.0*f[l] + f[n1[l]]) / hx / hx
#define dy2(l, f) (f[n2[l]] - 2.0*f[l] + f[n4[l]]) / hy / hy


#define dx1_eq_0_forward(l, f)  (4.0*f[n3[l]] - f[n3[n3[l]]]) / 3.0
#define dx1_eq_0_back(l, f)  (4.0*f[n1[l]] - f[n1[n1[l]]]) / 3.0
#define dy1_eq_0_up(l, f)  (4.0*f[n2[l]] - f[n2[n2[l]]]) / 3.0
#define dy1_eq_0_down(l, f)  (4.0*f[n4[l]] - f[n4[n4[l]]]) / 3.0
#define dx1_forward(l, f)  -0.5*(3.0*f[l] - 4.0*f[n3[l]] + f[n3[n3[l]]]) / hx
#define dx1_back(l, f)   0.5*(3.0*f[l] - 4.0*f[n1[l]] + f[n1[n1[l]]]) / hx
#define dy1_up(l, f)   -0.5*(3.0*f[l] - 4.0*f[n2[l]] + f[n2[n2[l]]]) / hy
#define dy1_down(l, f)  0.5*(3.0*f[l] - 4.0*f[n4[l]] + f[n4[n4[l]]]) / hy


#define dx2_forward(l, f)  (2.0 * f[l] - 5.0 * f[n3[l]] + 4.0 * f[n3[n3[l]]] - f[n3[n3[n3[l]]]]) / hx / hx
#define dx2_back(l, f)  (2.0 * f[l] - 5.0 * f[n1[l]] + 4.0 * f[n1[n1[l]]] - f[n1[n1[n1[l]]]]) / hx / hx
#define dy2_up(l, f)  (2.0 * f[l] - 5.0 * f[n2[l]] + 4.0 * f[n2[n2[l]]] - f[n2[n2[n2[l]]]]) / hy / hy
#define dy2_down(l, f)  (2.0 * f[l] - 5.0 * f[n4[l]] + 4.0 * f[n4[n4[l]]] - f[n4[n4[n4[l]]]]) / hy / hy
*/




__device__  double r_gamma(unsigned int l)
{
	return (J_back[l] - (J_back[l] / OFFSET)*OFFSET) * hx*cosA +    //cosA*x
		(J_back[l] / OFFSET) * hy*sinA;					   //sinA*y
}
__device__  double x_gamma(unsigned int l) {
	return (J_back[l] - (J_back[l] / OFFSET)*OFFSET) * hx*cosA;    //cosA*x
}
__device__  double y_gamma(unsigned int l) {
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
			mu[l] = -MM*Gr* r_gamma(l) //nu takoe // da norm
				+ 2.0 * A * C[l] + 4.0 * pow(C[l], 3) - Ca*(dx2(l, C) + dy2(l, C));
			break;
		case 1: //left rigid
			mu[l] = dx1_eq_0_forward(l, mu);
			break;
		case 2: //upper rigid
			mu[l] = dy1_eq_0_down(l, mu);
			/*
			int down1 = n4[l];
			int down2 = n4[n4[l]];
			double m1 = -MM*Gr* r_gamma(down1)	+2.0 * A * C[down1] + 4.0 * pow(C[down1], 3) - Ca*(dx2(down1, C) + dy2(down1, C));
			double m2 = -MM*Gr* r_gamma(down2) +2.0 * A * C[down2] + 4.0 * pow(C[down2], 3) - Ca*(dx2(down2, C) + dy2(down2, C));
			mu[l] = (4.0*m1 - m2) / 3.0;
			*/
			break;
		case 3: //right rigid
			mu[l] = dx1_eq_0_back(l, mu);
			break;
		case 4: //lower rigid
			mu[l] = dy1_eq_0_up(l, mu);
			/*
			int up1 = n2[l];
			int up2 = n2[n2[l]];
			double m1_ = -MM*Gr* r_gamma(up1) +2.0 * A * C[up1] + 4.0 * pow(C[up1], 3) - Ca*(dx2(up1, C) + dy2(up1, C));
			double m2_ = -MM*Gr* r_gamma(up2) +2.0 * A * C[up2] + 4.0 * pow(C[up2], 3) - Ca*(dx2(up2, C) + dy2(up2, C));
			mu[l] = (4.0*m1_ - m2_) / 3.0;
			*/
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
			mu[l] = -Ca*dx2_forward(l, C) - Ca*dy2(l, C) + 2.0 * A * C[l] + 4.0 * pow(C[l], 3) - MM*Gr* r_gamma(l); //dx1_eq_0_forward(l, mu);
			break;
		case 10://outlet (to right)
			mu[l] = -Ca*dx2_back(l, C) - Ca*dy2(l, C) + 2.0 * A * C[l] + 4.0 * pow(C[l], 3) - MM*Gr* r_gamma(l); //dx1_eq_0_back(l, mu);
			break;
		default:
			break;
		}

	}
}

__global__ void chemical_potential_border(double *mu, double *C)
{
	unsigned int l = threadIdx.x + blockIdx.x*blockDim.x;

	if (l < n)
	{
		switch (t[l])
		{
		case 0: //inner

			break;
		case 1: //left rigid
			mu[l] = dx1_eq_0_forward(l, mu);
			break;
		case 2: //upper rigid
			mu[l] = dy1_eq_0_down(l, mu);// -2.0 / 3.0*hy*(-Gr*r_gamma(l)*C[l]);
			/*
			int down1 = n4[l];
			int down2 = n4[n4[l]];
			double m1 = -MM*Gr* r_gamma(down1)	+2.0 * A * C[down1] + 4.0 * pow(C[down1], 3) - Ca*(dx2(down1, C) + dy2(down1, C));
			double m2 = -MM*Gr* r_gamma(down2) +2.0 * A * C[down2] + 4.0 * pow(C[down2], 3) - Ca*(dx2(down2, C) + dy2(down2, C));
			mu[l] = (4.0*m1 - m2) / 3.0;
			*/
			break;
		case 3: //right rigid
			mu[l] = dx1_eq_0_back(l, mu);
			break;
		case 4: //lower rigid
			mu[l] = dy1_eq_0_up(l, mu);// +2.0 / 3.0*hy*(-Gr*r_gamma(l)*C[l]);;
			/*
			int up1 = n2[l];
			int up2 = n2[n2[l]];
			double m1_ = -MM*Gr* r_gamma(up1) +2.0 * A * C[up1] + 4.0 * pow(C[up1], 3) - Ca*(dx2(up1, C) + dy2(up1, C));
			double m2_ = -MM*Gr* r_gamma(up2) +2.0 * A * C[up2] + 4.0 * pow(C[up2], 3) - Ca*(dx2(up2, C) + dy2(up2, C));
			mu[l] = (4.0*m1_ - m2_) / 3.0;
			*/
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
			//mu[l] = dx1_eq_0_forward(l, mu);
			//mu[l] = dx2_eq_0_forward(l, mu);
			break;
		case 10://outlet (to right)
			//mu[l] = dx1_eq_0_back(l, mu);
			//mu[l] = dx2_eq_0_back(l, mu);
			break;
		default:
			break;
		}

	}
}

__global__ void chemical_potential_inside(double *mu, double *C)
{
	unsigned int l = threadIdx.x + blockIdx.x*blockDim.x;

	if (l < n)
	{
		switch (t[l])
		{
		case 0: //inner
			mu[l] = -MM*Gr* r_gamma(l) //nu takoe // da norm
				+ 2.0 * A * C[l] + 4.0 * pow(C[l], 3) - Ca*(dx2(l, C) + dy2(l, C));
			break;
		case 1: //left rigid

			break;
		case 2: //upper rigid

			break;
		case 3: //right rigid

			break;
		case 4: //lower rigid

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
			mu[l] = -Ca*dx2_forward(l, C) - Ca*dy2(l, C) + 2.0 * A * C[l] + 4.0 * pow(C[l], 3) - MM*Gr* r_gamma(l); 
			break;
		case 10://outlet (to right)
			mu[l] = -Ca*dx2_back(l, C) - Ca*dy2(l, C) + 2.0 * A * C[l] + 4.0 * pow(C[l], 3) - MM*Gr* r_gamma(l);
			break;
		default:
			break;
		}

	}
}

__global__ void chemical_potential_Gr(double *mu)
{
	unsigned int l = threadIdx.x + blockIdx.x*blockDim.x;

	if (l < n)
	{
		mu[l] = -MM*Gr* r_gamma(l);
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
			ux[l] = vx[l] //+ Gr*C0[l] * x_gamma(l)
				+ tau  * (
					-vx[l] * dx1(l, vx) - vy[l] * dy1(l, vx)
					+ (dx2(l, vx) + dy2(l, vx)) / Re
					- C0[l] * dx1(l, mu) / MM
					);
			//uy_d
			uy[l] = vy[l] //+ Gr*C0[l] * y_gamma(l)
				+ tau  * (
					-vx[l] * dx1(l, vy) - vy[l] * dy1(l, vy)
					+ (dx2(l, vy) + dy2(l, vy)) / Re
					- C0[l] * dy1(l, mu) / MM
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
				-vx[l] * dx1_forward(l, vy) - vy[l] * dy1(l, vy)
				+ (dx2_forward(l, vy) + dy2(l, vy)) / Re  //  !быть может, !тут нужно дополнить
				- C0[l] * dy1(l, mu) / MM
				);
			break;
		case 10: //outlet (to right)
			ux[l] = vx[l] + tau*(
				-vx[l] * dx1_back(l, vx) - vy[l] * dy1(l, vx)
				+ (dx2_back(l, vx) + dy2(l, vx)) / Re
				- C0[l] * dx1_back(l, mu) / MM  //!
				);
			uy[l] = tau * (
				-vx[l] * dx1_back(l, vy) - vy[l] * dy1(l, vy)
				+ (dx2_back(l, vy) + dy2(l, vy)) / Re
				- C0[l] * dy1(l, mu) / MM //!
				);
			break;
		default:
			break;
		}

	}
}

__global__ void quasi_velocity_pulsation(double *ux, double *uy, double *vx, double *vy, double *C0, double *mu, double time) {

	unsigned int l = threadIdx.x + blockIdx.x*blockDim.x;

	if (l < n)
	{

		switch (t[l])
		{
		case 0: //inner
				//ux_d

			ux[l] = vx[l] //+ Gr*C0[l] * x_gamma(l)
				+ tau  * (
					-vx[l] * dx1(l, vx) - vy[l] * dy1(l, vx)
					+ (dx2(l, vx) + dy2(l, vx)) / Re
					- C0[l] * dx1(l, mu) / MM
					);
			//uy_d


			uy[l] = vy[l] //+ Gr*C0[l] * y_gamma(l)
				+ tau  * (
					-vx[l] * dx1(l, vy) - vy[l] * dy1(l, vy)
					+ (dx2(l, vy) + dy2(l, vy)) / Re
					- C0[l] * dy1(l, mu) / MM
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
				-vx[l] * dx1_forward(l, vy) - vy[l] * dy1(l, vy)
				+ (dx2_forward(l, vy) + dy2(l, vy)) / Re  //  !быть может, !тут нужно дополнить
				- C0[l] * dy1(l, mu) / MM
				);
			break;
		case 10: //outlet (to right)
			ux[l] = vx[l] + tau*(
				-vx[l] * dx1_back(l, vx) - vy[l] * dy1(l, vx)
				+ (dx2_back(l, vx) + dy2(l, vx)) / Re
				- C0[l] * dx1_back(l, mu) / MM  //!
				);
			uy[l] = tau * (
				-vx[l] * dx1_back(l, vy) - vy[l] * dy1(l, vy)
				+ (dx2_back(l, vy) + dy2(l, vy)) / Re
				- C0[l] * dy1(l, mu) / MM //!
				);
			break;
		default:
			break;
		}
		ux[l] += tau*Amp*sin(Omega*time)*vibr_X;
		uy[l] += tau*Amp*sin(Omega*time)*vibr_Y;

	}
}

__global__ void quasi_velocity_pulsation_with_Phi(double *ux, double *uy, double *vx, double *vy, double *C0, double *mu, double time, double *Phi, double *WX, double *WY) {

	unsigned int l = threadIdx.x + blockIdx.x*blockDim.x;

	if (l < n)
	{
		double W0_W = WX[l] * vibr_X + WY[l] * vibr_Y;

		switch (t[l])
		{
		case 0: //inner
				//ux_d

			ux[l] = vx[l] //+ Gr*C0[l] * x_gamma(l)
				+ tau  * (
					-vx[l] * dx1(l, vx) - vy[l] * dy1(l, vx)
					+ (dx2(l, vx) + dy2(l, vx)) / Re
					- C0[l] * dx1(l, mu) / MM
					- VV*(W0_W)*dx1(l, C0)
					);
			//uy_d


			uy[l] = vy[l] //+ Gr*C0[l] * y_gamma(l)
				+ tau  * (
					-vx[l] * dx1(l, vy) - vy[l] * dy1(l, vy)
					+ (dx2(l, vy) + dy2(l, vy)) / Re
					- C0[l] * dy1(l, mu) / MM
					- VV*(W0_W)*dy1(l, C0)
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
				- VV*(W0_W)*dx1_forward(l, C0)
				);

			uy[l] = tau * (
				-vx[l] * dx1_forward(l, vy) - vy[l] * dy1(l, vy)
				+ (dx2_forward(l, vy) + dy2(l, vy)) / Re  //  !быть может, !тут нужно дополнить
				- C0[l] * dy1(l, mu) / MM
				- VV*(W0_W)*dy1(l, C0)
				);
			break;
		case 10: //outlet (to right)

			ux[l] = vx[l] + tau*(
				-vx[l] * dx1_back(l, vx) - vy[l] * dy1(l, vx)
				+ (dx2_back(l, vx) + dy2(l, vx)) / Re
				- C0[l] * dx1_back(l, mu) / MM  //!
				- VV*(W0_W)*dx1_back(l, C0)
				);
			uy[l] = tau * (
				-vx[l] * dx1_back(l, vy) - vy[l] * dy1(l, vy)
				+ (dx2_back(l, vy) + dy2(l, vy)) / Re
				- C0[l] * dy1(l, mu) / MM //!
				- VV*(W0_W)*dy1(l, C0)
				);
			break;
		default:
			break;
		}
	}
}

__global__ void quasi_velocity_no_phase_field(double *ux, double *uy, double *vx, double *vy, double *C0, double *mu) {

	unsigned int l = threadIdx.x + blockIdx.x*blockDim.x;

	if (l < n)
	{

		switch (t[l])
		{
		case 0: //inner
				//ux_d
			ux[l] = vx[l] //+ Gr*C0[l] * x_gamma(l)
				+ tau  * (
					-vx[l] * dx1(l, vx) - vy[l] * dy1(l, vx)
					+ (dx2(l, vx) + dy2(l, vx)) / Re
					+ Gr*C0[l] * cosA
					);
			//uy_d
			uy[l] = vy[l] //+ Gr*C0[l] * y_gamma(l)
				+ tau  * (
					-vx[l] * dx1(l, vy) - vy[l] * dy1(l, vy)
					+ (dx2(l, vy) + dy2(l, vy)) / Re
					+ Gr*C0[l] * sinA
					);
			break;
		case 1: //left rigid
			ux[l] = tau / Re * dx2_forward(l, vx) + tau*Gr*C0[l] * cosA;
			break;
		case 2: //upper rigid
			uy[l] = tau / Re * dy2_down(l, vy) + tau*Gr*C0[l] * sinA;
			break;
		case 3: //right rigid
			ux[l] = tau / Re * dx2_back(l, vx) + tau*Gr*C0[l] * cosA;
			break;
		case 4: //lower rigid
			uy[l] = tau / Re * dy2_up(l, vy) + tau*Gr*C0[l] * sinA;
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
				+ Gr*C0[l] * cosA
				);

			uy[l] = tau * (
				-vx[l] * dx1_forward(l, vy) - vy[l] * dy1(l, vy)
				+ (dx2_forward(l, vy) + dy2(l, vy)) / Re  //  !быть может, !тут нужно дополнить
				+ Gr*C0[l] * sinA
				);
			break;
		case 10: //outlet (to right)
			ux[l] = vx[l] + tau*(
				-vx[l] * dx1_back(l, vx) - vy[l] * dy1(l, vx)
				+ (dx2_back(l, vx) + dy2(l, vx)) / Re
				+ Gr*C0[l] * cosA
				);
			uy[l] = tau * (
				-vx[l] * dx1_back(l, vy) - vy[l] * dy1(l, vy)
				+ (dx2_back(l, vy) + dy2(l, vy)) / Re
				+ Gr*C0[l] * sinA
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
					+ (dx2(l, mu) + dy2(l, mu)) / Pe
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


__global__ void concentration_surface_energy_wetting(double *C, double *C0, double *vx, double *vy, double *mu) {


	unsigned int l = threadIdx.x + blockIdx.x*blockDim.x;


	double Ca_test = sqrt(Ca) * 5;

	if (l < n)
	{

		switch (t[l])
		{
		case 0: //inner
			C[l] = C0[l]
				+ tau * (
					-vx[l] * dx1(l, C0)
					- vy[l] * dy1(l, C0)
					+ (dx2(l, mu) + dy2(l, mu)) / Pe
					);
			break;
		case 1: //left rigid
			C[l] = dx1_eq_0_forward(l, C0) - 0.75*cosTh*(1.0 - 4.0*C0[l] * C0[l]) * 2.0 / 3.0*hx / Ca_test;
			break;
		case 2: //upper rigid
			C[l] = dy1_eq_0_down(l, C0) - 0.75*cosTh*(1.0 - 4.0*C0[l] * C0[l]) * 2.0 / 3.0*hy / Ca_test;
			break;
		case 3: //right rigid
			C[l] = dx1_eq_0_back(l, C0) - 0.75*cosTh*(1.0 - 4.0*C0[l] * C0[l]) * 2.0 / 3.0*hx / Ca_test;
			break;
		case 4: //lower rigid
			C[l] = dy1_eq_0_up(l, C0) - 0.75*cosTh*(1.0 - 4.0*C0[l] * C0[l]) * 2.0 / 3.0*hy / Ca_test;
			break;
		case 5: //left upper rigid corner
			C[l] = 0.5* (dx1_eq_0_forward(l, C0) - 0.75*cosTh*(1.0 - C0[l] * C0[l]) * 2.0 / 3.0*hx / Ca_test + dy1_eq_0_down(l, C0) + 0.75*cosTh*(1.0 - C0[l] * C0[l]) * 2.0 / 3.0*hy / Ca_test);
			break;
		case 6: //right upper rigid corner
			C[l] = 0.5* (dx1_eq_0_back(l, C0) + 0.75*cosTh*(1.0 - C0[l] * C0[l]) * 2.0 / 3.0*hx / Ca_test + dy1_eq_0_down(l, C0) - 0.75*cosTh*(1.0 - C0[l] * C0[l]) * 2.0 / 3.0*hy / Ca_test);
			break;
		case 7: //right lower rigid corner
			C[l] = 0.5* (dx1_eq_0_back(l, C0) + 0.75*cosTh*(1.0 - C0[l] * C0[l]) * 2.0 / 3.0*hx / Ca_test + dy1_eq_0_up(l, C0) - 0.75*cosTh*(1.0 - C0[l] * C0[l]) * 2.0 / 3.0*hy / Ca_test);
			break;
		case 8: //left lower rigid corner
			C[l] = 0.5* (dx1_eq_0_forward(l, C0) - 0.75*cosTh*(1.0 - C0[l] * C0[l]) * 2.0 / 3.0*hx / Ca_test + dy1_eq_0_up(l, C0) - 0.75*cosTh*(1.0 - C0[l] * C0[l]) * 2.0 / 3.0*hy / Ca_test);
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

__global__ void concentration_geometrical_wetting(double *C, double *C0, double *vx, double *vy, double *mu) {


	unsigned int l = threadIdx.x + blockIdx.x*blockDim.x;


	double Ca_test = sqrt(Ca) * 5;

	if (l < n)
	{

		switch (t[l])
		{
		case 0: //inner
			C[l] = C0[l]
				+ tau * (
					-vx[l] * dx1(l, C0)
					- vy[l] * dy1(l, C0)
					+ (dx2(l, mu) + dy2(l, mu)) / Pe
					);
			break;
		case 1: //left rigid
			C[l] = dx1_eq_0_forward(l, C0) - cosTh * 2.0 / 3.0*hx * sqrt(pow(dx1_forward(l, C0), 2) + pow(dy1(l, C0), 2));
			break;
		case 2: //upper rigid
			C[l] = dy1_eq_0_down(l, C0) - 0.75*cosTh*(1.0 - 4.0*C0[l] * C0[l]) * 2.0 / 3.0*hy / Ca_test;
			break;
		case 3: //right rigid
			C[l] = dx1_eq_0_back(l, C0) - 0.75*cosTh*(1.0 - 4.0*C0[l] * C0[l]) * 2.0 / 3.0*hx / Ca_test;
			break;
		case 4: //lower rigid
			C[l] = dy1_eq_0_up(l, C0) - 0.75*cosTh*(1.0 - 4.0*C0[l] * C0[l]) * 2.0 / 3.0*hy / Ca_test;
			break;
		case 5: //left upper rigid corner
			C[l] = 0.5* (dx1_eq_0_forward(l, C0) - 0.75*cosTh*(1.0 - C0[l] * C0[l]) * 2.0 / 3.0*hx / Ca_test + dy1_eq_0_down(l, C0) + 0.75*cosTh*(1.0 - C0[l] * C0[l]) * 2.0 / 3.0*hy / Ca_test);
			break;
		case 6: //right upper rigid corner
			C[l] = 0.5* (dx1_eq_0_back(l, C0) + 0.75*cosTh*(1.0 - C0[l] * C0[l]) * 2.0 / 3.0*hx / Ca_test + dy1_eq_0_down(l, C0) - 0.75*cosTh*(1.0 - C0[l] * C0[l]) * 2.0 / 3.0*hy / Ca_test);
			break;
		case 7: //right lower rigid corner
			C[l] = 0.5* (dx1_eq_0_back(l, C0) + 0.75*cosTh*(1.0 - C0[l] * C0[l]) * 2.0 / 3.0*hx / Ca_test + dy1_eq_0_up(l, C0) - 0.75*cosTh*(1.0 - C0[l] * C0[l]) * 2.0 / 3.0*hy / Ca_test);
			break;
		case 8: //left lower rigid corner
			C[l] = 0.5* (dx1_eq_0_forward(l, C0) - 0.75*cosTh*(1.0 - C0[l] * C0[l]) * 2.0 / 3.0*hx / Ca_test + dy1_eq_0_up(l, C0) - 0.75*cosTh*(1.0 - C0[l] * C0[l]) * 2.0 / 3.0*hy / Ca_test);
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



__global__ void concentration_wetting(double *C, double *C0, double *vx, double *vy, double *mu) {


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
					+ (dx2(l, mu) + dy2(l, mu)) / Pe
					);
			break;
		case 1: //left rigid
			if (C0[n3[l]] < C0[l])
				C[l] = C0[n3[l]];
			break;
		case 2: //upper rigid
			if (C0[n4[l]] < C0[l])
				C[l] = C0[n4[l]];
			break;
		case 3: //right rigid
			if (C0[n1[l]] < C0[l])
				C[l] = C0[n1[l]];
			break;
		case 4: //lower rigid
			if (C0[n2[l]] < C0[l])
				C[l] = C0[n2[l]];
			break;
		case 5: //left upper rigid corner
			if (C0[n3[n4[l]]] < C0[l])
				C[l] = C0[n3[n4[l]]];
			break;
		case 6: //right upper rigid corner
			if (C0[n1[n4[l]]] < C0[l])
				C[l] = C0[n1[n4[l]]];
			break;
		case 7: //right lower rigid corner
			if (C0[n2[n1[l]]] < C0[l])
				C[l] = C0[n2[n1[l]]];
			break;
		case 8: //left lower rigid corner
			if (C0[n2[n3[l]]] < C0[l])
				C[l] = C0[n2[n3[l]]];
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
__global__ void concentration_no_wetting(double *C, double *C0, double *vx, double *vy, double *mu) {


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
					+ (dx2(l, mu) + dy2(l, mu)) / Pe
					);
			break;
		case 1: //left rigid
			C[l] = 0.5;
			break;
		case 2: //upper rigid
			C[l] = 0.5;
			break;
		case 3: //right rigid
			C[l] = 0.5;
			break;
		case 4: //lower rigid
			C[l] = 0.5;
			break;
		case 5: //left upper rigid corner
			C[l] = 0.5;
			break;
		case 6: //right upper rigid corner
			C[l] = 0.5;
			break;
		case 7: //right lower rigid corner
			C[l] = 0.5;
			break;
		case 8: //left lower rigid corner
			C[l] = 0.5;
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

__global__ void concentration_no_input_C(double *C, double *C0, double *vx, double *vy, double *mu) {


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
					+ (dx2(l, mu) + dy2(l, mu)) / Pe
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
			C[l] = 0.5;  dx1_eq_0_forward(l, C0);
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

			//double vx1 = ux[n3[l]] - tau * dx1(n3[l], p);
			//double vx2 = ux[n3[n3[l]]] - tau * dx1(n3[n3[l]], p);

			//double vy1 = uy[n3[l]] - tau * dy1(n3[l], p);
			//double vy2 = uy[n3[n3[l]]] - tau * dy1(n3[n3[l]], p);

			//vx[l] = 2.0*vx1 - vx2;
			//vy[l] = 2.0*vy1 - vy2;
			break;
		case 10: //outlet (to right)
			vx[l] = ux[l] - tau * dx1_back(l, p);
			vy[l] = uy[l] - tau * dy1(l, p);

			//vx[l] = ux[n1[l]] - tau * dx1(n1[l], p);
			//vy[l] = uy[n1[l]] - tau * dy1(n1[l], p);


			break;
		default:
			break;
		}
	}
}

//for solely pressure
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
			p[l] = dx1_eq_0_forward(l, p0) - ux[l] * 2.0 * hx / tau / 3.0;
			break;
		case 2: //upper rigid
			p[l] = dy1_eq_0_down(l, p0) + uy[l] * 2.0 * hy / tau / 3.0;
			break;
		case 3: //right rigid
			p[l] = dx1_eq_0_back(l, p0) + ux[l] * 2.0 * hx / tau / 3.0;
			break;
		case 4: //lower rigid
			p[l] = dy1_eq_0_up(l, p0) - uy[l] * 2.0 * hy / tau / 3.0;
			break;
		case 5: //left upper rigid corner
			p[l] = 0.5* (dx1_eq_0_forward(l, p0) - ux[l] * 2.0 * hx / tau / 3.0
				+ dy1_eq_0_down(l, p0) + uy[l] * 2.0 * hy / tau / 3.0);
			break;
		case 6: //right upper rigid corner
			p[l] = 0.5* (dx1_eq_0_back(l, p0) + ux[l] * 2.0 * hx / tau / 3.0
				+ dy1_eq_0_down(l, p0) + uy[l] * 2.0 * hy / tau / 3.0);
			break;
		case 7: //right lower rigid corner
			p[l] = 0.5* (dx1_eq_0_back(l, p0) + ux[l] * 2.0 * hx / tau / 3.0
				+ dy1_eq_0_up(l, p0) - uy[l] * 2.0 * hy / tau / 3.0);
			break;
		case 8: //left lower rigid corner
			p[l] = 0.5* (dx1_eq_0_forward(l, p0) - ux[l] * 2.0 * hx / tau / 3.0
				+ dy1_eq_0_up(l, p0) - uy[l] * 2.0 * hy / tau / 3.0);
			break;
		case 9: //inlet (from left)
			p[l] = 8.0 / Re*Lx*dP 
				+ PHASE*((0.5*Ca*pow(dx1_forward(l, C), 2)
					- mu[l] * C[l]
					+ A*pow(C[l], 2) + pow(C[l], 4)) / MM
					- Gr*C[l] * r_gamma(l));
			break;
		case 10://outlet (to right)
			p[l] = 0 
				+ PHASE*((0.5*Ca*pow(dx1_back(l, C), 2)
					- mu[l] * C[l]
					+ A*pow(C[l], 2) + pow(C[l], 4)) / MM
					- Gr*C[l] * r_gamma(l));
			break;
		default:
			break;
		}
	}
}

//for pressure with Phi
__global__ void Poisson_pulsation_Phi(double *p, double *p0, double *ux, double *uy, double *mu, double *C, double *Phi, double *WX, double *WY)
{
	//vibration = 1
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
			p[l] = dx1_eq_0_forward(l, p0) - ux[l] * 2.0 * hx / tau / 3.0;
			break;
		case 2: //upper rigid
			p[l] = dy1_eq_0_down(l, p0) + uy[l] * 2.0 * hy / tau / 3.0;
			break;
		case 3: //right rigid
			p[l] = dx1_eq_0_back(l, p0) + ux[l] * 2.0 * hx / tau / 3.0;
			break;
		case 4: //lower rigid
			p[l] = dy1_eq_0_up(l, p0) - uy[l] * 2.0 * hy / tau / 3.0;
			break;
		case 5: //left upper rigid corner
			p[l] = 0.5* (dx1_eq_0_forward(l, p0) - ux[l] * 2.0 * hx / tau / 3.0
				+ dy1_eq_0_down(l, p0) + uy[l] * 2.0 * hy / tau / 3.0);
			break;
		case 6: //right upper rigid corner
			p[l] = 0.5* (dx1_eq_0_back(l, p0) + ux[l] * 2.0 * hx / tau / 3.0
				+ dy1_eq_0_down(l, p0) + uy[l] * 2.0 * hy / tau / 3.0);
			break;
		case 7: //right lower rigid corner
			p[l] = 0.5* (dx1_eq_0_back(l, p0) + ux[l] * 2.0 * hx / tau / 3.0
				+ dy1_eq_0_up(l, p0) - uy[l] * 2.0 * hy / tau / 3.0);
			break;
		case 8: //left lower rigid corner
			p[l] = 0.5* (dx1_eq_0_forward(l, p0) - ux[l] * 2.0 * hx / tau / 3.0
				+ dy1_eq_0_up(l, p0) - uy[l] * 2.0 * hy / tau / 3.0);
			break;
		case 9: //inlet (from left)
			p[l] = 8.0 / Re*Lx*dP 
				+ PHASE*((0.5*Ca*pow(dx1_forward(l, C), 2)
					- mu[l] * C[l]
					+ A*pow(C[l], 2) + pow(C[l], 4)) / MM
					- Gr*C[l] * r_gamma(l));
			p[l] += (WX[l] * WX[l] + WY[l] * WY[l])*0.5*VV;

			break;
		case 10://outlet (to right)
			p[l] = 0 
				+ PHASE*((0.5*Ca*pow(dx1_back(l, C), 2)
					- mu[l] * C[l]
					+ A*pow(C[l], 2) + pow(C[l], 4)) / MM
					- Gr*C[l] * r_gamma(l));
			p[l] += (WX[l] * WX[l] + WY[l] * WY[l])*0.5*VV;
			break;
		default:
			break;
		}
	}
}


__global__ void Poisson_pulsation(double *p, double *p0, double *ux, double *uy, double *mu, double *C, double time)
{
	//vibration = 3
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
			p[l] = dx1_eq_0_forward(l, p0) - ux[l] * 2.0 * hx / tau / 3.0;
			break;
		case 2: //upper rigid
			p[l] = dy1_eq_0_down(l, p0) + uy[l] * 2.0 * hy / tau / 3.0;
			break;
		case 3: //right rigid
			p[l] = dx1_eq_0_back(l, p0) + ux[l] * 2.0 * hx / tau / 3.0;
			break;
		case 4: //lower rigid
			p[l] = dy1_eq_0_up(l, p0) - uy[l] * 2.0 * hy / tau / 3.0;
			break;
		case 5: //left upper rigid corner
			p[l] = 0.5* (dx1_eq_0_forward(l, p0) - ux[l] * 2.0 * hx / tau / 3.0
				+ dy1_eq_0_down(l, p0) + uy[l] * 2.0 * hy / tau / 3.0);
			break;
		case 6: //right upper rigid corner
			p[l] = 0.5* (dx1_eq_0_back(l, p0) + ux[l] * 2.0 * hx / tau / 3.0
				+ dy1_eq_0_down(l, p0) + uy[l] * 2.0 * hy / tau / 3.0);
			break;
		case 7: //right lower rigid corner
			p[l] = 0.5* (dx1_eq_0_back(l, p0) + ux[l] * 2.0 * hx / tau / 3.0
				+ dy1_eq_0_up(l, p0) - uy[l] * 2.0 * hy / tau / 3.0);
			break;
		case 8: //left lower rigid corner
			p[l] = 0.5* (dx1_eq_0_forward(l, p0) - ux[l] * 2.0 * hx / tau / 3.0
				+ dy1_eq_0_up(l, p0) - uy[l] * 2.0 * hy / tau / 3.0);
			break;
		case 9: //inlet (from left)
			p[l] = 8.0 / Re*Lx*dP*(1.0 + Amp*sin(Omega*time))
				+ PHASE*((0.5*Ca*pow(dx1_forward(l, C), 2)
					- mu[l] * C[l]
					+ A*pow(C[l], 2) + pow(C[l], 4)) / MM
					- Gr*C[l] * r_gamma(l));
			break;
		case 10://outlet (to right)
			p[l] = 0
				+ PHASE*((0.5*Ca*pow(dx1_back(l, C), 2)
					- mu[l] * C[l]
					+ A*pow(C[l], 2) + pow(C[l], 4)) / MM
					- Gr*C[l] * r_gamma(l));
			break;
		default:
			break;
		}
	}
}

__global__ void Poisson_Phi(double *Phi, double *Phi0, double *C, double *WX, double *WY)
{
	//vibration = 1
	unsigned int l = threadIdx.x + blockIdx.x*blockDim.x;
	if (l < n)
	{
		//if (l == 1) Phi0[l] = 0;
		switch (t[l])
		{
		case 0: //inner
			Phi[l] = Phi0[l] + tau_p*(
				-(dx1(l, C)*vibr_X + dy1(l, C)*vibr_Y)
				+ dx2(l, Phi0) + dy2(l, Phi0)
				);
			break;
		case 1: //left rigid
			Phi[l] = dx1_eq_0_forward(l, Phi0) - C[l] * vibr_X * 2.0 * hx / 3.0;
			break;
		case 2: //upper rigid
			Phi[l] = dy1_eq_0_down(l, Phi0) + C[l] * vibr_Y * 2.0 * hy / 3.0;
			break;
		case 3: //right rigid
			Phi[l] = dx1_eq_0_back(l, Phi0) + C[l] * vibr_X * 2.0 * hx / 3.0;
			break;
		case 4: //lower rigid
			Phi[l] = dy1_eq_0_up(l, Phi0) - C[l] * vibr_Y * 2.0 * hy / 3.0;
			break;
		case 5: //left upper rigid corner
			Phi[l] = 0.5* (dx1_eq_0_forward(l, Phi0) - C[l] * vibr_X * 2.0 * hx / 3.0
				+ dy1_eq_0_down(l, Phi0) + C[l] * vibr_Y * 2.0 * hy / 3.0);
			break;
		case 6: //right upper rigid corner
			Phi[l] = 0.5* (dx1_eq_0_back(l, Phi0) + C[l] * vibr_X * 2.0 * hx / 3.0
				+ dy1_eq_0_down(l, Phi0) + C[l] * vibr_Y * 2.0 * hy / 3.0);
			break;
		case 7: //right lower rigid corner
			Phi[l] = 0.5* (dx1_eq_0_back(l, Phi0) + C[l] * vibr_X * 2.0 * hx / 3.0
				+ dy1_eq_0_up(l, Phi0) - C[l] * vibr_Y * 2.0 * hy / 3.0);
			break;
		case 8: //left lower rigid corner
			Phi[l] = 0.5* (dx1_eq_0_forward(l, Phi0) - C[l] * vibr_X * 2.0 * hx / 3.0
				+ dy1_eq_0_up(l, Phi0) - C[l] * vibr_Y * 2.0 * hy / 3.0);
			break;
		case 9: //inlet (from left)
			if (PHI_border_left == 0) {
				Phi[l] = PHI_value_left;
			}
			else if (PHI_border_left == 1) {
				Phi[l] = dx1_eq_0_forward(l, Phi0) - (PHI_value_left) * 2.0 * hx / 3.0;
			}
			else if (PHI_border_left == 2) {
				Phi[l] = dx2_eq_0_forward(l, Phi0) - (PHI_value_left)* hx * hx / 2.0;
			}
			else if (PHI_border_left == 3) {
				Phi[l] = dx1_eq_0_forward(l, Phi0) - (C[l] * vibr_X) * 2.0 * hx / 3.0;
			}
			else if (PHI_border_left == 4) {
				Phi[l] = 0.5*(-hy*hy*C[l] + Phi0[n2[l]] + Phi0[n4[l]]);
			}

			break;
		case 10://outlet (to right)
			if (PHI_border_right == 0) {
				Phi[l] = PHI_value_right;
			}
			else if (PHI_border_right == 1) {
				Phi[l] = dx1_eq_0_back(l, Phi0) + (PHI_value_right) * 2.0 * hx / 3.0;
			}
			else if (PHI_border_right == 2) {
				Phi[l] = dx2_eq_0_back(l, Phi0) - (PHI_value_right)* hx * hx / 2.0;
			}
			else if (PHI_border_right == 3) {
				Phi[l] = dx1_eq_0_back(l, Phi0) + (C[l] * vibr_X) * 2.0 * hx / 3.0;
			}
			else if (PHI_border_right == 4) {
				Phi[l] = 0.5*(-hy*hy*C[l] + Phi0[n2[l]] + Phi0[n4[l]]);
			}
			break;
		default:
			break;
		}
		if (l == 1) Phi_reference = Phi[l];
	}
}

__global__ void Phi_normalization(double *Phi) {
	//vibration = 1
	unsigned int l = threadIdx.x + blockIdx.x*blockDim.x;
	if (l < n)
	{
		Phi[l] = Phi[l] - Phi_reference;
	}
}

__global__ void WW_from_Phi(double *WX, double *WY, double *Phi, double *C) {
	{
		//vibration = 1
		unsigned int l = threadIdx.x + blockIdx.x*blockDim.x;
		if (l < n)
		{
			switch (t[l])
			{
			case 0: //inner
				WX[l] = -vibr_X*C[l] + dx1(l, Phi);
				WY[l] = -vibr_Y*C[l] + dy1(l, Phi);
				break;
			case 1: //left rigid
				WX[l] = 0;
				WY[l] = 0; // -vibr_Y*C[n3[l]] + dy1(n3[l], Phi);;  //well, think of it if dC/dn != 0
				break;
			case 2: //upper rigid
				WX[l] = 0; // -vibr_X*C[n4[l]] + dx1(n4[l], Phi);;
				WY[l] = 0;
				break;
			case 3: //right rigid
				WX[l] = 0;
				WY[l] = 0; // -vibr_Y*C[n1[l]] + dy1(n1[l], Phi);
				break;
			case 4: //lower rigid
				WX[l] = 0; // -vibr_X*C[n2[l]] + dx1(n2[l], Phi);
				WY[l] = 0;
				break;
			case 5: //left upper rigid corner
				WX[l] = 0; WY[l] = 0;
				break;
			case 6: //right upper rigid corner
				WX[l] = 0; WY[l] = 0;
				break;
			case 7: //right lower rigid corner
				WX[l] = 0; WY[l] = 0;
				break;
			case 8: //left lower rigid corner
				WX[l] = 0; WY[l] = 0;
				break;
			case 9: //inlet (from left)
				WX[l] = -vibr_X * C[l] + dx1_forward(l, Phi);
				WY[l] = -vibr_Y * C[l] + dy1(l, Phi);
				break;
				/*

				if (W_BORDER == 0) { //dPhi/dn = 0
				WX[l] = -vibr_X * C[l];
				WY[l] = -vibr_Y*C[l] + dy1(l, Phi);
				}
				else if (W_BORDER == 1) { //W = 0
				WX[l] = 0;
				WY[l] = 0;
				}
				else if (W_BORDER == 2) {
				WX[l] = 0;
				WY[l] = -vibr_Y*C[l] + dy1(l, Phi);
				}
				else if (W_BORDER == 3) { //dW/dn = 0
				WX[l] = (4.0*(-vibr_X*C[n3[l]] + dx1(n3[l], Phi)) - (-vibr_X*C[n3[n3[l]]] + dx1(n3[n3[l]], Phi))) / 3.0;
				WY[l] = -vibr_Y*C[l] + dy1(l, Phi);
				}
				*/
			case 10://outlet (to right)
				WX[l] = -vibr_X * C[l] + dx1_back(l, Phi);
				WY[l] = -vibr_Y * C[l] + dy1(l, Phi);
				break;
				/*
				if (W_BORDER == 0) { //dPhi/dn = 0
				WX[l] = -vibr_X*C[l];
				WY[l] = -vibr_Y*C[l] + dy1(l, Phi);
				}
				else if (W_BORDER == 1) { //W = 0
				WX[l] = 0;
				WY[l] = 0;
				}
				else if (W_BORDER == 2) {
				WX[l] = 0;
				WY[l] = -vibr_Y*C[l] + dy1(l, Phi);
				}
				else if (W_BORDER == 3) { //dW/dn = 0
				WX[l] = (4.0*(-vibr_X*C[n1[l]] + dx1(n1[l], Phi)) - (-vibr_X*C[n1[n1[l]]] + dx1(n1[n1[l]], Phi))) / 3.0;
				WY[l] = -vibr_Y*C[l] + dy1(l, Phi);
				}
				*/
			default:
				break;
			}
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

__global__ void quasi_velocity_upstream(double *ux, double *uy, double *vx, double *vy, double *C0, double *mu) {

	unsigned int l = threadIdx.x + blockIdx.x*blockDim.x;

	if (l < n)
	{

		switch (t[l])
		{
		case 0: //inner
				//ux_d
			ux[l] = vx[l] //+ Gr*C0[l] * x_gamma(l)
				+ tau  * (
					-VgradF(l, vx, vx, vy)
					//-vx[l] * dx1(l, vx) - vy[l] * dy1(l, vx)
					+ (dx2(l, vx) + dy2(l, vx)) / Re
					- C0[l] * dx1(l, mu) / MM
					);
			//uy_d
			uy[l] = vy[l] //+ Gr*C0[l] * y_gamma(l)
				+ tau  * (
					-VgradF(l, vy, vx, vy)
					//-vx[l] * dx1(l, vy) - vy[l] * dy1(l, vy)
					+ (dx2(l, vy) + dy2(l, vy)) / Re
					- C0[l] * dy1(l, mu) / MM
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
				-vx[l] * dx1_forward(l, vy) - vy[l] * dy1(l, vy)
				+ (dx2_forward(l, vy) + dy2(l, vy)) / Re
				- C0[l] * dy1(l, mu) / MM
				);
			break;
		case 10: //outlet (to right)
			ux[l] = vx[l] + tau*(
				-vx[l] * dx1_back(l, vx) - vy[l] * dy1(l, vx)
				+ (dx2_back(l, vx) + dy2(l, vx)) / Re
				- C0[l] * dx1_back(l, mu) / MM  //!
				);
			uy[l] = tau * (
				-vx[l] * dx1_back(l, vy) - vy[l] * dy1(l, vy)
				+ (dx2_back(l, vy) + dy2(l, vy)) / Re
				- C0[l] * dy1(l, mu) / MM //!
				);
			break;
		default:
			break;
		}

	}
}
__global__ void concentration_upstream(double *C, double *C0, double *vx, double *vy, double *mu) {


	unsigned int l = threadIdx.x + blockIdx.x*blockDim.x;


	if (l < n)
	{

		switch (t[l])
		{
		case 0: //inner
			C[l] = C0[l]
				+ tau * (
					-VgradF(l, C0, vx, vy)
					+ (dx2(l, mu) + dy2(l, mu)) / Pe
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
				//C[l] = dx1_eq_0_back(l, C0);
			C[l] = extrapolate_forward(l, C0);
			break;
		default:
			break;
		}


	}


}
__global__ void chemical_potential_upstream(double *mu, double *C)
{
	unsigned int l = threadIdx.x + blockIdx.x*blockDim.x;

	if (l < n)
	{
		switch (t[l])
		{
		case 0: //inner
			mu[l] = -Gr* r_gamma(l) //nu takoe
				+ 2.0 * A * C[l] + 4.0 * pow(C[l], 3) - Ca*(dx2(l, C) + dy2(l, C));
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
			mu[l] = -Ca*dx2_forward(l, C) + 2.0 * A * C[l] + 4.0 * pow(C[l], 3) - Gr* r_gamma(l); //dx1_eq_0_forward(l, mu);
			break;
		case 10://outlet (to right)
				//mu[l] = -Ca*dx2_back(l, C) - Ca*dy2(l, C) + 2.0 * A * C[l] + 4.0 * pow(C[l], 3) - Gr* r_gamma(l); //dx1_eq_0_back(l, mu);
			mu[l] = extrapolate_forward(l, mu);
			break;
		default:
			break;
		}

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

	__host__ __device__ void set_geometry_narrow_tubes(unsigned int L, /*length of horizontal tube*/
		unsigned int H, /*length(height) of vertical tube*/
		unsigned int D /*diameter(width) of tube*/)
	{
		nx[0] = D; ny[0] = D;
		nx[1] = L - 1; ny[1] = D;
		nx[2] = D; ny[2] = H - 1;
		nx[3] = L - 1; ny[3] = D;
		nx[4] = D; ny[4] = H - 1;

		total_size = 0;
		for (int i = 0; i < 5; i++)
		{
			size[i] = (nx[i] + 1)*(ny[i] + 1);
			offset[i] = nx[i] + 1;
			total_size += size[i];
		}
	}


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
	unsigned int nxg, nyg, ng;
	unsigned int nx, ny, offset;
	double *C0, *C, *p, *p0, *ux, *uy, *vx, *vy, *mu;
	double LX, LY;

	//integer global inxed i
	unsigned int iG(unsigned int l) {
		return 	(J_back[l] - (J_back[l] / OFFSET)*OFFSET);
	}
	//integer global index j
	unsigned int jG(unsigned int l) {
		return 	(J_back[l] / OFFSET);
	}


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
	}

	void set_global_size_narrow_tubes(int input_L, int input_H, int input_D, int input_Mx, int input_My) {
		Mx = input_Mx - 1; My = input_My - 1; Msize = input_Mx*input_My; Moffset = input_Mx;
		//Mcr.resize(Msize, cr);
		Mcr = new cross[Msize];

		for (int i = 0; i < Msize; i++) {
			Mcr[i].set_geometry_narrow_tubes(input_L, input_H, input_D);
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
	}


	void set_type() {
		if (Msize == 0) {
			printf("hop hey la la ley, stop it, bro, ya doin it wron' \n");
		}

		l = new int[TOTAL_SIZE];
		t = new int[TOTAL_SIZE];
		for (unsigned int i = 0; i < TOTAL_SIZE; i++) {
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
	void set_type_B() {
		int l, L;
		//int l1, l2, l3, l4;

		//inner
		for (unsigned int i = 0; i <= nxg; i++) {
			for (unsigned int j = 0; j <= nyg; j++) {
				l = i + OFFSET*j; L = J[l];
				if (I[l] == 1) {
					if (n1[L] != -1 && n2[L] != -1 && n3[L] != -1 && n4[L] != -1)
						t[L] = 0;

				}
			}
		}



		//rigid walls
		for (unsigned int i = 0; i <= nxg; i++) {
			for (unsigned int j = 0; j <= nyg; j++) {
				l = i + OFFSET*j; L = J[l];
				//l1 = i - 1 + OFFSET*j; 	l2 = i + OFFSET*j + OFFSET; l3 = i + 1 + OFFSET*j; 	l4 = i + OFFSET*j - OFFSET;
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
		for (unsigned int i = 0; i <= nxg; i++) {
			for (unsigned int j = 0; j <= nyg; j++) {
				l = i + OFFSET*j; L = J[l];

				//cout << i << " " << j << " " << l << " " <<  endl; pause
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
		for (unsigned int i = 0; i <= nxg; i = i + nxg) {
			for (unsigned int j = 0; j <= nyg; j++) {
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

		/*
		//near border
		for (int i = 0; i <= nxg; i = i + nxg) {
		for (int j = 0; j <= nyg; j++) {
		l = i + OFFSET*j; L = J[l];
		if (t[n1[L]] == 1) t[L] = 11;
		if (t[n2[L]] == 2) t[L] = 12;
		if (t[n3[L]] == 3) t[L] = 13;
		if (t[n4[L]] == 4) t[L] = 14;

		if (t[n1[L]] == 1 && t[n2[L]] == 2) t[L] = 15;
		if (t[n2[L]] == 2 && t[n3[L]] == 3) t[L] = 16;
		if (t[n3[L]] == 3 && t[n4[L]] == 4) t[L] = 17;
		if (t[n4[L]] == 4 && t[n1[L]] == 1) t[L] = 18;
		}
		}
		*/


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
		for (unsigned int i = 0; i < TOTAL_SIZE; i++) {
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
	void set_neighbor_B() {
		int l, L, l1, l2, l3, l4;

		for (unsigned int j = 0; j <= nyg; j++) {
			for (unsigned int i = 0; i <= nxg; i++) {


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
		for (unsigned int i = 0; i < TOTAL_SIZE; i++) {
			J_back[i] = -1;
		}

		int *shift_x, *shift_y;
		shift_x = new int[Mx + 1];
		shift_y = new int[My + 1];
		shift_x[0] = 0;
		shift_y[0] = 0;

		for (int im = 1; im <= Mx; im++) shift_x[im] = (Mcr[im - 1].nx[0] + 1 + Mcr[im - 1].nx[1] + 1 + Mcr[im - 1].nx[3] + 1 + shift_x[im - 1]);
		for (int jm = 1; jm <= My; jm++) shift_y[jm] = (Mcr[(jm - 1)*Moffset].ny[0] + 1 + Mcr[(jm - 1)*Moffset].ny[2] + 1 + Mcr[(jm - 1)*Moffset].ny[4] + 1 + shift_y[jm - 1]);

		if (Msize == 0) {
			printf("set_global_id , hop hey la la ley, stop it, bro, ya doin it wron' \n");
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
	void set_global_id_B() {
		nxg = 0, nyg = 0;
		for (int im = 0; im <= Mx; im++) 	nxg += Mcr[im].nx[0] + 1 + Mcr[im].nx[1] + 1 + Mcr[im].nx[3] + 1;
		for (int jm = 0; jm <= My; jm++)	nyg += Mcr[jm*Moffset].ny[0] + 1 + Mcr[jm*Moffset].ny[2] + 1 + Mcr[jm*Moffset].ny[4] + 1;
		if (nxg != 0) nxg--; if (nyg != 0) nyg--;


		I = new int[(nxg + 1)*(nyg + 1)];
		J = new int[(nxg + 1)*(nyg + 1)];
		J_back = new int[TOTAL_SIZE];
		n1 = new int[TOTAL_SIZE];
		n2 = new int[TOTAL_SIZE];
		n3 = new int[TOTAL_SIZE];
		n4 = new int[TOTAL_SIZE];
		t = new int[TOTAL_SIZE];
		OFFSET = nxg + 1;


		for (unsigned int i = 0; i < (nxg + 1)*(nyg + 1); i++) {
			I[i] = 0; J[i] = -1;
		}
		for (unsigned int i = 0; i < TOTAL_SIZE; i++) {
			J_back[i] = -1;
			n1[i] = -1;
			n2[i] = -1;
			n3[i] = -1;
			n4[i] = -1;
			t[i] = -1;
		}

		int *shift_x, *shift_y;
		shift_x = new int[Mx + 1];
		shift_y = new int[My + 1];
		shift_x[0] = 0;
		shift_y[0] = 0;

		for (int im = 1; im <= Mx; im++) shift_x[im] = (Mcr[im - 1].nx[0] + 1 + Mcr[im - 1].nx[1] + 1 + Mcr[im - 1].nx[3] + 1 + shift_x[im - 1]);
		for (int jm = 1; jm <= My; jm++) shift_y[jm] = (Mcr[(jm - 1)*Moffset].ny[0] + 1 + Mcr[(jm - 1)*Moffset].ny[2] + 1 + Mcr[(jm - 1)*Moffset].ny[4] + 1 + shift_y[jm - 1]);

		if (Msize == 0) {
			printf("set_global_id , hop hey la la ley, stop it, bro, ya doin it wron' \n");
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



	void set_global_size_box(int input_nx, int input_ny) {
		nx = input_nx; nxg = nx;
		ny = input_ny; nyg = ny;
		offset = nx + 1;
		OFFSET = offset;
		TOTAL_SIZE = (input_nx + 1) * (input_ny + 1);
	}
	void set_global_id_box() {
		I = new int[(nx + 1)*(ny + 1)];
		J = new int[(nx + 1)*(ny + 1)];
		J_back = new int[TOTAL_SIZE];

		OFFSET = nx + 1;


		for (unsigned int i = 0; i < (nx + 1)*(ny + 1); i++) {
			I[i] = 0; J[i] = -1;
		}
		for (unsigned int i = 0; i < TOTAL_SIZE; i++) {
			J_back[i] = -1;
		}



		unsigned int k, it = 0; // in, ii, jj;
		for (unsigned int i = 0; i <= nx; i++) {
			for (unsigned int j = 0; j <= ny; j++) {

				k = i + offset*j;
				I[k] = 1;
				J[k] = k;
				J_back[k] = k;

				it++;

			}
		}
	}
	void set_type_box() {
		l = new int[TOTAL_SIZE];
		t = new int[TOTAL_SIZE];
		for (unsigned int i = 0; i < TOTAL_SIZE; i++) {
			l[i] = 0;
			t[i] = 0;
		}

		unsigned int k;
		for (unsigned int i = 0; i <= nx; i++) {
			for (unsigned int j = 0; j <= ny; j++) {
				k = i + offset*j;
				if (i == 0) t[k] = 1;
				if (i == nx) t[k] = 3;
				if (j == 0) t[k] = 4;
				if (j == ny) t[k] = 2;

				iter++;


			}
		}
	}
	void set_neighbor_box() {
		n1 = (int*)malloc(TOTAL_SIZE * sizeof(int));
		n2 = (int*)malloc(TOTAL_SIZE * sizeof(int));
		n3 = (int*)malloc(TOTAL_SIZE * sizeof(int));
		n4 = (int*)malloc(TOTAL_SIZE * sizeof(int));
		for (unsigned int i = 0; i < TOTAL_SIZE; i++) {
			n1[i] = -1; n2[i] = -1; n3[i] = -1; n4[i] = -1;
		}

		unsigned int k, it = 0;
		for (unsigned int i = 0; i <= nx; i++) {
			for (unsigned int j = 0; j <= ny; j++) {
				k = i + offset*j;
				if (t[k] == 0) {
					n1[k] = k - 1;
					n2[k] = k + offset;
					n3[k] = k + 1;
					n4[k] = k - offset;
				}

				if (t[k] == 2) {
					n1[k] = k - 1;
					n3[k] = k + 1;
					n4[k] = k - offset;
				}
				if (t[k] == 4) {
					n1[k] = k - 1;
					n2[k] = k + offset;
					n3[k] = k + 1;
				}
				if (t[k] == 9 || t[k] == 1) {
					n3[k] = k + 1;
					n4[k] = k - offset;
					n2[k] = k + offset;
				}
				if (t[k] == 10 || t[k] == 3) {
					n1[k] = k - 1;
					n4[k] = k - offset;
					n2[k] = k + offset;
				}

				if (i == nxg) {
					n3[k] = -1;
				}

				if (i == 0) {
					n1[k] = -1;
				}

				if (j == nyg) {
					n2[k] = -1;
				}

				if (j == 0) {
					n4[k] = -1;
				}



				it++;
			}
		}
	}

	void set_type_tube() {
		l = new int[TOTAL_SIZE];
		t = new int[TOTAL_SIZE];
		for (unsigned int i = 0; i < TOTAL_SIZE; i++) {
			l[i] = 0;
			t[i] = 0;
		}

		unsigned int k;
		for (unsigned int i = 0; i <= nx; i++) {
			for (unsigned int j = 0; j <= ny; j++) {
				k = i + offset*j;
				if (i == 0) t[k] = 9;
				if (i == nx) t[k] = 10;
				if (j == 0) t[k] = 4;
				if (j == ny) t[k] = 2;

				iter++;


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
		for (unsigned int j = 0; j <= nyg; j = j + step) {
			for (unsigned int i = 0; i <= nxg; i = i + step) {
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

	void write_linear_profile(string file_name, string head, double time, int step, double hx, double **f, int N_fields, int j_ = -1) {
#ifdef __linux__ 
		ofstream to_file(("horizontal_profile/" + file_name + ".dat").c_str());
#endif
#ifdef _WIN32
		ofstream to_file(("horizontal_profile\\" + file_name + ".dat").c_str());
#endif

		int l, L;
		to_file << head << " t=" << time << endl;
		//for (unsigned int j = 0; j <= nyg; j = j + step) {
		unsigned int j = nyg / 2;
		if (j_ > -1) j = j_;
		for (unsigned int i = 0; i <= nxg; i = i + step) {
			l = i + OFFSET*j; L = J[l];
			//if (J[l] == J[l]) to_file << i << " " << j << " " << f[L] << endl;
			if (I[l] == 1) {
				to_file << i << " " << hx*i;
				for (int k = 0; k < N_fields; k++) {
					to_file << " " << f[k][L];
				}
				to_file << endl;
			}
			else
			{
				to_file << "skip" << endl;
				//to_file << i << " " << j << " " << NAN << endl;
				//to_file << i << " " << j << " " << 0 << " " << 0 << " " << 0 << " " << 0 << " " << 0 << " " << 0 << " " << 0 	<< endl;
			}

		}
		//}
		to_file.close();

	}

	void write_section_profile(string file_name, string head, double time, int step, double hy, double **f, int N_fields, unsigned int i) {
#ifdef __linux__ 
		ofstream to_file(("vertical_profile/" + file_name + ".dat").c_str());
#endif
#ifdef _WIN32
		ofstream to_file(("vertical_profile\\" + file_name + ".dat").c_str());
#endif

		int l, L;
		to_file << head << " t=" << time << endl;
		for (unsigned int j = 0; j <= nyg; j = j + step) {
			l = i + OFFSET*j; L = J[l];
			if (I[l] == 1) {
				to_file << j << " " << hy*j;
				for (int k = 0; k < N_fields; k++) {
					to_file << " " << f[k][L];
				}
				to_file << endl;
			}
			else
			{
				to_file << "skip" << endl;
			}

		}
		to_file.close();

	}


	void write_field_tecplot(double blank, double hx, double hy, string file_name, double time, int step, int iter, double **f, unsigned int N_fields, string head) {

		ofstream to_file;
		if (iter == 1)
			to_file.open((file_name + ".dat").c_str());
		else
			to_file.open((file_name + ".dat").c_str(), ofstream::app);

		//make time to be string type
		stringstream ss;
		ss << time;
		string str_time = ss.str();

		//count the number of x and y elements
		unsigned int II = 0, JJ = 0;
		for (unsigned int j = 0; j <= nyg; j = j + step)
			JJ++;
		for (unsigned int i = 0; i <= nxg; i = i + step)
			II++;

		//to_file << "VARIABLES=\"x\",\"y\",\"C\",\"mu\",\"vx\",\"vy\",\"p\"" << endl;
		to_file << head << endl;
		to_file << "ZONE T=\"" + str_time + "\", " << "I=" << II << ", J=" << JJ << endl;

		int l, L;
		//to_file << time << endl;
		for (unsigned int j = 0; j <= nyg; j = j + step) {
			for (unsigned int i = 0; i <= nxg; i = i + step) {
				l = i + OFFSET*j; L = J[l];
				if (I[l] == 1) {
					//to_file << hx*i << " " << hy*j << " " << C[L] << " " << mu[L] << " " << vx[L] << " " << vy[L] << " " << p[L] << endl;
					to_file << hx*i << " " << hy*j;
					for (int k = 0; k < N_fields; k++) {
						to_file << " " << f[k][L];
					}
					to_file << endl;
				}
				else
				{
					to_file << hx*i << " " << hy*j;
					for (int k = 0; k < N_fields; k++) {
						to_file << " " << blank;
					}
					to_file << endl;
				}

			}
		}
		//to_file.close();
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

	void save(double *vx, double *vy, double *p, double *C, double *mu, unsigned int i_time, unsigned int i_write, double timeq, double kk, unsigned int extended = 0, double* vib = NULL) {

		ofstream to_file("recovery.dat");
		ofstream to_file2("recovery2.dat");

		to_file << i_time << " " << i_write << " " << timeq << " " << kk << endl;
		to_file2 << i_time << " " << i_write << " " << timeq << " " << kk << endl;


		for (unsigned int i = 0; i < TOTAL_SIZE; i++)
		{
			to_file << vx[i] << " " << vy[i] << " " << p[i] << " " << C[i] << " " << mu[i] << endl;
		}
		for (unsigned int i = 0; i < TOTAL_SIZE; i++)
		{
			to_file2 << vx[i] << " " << vy[i] << " " << p[i] << " " << C[i] << " " << mu[i] << endl;
		}




		to_file.close();
		to_file2.close();

	}
	void save(double** f, unsigned int n, unsigned int i_time, unsigned int i_write, double timeq, double kk) {

		ofstream to_file("recovery.dat");
		ofstream to_file2("recovery2.dat");

		to_file << i_time << " " << i_write << " " << timeq << " " << kk << endl;
		to_file2 << i_time << " " << i_write << " " << timeq << " " << kk << endl;


		for (unsigned int i = 0; i < TOTAL_SIZE; i++)
		{
			for (unsigned int k = 0; k < n; k++) {
				to_file << f[k][i] << " ";
			}
			to_file << endl;
		}

		for (unsigned int i = 0; i < TOTAL_SIZE; i++)
		{
			for (unsigned int k = 0; k < n; k++) {
				to_file2 << f[k][i] << " ";
			}
			to_file2 << endl;
		}

		to_file.close();
		to_file2.close();

	}

	void recover(double *vx, double *vy, double *p, double *C, double *mu, unsigned int &i_time, unsigned int &i_write, double &timeq, unsigned int &kk, unsigned int extended = 0, double *vib = NULL) {
		ifstream from_file("recovery.dat");

		string str;
		string substr;
		stringstream ss;


		getline(from_file, str);

		ss << str;
		ss >> substr; i_time = atoi(substr.c_str());
		ss >> substr; i_write = atoi(substr.c_str());
		ss >> substr; timeq = atof(substr.c_str());
		ss >> substr; kk = atoi(substr.c_str());

		for (unsigned int i = 0; i < TOTAL_SIZE; i++) {
			getline(from_file, str);
			ss.str(""); ss.clear();
			ss << str;
			ss >> substr; vx[i] = atof(substr.c_str());
			ss >> substr; vy[i] = atof(substr.c_str());
			ss >> substr; p[i] = atof(substr.c_str());
			ss >> substr; C[i] = atof(substr.c_str());
			if (extended) ss >> substr; vib[i] = atof(substr.c_str());
		}






		from_file.close();
	}

	void recover(double** f, unsigned int n, unsigned int& i_time, unsigned int& i_write, double& timeq, unsigned int& kk) {
		ifstream from_file("recovery.dat");

		string str;
		string substr;
		stringstream ss;


		getline(from_file, str);

		ss << str;
		ss >> substr; i_time = atoi(substr.c_str());
		ss >> substr; i_write = atoi(substr.c_str());
		ss >> substr; timeq = atof(substr.c_str());
		ss >> substr; kk = atoi(substr.c_str());


		for (unsigned int i = 0; i < TOTAL_SIZE; i++) {
			getline(from_file, str);
			ss.str(""); ss.clear(); ss << str;
			for (unsigned int k = 0; k < n; k++) {
				ss >> substr; f[k][i] = atof(substr.c_str());
			}
		}


		from_file.close();
	}


	void read_concentration(double *C, string file_name, int column, int skip_lines = 1, int invert_C = 1) {
		ifstream from_file(file_name);

		string str;
		string substr;
		stringstream ss;


		for (int k = 0; k < skip_lines; k++) {
			getline(from_file, str);
		}


		for (unsigned int i = 0; i < TOTAL_SIZE; i++) {
			getline(from_file, str);
			ss.str(""); ss.clear();
			ss << str;

			for (int k = 0; k < column; k++) {
				ss >> substr;
			}
			C[i] = atof(substr.c_str());
			if (invert_C == 1)
				C[i] = C[i] * (-1);
		}


		from_file.close();
	}

	void read_grid_geometry() {
		ifstream from_file("GRID.dat");
		if (from_file.good()) {
			cout << endl << "GRID.dat has been read" << endl << endl;
		}


		string str;
		string substr;
		stringstream ss;


		getline(from_file, str);

		ss << str;
		ss >> substr; nx = nxg = atoi(substr.c_str());
		ss >> substr; ny = nyg = atoi(substr.c_str());
		ss >> substr; offset = OFFSET = atoi(substr.c_str());
		ss >> substr; TOTAL_SIZE = atoi(substr.c_str());
		getline(from_file, str); //head 

		n1 = (int*)malloc(TOTAL_SIZE * sizeof(int));
		n2 = (int*)malloc(TOTAL_SIZE * sizeof(int));
		n3 = (int*)malloc(TOTAL_SIZE * sizeof(int));
		n4 = (int*)malloc(TOTAL_SIZE * sizeof(int));
		J_back = (int*)malloc(TOTAL_SIZE * sizeof(int));
		t = (int*)malloc(TOTAL_SIZE * sizeof(int));

		I = new int[(nxg + 1) * (nyg + 1)];
		J = new int[(nxg + 1) * (nyg + 1)];

		int L, l;
		for (int j = 0; j <= nyg; j++) {
			for (int i = 0; i <= nxg; i++) {
				getline(from_file, str);  ss.str(""); ss.clear(); ss << str; //read a line
				L = i + OFFSET * j;
				ss >> substr; I[L] = atoi(substr.c_str());
				ss >> substr; // i
				ss >> substr; // j
				ss >> substr; J[L] = atoi(substr.c_str());
				if (I[L] == 1) {
					l = J[L];
					ss >> substr; t[l] = atoi(substr.c_str());
					ss >> substr; J_back[l] = atoi(substr.c_str());
					ss >> substr; n1[l] = atoi(substr.c_str());
					ss >> substr; n2[l] = atoi(substr.c_str());
					ss >> substr; n3[l] = atoi(substr.c_str());
					ss >> substr; n4[l] = atoi(substr.c_str());
				}



			}
		}


		from_file.close();
	}

	void linear_pressure(double *p, double hx, double hy, double cosA, double sinA, double Lx, double Ly, double coefficient = 1) {
		for (unsigned int l = 0; l < TOTAL_SIZE; l++) {
			p[l] = coefficient*((Lx - hx*iG(l))*cosA - (Ly - hy*jG(l))*sinA);
		}

	}

	//C0(qx,qy)=0.5d0*dtanh((qx*hx-0.5d0)/delta)
	void fill_gradually(double *C, double hx, double hy, double delta, double shift) {
		unsigned int i, j;
		for (unsigned int l = 0; l < TOTAL_SIZE; l++) {
			i = iG(l); j = jG(l);
			C[l] = 0.5*tanh((i*hx - shift) / delta);
		}
	}
	void fill_with_sphere(double *C, double hx, double hy, double x0, double y0, double R0, double C_outer, double C_inner) {
		unsigned int i, j;
		for (unsigned int l = 0; l < TOTAL_SIZE; l++) {
			i = iG(l); 	j = jG(l);
			double x = i*hx, y = j*hy;

			if (sqrt(pow(x - x0, 2) + pow(y - y0, 2)) < R0) {
				C[l] = C_inner;
			}
			else {
				C[l] = C_outer;
			}
		}
	}

	void fill_horizontal_way(double *C, double hx, double hy, double eq_C, double y0, double amp, double k, double delta) {
		unsigned int i, j;
		for (unsigned int l = 0; l < TOTAL_SIZE; l++) {
			i = iG(l); 	j = jG(l);
			double x = i*hx, y = j*hy;

			C[l] = eq_C*(tanh((y - y0 - amp*cos(k*x)) / delta));

		}
	}

	void fast_test_writing(double *f) {

		ofstream to_file("test_field.dat");


		to_file << "i, j, f" << endl;

		int l, L;

		for (unsigned int j = 0; j <= nyg; j = j++) {
			for (unsigned int i = 0; i <= nxg; i = i++) {
				l = i + OFFSET*j; L = J[l];

				if (I[l] == 1) {
					to_file << i << " " << j << " " << f[L] << endl;
				}
				else
				{
					to_file << "skip" << endl;
				}

			}
		}


		to_file.close();
	}

	double isoline(double hx, double hy, double *C, signed char *mark, double *fx, double *fy, double val) {
		//integer, intent(in) ::nx, ny
		//	real * 8, intent(in) ::Lx, C(0:nx, 0 : ny), val
		//real * 8, intent(inout)	::fx(0:nx - 1, 0 : ny), fy(0:nx, 0 : ny - 1)
		//integer, intent(inout)	::mark(0:nx, 0 : ny)
		//integer qx, qy, i, j
		//real * 8 hx, hy, len
		//real(8), parameter::nan = transfer(-2251799813685248_int64, 1._real64)


		double len = 0;
		double nan = NAN;
		unsigned int i, j, ii, jj;
		unsigned int lr, lu, lru; //l right, up and right-up

		for (unsigned int l = 0; l < TOTAL_SIZE; l++) {
			fx[l] = nan;
			fy[l] = nan;
		}
		//		int l = 0;


		for (unsigned int l = 0; l < TOTAL_SIZE; l++) {
			if (C[l] < val)
				mark[l] = -1;
			else if (C[l] > val)
				mark[l] = +1;
			else
				mark[l] = 0;
		}

		for (unsigned int l = 0; l < TOTAL_SIZE; l++) {
			i = iG(l); 	j = jG(l);

			if (t[l] == 2 || t[l] == 3 || t[l] == 10 || t[l] == 6 || t[l] == 7) continue;
			if (t[n2[l]] == 10 || t[n4[l]] == 10) continue;
			if (n3[l] == -1 || n2[l] == -1) continue;
			ii = iG(n3[l]); jj = jG(n2[l]);

			//if (ii > nxg || jj > nyg) continue;


			//cout << "l " << l << endl;

			lr = n3[l]; lu = n2[l]; lru = n3[n2[l]];

			if (abs(mark[l] + mark[lr] + mark[lu] + mark[lru]) == 4)
				continue;
			else {
				//case a

				//************
				//************
				//************
				//************
				//−−−−−−−−−−−−

				if (mark[l] == 0 && mark[lr] == 0) {
					fy[l] = hy*j;
					fy[lr] = hy*j;

					len = len + hx;
					continue;
				}

				//case b

				//| ***********
				//| ***********
				//| ***********
				//| ***********
				//| ***********

				if (mark[l] == 0 && mark[lu] == 0) {
					fx[l] = hx*i;
					fx[lu] = hx*i;

					len = len + hy;
					continue;
				}

				//case 1

				//************
				//************
				//−−−−−−−−−−−−
				//************
				//************

				if (mark[l] * mark[lu] <= 0 && mark[lr] * mark[lru] <= 0 && mark[l] * mark[lu] + mark[lr] * mark[lru] != 0) {

					fy[l] = (val - C[l])*hy / (C[lu] - C[l]) + hy*j;  //left
					fy[lr] = (val - C[lr])*hy / (C[lru] - C[lr]) + hy*j; //right

					len = len + sqrt(hx*hx + pow(fy[lr] - fy[l], 2));
					continue;
				}

				//case 2

				//***** | ******
				//***** | ******
				//***** | ******
				//***** | ******
				//***** | ******

				if (mark[l] * mark[lr] <= 0 && mark[lu] * mark[lru] <= 0 && mark[l] * mark[lr] + mark[lu] * mark[lru] != 0) {
					fx[l] = (val - C[l])*hx / (C[lr] - C[l]) + hx*i; //down
					fx[lu] = (val - C[lu])*hx / (C[lru] - C[lu]) + hx*i; //up

					len = len + sqrt(pow(fx[lu] - fx[l], 2) + hy*hy);
					continue;
				}

				//case 3

				//***** | ******
				//***** | ******
				//−−−−−−******
				//************
				//************

				if (mark[l] * mark[lu] <= 0 && mark[lu] * mark[lru] <= 0 && mark[l] * mark[lu] + mark[lu] * mark[lru] != 0) {
					fx[lu] = (val - C[lu])*hx / (C[lru] - C[lu]) + hx*i; //up
					fy[l] = (val - C[l])*hy / (C[lu] - C[l]) + hy*j;  //left

					len = len + sqrt(pow(fx[lu] - hx*i, 2) + pow(fy[l] - hy*(jj), 2));
					continue;
				}

				//case 4

				//***** | ******
				//***** | ******
				//*****−−−−−−−
				//************
				//************

				if (mark[lr] * mark[lru] <= 0 && mark[lu] * mark[lru] <= 0 && mark[lr] * mark[lru] + mark[lu] * mark[lru] != 0) {
					fx[lu] = (val - C[lu])*hx / (C[lru] - C[lu]) + hx*i; //up
					fy[lr] = (val - C[lr])*hy / (C[lru] - C[lr]) + hy*j; //right

					len = len + sqrt(pow(fx[lu] - hx*(ii), 2) + pow(fy[lr] - hy*(jj), 2));
					continue;
				}

				//case 5

				//************
				//************
				//******−−−−−−
				//***** | ******
				//***** | ******

				if (mark[l] * mark[lr] <= 0 && mark[lr] * mark[lru] <= 0 && mark[l] * mark[lr] + mark[lr] * mark[lru] != 0) {
					fy[lr] = (val - C[lr])*hy / (C[lru] - C[lr]) + hy*j; //right
					fx[l] = (val - C[l])*hx / (C[lr] - C[l]) + hx*i; //down

					len = len + sqrt(pow(fx[l] - hx*(ii), 2) + pow(fy[lr] - hy*j, 2));
					continue;
				}

				//case 6

				//************
				//************
				//−−−−−−******
				//***** | ******
				//***** | ******

				if (mark[l] * mark[lr] <= 0 && mark[l] * mark[lu] <= 0 && mark[l] * mark[lr] + mark[l] * mark[lu] != 0) {
					fy[l] = (val - C[l])*hy / (C[lu] - C[l]) + hy*j; //left
					fx[l] = (val - C[l])*hx / (C[lr] - C[l]) + hx*i; //down

					len = len + sqrt(pow(fx[l] - hx*i, 2) + pow(fy[l] - hy*j, 2));
					continue;
				}




			}//end of main if


		}

		return len;

	}

	double volume(double hx, double hy, double *C, double lim) {
		double vol = 0;
		//unsigned int i, j;
		for (unsigned int l = 0; l < TOTAL_SIZE; l++) {
			if (t[l] == 2 || t[l] == 3 || t[l] == 10 || t[l] == 6 || t[l] == 7) continue;
			if (abs(C[l]) < lim)
				vol += hx*hy;
		}

		return vol;
	}

	double change_sign_at_X(double hx, double hy, double *F, unsigned int j) {
		int l1, l2, L1, L2;
		double x = 0;
		double F_ = 0;
		//unsigned int j = nyg / 2;
		l1 = 0 + OFFSET*j; L1 = J[l1];

		for (unsigned int i = 1; i <= nxg; i++) {
			l2 = i + OFFSET*j; L2 = J[l2];


			if (I[l2] == 1)
			{
				if (F[L2] * F[L1] <= 0) {
					x = hx*(F_ - F[L1]) / (F[L2] - F[L1]) + ((i - 1)*hx);
					return x;
				}
				L1 = L2;
			}
			else
			{
				cout << "not a good sign you see it" << endl;
				exit(0);
			}


		}
		return x;
	}
	double pressure_jump(double hx, double hy, double *p, double x_, double border_width) {
		int l, L;
		unsigned int j = nyg / 2;

		double P1 = 0, P2 = 0;
		int i1, i2, n1 = 0, n2 = 0;


		i1 = (int)((x_ - border_width) / hx);
		i2 = (int)((x_ + border_width) / hx);
		for (unsigned int i = 0; i <= nxg; i++) {
			l = i + OFFSET*j; L = J[l];

			if (I[l] == 1) {

				if (i < i1) {
					P1 += p[L];
					n1++;
				}
				if (i > i2) {
					P2 += p[L];
					n2++;
				}

			}
			else
			{
				cout << "not a good sign you see it" << endl;
				exit(0);
			}


		}

		P1 = P1 / n1;
		P2 = P2 / n2;
		double ret = abs(P2 - P1);
		if (!std::isfinite(ret)) return 0.0;
		else return ret;
	}

	double flow_rate(double hx, double hy, double *vx, double Ly, unsigned int i) {
		int l, L;
		double Q = 0.0;
		for (unsigned int j = 0; j <= nyg; j++) {
			l = i + OFFSET*j; L = J[l];


			if (I[l] == 1) {
				Q += vx[L];
			}
			else
			{
				cout << "not a good sign you see it" << endl;
				exit(0);
			}
		}

		Q = Q * hy / Ly;
		return Q;
	}






	double tension(double hx, double hy, double *C) {
		double ten = 0;
		//unsigned int lr, lu, lru;
		for (unsigned int l = 0; l < TOTAL_SIZE; l++) {
			if (t[l] == 0) {
				ten += 0.25 / hx / hx*pow(C[n3[l]] - C[n1[l]], 2) + 0.25 / hy / hy*pow(C[n2[l]] - C[n4[l]], 2);
			}
		}

		return ten*hx*hy;
	}
	void X_averaged_in_each_phase(double hx, double hy, double *C, double *X, double &X1av, double &X2av, double &Xav, double level = 0.0) {
		Xav = 0; X1av = 0; /*plus*/ X2av = 0; /*minus*/
		unsigned int n = 0, n2 = 0, n_plus = 0, n2_plus = 0, n_minus = 0, n2_minus = 0;

		for (unsigned int l = 0; l < TOTAL_SIZE; l++) {
			if (t[l] == 0) {
				Xav += X[l];
				n++;
				if (C[l] > level) {
					X1av += X[l];
					n_plus++;
				}
				if (C[l] < -level) {
					X2av += X[l];
					n_minus++;
				}
			}

			else
			{
				Xav += X[l] / 2;
				n2++;
				if (C[l] > level) {
					X1av += X[l] / 2;
					n2_plus++;
				}
				if (C[l] < -level) {
					X2av += X[l] / 2;
					n2_minus++;
				}
			}
		}

		if (n + n2 > 0) Xav /= (n + 0.5*n2);
		if (n_plus + n2_plus > 0) X1av /= (n_plus + 0.5*n2_plus);
		if (n_minus + n2_minus > 0) X2av /= (n_minus + 0.5*n2_minus);
	}

#define DX(F) 0.5 / hx * (F[n3[l]] - F[n1[l]])
#define DY(F) 0.5 / hy * (F[n2[l]] - F[n4[l]])
#define DX2(F) 1.0 / (hx * hx) * (F[n3[l]] + F[n1[l]] - 2.0 * F[l])
#define DY2(F) 1.0 / (hy * hy) * (F[n2[l]] + F[n4[l]] - 2.0 * F[l])
	//#define DXY(F) (-F[l - 1 + OFFSET] + F[l + 1 + OFFSET] - F[l + 1 - OFFSET] + f[l - 1 - OFFSET]) / hx / hy / 4.0;
#define DXY(F) (-F[n1[n2[l]]] + F[n3[n2[l]]] - F[n3[n4[l]]] + F[n1[n4[l]]]) / hx / hy / 4.0

	void curvature_direct(double *C, double hx, double hy, double *curv, double add = 0.0) {
		double dCx, dCy, abs_dC;
		unsigned int i, j;
		for (unsigned int l = 0; l < TOTAL_SIZE; l++) {
			i = iG(l); 	j = jG(l);

			if (t[l] == 0) {
				dCx = DX(C); dCy = DY(C);
				abs_dC = sqrt(dCx*dCx + dCy*dCy);
				double abs_dC3 = abs_dC*abs_dC*abs_dC + add;

				curv[l] = (dCx*dCx*DY2(C) + dCy*dCy*DX2(C) - 2.0*dCx*dCy*DXY(C)) / abs_dC3;
				//if (abs_dC < 1e-6) curv[l] = 0;
			}
			else {
				curv[l] = 0.0;
			}
		}
	}

	void curvature_direct2(double *C, double hx, double hy, double *curv) {
		double dCx, dCy, abs_dC;
		unsigned int i, j;
		for (unsigned int l = 0; l < TOTAL_SIZE; l++) {
			i = iG(l); 	j = jG(l);

			if (t[l] == 0) {
				dCx = DX(C); dCy = DY(C);
				abs_dC = sqrt(dCx*dCx + dCy*dCy);
				double abs_dC3 = abs_dC*abs_dC*abs_dC;

				curv[l] = (dCx*dCx*DY2(C) + dCy*dCy*DX2(C) - 2.0*dCx*dCy*DXY(C)) / abs_dC3;
				if (abs_dC < 1e-6) curv[l] = 0;
			}
			else {
				curv[l] = 0.0;
			}
		}
	}



	void curvature_2_steps(double *C, double *nx, double *ny, double hx, double hy, double *curv) {
		//1
		for (unsigned int l = 0; l < TOTAL_SIZE; l++) {
			if (t[l] == 0) {
				double dCx = DX(C);
				double dCy = DY(C);
				double abs_dC = sqrt(dCx*dCx + dCy*dCy) + 0.001;
				nx[l] = dCx / abs_dC;
				ny[l] = dCy / abs_dC;

			}
			else {
				nx[l] = 0.0;
				ny[l] = 0.0;
			}
		}
		//2
		for (unsigned int l = 0; l < TOTAL_SIZE; l++) {
			if (t[l] == 0) {
				curv[l] = DX(nx) + DY(ny);
			}
			else {
				curv[l] = 0.0;
			}
		}
	}

	void check() {
		int l, L;
		ofstream write("geomSettingCheck.txt");
		write << "i, j, 1, L, t[L], n1[L], n2[L], n3[L], n4[L]" << endl;
		for (unsigned int i = 0; i <= nxg; i++) {
			for (unsigned int j = 0; j <= nyg; j++) {
				l = i + OFFSET*j; L = J[l];

				if (I[i + OFFSET*j] == 1)
					write << i << " " << j << " " << 1 << " " << L << " " << t[L] << " " << n1[L] << " " << n2[L] << " " << n3[L] << " " << n4[L] << endl;
				else write << i << " " << j << " " << -1 << " " << -1 << " " << -1 << " " << -1 << " " << -1 << " " << -1 << " " << -1 << endl;
			}



		}
		write.close();


	}

	unsigned int checkExit(double *C) {
		for (unsigned int l = 0; l < TOTAL_SIZE; l++) {
			if (t[l] == 10) {
				if (C[l] < 0)
					return 1;
			}
		}
		return 0;
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
		for (unsigned int i = 0; i < (nxg + 1)*(nyg + 1); i++) {
			J[i] = -1; I[i] = 0;
		}

		// zero-level
		while (shiftY < line_y) {
			for (unsigned int j = 0; j <= z; j++) {
				for (unsigned int i = 0; i <= x - 1; i++) {
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
		for (unsigned int C = 1; C <= line_N; C++)
		{

			//column
			shiftX += x;
			for (unsigned int i = 0; i <= z; i++) {
				for (unsigned int j = 0; j <= line_y; j++) {
					li.push_back(i + shiftX);
					lj.push_back(j);
					iter++;
				}
			}

			//blocks
			shiftX += z;
			shiftY = (y + z) / 2;
			while (shiftY < line_y) {
				for (unsigned int i = 1; i <= x - 1; i++) {
					for (unsigned int j = 0; j <= z; j++) {
						li.push_back(i + shiftX);
						lj.push_back(j + shiftY);

						iter++;
					}
				}
				shiftY += y + z;
			}


			//column
			shiftX += x;

			for (unsigned int i = 0; i <= z; i++) {
				for (unsigned int j = 0; j <= line_y; j++) {
					li.push_back(i + shiftX);
					lj.push_back(j);
					iter++;
				}
			}

			//blocks
			shiftX += z;
			shiftY = 0;
			while (shiftY < line_y) {
				for (unsigned int j = 0; j <= z; j++) {
					for (unsigned int i = 1; i <= x - 1; i++) {
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

		for (unsigned int i = 0; i < TOTAL_SIZE; i++) {
			n1[i] = -1;
			n2[i] = -1;
			n3[i] = -1;
			n4[i] = -1;
			t[i] = -1;
		}


		for (unsigned int i = 0; i < iter; i++) {
			J_back[i] = li[i] + OFFSET*lj[i];
			J[J_back[i]] = i;
			I[J_back[i]] = 1;
		}










	}

	void set_neighbor() {
		int l, L, l1, l2, l3, l4;


		for (unsigned int i = 0; i <= nxg; i++) {
			for (unsigned int j = 0; j <= nyg; j++) {
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
		int l, L;
		//int l1, l2, l3, l4;

		//inner
		for (unsigned int i = 0; i <= nxg; i++) {
			for (unsigned int j = 0; j <= nyg; j++) {
				l = i + OFFSET*j; L = J[l];
				if (I[l] == 1) {
					if (n1[L] != -1 && n2[L] != -1 && n3[L] != -1 && n4[L] != -1)
						t[L] = 0;

				}
			}
		}



		//rigid walls
		for (unsigned int i = 0; i <= nxg; i++) {
			for (unsigned int j = 0; j <= nyg; j++) {
				l = i + OFFSET*j; L = J[l];
				//	l1 = i - 1 + OFFSET*j; 				l2 = i + OFFSET*j + OFFSET; 				l3 = i + 1 + OFFSET*j;				l4 = i + OFFSET*j - OFFSET;
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
		for (unsigned int i = 0; i <= nxg; i++) {
			for (unsigned int j = 0; j <= nyg; j++) {
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
		for (unsigned int i = 0; i <= nxg; i = i + nxg) {
			for (unsigned int j = 0; j <= nyg; j++) {
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
		for (unsigned int i = 0; i <= nxg; i++) {
			for (unsigned int j = 0; j <= nyg; j++) {
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
		for (unsigned int j = 0; j <= nyg; j = j + step) {
			for (unsigned int i = 0; i <= nxg; i = i + step) {
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


		for (unsigned int i = 0; i < TOTAL_SIZE; i++)
			to_file << vx[i] << " " << vy[i] << " " << p[i] << " " << C[i] << " " << mu[i] << endl;
		for (unsigned int i = 0; i < TOTAL_SIZE; i++)
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

		for (unsigned int i = 0; i < TOTAL_SIZE; i++) {
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

struct box {

	cross *Mcr;
	int *l, *t, *I, *J, *J_back;
	int Mx, My, Msize, Moffset, OFFSET;
	unsigned int iter = 0;
	unsigned int TOTAL_SIZE = 0;
	int *n1, *n2, *n3, *n4;
	unsigned int nx, ny, offset;
	unsigned int nxg, nyg;
	double *C0, *C, *p, *p0, *ux, *uy, *vx, *vy, *mu;
	double LX, LY;

	void set_global_size(int input_nx, int input_ny) {
		nx = input_nx; nxg = nx;
		ny = input_ny; nyg = ny;
		offset = nx + 1;
		OFFSET = offset;
		TOTAL_SIZE = (input_nx + 1) * (input_ny + 1);
	}


	void set_type() {
		l = new int[TOTAL_SIZE];
		t = new int[TOTAL_SIZE];
		for (unsigned int i = 0; i < TOTAL_SIZE; i++) {
			l[i] = 0;
			t[i] = 0;
		}

		unsigned int k;
		for (unsigned int i = 0; i <= nx; i++) {
			for (unsigned int j = 0; j <= ny; j++) {
				k = i + offset*j;
				if (i == 0) t[k] = 9;
				if (i == nx) t[k] = 10;
				if (j == 0) t[k] = 4;
				if (j == ny) t[k] = 2;

				iter++;


			}
		}

	}

	void set_neighbor()
	{


		n1 = (int*)malloc(TOTAL_SIZE * sizeof(int));
		n2 = (int*)malloc(TOTAL_SIZE * sizeof(int));
		n3 = (int*)malloc(TOTAL_SIZE * sizeof(int));
		n4 = (int*)malloc(TOTAL_SIZE * sizeof(int));
		for (unsigned int i = 0; i < TOTAL_SIZE; i++) {
			n1[i] = -1; n2[i] = -1; n3[i] = -1; n4[i] = -1;
		}

		unsigned int k, it = 0;
		for (unsigned int i = 0; i <= nx; i++) {
			for (unsigned int j = 0; j <= ny; j++) {
				k = i + offset*j;
				if (t[k] == 0) {
					n1[k] = k - 1;
					n2[k] = k + offset;
					n3[k] = k + 1;
					n4[k] = k - offset;
				}

				if (t[k] == 2) 	n4[k] = k - offset;
				if (t[k] == 4)  n2[k] = k + offset;
				if (t[k] == 9) {
					n3[k] = k + 1;
					n4[k] = k - offset;
					n2[k] = k + offset;
				}
				if (t[k] == 10) {
					n1[k] = k - 1;
					n4[k] = k - offset;
					n2[k] = k + offset;
				}

				it++;
			}
		}



	}

	void set_global_id() {


		I = new int[(nx + 1)*(ny + 1)];
		J = new int[(nx + 1)*(ny + 1)];
		J_back = new int[TOTAL_SIZE];

		OFFSET = nx + 1;


		for (unsigned int i = 0; i < (nx + 1)*(ny + 1); i++) {
			I[i] = 0; J[i] = -1;
		}
		for (unsigned int i = 0; i < TOTAL_SIZE; i++) {
			J_back[i] = -1;
		}



		unsigned int k, it = 0; // , in, ii, jj;
		for (unsigned int i = 0; i <= nx; i++) {
			for (unsigned int j = 0; j <= ny; j++) {

				k = i + offset*j;
				I[k] = 1;
				J[k] = k;
				J_back[k] = k;

				it++;

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


		unsigned int l, L;
		to_file << time << endl;
		for (unsigned int j = 0; j <= nyg; j = j + step) {
			for (unsigned int i = 0; i <= nxg; i = i + step) {
				l = i + OFFSET*j; L = J[l];
				//if (J[l] == J[l]) to_file << i << " " << j << " " << f[L] << endl;
				if (I[l] == 1) {
					//to_file << i << " " << j << " " << f[L] << " " << t[L] << " " << L << " " << n1[L] << " " << n2[L] << " " << n3[L] << " " << n4[L] << " " <<
					//(J_back[L] - (J_back[L] / OFFSET)*OFFSET) << " " << (J_back[L] / OFFSET) << endl;
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


		for (unsigned int i = 0; i < TOTAL_SIZE; i++)
			to_file << vx[i] << " " << vy[i] << " " << p[i] << " " << C[i] << " " << mu[i] << endl;
		for (unsigned int i = 0; i < TOTAL_SIZE; i++)
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

		for (unsigned int i = 0; i < TOTAL_SIZE; i++) {
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

struct box_inherited :multi_cross
{
	unsigned int nx, ny, offset;

	void set_global_size(int input_nx, int input_ny) {
		nx = input_nx; nxg = nx;
		ny = input_ny; nyg = ny;
		offset = nx + 1;
		OFFSET = offset;
		TOTAL_SIZE = (input_nx + 1) * (input_ny + 1);
	}
	void set_type() {
		l = new int[TOTAL_SIZE];
		t = new int[TOTAL_SIZE];
		for (unsigned int i = 0; i < TOTAL_SIZE; i++) {
			l[i] = 0;
			t[i] = 0;
		}

		unsigned int k;
		for (unsigned int i = 0; i <= nx; i++) {
			for (unsigned int j = 0; j <= ny; j++) {
				k = i + offset*j;
				if (i == 0) t[k] = 9;
				if (i == nx) t[k] = 10;
				if (j == 0) t[k] = 4;
				if (j == ny) t[k] = 2;

				iter++;


			}
		}

	}

	void set_neighbor()
	{


		n1 = (int*)malloc(TOTAL_SIZE * sizeof(int));
		n2 = (int*)malloc(TOTAL_SIZE * sizeof(int));
		n3 = (int*)malloc(TOTAL_SIZE * sizeof(int));
		n4 = (int*)malloc(TOTAL_SIZE * sizeof(int));
		for (unsigned int i = 0; i < TOTAL_SIZE; i++) {
			n1[i] = -1; n2[i] = -1; n3[i] = -1; n4[i] = -1;
		}

		unsigned int k, it = 0;
		for (unsigned int i = 0; i <= nx; i++) {
			for (unsigned int j = 0; j <= ny; j++) {
				k = i + offset*j;
				if (t[k] == 0) {
					n1[k] = k - 1;
					n2[k] = k + offset;
					n3[k] = k + 1;
					n4[k] = k - offset;
				}

				if (t[k] == 2) 	n4[k] = k - offset;
				if (t[k] == 4)  n2[k] = k + offset;
				if (t[k] == 9) {
					n3[k] = k + 1;
					n4[k] = k - offset;
					n2[k] = k + offset;
				}
				if (t[k] == 10) {
					n1[k] = k - 1;
					n4[k] = k - offset;
					n2[k] = k + offset;
				}

				it++;
			}
		}



	}

	void set_global_id() {


		I = new int[(nx + 1)*(ny + 1)];
		J = new int[(nx + 1)*(ny + 1)];
		J_back = new int[TOTAL_SIZE];

		OFFSET = nx + 1;


		for (unsigned int i = 0; i < (nx + 1)*(ny + 1); i++) {
			I[i] = 0; J[i] = -1;
		}
		for (unsigned int i = 0; i < TOTAL_SIZE; i++) {
			J_back[i] = -1;
		}



		unsigned int k, it = 0; // in, ii, jj;
		for (unsigned int i = 0; i <= nx; i++) {
			for (unsigned int j = 0; j <= ny; j++) {

				k = i + offset*j;
				I[k] = 1;
				J[k] = k;
				J_back[k] = k;

				it++;

			}
		}


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



//this "stupid" step is designed to keep some objects in the global GPU scope
//it is supposed to simplify some other parts of the code
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

//pressure transformation to real pressure
void true_pressure(double *p, double *p_true, double *C, double *mu, int *t, int *n1, int *n2, int *n3, int *n4, int *J_back,
	double tau, unsigned int size, double hx, double hy, double Ca, double A, double Gr, double M, int OFFSET, double sinA, double cosA, unsigned int PHASE, double VV_h, double vibrX, double vibrY) {
	/*функция написана не совсем интуитивно понятно, были ошибки, ошибки исправлялись,
	осознание того, как надо было, пришло потом, когда переписывать заново стало долго*/
	double WX = 0.0, WY = 0.0;
	double dxC = 0, dyC = 0;
	int left, right, up, down, left2, right2, up2, down2;


	
	for (unsigned int l = 0; l < size; l++) {
		if (PHASE == 0) {
			p_true[l] = p[l];
			continue;
		}

		left = n1[l]; right = n3[l]; up = n2[l]; down = n4[l];
		if (left == -1) left = right;
		if (right == -1) right = left;
		if (up == -1) up = down;
		if (down == -1) down = up;
		left2 = n1[left]; right2 = n3[right]; up2 = n2[up]; down2 = n4[down];

		p_true[l] = p[l] +
			(+mu[l] * C[l] - A*pow(C[l], 2) - pow(C[l], 4)) / M
			+ C[l] * Gr*((J_back[l] - (J_back[l] / OFFSET)*OFFSET) * hx*cosA + (J_back[l] / OFFSET) * hy*sinA);



		switch (t[l])
		{
		case 0: //inner
			p_true[l] += -0.5*Ca / M*(
				pow((0.5*(C[right] - C[left]) / hx), 2)
				+ pow((0.5*(C[up] - C[down]) / hy), 2));
			if (VV_h > 0) {
				WX = -C[l] * vibrX + 0.5*(C[right] - C[left]) / hx;
				WY = -C[l] * vibrY + 0.5*(C[up] - C[down]) / hy;

				p_true[l] += -VV_h*0.5*(WX*WX + WY*WY);
			}
			break;
		case 1: //left rigid
			p_true[l] += -0.5*Ca / M*(
				pow((-0.5*(3.0*C[l] - 4.0*C[right] + C[right2]) / hx), 2)
				+ pow((0.5*(C[up] - C[down]) / hy), 2));
			break;
		case 2: //upper rigid
			p_true[l] += -0.5*Ca / M*(
				pow((0.5*(C[right] - C[left]) / hx), 2)
				+ pow((0.5*(3.0*C[l] - 4.0*C[down] + C[down2]) / hy), 2));
			break;
		case 3: //right rigid
			p_true[l] += -0.5*Ca / M*(
				pow((0.5*(3.0*C[l] - 4.0*C[left] + C[left2]) / hx), 2)
				+ pow((0.5*(C[up] - C[down]) / hy), 2));
			break;
		case 4: //lower rigid
			p_true[l] += -0.5*Ca / M*(
				pow((0.5*(C[right] - C[left]) / hx), 2)
				+ pow((-0.5*(3.0*C[l] - 4.0*C[up] + C[up2]) / hy), 2));
			break;
		case 5: //left upper rigid corner
			p_true[l] += -0.5*Ca / M*(
				pow((-0.5*(3.0*C[l] - 4.0*C[right] + C[right2]) / hx), 2)
				+ pow((0.5*(3.0*C[l] - 4.0*C[down] + C[down2]) / hy), 2));
			break;
		case 6: //right upper rigid corner
			p_true[l] += -0.5*Ca / M*(
				pow((0.5*(3.0*C[l] - 4.0*C[left] + C[left2]) / hx), 2)
				+ pow((0.5*(3.0*C[l] - 4.0*C[down] + C[down2]) / hy), 2));
			break;
		case 7: //right lower rigid corner
			p_true[l] += -0.5*Ca / M*(
				pow((0.5*(3.0*C[l] - 4.0*C[left] + C[left2]) / hx), 2)
				+ pow((-0.5*(3.0*C[l] - 4.0*C[up] + C[up2]) / hy), 2));
			break;
		case 8: //left lower rigid corner
			p_true[l] += -0.5*Ca / M*(
				pow((-0.5*(3.0*C[l] - 4.0*C[right] + C[right2]) / hx), 2)
				+ pow((-0.5*(3.0*C[l] - 4.0*C[up] + C[up2]) / hy), 2));
			break;
		case 9: //inlet (from left)
			dxC = -0.5*(3.0*C[l] - 4.0*C[right] + C[right2]) / hx;
			dyC = 0.5*(C[up] - C[down]) / hy;
			p_true[l] += -0.5*Ca / M*(pow(dxC, 2) + pow(dyC, 2));
			if (VV_h > 0) {
				WX = -C[l] * vibrX + dxC;
				WY = -C[l] * vibrY + dyC;

				p_true[l] += -VV_h*0.5*(WX*WX + WY*WY);
			}
			break;
		case 10://outlet (to right)
			dxC = 0.5*(3.0*C[l] - 4.0*C[left] + C[left2]) / hx;
			dyC = 0.5*(C[up] - C[down]) / hy;
			p_true[l] += -0.5*Ca / M*(pow(dxC, 2)+ pow(dyC, 2));
			if (VV_h > 0) {
				WX = -C[l] * vibrX + dxC;
				WY = -C[l] * vibrY + dyC;

				p_true[l] += -VV_h*0.5*(WX*WX + WY*WY);
			}
			break;
		default:
			break;
		}

	}

}





void signalHandler(int signum) {

	cout << "Interrupt signal (" << signum << ") received.\n";
	cout << "state: " << state << endl;
	// cleanup and close up stuff here  
	// terminate program  

	exit(signum);
}
void create_folder(string name) {
#ifdef __linux__ 
	string str = "mkdir -p " + name + "/";
	system(str.c_str());
#endif
#ifdef _WIN32
	CreateDirectoryA(name.c_str(), NULL);
#endif
}





int main(int argc, char **argv) {
	state = 0;
	signal(SIGINT, signalHandler);

	cout << "Version: " << ThisSoftwareVersion << endl;
	cout << "Compilation time: " << __DATE__ << " " << __TIME__ << endl;
	cout << "command line: " << endl; for (int i = 0; i < argc; i++) cout << i << ": " << argv[i] << endl;

	int devID = 0, deviceCount = 0;
	cudaGetDeviceCount(&deviceCount);
	if (deviceCount == 0) cout << "there is no detected GPU" << endl;
	double heap_GB = 1.0;

	double timer1, timer2;
	double pi = 3.1415926535897932384626433832795;
	double eps0 = 1e-5;
	double *C0, *C, *p, *p0, *ux, *uy, *vx, *vy, *mu, *zero, *Phi, *Phi0, *WX, *WY;  //_d - device (GPU) 
	double *C_h, *p_h, *vx_h, *vy_h, *mu_h, *p_true_h, *zero_h, *Phi_h, *WX_h, *WY_h;  //_h - host (CPU)
	double *curv1, *curv2, *nx_dC, *ny_dC;
	double *psiav_array, *psiav_array_Phi; 		 //  temporal variables //psiav0_h, eps_h *psiav_d, *psiav_array_h,   *psiav_h;
	double hx_h, hy_h, Lx_h, Ly_h, tau_h, tau_p_h, psiav, psiav0, eps, A_h, Ca_h, Gr_h, Pe_h, Re_h, MM_h, dP_h, Gs_h; //parameters 
	double alpha_h, sinA_h, cosA_h, theta_h, sinTh_h, cosTh_h;
	double Ek, Ek_old, Vmax, Q_in, Q_out, C_average, Cv;
	unsigned int nx_h, ny_h, Matrix_X, Matrix_Y, iter = 0, offset_h, Phi_kk, kk, k = 0, tt, write_i = 0, each = 1, stop = 0;					  //parameters
	double time_fields, time_recovery, time_display;
	double timeq = 0.0, C_av, C_plus, C_minus;
	double tecplot, limit_timeq;
	bool copied = false;
	unsigned int linear_pressure, fill_gradually, wetting, read_C, stop_at_exit;
	unsigned int sphere_distribution, curv_calc, vibration, simple_geometry = 0;
	double fill_gradually_x;
	unsigned int reset_timeq, invert_initial_C, reset_velocity, reset_pressure;
	unsigned int PHASE_h, DIFFUSION_h;
	string geometry;

	//1 is 'yes' / true, 0 is 'no' / false
	//unsigned int clean_fields;
	unsigned int picture_switch = 1; //write fields to a file?
	unsigned int read_switch = 1; //read to continue or not? 


								  //reading_parameters(ny_h, nx_h, each_t, each, Matrix_X, Matrix_Y, tau_h, A_h, Ca_h, Gr_h, Pe_h, Re_h, alpha_h, MM_h, tecplot, PHASE_h);
	create_folder("fields");
	string file_name = "inp.dat";
	if (argc == 2) file_name = argv[1];
	ReadingFile File(file_name);
#define constUint(VAR)  \
	unsigned int VAR##_h; File.reading<unsigned int>(VAR##_h, #VAR, 0); \
	cudaMemcpyToSymbol(VAR, &VAR##_h, sizeof(unsigned int), 0, cudaMemcpyHostToDevice);
#define constDouble(VAR)  \
	double VAR##_h; File.reading<double>(VAR##_h, #VAR, 0.0); \
	cudaMemcpyToSymbol(VAR, &VAR##_h, sizeof(double), 0, cudaMemcpyHostToDevice);



	File.reading<unsigned int>(ny_h, "ny", 200);
	File.reading<unsigned int>(nx_h, "nx", 200);
	File.reading<double>(time_fields, "time_fields", 0.5);
	File.reading<double>(time_recovery, "time_recovery", 0.5);
	File.reading<double>(time_display, "time_display", 0.1);
	File.reading<unsigned int>(each, "each_xy", 10);
	File.reading<unsigned int>(Matrix_X, "Matrix_X", 3);
	File.reading<unsigned int>(Matrix_Y, "Matrix_Y", 3);
	File.reading<double>(tau_h, "tau", 5.0e-5);
	File.reading<double>(A_h, "A", -0.5);
	if (File.reading<double>(Ca_h, "Ca", 4e-4) == 0) File.reading<double>(Ca_h, "Cn", 4e-4);
	File.reading<double>(Gr_h, "Gr", 0.0);
	if (File.reading<double>(Pe_h, "Pe", 1e+4) == 0) File.reading<double>(Pe_h, "Sc", 1e+4);
	File.reading<double>(Re_h, "Re", 1.0);
	File.reading<double>(alpha_h, "alpha", 0.0);
	File.reading<double>(theta_h, "theta", 90.0);
	File.reading<double>(MM_h, "MM", 1.0);
	File.reading<double>(Gs_h, "Gs", 0.0);
	File.reading<double>(dP_h, "dP", 1.0);
	File.reading<double>(tecplot, "tecplot", 10000);
	File.reading<unsigned int>(PHASE_h, "Phase_field", 1, 0, 1);
	File.reading<unsigned int>(read_switch, "read_recovery", 1, 0, 1);
	File.reading<unsigned int>(picture_switch, "picture_switch", 1, 0, 1);
	File.reading<double>(limit_timeq, "time_limit", 5000.0);
	File.reading<unsigned int>(linear_pressure, "linear_pressure", 0, 0, 1);
	File.reading<unsigned int>(fill_gradually, "fill_gradually", 0, 0, 1);
	File.reading<double>(fill_gradually_x, "fill_gradually_x", 0.5);
	File.reading<unsigned int>(DIFFUSION_h, "pure_diffusion", 0, 0, 1);
	File.reading<unsigned int>(wetting, "wetting", 0, 0, 4);
	File.reading_string(geometry, "geometry", "matrix");
	File.reading<unsigned int>(reset_timeq, "reset_time", 0, 0, 1);
	File.reading<unsigned int>(invert_initial_C, "invert_C", 0, 0, 1);
	File.reading<unsigned int>(reset_velocity, "reset_velocity", 0, 0, 1);
	File.reading<unsigned int>(reset_pressure, "reset_pressure", 0, 0, 1);
	File.reading<double>(heap_GB, "heap_GB", 1.0);
	File.reading<int>(devID, "GPU_id", 0, 0, deviceCount - 1);
	File.reading<unsigned int>(read_C, "read_concentration", 0, 0, 1);
	File.reading<unsigned int>(stop_at_exit, "stop_at_exit", 0, 0, 1);
	File.reading<unsigned int>(sphere_distribution, "sphere", 0, 0, 1);
	File.reading<unsigned int>(curv_calc, "curv_calc", 0, 0, 1);
	unsigned int horizontal_profile;	File.reading<unsigned int>(horizontal_profile, "horizontal_profile", 0, 0, 1); if (horizontal_profile) create_folder("horizontal_profile");
	unsigned int vertical_profile;		File.reading<unsigned int>(vertical_profile, "vertical_profile", 0, 0, 1); if (vertical_profile) create_folder("vertical_profile");
	File.reading<double>(Lx_h, "Lx", 0.0);
	File.reading<double>(Ly_h, "Ly", 0.0);
	File.reading<unsigned int>(vibration, "vibration", 0, 0, 3);
	double Amp_h;		File.reading<double>(Amp_h, "Amp", 0.0);
	double Omega_h;		File.reading<double>(Omega_h, "Omega", 0.0);
	double vibr_X_h;		File.reading<double>(vibr_X_h, "vibr_X", 0.0, 0.0, 1.0);
	double vibr_Y_h;		File.reading<double>(vibr_Y_h, "vibr_Y", 0.0, 0.0, 1.0);
	double VV_h;			File.reading<double>(VV_h, "VV", 0.0);  if (vibration == 0) { VV_h = 0; }
	unsigned int integrals_add1; File.reading<unsigned int>(integrals_add1, "integrals_add1", 0);
	string filling;  File.reading_string(filling, "filling", "no");
	//unsigned int W_BORDER_h; File.reading<unsigned int>(W_BORDER_h, "W_BORDER", 0);
	//cudaMemcpyToSymbol(W_BORDER, &W_BORDER_h, sizeof(unsigned int), 0, cudaMemcpyHostToDevice);
	//unsigned int PHI_BORDER_LEFT_h; File.reading<unsigned int>(PHI_BORDER_LEFT_h, "PHI_BORDER_LEFT", 0);
	//cudaMemcpyToSymbol(PHI_BORDER_LEFT, &PHI_BORDER_LEFT_h, sizeof(unsigned int), 0, cudaMemcpyHostToDevice);
	//unsigned int PHI_BORDER_RIGHT_h; File.reading<unsigned int>(PHI_BORDER_RIGHT_h, "PHI_BORDER_RIGHT", 0);
	//cudaMemcpyToSymbol(PHI_BORDER_RIGHT, &PHI_BORDER_RIGHT_h, sizeof(unsigned int), 0, cudaMemcpyHostToDevice);
	//File.reading<unsigned int>(clean_fields, "clean_fields", 1, 0, 1);
	constUint(W_BORDER);
	constUint(PHI_border_left);
	constUint(PHI_border_right);
	constDouble(PHI_value_left);
	constDouble(PHI_value_right);


	//GPU setting
	cudaSetDevice(devID);
	cudaDeviceProp deviceProp;
	cudaGetDeviceProperties(&deviceProp, devID);
	printf("\nDevice %d: \"%s\"\n", devID, deviceProp.name);


	//allocate heap size
	size_t limit = (size_t)(1024 * 1024 * 1024 * heap_GB);
	cudaDeviceSetLimit(cudaLimitMallocHeapSize, limit);
	cudaDeviceGetLimit(&limit, cudaLimitMallocHeapSize);

	//if (clean_fields == 1) {
	//system("exec rm -r /fields/*");
	//cout << "fields cleaned" << endl;
	//}


	//the main class for geometry
	multi_cross Geom;

	hy_h = 1.0 / ny_h;	hx_h = hy_h;
	tt = (unsigned int)round(1.0 / tau_h);
	cosA_h = cos(alpha_h*pi / 180);
	sinA_h = sin(alpha_h*pi / 180);
	tau_p_h = 0.20*hx_h*hx_h;
	Ek = 0; Ek_old = 0;
	kk = 1000000; //Poisson iteration limit 
	Phi_kk = 1000000;
	//geometry
	{
		if (geometry == "matrix") {
			Geom.set_global_size(nx_h, ny_h, Matrix_X, Matrix_Y);
			//Geom.set_global_size_narrow_tubes(2*nx_h, nx_h, ny_h/2, Matrix_X, Matrix_Y);
			Geom.set_type();
			//Geom.left_normal_in((Matrix_Y - 1) / 2, (Matrix_Y - 1) / 2);
			//Geom.left_normal_out((Matrix_Y - 1) / 2, (Matrix_Y - 1) / 2);
			Geom.set_neighbor();
			Geom.set_global_id();
			cout << "Matrix_X = " << Matrix_X << ", Matrix_Y = " << Matrix_Y << endl;
			Geom.check();
		}

		else if (geometry == "matrix2") {
			Geom.set_global_size(nx_h, ny_h, Matrix_X, Matrix_Y);
			Geom.set_global_id_B();
			Geom.set_neighbor_B();
			Geom.set_type_B();
			Geom.check();
			cout << "Matrix_X = " << Matrix_X << ", Matrix_Y = " << Matrix_Y << endl;
		}

		else if (geometry == "box") {
			if (Ly_h != 0) hy_h = Ly_h / ny_h;
			if (Lx_h != 0) hx_h = Lx_h / nx_h;

			Geom.set_global_size_box(nx_h, ny_h);
			Geom.set_type_box();
			Geom.set_neighbor_box();
			Geom.set_global_id_box();
			Geom.check();
			simple_geometry = 1;
		}
		else if (geometry == "tube") {
			Geom.set_global_size_box(nx_h, ny_h);
			Geom.set_type_tube();
			Geom.set_neighbor_box();
			Geom.set_global_id_box();
			Geom.check();
			simple_geometry = 1;
		}
		else if (geometry == "grid") {
			Geom.read_grid_geometry();
			File.reading<double>(hy_h, "h_step", 0.001);
			hx_h = hy_h;
			tau_p_h = 0.20 * hx_h * hx_h;
			nx_h = Geom.nx;
			ny_h = Geom.ny;
			offset_h = Geom.offset; //sure you need all this?
			Geom.check();
		}



		else {
			cout << "what are you trying to do?" << endl;
			return 0;
		}
	}




	/*
	box_inherited Geom;
	Geom.set_global_size(nx_h, ny_h);
	Geom.set_type();
	Geom.set_neighbor();
	Geom.set_global_id();
	cout << "SIZE = " << " " << Geom.TOTAL_SIZE << endl;
	*/

	//alternative geometry
	/*
	multi_line Geom;
	Geom.generate_levels(30, 30, 30, 3, 5);
	cout << "approximate memory amount = " << 100 * Geom.TOTAL_SIZE / 1024 / 1024 << " MB" << endl << endl << endl;
	Geom.set_neighbor();
	Geom.set_type();
	pause
	*/




	//int sss = 0;	for (int i = 0; i < Geom.TOTAL_SIZE; i++) if (Geom.t[i] == 9) sss++;	cout << "S=" << sss << endl; 




	//here we copy the arrays responsible for the geometry to GPU
	stupid_step(Geom.n1, Geom.n2, Geom.n3, Geom.n4, Geom.t, Geom.J_back, Geom.TOTAL_SIZE);
	cudaCheckError()

		//total Length and Width of the porous matrix
		Lx_h = hx_h * (Geom.nxg);
	Ly_h = hy_h * (Geom.nyg);

	cudaDeviceSynchronize();


	//size setting
	//you may just skip it, that is weird
	offset_h = nx_h + 1;
	unsigned int size_l = Geom.TOTAL_SIZE; //Number of all nodes/elements 

	if (size_l <= 1024 || size_l >= 1024 * 1024 * 1024) { cout << "data is too small or too large" << endl; return 0; }
	std::cout << "size_l=" << size_l << endl;
	size_t size_b /*size (in) bytes*/ = size_l * sizeof(double); //sizeof(double) = 8 bytes
	size_t thread_x_d /*the dimension of x in a block*/ = 1024;
	//size_t threads_per_block = thread_x_d;

	dim3 gridD((unsigned int)ceil((size_l + 0.0) / thread_x_d));
	dim3 blockD((unsigned int)thread_x_d);
	std::cout << "gridD.x=" << gridD.x << endl;
	std::cout << "blockD.x=" << blockD.x << endl;

	//setting for the reduction procedure 
	//that is even weirder, skip it, don't hesitate 
	unsigned long long int *Gp, *Np;
	unsigned int s = 0;



	unsigned int GN = size_l;
	while (true)
	{
		s++;
		GN = (unsigned int)ceil(GN / (thread_x_d + 0.0));
		if (GN == 1)  break;
	}
	GN = size_l;
	std::cout << "the number of reduction = " << s << endl;
	Gp = new unsigned long long int[s];
	Np = new unsigned long long int[s];
	for (unsigned int i = 0; i < s; i++)
		Gp[i] = GN = (unsigned int)ceil(GN / (thread_x_d + 0.0));
	Np[0] = size_l;
	for (unsigned int i = 1; i < s; i++)
		Np[i] = Gp[i - 1];
	int last_reduce = (int)pow(2, ceil(log2(Np[s - 1] + 0.0)));
	std::cout << "last reduction = " << last_reduce << endl;
	(s != 1) ? std::cout << "sub array for the Poisson solver = " << Np[1] << endl :
		std::cout << "it shouldn't be here" << endl;
	double *arr[10];
	double *arr2[10];

	//allocating memory for arrays on CPU and initializing them 
	{
		if (DIFFUSION_h == 1) {
			C_h = (double*)malloc(size_b); 		mu_h = (double*)malloc(size_b);
			p_true_h = (double*)malloc(size_b); zero_h = (double*)malloc(size_b);
			p_h = vx_h = vy_h = zero_h;
			for (unsigned int l = 0; l < size_l; l++) { C_h[l] = 0.5; mu_h[l] = 0; p_true_h[l] = 0.0; zero_h[l] = 0.0; }
		}
		else {
			C_h = (double*)malloc(size_b); 		mu_h = (double*)malloc(size_b);
			p_h = (double*)malloc(size_b);		p_true_h = (double*)malloc(size_b);
			vx_h = (double*)malloc(size_b);		vy_h = (double*)malloc(size_b);
			//  psiav_h = (double*)malloc(sizeof(double)); 
			//	psiav_array_h = (double*)malloc(size_b / threads_per_block);
			for (unsigned int l = 0; l < size_l; l++) { C_h[l] = 0.5; mu_h[l] = 0; p_h[l] = 0.0; p_true_h[l] = 0.0; vx_h[l] = 0.0; vy_h[l] = 0.0; }
		}
		if (curv_calc == 1) {
			curv1 = (double*)malloc(size_b);
			curv2 = (double*)malloc(size_b);
			nx_dC = (double*)malloc(size_b);
			ny_dC = (double*)malloc(size_b);
			for (unsigned int l = 0; l < size_l; l++) {
				curv1[l] = 0.0;
				curv2[l] = 0.0;
				nx_dC[l] = 0.0;
				ny_dC[l] = 0.0;
			}
		}
		if (vibration == 1) {
			Phi_h = (double*)malloc(size_b);
			WX_h = (double*)malloc(size_b);
			WY_h = (double*)malloc(size_b);
			for (unsigned int l = 0; l < size_l; l++) {
				Phi_h[l] = 0.0;
				WX_h[l] = 0.0;
				WY_h[l] = 0.0;
			}
		}
	}


	if (linear_pressure == 1) {
		Geom.linear_pressure(p_h, hx_h, hy_h, cosA_h, sinA_h, Lx_h, Ly_h, 8.0 / Re_h);
	}
	if (filling == "shift") {
		double delta = sqrt(Ca_h / 0.5);
		Geom.fill_gradually(C_h, hx_h, hy_h, delta, fill_gradually_x);
	}
	if (filling == "sphere") {
		double sphere_x0, sphere_y0, sphere_R0;
		double C_outer, C_inner;
		File.reading<double>(sphere_x0, "sphere_x0", Lx_h * 0.5);
		File.reading<double>(sphere_y0, "sphere_y0", Ly_h * 0.5);
		File.reading<double>(sphere_R0, "sphere_R0", 0.1);
		File.reading<double>(C_outer, "sphere_C_outer", +sqrt(abs(-A_h) / 2.0));
		File.reading<double>(C_inner, "sphere_C_inner", -sqrt(abs(-A_h) / 2.0));

		Geom.fill_with_sphere(C_h, hx_h, hy_h, sphere_x0, sphere_y0, sphere_R0, C_outer, C_inner);
	}
	if (filling == "horizontal") {
		double delta = sqrt(Ca_h / 0.5);
		double horizontal_amp; File.reading<double>(horizontal_amp, "horizontal_amp", 0);
		Geom.fill_horizontal_way(C_h, hx_h, hy_h, 0.5, Ly_h*0.5, horizontal_amp, 2.0*Pi / Lx_h, delta);

	}




	//additional allocation on CPU for statistics if necessary // при какой еще необходимости!?
	double *fx, *fy; signed char *mark;
	{
		fx = (double*)malloc(sizeof(double)*size_l);
		fy = (double*)malloc(sizeof(double)*size_l);
		mark = (signed char*)malloc(sizeof(signed char)*size_l);
	}




	//allocating memory for arrays on GPU
	{
		if (DIFFUSION_h == 1) {
			cudaMalloc((void**)&C, size_b); 	cudaMalloc((void**)&C0, size_b);
			cudaMalloc((void**)&mu, size_b);    cudaMalloc((void**)&zero, size_b);
			p = p0 = ux = uy = vx = vy = zero;
		}
		else {
			cudaMalloc((void**)&C, size_b); 	cudaMalloc((void**)&C0, size_b);
			cudaMalloc((void**)&p, size_b); 	cudaMalloc((void**)&p0, size_b);
			cudaMalloc((void**)&ux, size_b);	cudaMalloc((void**)&uy, size_b);
			cudaMalloc((void**)&vx, size_b);	cudaMalloc((void**)&vy, size_b);
			cudaMalloc((void**)&mu, size_b);
			(s != 1) ? cudaMalloc((void**)&psiav_array, sizeof(double)*Np[1]) : cudaMalloc((void**)&psiav_array, sizeof(double));
		}
		if (vibration == 1) {
			cudaMalloc((void**)&Phi, size_b);
			cudaMalloc((void**)&Phi0, size_b);
			cudaMalloc((void**)&WX, size_b);
			cudaMalloc((void**)&WY, size_b);
			(s != 1) ? cudaMalloc((void**)&psiav_array_Phi, sizeof(double)*Np[1]) : cudaMalloc((void**)&psiav_array_Phi, sizeof(double));
		}
	}

	//for Poisson procedure shortness 
	arr[0] = p;
	for (unsigned int i = 1; i <= s; i++)
		arr[i] = psiav_array;

	if (vibration == 1) {
		arr2[0] = Phi;
		for (unsigned int i = 1; i <= s; i++)
			arr2[i] = psiav_array_Phi;
	}




	//ofstream is a class to write data in a file, ifstream is a class to read data from a file
	ofstream integrals;
	ofstream test_output; int test_output_switch;
	File.reading<int>(test_output_switch, "test_output", 0, 0, 1);
	if (test_output_switch) test_output.open("test_output.dat");
	ofstream k_number;
	ifstream read;
	read.open("recovery.dat");

	//checking whether a recovery file exists or not
	//if not we start at t = 0, otherwise we continue from the saved data
	bool file_exists = read.good();
	if (read_switch == 0) file_exists = false;

	if (file_exists == true) {
		read_switch = 1;
		std::cout << endl << "CONTINUE" << endl;
	}
	else {
		read_switch = 0;
		iter = 0;
		std::cout << endl << "from the Start" << endl;
		if (read_C == 1) {
			std::cout << "initial concentration reading" << endl;
			Geom.read_concentration(C_h, "recovery.dat", 4, 1, 1);
		}
	}

	//continue
	if (read_switch == 1) {


		//Geom.recover(vx_h, vy_h, p_h, C_h, mu_h, iter, write_i, timeq, kk);

		double* var[10];
		unsigned int n = 0;
		var[n] = vx_h; n++;
		var[n] = vy_h; n++;
		var[n] = p_h; n++;
		var[n] = C_h; n++;
		var[n] = mu_h; n++;
		if (vibration == 1) { var[n] = Phi_h; n++; }
		Geom.recover(var, n, iter, write_i, timeq, kk);

		if (reset_timeq == 0) {
			integrals.open("integrals.dat", std::ofstream::app);
			cout << "from time: " << timeq << " iter:" << iter << endl;
		}
		else if (reset_timeq == 1)
		{
			cout << "reset time" << endl;
			integrals.open("integrals.dat");
			iter = 0;
			write_i = 0;
			timeq = 0;
			if (invert_initial_C == 1)
				for (unsigned int l = 0; l < size_l; l++)
					C_h[l] = C_h[l] * (-1);
			if (reset_velocity == 1)
				for (unsigned int l = 0; l < size_l; l++) {
					vx_h[l] = 0.0;
					vy_h[l] = 0.0;
				}
			if (reset_pressure == 1)
				for (unsigned int l = 0; l < size_l; l++)
					p_h[l] = 0.0;
		}

	}

	//from the start
	if (read_switch == 0)
		integrals.open("integrals.dat");



	//copying values from host variables to device ones
	{
		cudaMemcpy(C, C_h, size_b, cudaMemcpyHostToDevice); 	cudaMemcpy(C0, C_h, size_b, cudaMemcpyHostToDevice);
		cudaMemcpy(p0, p_h, size_b, cudaMemcpyHostToDevice); 	cudaMemcpy(p, p_h, size_b, cudaMemcpyHostToDevice);
		cudaMemcpy(ux, vx_h, size_b, cudaMemcpyHostToDevice); 	cudaMemcpy(uy, vy_h, size_b, cudaMemcpyHostToDevice);
		cudaMemcpy(vx, vx_h, size_b, cudaMemcpyHostToDevice); 	cudaMemcpy(vy, vy_h, size_b, cudaMemcpyHostToDevice);
		cudaMemcpy(mu, mu_h, size_b, cudaMemcpyHostToDevice);
		if (vibration == 1) {
			cudaMemcpy(Phi, Phi_h, size_b, cudaMemcpyHostToDevice);
			cudaMemcpy(Phi0, Phi_h, size_b, cudaMemcpyHostToDevice);
		}
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
		cudaMemcpyToSymbol(Gs, &Gs_h, sizeof(double), 0, cudaMemcpyHostToDevice);
		cudaMemcpyToSymbol(Pe, &Pe_h, sizeof(double), 0, cudaMemcpyHostToDevice);
		cudaMemcpyToSymbol(Re, &Re_h, sizeof(double), 0, cudaMemcpyHostToDevice);
		cudaMemcpyToSymbol(MM, &MM_h, sizeof(double), 0, cudaMemcpyHostToDevice);

		cudaMemcpyToSymbol(tau, &tau_h, sizeof(double), 0, cudaMemcpyHostToDevice);
		cudaMemcpyToSymbol(tau_p, &tau_p_h, sizeof(double), 0, cudaMemcpyHostToDevice);
		cudaMemcpyToSymbol(alpha, &alpha_h, sizeof(double), 0, cudaMemcpyHostToDevice);
		cudaMemcpyToSymbol(sinA, &sinA_h, sizeof(double), 0, cudaMemcpyHostToDevice);
		cudaMemcpyToSymbol(cosA, &cosA_h, sizeof(double), 0, cudaMemcpyHostToDevice);
		cudaMemcpyToSymbol(theta, &theta_h, sizeof(double), 0, cudaMemcpyHostToDevice);
		cudaMemcpyToSymbol(sinTh, &sinTh_h, sizeof(double), 0, cudaMemcpyHostToDevice);
		cudaMemcpyToSymbol(cosTh, &cosTh_h, sizeof(double), 0, cudaMemcpyHostToDevice);
		cudaMemcpyToSymbol(OFFSET, &Geom.OFFSET, sizeof(int), 0, cudaMemcpyHostToDevice);
		//		cudaMemcpyToSymbol(Mx, &Geom.Mx, sizeof(int), 0, cudaMemcpyHostToDevice);
		//		cudaMemcpyToSymbol(My, &Geom.My, sizeof(int), 0, cudaMemcpyHostToDevice);
		//		cudaMemcpyToSymbol(Msize, &Geom.Msize, sizeof(int), 0, cudaMemcpyHostToDevice);
		//		cudaMemcpyToSymbol(Moffset, &Geom.Moffset, sizeof(int), 0, cudaMemcpyHostToDevice);
		cudaMemcpyToSymbol(TOTAL_SIZE, &Geom.TOTAL_SIZE, sizeof(int), 0, cudaMemcpyHostToDevice);
		cudaMemcpyToSymbol(PHASE, &PHASE_h, sizeof(unsigned int), 0, cudaMemcpyHostToDevice);
		cudaMemcpyToSymbol(dP, &dP_h, sizeof(double), 0, cudaMemcpyHostToDevice);
		cudaMemcpyToSymbol(Amp, &Amp_h, sizeof(double), 0, cudaMemcpyHostToDevice);
		cudaMemcpyToSymbol(Omega, &Omega_h, sizeof(double), 0, cudaMemcpyHostToDevice);
		cudaMemcpyToSymbol(VV, &VV_h, sizeof(double), 0, cudaMemcpyHostToDevice);
		cudaMemcpyToSymbol(vibr_X, &vibr_X_h, sizeof(double), 0, cudaMemcpyHostToDevice);
		cudaMemcpyToSymbol(vibr_Y, &vibr_Y_h, sizeof(double), 0, cudaMemcpyHostToDevice);
	}




	{
		cout << "approximate memory amount = " << 100 * Geom.TOTAL_SIZE / 1024 / 1024 << " MB" << endl;
		PrintVar(wetting)
			PrintVar(DIFFUSION_h)
			PrintVar(geometry)
			PrintVar(filling)
	}



	//just printing parameters from GPU to be confident they are passed correctly 
	hello << <1, 1 >> > ();
	cudaDeviceSynchronize();








	//Geom.fast_test_writing(C_h);




	Geom.write_field(C_h, "0", timeq, each);






	//measure real time of calculating 
	timer1 = clock() / CLOCKS_PER_SEC;


	//Geom.write_field(C_h, "test", 0, 1);

	//write the file with parameters
	//this step was written for making movies
	{
#ifdef __linux__ 
		ofstream to_file("fields/param.dat");
#endif
#ifdef _WIN32
		ofstream to_file("fields\\param.dat");
#endif
#define space << " " << 
		to_file << Geom.nxg / each space Geom.nyg / each space hx_h*each space hy_h*each space Lx_h space Ly_h
			space Gr_h space Ca_h space Pe_h space Re_h space A_h space MM_h space alpha_h
			<< endl;
		to_file.close();
	}







	true_pressure(p_h, p_true_h, C_h, mu_h, Geom.t, Geom.n1, Geom.n2, Geom.n3, Geom.n4, Geom.J_back, tau_h, Geom.TOTAL_SIZE, hx_h, hy_h, Ca_h, A_h, Gr_h, MM_h, Geom.OFFSET, sinA_h, cosA_h, PHASE_h, VV_h, vibr_X_h, vibr_Y_h);







	//pause
		// the main time loop of the whole calculation procedure
		while (true) {

			iter = iter + 1; 	timeq = timeq + tau_h;

			//Poisson equation for pulsation potential
			if (vibration == 1)
			{

				double eps_Phi = 1.0;
				double psiav0_Phi = -1.0;
				double psiav_Phi = 0.0;
				unsigned int k_Phi = 0;
				//while ((eps_Phi > eps0*psiav0_Phi && k_Phi < Phi_kk))
				while ((eps_Phi > eps0*psiav0_Phi))
				{

					psiav_Phi = 0.0;  k_Phi++;
					Poisson_Phi << <gridD, blockD >> >(Phi, Phi0, C0, WX, WY);


					for (unsigned int i = 0; i < s; i++)
						reduction00 << < Gp[i], 1024, 1024 * sizeof(double) >> > (arr2[i], Np[i], arr2[i + 1]);
					swap_one << <gridD, blockD >> > (Phi0, Phi);
					cudaMemcpy(&psiav_Phi, psiav_array_Phi, sizeof(double), cudaMemcpyDeviceToHost);

					eps_Phi = abs(psiav_Phi - psiav0_Phi); 	psiav0_Phi = psiav_Phi;

					if (k_Phi % 1000 == 0) {
						cout << "Phi_iter=" << k_Phi << " " << eps_Phi << endl;
					}
				}
				Phi_kk = k_Phi;

				if (iter % (int)(tt *time_display) == 0 || iter == 1) {
					cout << "Phi_iter=" << Phi_kk << endl;
				}
				Phi_normalization << < gridD, blockD >> > (Phi);
				WW_from_Phi << < gridD, blockD >> > (WX, WY, Phi, C0);
			}


			if (DIFFUSION_h == 1) { //only diffusion

				if (PHASE_h == 1) {
					chemical_potential << <gridD, blockD >> > (mu, C);
					concentration << < gridD, blockD >> > (C, C0, vx, vy, mu);
					//concentration_surface_energy_wetting << < gridD, blockD >> > (C, C0, vx, vy, mu);
				}
				else if (PHASE_h == 0) {

					concentration << < gridD, blockD >> > (C, C0, vx, vy, C0);

				}

				swap_one << <gridD, blockD >> > (C0, C);

			}
			else { //flow

				   //1st step, calculating of time evolutionary parts of velocity (quasi-velocity) and concentration and chemical potential
				{
					if (PHASE_h == 1) {
						chemical_potential << <gridD, blockD >> > (mu, C); 
						
						//chemical_potential_inside << <gridD, blockD >> > (mu, C);
						//chemical_potential_border << <gridD, blockD >> > (mu, C);
						//quasi_velocity_upstream << < gridD, blockD >> > (ux, uy, vx, vy, C0, mu);

						switch (vibration)
						{
						case 0: //as it is
							quasi_velocity << < gridD, blockD >> > (ux, uy, vx, vy, C0, mu);
							break;
						case 1: //const initial concentration at walls, which is not washed out
							quasi_velocity_pulsation_with_Phi << < gridD, blockD >> > (ux, uy, vx, vy, C0, mu, timeq, Phi, WX, WY);
							break;
						case 2: //ongoing concentration devours initial one
							quasi_velocity_pulsation << < gridD, blockD >> > (ux, uy, vx, vy, C0, mu, timeq);
							break;
						case 3: //surface energy formulation by Jacqmin // not finished 
							quasi_velocity << < gridD, blockD >> > (ux, uy, vx, vy, C0, mu);
							break;
						default:
							break;
						}

						switch (wetting)
						{
						case 0: //as it is
							concentration << < gridD, blockD >> > (C, C0, vx, vy, mu);
							//concentration_upstream << < gridD, blockD >> > (C, C0, vx, vy, mu);
							break;
						case 1: //const initial concentration at walls, which is not washed out
							concentration_no_wetting << < gridD, blockD >> > (C, C0, vx, vy, mu);
							break;
						case 2: //ongoing concentration devours initial one
							concentration_wetting << < gridD, blockD >> > (C, C0, vx, vy, mu);
							break;
						case 3: //surface energy formulation by Jacqmin // not finished 
							concentration_surface_energy_wetting << < gridD, blockD >> > (C, C0, vx, vy, mu);
							break;
						default:
							break;
						}
					}
					else if (PHASE_h == 0) {
						chemical_potential_Gr << <gridD, blockD >> > (mu);

						switch (vibration)
						{
						case 0: //as it is
							quasi_velocity << < gridD, blockD >> > (ux, uy, vx, vy, C0, mu);
							//quasi_velocity_no_phase_field << < gridD, blockD >> > (ux, uy, vx, vy, C0, mu);
							break;
						case 1: //const initial concentration at walls, which is not washed out
							quasi_velocity_pulsation_with_Phi << < gridD, blockD >> > (ux, uy, vx, vy, C0, mu, timeq, Phi, WX, WY);
							break;
						case 2: //ongoing concentration devours initial one
							quasi_velocity_pulsation << < gridD, blockD >> > (ux, uy, vx, vy, C0, mu, timeq);
							break;
						case 3: //surface energy formulation by Jacqmin // not finished 
							quasi_velocity << < gridD, blockD >> > (ux, uy, vx, vy, C0, mu);
							break;
						default:
							break;
						}

						//if (timeq < 1)
						concentration << < gridD, blockD >> > (C, C0, vx, vy, C0);
						//else concentration_no_input_C << < gridD, blockD >> > (C, C0, vx, vy, C0);
					}
				}





				//2nd step, Poisson equation for pressure 
				{
					eps = 1.0; 		psiav0 = -1.0;		psiav = 0.0;		k = 0;
					//while (eps > eps0*psiav0 || k < 10)
					//while (eps > eps0*psiav0 )
					while ((eps > eps0*psiav0 && k < kk))
					{

						psiav = 0.0;  k++;
						if (vibration == 1) Poisson_pulsation_Phi << <gridD, blockD >> > (p, p0, ux, uy, mu, C, Phi, WX, WY);
						else if (vibration == 3) Poisson_pulsation << <gridD, blockD >> >(p, p0, ux, uy, mu, C, timeq);
						else Poisson << <gridD, blockD >> > (p, p0, ux, uy, mu, C);




						for (unsigned int i = 0; i < s; i++)
							reduction00 << < Gp[i], 1024, 1024 * sizeof(double) >> > (arr[i], Np[i], arr[i + 1]);
						swap_one << <gridD, blockD >> > (p0, p);
						cudaMemcpy(&psiav, psiav_array, sizeof(double), cudaMemcpyDeviceToHost);

						eps = abs(psiav - psiav0); 	psiav0 = psiav;

						if (k % 1000 == 0) {
							cout << "p_iter=" << k << endl;
						}
					}


				}
				kk = k;

				if (iter % (int)(tt *time_display) == 0 || iter == 1) {
					cout << "p_iter=" << k << endl;
				}


				//3rd step, velocity correction and swapping field values
				velocity_correction << <gridD, blockD >> > (vx, vy, ux, uy, p);

				swap_3 << <gridD, blockD >> > (ux, vx, uy, vy, C0, C);


			}

			//4th step, printing results, writing data and whatever you want
			if (iter % (int)(tt *time_display) == 0 || iter == 1) {
				cout << setprecision(15) << endl;
				cout << fixed << endl;
				cudaMemcpy(vx_h, vx, size_b, cudaMemcpyDeviceToHost);
				cudaMemcpy(vy_h, vy, size_b, cudaMemcpyDeviceToHost);
				cudaMemcpy(p_h, p, size_b, cudaMemcpyDeviceToHost);
				cudaMemcpy(C_h, C, size_b, cudaMemcpyDeviceToHost);
				cudaMemcpy(mu_h, mu, size_b, cudaMemcpyDeviceToHost);
				copied = true;

				true_pressure(p_h, p_true_h, C_h, mu_h, Geom.t, Geom.n1, Geom.n2, Geom.n3, Geom.n4, Geom.J_back,
					tau_h, Geom.TOTAL_SIZE, hx_h, hy_h, Ca_h, A_h, Gr_h, MM_h, Geom.OFFSET, sinA_h, cosA_h, PHASE_h, VV_h, vibr_X_h, vibr_Y_h);

				double len, ten, vol, width, p_plusAv, p_minusAv, p_Av, vx_plusAv, vx_minusAv, vx_Av;
				velocity(size_l, hx_h, hy_h, vx_h, vy_h, Ek, Vmax);
				VFR(vx_h, Geom.t, size_l, hy_h, Q_in, Q_out, C_h, C_average, Cv);
				C_statistics(Geom.TOTAL_SIZE, hx_h, hy_h, Geom.t, C_h, C_av, C_plus, C_minus);
				len = Geom.isoline(hx_h, hy_h, C_h, mark, fx, fy, 0.0);
				ten = Ca_h / len / MM_h * Geom.tension(hx_h, hy_h, C_h);   if (!std::isfinite(ten)) ten = 0; //if (ten != ten) ten = 0;
				vol = Geom.volume(hx_h, hy_h, C_h, 0.2);
				width = vol / len;	if (!std::isfinite(width)) width = 0;  //if (abs(width) > 100000) width = 0;	
				Geom.X_averaged_in_each_phase(hx_h, hy_h, C_h, p_true_h, p_plusAv, p_minusAv, p_Av, 0.05);
				Geom.X_averaged_in_each_phase(hx_h, hy_h, C_h, vx_h, vx_plusAv, vx_minusAv, vx_Av);


				//Display
				timer
					cout << "t= " << tau_h*iter << endl;
				cout << "Vmax= " << Vmax << endl;
				cout << "Ek= " << Ek << endl;
				cout << "dEk= " << (Ek - Ek_old) << endl;
				cout << "p_iter=" << k << endl;
				cout << "Q_in=" << Q_in << endl;
				cout << "Q_out=" << Q_out << endl;
				cout << "Vx_max=" << maxval(vx_h, size_l) << endl;
				cout << "C_max=" << maxval(C_h, size_l) << endl;
				cout << "p_max=" << maxval(p_h, size_l) << endl;
				cout << "C_av=" << C_av << endl;
				cout << "C_plus=" << C_plus << endl;
				cout << "C_minus=" << C_minus << endl;



				//Integrals
				if (iter == 1) {
					integrals << "t, Ek, Vmax,  time(min), dEk, Q_in, Q_out, C_average, Q_per_cap, Q_per_width"
						<< ", Cv_per_cap, Cv_per_width, C_av, C_plus, C_minus, L, ten, width"
						<< ", p_plusAv, p_minusAv, vx_plusAv, vx_minusAv";
					if (integrals_add1) {
						integrals << ", Xtip, Xwall, Qtip, Qwall, Cap_pres";
					}
					integrals << endl;
				}
				integrals << setprecision(20) << fixed;
				integrals << timeq << " " << Ek << " " << Vmax << " " << (timer2 - timer1) / 60
					<< " " << abs(Ek - Ek_old) << " " << Q_in << " " << Q_out << " " << C_average / Matrix_Y << " " << Q_out / Matrix_Y << " " << Q_out / Ly_h
					<< " " << Cv / Matrix_Y << " " << Cv / Ly_h
					<< " " << C_av << " " << C_plus << " " << C_minus
					<< " " << len << " " << ten << " " << width
					<< " " << p_plusAv << " " << p_minusAv << " " << vx_plusAv << " " << vx_minusAv;
				if (integrals_add1) {
					double x_tip = Geom.change_sign_at_X(hx_h, hy_h, C_h, Geom.nyg / 2);
					double x_wall = Geom.change_sign_at_X(hx_h, hy_h, C_h, 0);
					double cap_pres = Geom.pressure_jump(hx_h, hy_h, p_true_h, x_tip, 0.1);
					double Q_tip = Geom.flow_rate(hx_h, hy_h, vx_h, Ly_h, (unsigned int)(x_tip / hx_h));
					double Q_wall = Geom.flow_rate(hx_h, hy_h, vx_h, Ly_h, (unsigned int)(x_wall / hx_h));

					integrals << " " << x_tip << " " << x_wall << " " << Q_tip << " " << Q_wall << " " << cap_pres;
					cout << "x_tip=" << x_tip << endl;
				}

				integrals << endl;

				Ek_old = Ek;

				if (stop_at_exit == 1) {
					stop = Geom.checkExit(C_h);
					if (stop == 1)
						cout << "stop command is applied" << endl;
				}
			}


			//fields writing
			if (iter % (int(time_fields * tt)) == 0 || iter == 1 || stop == 1)
			{
				if (copied == false) {
					cudaMemcpy(vx_h, vx, size_b, cudaMemcpyDeviceToHost);
					cudaMemcpy(vy_h, vy, size_b, cudaMemcpyDeviceToHost);
					cudaMemcpy(p_h, p, size_b, cudaMemcpyDeviceToHost);
					cudaMemcpy(C_h, C, size_b, cudaMemcpyDeviceToHost);
					cudaMemcpy(mu_h, mu, size_b, cudaMemcpyDeviceToHost);
					true_pressure(p_h, p_true_h, C_h, mu_h, Geom.t, Geom.n1, Geom.n2, Geom.n3, Geom.n4, Geom.J_back,
						tau_h, Geom.TOTAL_SIZE, hx_h, hy_h, Ca_h, A_h, Gr_h, MM_h, Geom.OFFSET, sinA_h, cosA_h, PHASE_h, VV_h, vibr_X_h, vibr_Y_h);
					copied = true;
				}
				write_i++;
				stringstream ss; string file_name;	ss.str(""); ss.clear();
				ss << write_i;		file_name = ss.str();
				Geom.write_field(C_h, file_name, timeq, each);
				Geom.write_field(p_true_h, "true_p_" + file_name, timeq, each);
				Geom.write_field(mu_h, "mu_" + file_name, timeq, each);

				if (curv_calc) {
					Geom.curvature_direct(C_h, hx_h, hy_h, curv1, 0.1);
					Geom.curvature_direct(C_h, hx_h, hy_h, curv2, 0.001);
					//Geom.curvature_2_steps(C_h, nx_dC, ny_dC, hx_h, hy_h, curv2);
					Geom.write_field(curv1, "curv_" + file_name, timeq, each);
					Geom.write_field(curv2, "curv2_" + file_name, timeq, each);
				}

				if (vibration == 1) {


					WW_from_Phi << <gridD, blockD >> > (WX, WY, Phi, C);
					cudaMemcpy(Phi_h, Phi, size_b, cudaMemcpyDeviceToHost);
					cudaMemcpy(WX_h, WX, size_b, cudaMemcpyDeviceToHost);
					cudaMemcpy(WY_h, WY, size_b, cudaMemcpyDeviceToHost);
					Geom.write_field(Phi_h, "Phi_" + file_name, timeq, each);
					Geom.write_field(WX_h, "WX_" + file_name, timeq, each);
					Geom.write_field(WY_h, "WY_" + file_name, timeq, each);

					if (test_output_switch) test_output << timeq << " " << MAXval(Phi_h, size_l) << " " << MINval(Phi_h, size_l) << " " << Phi_h[0 + Geom.OFFSET*Geom.nyg] - Phi_h[Geom.nxg + Geom.OFFSET*Geom.nyg] << endl;
				}


				if (horizontal_profile) {
					double *var[20];
					string head = "i x C P_true P Mu vx vy";
					var[0] = C_h; var[1] = p_true_h; var[2] = p_h; var[3] = mu_h;
					var[4] = vx_h; var[5] = vy_h;
					int n = 6;
					if (curv_calc) {
						head.append(" curv1 curv2");
						var[n] = curv1; n++;
						var[n] = curv2; n++;
					}
					if (vibration == 1) {
						head.append(" Phi");
						head.append(" WX");
						head.append(" WY");
						var[n] = Phi_h; n++;
						var[n] = WX_h; n++;
						var[n] = WY_h; n++;
					}
					Geom.write_linear_profile(file_name, head, timeq, 1, hx_h, var, n);
					Geom.write_linear_profile(file_name + "_bot", head, timeq, 1, hx_h, var, n, 0);
					Geom.write_linear_profile(file_name + "_top", head, timeq, 1, hx_h, var, n, Geom.nyg);


				}
				if (vertical_profile && integrals_add1) {
					double *var[20];
					string head = "j y C P_true P Mu vx vy";
					var[0] = C_h; var[1] = p_true_h; var[2] = p_h; var[3] = mu_h;
					var[4] = vx_h; var[5] = vy_h;
					int n = 6;
					double x_tip = Geom.change_sign_at_X(hx_h, hy_h, C_h, Geom.nyg / 2);
					double x_wall = Geom.change_sign_at_X(hx_h, hy_h, C_h, 0);
					if (vibration == 1) {
						head.append(" Phi");
						head.append(" WX");
						head.append(" WY");
						var[n] = Phi_h; n++;
						var[n] = WX_h; n++;
						var[n] = WY_h; n++;
					}
					Geom.write_section_profile(file_name + "_tip", head, timeq, 1, hy_h, var, n, (unsigned int)(x_tip / hx_h));
					Geom.write_section_profile(file_name + "_wall", head, timeq, 1, hy_h, var, n, (unsigned int)(x_wall / hx_h));
					Geom.write_section_profile(file_name + "_end", head, timeq, 1, hy_h, var, n, Geom.nxg - 1);
					Geom.write_section_profile(file_name + "_start", head, timeq, 1, hy_h, var, n, 0);
					//Geom.write_section_profile(file_name + "_j10", head, timeq, 1, hy_h, var, n, 10);
				}


				//Geom.write_field(mu_h, "mu" + file_name, timeq, each);
			}

			//fields writting for *** tecplot
			if (tecplot != 0 && (iter % (int(time_fields * tt)) == 0 || iter == 1 || stop == 1))
			{
				if (copied == false) {
					cudaMemcpy(vx_h, vx, size_b, cudaMemcpyDeviceToHost);
					cudaMemcpy(vy_h, vy, size_b, cudaMemcpyDeviceToHost);
					cudaMemcpy(p_h, p, size_b, cudaMemcpyDeviceToHost);
					cudaMemcpy(C_h, C, size_b, cudaMemcpyDeviceToHost);
					cudaMemcpy(mu_h, mu, size_b, cudaMemcpyDeviceToHost);
					true_pressure(p_h, p_true_h, C_h, mu_h, Geom.t, Geom.n1, Geom.n2, Geom.n3, Geom.n4, Geom.J_back,
						tau_h, Geom.TOTAL_SIZE, hx_h, hy_h, Ca_h, A_h, Gr_h, MM_h, Geom.OFFSET, sinA_h, cosA_h, PHASE_h, VV_h, vibr_X_h, vibr_Y_h);
					copied = true;
				}
				double *var[10];
				int n = 0;
				string head = "VARIABLES=\"x\",\"y\",\"C\",\"p\",\"mu\",\"vx\",\"vy\"";
				var[n] = C_h; n++;
				var[n] = p_true_h; n++;
				var[n] = mu_h; n++;
				var[n] = vx_h; n++;
				var[n] = vy_h; n++;
				if (vibration == 1) {
					head.append(",\"WX\",\"WY\",\"Phi\"");
					var[n] = WX_h; n++;
					var[n] = WY_h; n++;
					var[n] = Phi_h; n++;
				}
				Geom.write_field_tecplot(tecplot, hx_h, hy_h, "fields", timeq, each, iter, var, n, head);
			}



			//recovery fields writing
			if (iter % (int)(tt*time_recovery) == 0 || timeq > limit_timeq || stop == 1)
			{
				if (copied == false) {
					cudaMemcpy(vx_h, vx, size_b, cudaMemcpyDeviceToHost);
					cudaMemcpy(vy_h, vy, size_b, cudaMemcpyDeviceToHost);
					cudaMemcpy(p_h, p, size_b, cudaMemcpyDeviceToHost);
					cudaMemcpy(C_h, C, size_b, cudaMemcpyDeviceToHost);
					cudaMemcpy(mu_h, mu, size_b, cudaMemcpyDeviceToHost);
					if (vibration == 1) cudaMemcpy(Phi_h, Phi, size_b, cudaMemcpyDeviceToHost);
					copied = true;
				}
				//Geom.save(vx_h, vy_h, p_h, C_h, mu_h, iter, write_i, timeq, kk);
				double * var[10];
				unsigned int n = 0;
				var[n] = vx_h; n++;
				var[n] = vy_h; n++;
				var[n] = p_h; n++;
				var[n] = C_h; n++;
				var[n] = mu_h; n++;
				if (vibration == 1)
				{
					var[n] = Phi_h;
					n++;
				}
				Geom.save(var, n, iter, write_i, timeq, kk);
			}
			copied = false;
			// the end of 4th step


			if (timeq > limit_timeq || stop == 1) return 0;



		} //the end of the main time loop






}