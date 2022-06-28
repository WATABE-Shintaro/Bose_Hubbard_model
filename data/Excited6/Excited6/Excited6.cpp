// Excited6.cpp : このファイルには 'main' 関数が含まれています。プログラム実行の開始と終了がそこで行われます。
//

#include "pch.h"
#include <iostream>


//#ifdef _OPENMP
#include <omp.h>
//#endif

#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include <time.h>
#include<complex>
using namespace std;
/*
   A C-program for MT19937, with initialization improved 2002/1/26.
   Coded by Takuji Nishimura and Makoto Matsumoto.

   Before using, initialize the state by using init_genrand(seed)
   or init_by_array(init_key, key_length).

   Copyright (C) 1997 - 2002, Makoto Matsumoto and Takuji Nishimura,
   All rights reserved.

   Redistribution and use in source and binary forms, with or without
   modification, are permitted provided that the following conditions
   are met:

	 1. Redistributions of source code must retain the above copyright
		notice, this list of conditions and the following disclaimer.

	 2. Redistributions in binary form must reproduce the above copyright
		notice, this list of conditions and the following disclaimer in the
		documentation and/or other materials provided with the distribution.

	 3. The names of its contributors may not be used to endorse or promote
		products derived from this software without specific prior written
		permission.

   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
   "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
   LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
   A PARTICULAR PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
   CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
   EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
   PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
   PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
   LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
   NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
   SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


   Any feedback is very welcome.
   http://www.math.sci.hiroshima-u.ac.jp/~m-mat/MT/emt.html
   email: m-mat @ math.sci.hiroshima-u.ac.jp (remove space)
*/

/*
   The original version of http://www.math.sci.hiroshima-u.ac.jp/~m-mat/MT/MT2002/CODES/mt19937ar.c was modified by Takahiro Omi as
   - delete line 47 "#include<stdio.h>"
   - delete line 174 int main(void){...}
   - change N -> MT_N
   - change N -> MT_N
   - change the file name "mt19937ar.c" -> "MT.h"
*/


/* Period parameters */
#define MT_N 624
#define MT_M 397
#define MATRIX_A 0x9908b0dfUL   /* constant vector a */
#define UPPER_MASK 0x80000000UL /* most significant w-r bits */
#define LOWER_MASK 0x7fffffffUL /* least significant r bits */

static unsigned long mt[MT_N]; /* the array for the state vector  */
static int mti = MT_N + 1; /* mti==MT_N+1 means mt[MT_N] is not initialized */

/* initializes mt[MT_N] with a seed */
void init_genrand(unsigned long s)
{
	mt[0] = s & 0xffffffffUL;
	for (mti = 1; mti < MT_N; mti++) {
		mt[mti] =
			(1812433253UL * (mt[mti - 1] ^ (mt[mti - 1] >> 30)) + mti);
		/* See Knuth TAOCP Vol2. 3rd Ed. P.106 for multiplier. */
		/* In the previous versions, MSBs of the seed affect   */
		/* only MSBs of the array mt[].                        */
		/* 2002/01/09 modified by Makoto Matsumoto             */
		mt[mti] &= 0xffffffffUL;
		/* for >32 bit machines */
	}
}

/* initialize by an array with array-length */
/* init_key is the array for initializing keys */
/* key_length is its length */
/* slight change for C++, 2004/2/26 */
void init_by_array(unsigned long init_key[], int key_length)
{
	int i, j, k;
	init_genrand(19650218UL);
	i = 1; j = 0;
	k = (MT_N > key_length ? MT_N : key_length);
	for (; k; k--) {
		mt[i] = (mt[i] ^ ((mt[i - 1] ^ (mt[i - 1] >> 30)) * 1664525UL))
			+ init_key[j] + j; /* non linear */
		mt[i] &= 0xffffffffUL; /* for WORDSIZE > 32 machines */
		i++; j++;
		if (i >= MT_N) { mt[0] = mt[MT_N - 1]; i = 1; }
		if (j >= key_length) j = 0;
	}
	for (k = MT_N - 1; k; k--) {
		mt[i] = (mt[i] ^ ((mt[i - 1] ^ (mt[i - 1] >> 30)) * 1566083941UL))
			- i; /* non linear */
		mt[i] &= 0xffffffffUL; /* for WORDSIZE > 32 machines */
		i++;
		if (i >= MT_N) { mt[0] = mt[MT_N - 1]; i = 1; }
	}

	mt[0] = 0x80000000UL; /* MSB is 1; assuring non-zero initial array */
}

/* generates a random number on [0,0xffffffff]-interval */
unsigned long genrand_int32(void)
{
	unsigned long y;
	static unsigned long mag01[2] = { 0x0UL, MATRIX_A };
	/* mag01[x] = x * MATRIX_A  for x=0,1 */

	if (mti >= MT_N) { /* generate N words at one time */
		int kk;

		if (mti == MT_N + 1)   /* if init_genrand() has not been called, */
			init_genrand(5489UL); /* a default initial seed is used */

		for (kk = 0; kk < MT_N - MT_M; kk++) {
			y = (mt[kk] & UPPER_MASK) | (mt[kk + 1] & LOWER_MASK);
			mt[kk] = mt[kk + MT_M] ^ (y >> 1) ^ mag01[y & 0x1UL];
		}
		for (; kk < MT_N - 1; kk++) {
			y = (mt[kk] & UPPER_MASK) | (mt[kk + 1] & LOWER_MASK);
			mt[kk] = mt[kk + (MT_M - MT_N)] ^ (y >> 1) ^ mag01[y & 0x1UL];
		}
		y = (mt[MT_N - 1] & UPPER_MASK) | (mt[0] & LOWER_MASK);
		mt[MT_N - 1] = mt[MT_M - 1] ^ (y >> 1) ^ mag01[y & 0x1UL];

		mti = 0;
	}

	y = mt[mti++];

	/* Tempering */
	y ^= (y >> 11);
	y ^= (y << 7) & 0x9d2c5680UL;
	y ^= (y << 15) & 0xefc60000UL;
	y ^= (y >> 18);

	return y;
}

/* generates a random number on [0,0x7fffffff]-interval */
long genrand_int31(void)
{
	return (long)(genrand_int32() >> 1);
}

/* generates a random number on [0,1]-real-interval */
double genrand_real1(void)
{
	return genrand_int32()*(1.0 / 4294967295.0);
	/* divided by 2^32-1 */
}

/* generates a random number on [0,1)-real-interval */
double genrand_real2(void)
{
	return genrand_int32()*(1.0 / 4294967296.0);
	/* divided by 2^32 */
}

/* generates a random number on (0,1)-real-interval */
double genrand_real3(void)
{
	return (((double)genrand_int32()) + 0.5)*(1.0 / 4294967296.0);
	/* divided by 2^32 */
}

/* generates a random number on [0,1) with 53-bit resolution*/
double genrand_res53(void)
{
	unsigned long a = genrand_int32() >> 5, b = genrand_int32() >> 6;
	return(a*67108864.0 + b)*(1.0 / 9007199254740992.0);
}
/* These real versions are due to Isaku Wada, 2002/01/09 added */


int** bm2maker(int N, int M);
int bm2free(int** bm2, int N);
double** bm2makerDouble(int, int);
int bm2freeDouble(double**, int);
complex<double>neuralnetwork(int*, int);
complex<double> HmSum(int*, int, int, int, int);
int ncc(int, int, int*, int *, complex<double>*, int);
complex<double> cing(int);
int nccing(int tempN, int Mp, int * bmm, complex<double>* panswer, int frag);
complex<double> ning(int* bmn, int a, int b, int c, int frag);
int nfcount();
int nccnf(int tempN, int Mp, int* Nf);
double calculatehamiltonian(int*, int*);
double item1(int*, int*);
double item2(int*, int*);
double item3(int*, int*);
int shufflebmn(int*);
complex<double> metropolis(complex<double>(*func)(int*, int, int, int, int), int a, int b, int c, int frag);
complex <double> Hmmetropolis(int);
//double CnSum();
//int ncc2(int, int, int *, double*);
complex<double> Ow(int*, int, int, int, int);
//complex<double> HmOw(int* bmn, int a, int b, int c);
complex<double>HmOwmtr(int a, int b, int c, int frag);
int gdsd();
double dGdw(int a, int b, int c);
int nccdGdw(int tempN, int Mp, int * bmm, complex<double>* panswer1, complex<double>* panswer2, double* panswer3, complex<double>* panswer4, double* panswer5, int a, int b, int c);
int gdsd2();
double cdc0();
int nccdcdc0(int tempN, int Mp, int * bmm, complex<double>* panswer1, complex<double>* panswer2, double* panswer3, complex<double>* panswer4, double* panswer5);

double J;
double Uj;
int N;
int M;
int Nh;
int Nsample;
double mu;
int Nupdate;
double** bm2W1;
double** bm2W2;
double* bmh1;
double* bmh2;
const int Nm = 1;
const int n1real = 0;
const int n2img = 1;
int** bm2n;
double* bmH;
int** bm2nmtr;
complex<double>* bmHmtr;

double** ebm2W1;
double** ebm2W2;
double* ebmh1;
double* ebmh2;
double lambda;
const int grand = 0;
const int excited = 1;

int main()
{
	int i, j;
	printf("J=");
	J = 1;
	printf("U/J=");
	Uj = 2;
	printf("Nh=");
	Nh = 20;
	printf("mu=");
	mu = 0.001;
	printf("Number of updates=");
	Nupdate = 2000;
	printf("N=");
	N = 3;
	printf("M=");
	M = 3;
	printf("Nsample=");
	Nsample = 1000;
	bm2W1 = bm2makerDouble(Nh, M);
	bm2W2 = bm2makerDouble(Nm, Nh);
	bmh1 = (double*)malloc(sizeof(double)*Nh);
	bmh2 = (double*)malloc(sizeof(double)*Nm);
	init_genrand((unsigned)time(NULL));

	int Nf;
	Nf = nfcount();
	//bm2n = bm2maker(Nf, M);
	//bmH = (double*)malloc(sizeof(double)*Nf);
	bm2nmtr = bm2maker(Nsample, M);
	bmHmtr = (complex<double>*)malloc(sizeof(complex<double>)*Nsample);

	ebm2W1 = bm2makerDouble(Nh, M);
	ebm2W2 = bm2makerDouble(Nm, Nh);
	ebmh1 = (double*)malloc(sizeof(double)*Nh);
	ebmh2 = (double*)malloc(sizeof(double)*Nm);

	for (i = 0; i < Nh; i++)
	{
		for (j = 0; j < M; j++)
		{
			bm2W1[i][j] = (genrand_real1() - 0.5) * 2; //genrand_real1() / 20 + 0.05;
		}
	}
	for (i = 0; i < Nm; i++)
	{
		for (j = 0; j < Nh; j++)
		{
			bm2W2[i][j] = (genrand_real1() - 0.5) * 2;//  genrand_real1() / 20 + 0.05;
		}
	}
	//for (i = 1; i < 2; i++)
	//{
	//	for (j = 0; j < Nh; j++)
	//	{
	//		bm2W2[i][j] = 0;
	//	}
	//}
	for (i = 0; i < Nh; i++)
	{
		bmh1[i] = (genrand_real1() - 0.5) * 2;//-genrand_real1() / 20 - 0.05; 
	}
	for (i = 0; i < Nm; i++)
	{
		bmh2[i] = (genrand_real1() - 0.5) * 2;//-genrand_real1() / 20 - 0.05;
	}
	//for (i = 1; i < 2; i++)
	//{
	//	bmh2[i] = 0;
	//}

	//===============================================================================
	printf("\n");
	for (i = 0; i < Nupdate; i++)
	{
		gdsd();
		printf("%d\n", i + 1);
		printf("ararar %f+i%f\n", metropolis(HmSum, 0, 0, 0, grand).real(), metropolis(HmSum, 0, 0, 0, grand).imag());
		//for (j = 0; j < M; j++)
		//{
		//	printf("n_%d ,%f+i%f\n", j, metropolis(ning, j, 0, 0).real(), metropolis(ning, j, 0, 0).imag());
		//}
	}


	for (i = 0; i < Nh; i++)
	{
		for (j = 0; j < M; j++)
		{
			ebm2W1[i][j] = (genrand_real1() - 0.5) * 2; // bm2W1[i][j];//(genrand_real1() - 0.5) * 2; //genrand_real1() / 20 + 0.05;
		}
	}
	for (i = 0; i < Nm; i++)
	{
		for (j = 0; j < Nh; j++)
		{
			ebm2W2[i][j] = (genrand_real1() - 0.5) * 2;// bm2W2[i][j];// (genrand_real1() - 0.5) * 2;//  genrand_real1() / 20 + 0.05;
		}
	}
	for (i = 0; i < Nh; i++)
	{
		ebmh1[i] = (genrand_real1() - 0.5) * 2;//bmh1[i];// (genrand_real1() - 0.5) * 2;//-genrand_real1() / 20 - 0.05; 
	}
	for (i = 0; i < Nm; i++)
	{
		ebmh2[i] = (genrand_real1() - 0.5) * 2;//bmh2[i];// (genrand_real1() - 0.5) * 2;//-genrand_real1() / 20 - 0.05;
	}

	printf("\n");
	Nupdate = 2000;
	for (i = 0; i < Nupdate; i++)
	{
		lambda = 10;//pow(2, (double)i / 200.0) ;
		gdsd2();
		printf("%d\n", i + 1);
		printf("aaa %f\n", cdc0());
		printf("ararar %f+i%f\n", metropolis(HmSum, 0, 0, 0, excited).real(), metropolis(HmSum, 0, 0, 0, excited).imag());
		//for (j = 0; j < M; j++)
		//{
		//	printf("n_%d ,%f+i%f\n", j, metropolis(ning, j, 0, 0).real(), metropolis(ning, j, 0, 0).imag());
		//}
	}
	bm2free(bm2nmtr, Nsample);
	free(bmHmtr);
	/*
	for (i = 0; i < Nh; i++)
	{
		for (j = 0; j < M; j++)
		{
			printf("w1,%f\n", bm2W1[i][j]);
		}
	}
	for (i = 0; i < Nm; i++)
	{
		for (j = 0; j < Nh; j++)
		{
			printf("w2,%f\n", bm2W2[i][j]);
		}
	}
	for (i = 0; i < Nh; i++)
	{
		printf("h1,%f\n", bmh1[i]);
	}
	for (i = 0; i < Nm; i++)
	{
		printf("h2,%f\n", bmh2[i]);
	}
	cing();
	*/
	cing(excited);
	Nsample = 10000;
	printf("Uj=%f\n", Uj);
	printf("N=%d\n", N);
	printf("M=%d\n", M);
	printf("E grand ,%f+i%f\n", metropolis(HmSum, 0, 0, 0, grand).real(), metropolis(HmSum, 0, 0, 0, grand).imag());
	for (i = 0; i < M; i++)
	{
		printf("%f\n", metropolis(ning, i, 0, 0, grand).real());
	}
	printf("E excited1,%f+i%f\n", metropolis(HmSum, 0, 0, 0, excited).real(), metropolis(HmSum, 0, 0, 0, excited).imag());
	for (i = 0; i < M; i++)
	{
		printf("%f\n", metropolis(ning, i, 0, 0, excited).real());
	}

	/*
FILE *outputfile;         // 出力ストリーム
errno_t error;
error = fopen_s(&outputfile, "d.txt", "w");  // ファイルを書き込み用にオープン(開く)
if (error != 0) {          // オープンに失敗した場合
	printf("cannot open\n");         // エラーメッセージを出して
	exit(1);                         // 異常終了
}

fprintf(outputfile, "My name is Enokida Yuuichirou.\n"); // ファイルに書く

fclose(outputfile);          // ファイルをクローズ(閉じる)
*/
	bm2freeDouble(bm2W1, Nh);
	bm2freeDouble(bm2W2, Nm);
	free(bmh1);
	free(bmh2);

	bm2freeDouble(ebm2W1, Nh);
	bm2freeDouble(ebm2W2, Nm);
	free(ebmh1);
	free(ebmh2);

	//bm2free(bm2n, Nf);
	//free(bmH);


}

int** bm2maker(int N, int M) {									//二次元配列作る
	int i;
	int** bm2;
	bm2 = (int **)malloc(sizeof(int *)*N);
	for (i = 0; i < N; i++) {
		bm2[i] = (int *)malloc(sizeof(int)*M);
	}
	return bm2;
}
int bm2free(int** bm2, int N) {												//二次元配列消す
	int i;
	for (i = 0; i < N; i++) {
		free(bm2[i]);
	}
	free(bm2);
	return 0;
}


double** bm2makerDouble(int N, int M) {							//二次元配列作る
	int i;
	double** bm2;
	bm2 = (double **)malloc(sizeof(double *)*N);
	for (i = 0; i < N; i++) {
		bm2[i] = (double *)malloc(sizeof(double)*M);
	}
	return bm2;
}
int bm2freeDouble(double** bm2, int N) {
	int i;
	for (i = 0; i < N; i++) {
		free(bm2[i]);
	}
	free(bm2);
	return 0;
}

complex<double>neuralnetwork(int* bmn, int frag)
{
	complex<double> Cn, logCn;
	double* u1;
	double* u2;
	int i, j;
	u1 = (double*)malloc(sizeof(double)*Nh);
	u2 = (double*)malloc(sizeof(double)*Nm);

	if (frag == grand)
	{
		for (i = 0; i < Nh; i++)
		{
			u1[i] = 0;
			for (j = 0; j < M; j++)
			{
				u1[i] += bm2W1[i][j] * bmn[j];
			}
			u1[i] += bmh1[i];
		}
		for (i = 0; i < Nm; i++)
		{
			u2[i] = 0;
			for (j = 0; j < Nh; j++)
			{
				u2[i] += bm2W2[i][j] * tanh(u1[j]);
			}
			u2[i] += bmh2[i];
		}
		//u2[1] = 0;
	}
	else if (frag == excited)
	{
		for (i = 0; i < Nh; i++)
		{
			u1[i] = 0;
			for (j = 0; j < M; j++)
			{
				u1[i] += ebm2W1[i][j] * bmn[j];
			}
			u1[i] += ebmh1[i];
		}
		for (i = 0; i < Nm; i++)
		{
			u2[i] = 0;
			for (j = 0; j < Nh; j++)
			{
				u2[i] += ebm2W2[i][j] * tanh(u1[j]);
			}
			u2[i] += ebmh2[i];
		}
		//u2[1] = 0;
	}

	logCn = complex<double>(u2[n1real], 0);
	Cn = exp(logCn);
	free(u1);
	free(u2);
	return Cn;
}

int nfcount() {																//数数える
	int Mp, Nf;
	Mp = M - 1;
	Nf = 0;
	nccnf(N, Mp, &Nf);
	return Nf;
}
int nccnf(int tempN, int Mp, int* Nf) {												//数数える中身　再帰関数
	int i;
	if (Mp > 0) {
		for (i = 0; i <= tempN; i++)
		{
			nccnf(i, Mp - 1, Nf);
		}
	}
	else {
		*Nf += 1;
	}
	return 0;
}

complex<double> HmSum(int* bmn, int a, int b, int c, int frag) {																//ラベル付け
	int* bmm;
	//printf("%d,%d,%d\n aaa\n", bmn[0], bmn[1], bmn[2]);
	bmm = (int*)malloc(sizeof(int)*M);
	int Mp;
	Mp = M - 1;
	complex<double> answer;
	answer = complex<double>(0, 0);
	ncc(N, Mp, bmn, bmm, &answer, frag);
	//printf("aaaaaaaaaaaaaaaa");
	free(bmm);
	return answer;
}
int ncc(int tempN, int Mp, int* bmn, int * bmm, complex<double>* panswer, int frag) {												//ラベル付け中身　再帰関数
	int i;
	if (Mp > 0) {

		for (i = 0; i <= tempN; i++)
		{
			bmm[M - Mp - 1] = tempN - i;
			ncc(i, Mp - 1, bmn, bmm, panswer, frag);
		}

	}
	else {

		bmm[M - 1] = tempN;
		/*
		for ( i = 0; i <M; i++)
		{
			printf("%d",bmm[i]);
		}
		printf("\n");
		printf("oooo %f,%f,%f \n", calculatehamiltonian(bmn, bmm), neuralnetwork(bmn).real(), neuralnetwork(bmn).imag());
		*/
		if (abs(neuralnetwork(bmn, frag)) > 0.00000001)
		{
			*panswer += calculatehamiltonian(bmn, bmm) * neuralnetwork(bmm, frag) / neuralnetwork(bmn, frag);
		}

	}
	return 0;

}

complex<double> cing(int frag) {																//ラベル付け
	int* bmm;
	//printf("%d,%d,%d\n aaa\n", bmn[0], bmn[1], bmn[2]);
	bmm = (int*)malloc(sizeof(int)*M);
	int Mp;
	Mp = M - 1;
	complex<double> answer;
	answer = complex<double>(0, 0);
	nccing(N, Mp, bmm, &answer, frag);
	//printf("aaaaaaaaaaaaaaaa");
	free(bmm);
	return answer;
}
int nccing(int tempN, int Mp, int * bmm, complex<double>* panswer, int frag) {												//ラベル付け中身　再帰関数
	int i;
	if (Mp > 0) {

		for (i = 0; i <= tempN; i++)
		{
			bmm[M - Mp - 1] = tempN - i;
			nccing(i, Mp - 1, bmm, panswer, frag);
		}

	}
	else {

		bmm[M - 1] = tempN;
		/*
		for ( i = 0; i <M; i++)
		{
			printf("%d",bmm[i]);
		}
		printf("\n");
		printf("oooo %f,%f,%f \n", calculatehamiltonian(bmn, bmm), neuralnetwork(bmn).real(), neuralnetwork(bmn).imag());
		*/
		printf("Cn %f,%f\n", neuralnetwork(bmm, frag).real(), neuralnetwork(bmm, frag).imag());

	}
	return 0;
}

complex<double> ning(int* bmn, int a, int b, int c, int frag) {

	complex<double> answer;
	answer = complex<double>(bmn[a], 0);
	return answer;
}

double calculatehamiltonian(int* bmn, int* bmm)
{
	return -J * item1(bmn, bmm) + J * item2(bmn, bmm) + Uj * J / 2 * item3(bmn, bmm);
}

double item1(int* bmn, int* bmm)
{
	int answer1;
	double answer2;

	answer2 = 0;
	int i, j, k;
	for (i = 0; i < M - 1; i++)
	{
		answer1 = 1;
		j = i + 1;

		bmn[i] += 1;
		bmn[j] -= 1;
		for (k = 0; k < M; k++)
		{
			if (bmn[k] == bmm[k])
			{
				answer1 *= 1;
			}
			else
			{
				answer1 *= 0;
			}

		}
		bmn[i] -= 1;
		bmn[j] += 1;

		answer2 += sqrt(bmn[j])*sqrt(bmn[i] + 1)*answer1;


	}
	for (i = 1; i < M; i++)
	{
		j = i - 1;
		answer1 = 1;

		bmn[i] += 1;
		bmn[j] -= 1;
		for (k = 0; k < M; k++)
		{
			if (bmn[k] == bmm[k])
			{
				answer1 *= 1;
			}
			else
			{
				answer1 *= 0;
			}

		}
		bmn[i] -= 1;
		bmn[j] += 1;

		answer2 += sqrt(bmn[j])*sqrt(bmn[i] + 1)*answer1;


	}


	return answer2;
}

double item2(int* bmn, int* bmm)
{
	int answer1;
	double answer2;
	answer1 = 1;
	answer2 = 0;
	int i;
	for (i = 0; i < M; i++)
	{
		if (bmn[i] == bmm[i])
		{
			answer1 *= 1;
		}
		else
		{
			answer1 *= 0;
		}

	}
	for (i = 0; i < M; i++)
	{
		answer2 += pow((i - (M - 1) / 2.0), 2)*bmn[i];
	}
	return  (answer1*answer2);
}

double item3(int* bmn, int* bmm)
{
	int answer1, answer2;
	answer1 = 1;
	answer2 = 0;
	int i;
	for (i = 0; i < M; i++)
	{
		if (bmn[i] == bmm[i])
		{
			answer1 *= 1;
		}
		else
		{
			answer1 *= 0;
		}

	}
	for (i = 0; i < M; i++)
	{
		answer2 += bmn[i] * (bmn[i] - 1);
	}
	return (answer1*answer2);
}


int shufflebmn(int* bmn)
{
	int a, b;
	do
	{
		a = (int)floor(genrand_real2()*M);
	} while (bmn[a] == 0);
	do
	{
		b = (int)floor(genrand_real2()*M);
	} while (a == b);
	//printf("asa%d %d\n", a, b);
	bmn[a] -= 1;
	bmn[b] += 1;
	//printf("%d,%d,%d\n",bmn[0],bmn[1],bmn[2]);
	return 0;
}

complex <double> metropolis(complex<double>(*func)(int*, int, int, int, int), int aaa, int bbb, int ccc, int frag)
{
	int* bmn;
	int* bmn2;
	int i, j, k;
	double a, a2;
	double q1, q2;
	complex<double> answer(0, 0);
	//double cnsum;
	//cnsum = CnSum();
	bmn = (int*)malloc(sizeof(int)*M);
	bmn2 = (int*)malloc(sizeof(int)*M);
	bmn[0] = N;
	for (i = 1; i < M; i++)
	{
		bmn[i] = 0;
	}
	for (i = 0; i < N; i++)
	{
		shufflebmn(bmn);
	}
	//printf("a %d,%d,%d\n", bmn[0], bmn[1], bmn[2]);


	for (i = 0; i < Nsample; i++)
	{
		for (j = 0; j < N; j++)
		{
			for (k = 0; k < M; k++)
			{
				bmn2[k] = bmn[k];
			}
			shufflebmn(bmn2);

			q1 = 0;
			for (k = 0; k < M; k++)
			{
				if (bmn[k] == 0)
				{
					q1 += 1;
				}
			}
			q1 = M - q1;
			q2 = 0;
			for (k = 0; k < M; k++)
			{
				if (bmn2[k] == 0)
				{
					q2 += 1;
				}
			}
			q2 = M - q2;

			a = abs(neuralnetwork(bmn, frag));
			a2 = abs(neuralnetwork(bmn2, frag));
			if (genrand_real2() < pow(a2 / a, 2)*(q1 / q2))
			{

				for (k = 0; k < M; k++)
				{
					bmn[k] = bmn2[k];
				}

			}
			//printf("b %d,%d,%d\n", bmn[0], bmn[1], bmn[2]);
		}
		//printf("bababab %f\n", neuralnetwork(bmn) / cnsum );
		answer += func(bmn, aaa, bbb, ccc, frag);		// *  neuralnetwork(bmn) / cnsum;

	}
	//printf("%f\n", answer.real());
	answer = answer * (1.0 / (double)Nsample);
	//printf("%f\n", answer.real());

	free(bmn);
	free(bmn2);
	return answer;
}

complex <double> Hmmetropolis(int frag)
{
	int* bmn;
	int* bmn2;
	int i, j, k;
	double a, a2;
	double q1, q2;
	complex<double> answer(0, 0);
	//double cnsum;
	//cnsum = CnSum();
	bmn = (int*)malloc(sizeof(int)*M);
	bmn2 = (int*)malloc(sizeof(int)*M);
	bmn[0] = N;
	for (i = 1; i < M; i++)
	{
		bmn[i] = 0;
	}
	for (i = 0; i < N; i++)
	{
		shufflebmn(bmn);
	}
	//printf("a %d,%d,%d\n", bmn[0], bmn[1], bmn[2]);


	for (i = 0; i < Nsample; i++)
	{
		for (j = 0; j < N; j++)
		{
			for (k = 0; k < M; k++)
			{
				bmn2[k] = bmn[k];
			}
			shufflebmn(bmn2);

			q1 = 0;
			for (k = 0; k < M; k++)
			{
				if (bmn[k] == 0)
				{
					q1 += 1;
				}
			}
			q1 = M - q1;
			q2 = 0;
			for (k = 0; k < M; k++)
			{
				if (bmn2[k] == 0)
				{
					q2 += 1;
				}
			}
			q2 = M - q2;

			a = abs(neuralnetwork(bmn, frag));
			a2 = abs(neuralnetwork(bmn2, frag));
			if (genrand_real2() < pow(a2 / a, 2)*(q1 / q2))
			{

				for (k = 0; k < M; k++)
				{
					bmn[k] = bmn2[k];
				}

			}
			//printf("b %d,%d,%d\n", bmn[0], bmn[1], bmn[2]);
		}
		//printf("bababab %f\n", neuralnetwork(bmn) / cnsum );
		for (j = 0; j < M; j++)
		{
			bm2nmtr[i][j] = bmn[j];
		}
		bmHmtr[i] = HmSum(bmn, 0, 0, 0, frag);
		answer += bmHmtr[i];		// *  neuralnetwork(bmn) / cnsum;

	}

	answer = answer * (1.0 / (double)Nsample);

	free(bmn);
	free(bmn2);
	return answer;
}

complex<double>HmOwmtr(int a, int b, int c, int frag)
{
	complex<double> answer(0, 0);
	int i;
	for (i = 0; i < Nsample; i++)
	{
		answer += bmHmtr[i] * Ow(bm2nmtr[i], a, b, c, frag);
	}
	answer = answer * (1.0 / (double)Nsample);
	return answer;
}

int gdsd()
{
	double dhdw;
	complex<double> content;
	complex<double> content2;
	complex<double> content3;
	complex<double> content4;
	double** bm2W1temp;
	double** bm2W2temp;
	double* bmh1temp;
	double* bmh2temp;
	bm2W1temp = bm2makerDouble(Nh, M);
	bm2W2temp = bm2makerDouble(Nm, Nh);
	bmh1temp = (double*)malloc(sizeof(double)*Nh);
	bmh2temp = (double*)malloc(sizeof(double)*Nm);
	content2 = Hmmetropolis(grand);
	int i, j;
#pragma omp parallel for private(j,content3,content4 ,dhdw,content)
	for (i = 0; i < Nh; i++)
	{
		for (j = 0; j < M; j++)
		{
			content3 = HmOwmtr(1, i, j, grand);
			content4 = metropolis(Ow, 1, i, j, grand);
			content = content3 - content4 * content2;
			dhdw = 2 * content.real();
			bm2W1temp[i][j] = mu * dhdw;
			//printf("1,%f,%f,%f,%f\n", content2.real(), content4.real(), content3.real(), dhdw);
		}
	}
#pragma omp parallel for  private(j,content3,content4 ,dhdw,content)
	for (i = 0; i < Nm; i++)
	{
		for (j = 0; j < Nh; j++)
		{
			content3 = HmOwmtr(2, i, j, grand);
			content4 = metropolis(Ow, 2, i, j, grand);
			content = content3 - content4 * content2;
			dhdw = 2 * content.real();
			bm2W2temp[i][j] = mu * dhdw;
			//printf("2,%f,%f,%f,%f\n", content2.real(), content4.real(), content3.real(), dhdw);

		}
	}
#pragma omp parallel for  private(j,content3,content4 ,dhdw,content)
	for (i = 0; i < Nh; i++)
	{
		content3 = HmOwmtr(3, i, 0, grand);
		content4 = metropolis(Ow, 3, i, 0, grand);
		content = content3 - content4 * content2;
		dhdw = 2 * content.real();
		bmh1temp[i] = mu * dhdw;
		//printf("3,%f,%f,%f,%f\n", content2.real(), content4.real(), content3.real(), dhdw);
	}
#pragma omp parallel for   private(j,content3,content4 ,dhdw,content)
	for (i = 0; i < Nm; i++)
	{
		content3 = HmOwmtr(4, i, 0, grand);
		content4 = metropolis(Ow, 4, i, 0, grand);
		content = content3 - content4 * content2;
		dhdw = 2 * content.real();
		bmh2temp[i] = mu * dhdw;
		//printf("4,%f,%f,%f,%f\n", content2.real(), content4.real(), content3.real(), dhdw);
	}


	for (i = 0; i < Nh; i++)
	{
		for (j = 0; j < M; j++)
		{
			bm2W1[i][j] -= bm2W1temp[i][j];

		}
	}
	for (i = 0; i < Nm; i++)
	{
		for (j = 0; j < Nh; j++)
		{
			bm2W2[i][j] -= bm2W2temp[i][j];
		}
	}
	for (i = 0; i < Nh; i++)
	{
		bmh1[i] -= bmh1temp[i];
	}
	for (i = 0; i < Nm; i++)
	{
		bmh2[i] -= bmh2temp[i];
	}

	bm2freeDouble(bm2W1temp, Nh);
	bm2freeDouble(bm2W2temp, Nm);
	free(bmh1temp);
	free(bmh2temp);
	return 0;
}


double dGdw(int a, int b, int c) {																//ラベル付け
	int* bmm;
	//printf("%d,%d,%d\n aaa\n", bmn[0], bmn[1], bmn[2]);
	bmm = (int*)malloc(sizeof(int)*M);
	int Mp;
	Mp = M - 1;
	complex<double> answer1;
	answer1 = complex<double>(0, 0);
	complex<double> answer2;
	answer2 = complex<double>(0, 0);
	double answer3;
	answer3 = 0;
	complex<double> answer4;
	answer4 = complex<double>(0, 0);
	double answer5;
	complex<double> answer21;
	double answer22;
	double answer23;
	double answer31;
	nccdGdw(N, Mp, bmm, &answer1, &answer2, &answer3, &answer4, &answer5, a, b, c);
	//printf("aaaaaaaaaaaaaaaa");
	answer21 = answer1 * answer2;
	answer22 = answer3 * answer5;
	answer23 = pow(abs(answer2), 2);
	answer31 = 2 * answer21.real() / answer22 - answer23 / answer22 * 2 * answer4.real() / answer3;
	free(bmm);
	//printf("%f,%f,%f,%f\n", abs(answer1), abs(answer2), answer3, abs(answer4));
	//printf("dGdw %f\n", answer31);		
	return  answer31;
}
int nccdGdw(int tempN, int Mp, int * bmm, complex<double>* panswer1, complex<double>* panswer2, double* panswer3, complex<double>* panswer4, double* panswer5, int a, int b, int c) {												//ラベル付け中身　再帰関数
	int i;
	if (Mp > 0) {

		for (i = 0; i <= tempN; i++)
		{
			bmm[M - Mp - 1] = tempN - i;
			nccdGdw(i, Mp - 1, bmm, panswer1, panswer2, panswer3, panswer4, panswer5, a, b, c);
		}

	}
	else {

		bmm[M - 1] = tempN;
		/*
		for ( i = 0; i <M; i++)
		{
			printf("%d",bmm[i]);
		}
		printf("\n");
		printf("oooo %f,%f,%f \n", calculatehamiltonian(bmn, bmm), neuralnetwork(bmn).real(), neuralnetwork(bmn).imag());
		*/

		*panswer1 += Ow(bmm, a, b, c, excited) *conj(neuralnetwork(bmm, excited))*neuralnetwork(bmm, grand);
		*panswer2 += neuralnetwork(bmm, excited)*conj(neuralnetwork(bmm, grand));
		*panswer3 += pow(abs(neuralnetwork(bmm, excited)), 2);
		*panswer4 += Ow(bmm, a, b, c, excited) *conj(neuralnetwork(bmm, excited))*neuralnetwork(bmm, excited);
		*panswer5 += pow(abs(neuralnetwork(bmm, grand)), 2);
	}
	return 0;

}

double cdc0() {																//ラベル付け
	int* bmm;
	//printf("%d,%d,%d\n aaa\n", bmn[0], bmn[1], bmn[2]);
	bmm = (int*)malloc(sizeof(int)*M);
	int Mp;
	Mp = M - 1;
	complex<double> answer1;
	answer1 = complex<double>(0, 0);
	complex<double> answer2;
	answer2 = complex<double>(0, 0);
	double answer3;
	answer3 = 0;
	complex<double> answer4;
	answer4 = complex<double>(0, 0);
	double answer5;
	complex<double> answer21;
	double answer22;
	double answer23;
	double answer31;
	nccdcdc0(N, Mp, bmm, &answer1, &answer2, &answer3, &answer4, &answer5);
	//printf("aaaaaaaaaaaaaaaa");
	answer21 = 0;
	answer22 = 0;
	answer23 = 0;
	answer31 = pow(abs(answer2), 2) / (answer3*answer5);
	free(bmm);
	//printf("%f,%f,%f,%f\n", abs(answer1), abs(answer2), answer3, abs(answer4));
	//printf("dGdw %f\n", answer31);		
	return  answer31;
}
int nccdcdc0(int tempN, int Mp, int * bmm, complex<double>* panswer1, complex<double>* panswer2, double* panswer3, complex<double>* panswer4, double* panswer5) {												//ラベル付け中身　再帰関数
	int i;
	if (Mp > 0) {

		for (i = 0; i <= tempN; i++)
		{
			bmm[M - Mp - 1] = tempN - i;
			nccdcdc0(i, Mp - 1, bmm, panswer1, panswer2, panswer3, panswer4, panswer5);
		}

	}
	else {

		bmm[M - 1] = tempN;
		/*
		for ( i = 0; i <M; i++)
		{
			printf("%d",bmm[i]);
		}
		printf("\n");
		printf("oooo %f,%f,%f \n", calculatehamiltonian(bmn, bmm), neuralnetwork(bmn).real(), neuralnetwork(bmn).imag());
		*/

		*panswer1 += 0;
		*panswer2 += neuralnetwork(bmm, excited)*conj(neuralnetwork(bmm, grand));
		*panswer3 += pow(abs(neuralnetwork(bmm, excited)), 2);
		*panswer4 += 0;
		*panswer5 += pow(abs(neuralnetwork(bmm, grand)), 2);
	}
	return 0;

}

int gdsd2()
{
	double dhdw;
	complex<double> content;
	complex<double> content2;
	complex<double> content3;
	complex<double> content4;
	double** bm2W1temp;
	double** bm2W2temp;
	double* bmh1temp;
	double* bmh2temp;
	bm2W1temp = bm2makerDouble(Nh, M);
	bm2W2temp = bm2makerDouble(Nm, Nh);
	bmh1temp = (double*)malloc(sizeof(double)*Nh);
	bmh2temp = (double*)malloc(sizeof(double)*Nm);
	
	int i, j;
#pragma omp parallel for private(j,content3,content4 ,dhdw,content,content2)
	for (i = 0; i < Nh; i++)
	{
		for (j = 0; j < M; j++)
		{
			content2 = Hmmetropolis(excited);
			content3 = HmOwmtr(1, i, j, excited);
			content4 = metropolis(Ow, 1, i, j, excited);
			content = content3 - content4 * content2;
			dhdw = 2 * content.real();
			bm2W1temp[i][j] = mu * (dhdw + lambda * dGdw(1, i, j));
			//printf("1,%f,%f,%f,%f\n", content2.real(), content4.real(), content3.real(), dhdw);
			ebm2W1[i][j] -= bm2W1temp[i][j];
		}
	}
#pragma omp parallel for  private(j,content3,content4 ,dhdw,content,content2)
	for (i = 0; i < Nm; i++)
	{
		for (j = 0; j < Nh; j++)
		{
			content2 = Hmmetropolis(excited);
			content3 = HmOwmtr(2, i, j, excited);
			content4 = metropolis(Ow, 2, i, j, excited);
			content = content3 - content4 * content2;
			dhdw = 2 * content.real();
			bm2W2temp[i][j] = mu * (dhdw + lambda * dGdw(2, i, j));
			//printf("2,%f,%f,%f,%f\n", content2.real(), content4.real(), content3.real(), dhdw);
			ebm2W2[i][j] -= bm2W2temp[i][j];
		}
	}
#pragma omp parallel for  private(j,content3,content4 ,dhdw,content,content2)
	for (i = 0; i < Nh; i++)
	{
		content2 = Hmmetropolis(excited);
		content3 = HmOwmtr(3, i, 0, excited);
		content4 = metropolis(Ow, 3, i, 0, excited);
		content = content3 - content4 * content2;
		dhdw = 2 * content.real();
		bmh1temp[i] = mu * (dhdw + lambda * dGdw(3, i, 0));
		//printf("3,%f,%f,%f,%f\n", content2.real(), content4.real(), content3.real(), dhdw);
		ebmh1[i] -= bmh1temp[i];
	}
#pragma omp parallel for   private(j,content3,content4 ,dhdw,content,content2)
	for (i = 0; i < Nm; i++)
	{
		content2 = Hmmetropolis(excited);
		content3 = HmOwmtr(4, i, 0, excited);
		content4 = metropolis(Ow, 4, i, 0, excited);
		content = content3 - content4 * content2;
		dhdw = 2 * content.real();
		bmh2temp[i] = mu * (dhdw + lambda * dGdw(4, i, 0));
		//printf("4,%f,%f,%f,%f\n", content2.real(), content4.real(), content3.real(), dhdw);
		ebmh2[i] -= bmh2temp[i];
	}


	for (i = 0; i < Nh; i++)
	{
		for (j = 0; j < M; j++)
		{
			

		}
	}
	for (i = 0; i < Nm; i++)
	{
		for (j = 0; j < Nh; j++)
		{
			
		}
	}
	for (i = 0; i < Nh; i++)
	{
		
	}
	for (i = 0; i < Nm; i++)
	{
		
	}

	bm2freeDouble(bm2W1temp, Nh);
	bm2freeDouble(bm2W2temp, Nm);
	free(bmh1temp);
	free(bmh2temp);
	return 0;
}

complex<double> Ow(int* bmn, int a, int b, int c, int frag)
{
	int k, j, m, l;
	double sum1, sum2;
	double answers[2];
	complex<double> answer;

	if (frag == grand)
	{
		if (a == 2)
		{
			m = c;
			if (b == n1real)
			{




				sum1 = 0;
				for (j = 0; j < M; j++)
				{
					sum1 += bm2W1[m][j] * bmn[j];
				}
				sum1 += bmh1[m];
				sum2 = tanh(sum1);

				answers[0] = sum2;




				answers[1] = 0;

				answer = complex<double>(answers[0], 0);
			}
			else if (b == n2img)
			{

				answers[0] = 0;






				sum1 = 0;
				for (j = 0; j < M; j++)
				{
					sum1 += bm2W1[m][j] * bmn[j];
				}
				sum1 += bmh1[m];
				sum2 = tanh(sum1);

				answers[1] = sum2;



				answer = complex<double>(answers[0], 0);
			}
		}
		else if (a == 4)
		{
			if (b == n1real)
			{

				answers[0] = 1;


				answers[1] = 0;
				answer = complex<double>(answers[0], 0);
			}
			else if (b == n2img)
			{

				answers[0] = 0;


				answers[1] = 1;


				answer = complex<double>(answers[0], 0);
			}
		}
		else if (a == 1)
		{
			m = b;
			l = c;



			sum1 = bmn[l];

			sum2 = 0;
			for (j = 0; j < M; j++)
			{
				sum2 += bm2W1[m][j] * bmn[j];
			}
			sum2 += bmh1[m];

			answers[0] = bm2W2[n1real][m] * sum1 / pow(cosh(sum2), 2);


			sum1 = bmn[l];

			sum2 = 0;
			for (j = 0; j < M; j++)
			{
				sum2 += bm2W1[m][j] * bmn[j];
			}
			sum2 += bmh1[m];

			answers[1] = 0;// bm2W2[n2img][m] * sum1 / pow(cosh(sum2), 2);

			answer = complex<double>(answers[0], 0);
		}
		else if (a == 3)
		{
			m = b;




			sum1 = 1;


			sum2 = 0;
			for (j = 0; j < M; j++)
			{
				sum2 += bm2W1[m][j] * bmn[j];
			}
			sum2 += bmh1[m];

			answers[0] = bm2W2[n1real][m] * sum1 / pow(cosh(sum2), 2);




			sum1 = 1;



			sum2 = 0;
			for (j = 0; j < M; j++)
			{
				sum2 += bm2W1[m][j] * bmn[j];
			}
			sum2 += bmh1[m];

			answers[1] = 0;// bm2W2[n2img][m] * sum1 / pow(cosh(sum2), 2);

			answer = complex<double>(answers[0], 0);
		}
	}
	else if (frag == excited)
	{
		if (a == 2)
		{
			m = c;
			if (b == n1real)
			{




				sum1 = 0;
				for (j = 0; j < M; j++)
				{
					sum1 += ebm2W1[m][j] * bmn[j];
				}
				sum1 += ebmh1[m];
				sum2 = tanh(sum1);

				answers[0] = sum2;




				answers[1] = 0;

				answer = complex<double>(answers[0], 0);
			}
			else if (b == n2img)
			{

				answers[0] = 0;






				sum1 = 0;
				for (j = 0; j < M; j++)
				{
					sum1 += ebm2W1[m][j] * bmn[j];
				}
				sum1 += ebmh1[m];
				sum2 = tanh(sum1);

				answers[1] = sum2;



				answer = complex<double>(answers[0], 0);
			}
		}
		else if (a == 4)
		{
			if (b == n1real)
			{

				answers[0] = 1;


				answers[1] = 0;
				answer = complex<double>(answers[0], 0);
			}
			else if (b == n2img)
			{

				answers[0] = 0;


				answers[1] = 1;


				answer = complex<double>(answers[0], 0);
			}
		}
		else if (a == 1)
		{
			m = b;
			l = c;



			sum1 = bmn[l];

			sum2 = 0;
			for (j = 0; j < M; j++)
			{
				sum2 += ebm2W1[m][j] * bmn[j];
			}
			sum2 += ebmh1[m];

			answers[0] = ebm2W2[n1real][m] * sum1 / pow(cosh(sum2), 2);


			sum1 = bmn[l];

			sum2 = 0;
			for (j = 0; j < M; j++)
			{
				sum2 += ebm2W1[m][j] * bmn[j];
			}
			sum2 += ebmh1[m];

			answers[1] = ebm2W2[n2img][m] * sum1 / pow(cosh(sum2), 2);

			answer = complex<double>(answers[0], 0);
		}
		else if (a == 3)
		{
			m = b;




			sum1 = 1;


			sum2 = 0;
			for (j = 0; j < M; j++)
			{
				sum2 += ebm2W1[m][j] * bmn[j];
			}
			sum2 += ebmh1[m];

			answers[0] = ebm2W2[n1real][m] * sum1 / pow(cosh(sum2), 2);




			sum1 = 1;



			sum2 = 0;
			for (j = 0; j < M; j++)
			{
				sum2 += ebm2W1[m][j] * bmn[j];
			}
			sum2 += ebmh1[m];

			answers[1] = ebm2W2[n2img][m] * sum1 / pow(cosh(sum2), 2);

			answer = complex<double>(answers[0], 0);
		}
	}

	return answer;
}