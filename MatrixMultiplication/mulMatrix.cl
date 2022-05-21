// TODO: Add OpenCL kernel code here.

__kernel void mulMatrix(__global int *A, __global int *B, __global int *C, int N, int M, int P)
{
	int i = get_global_id(0);
	int j = get_global_id(1);
	int acc = 0;

	for (int k = 0 ; k < P; k++)
	{
		acc += A[i * P + k] * B[k * P + j];
	}
	C[i * N + j] = acc;
}