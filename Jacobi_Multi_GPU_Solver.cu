//////////////////////////////////////////////////////////////////////////
////This is code implements Jacobi Solver optimized for Quad GPU settings.
////Auhor: Anmol Chachra
////License: GNU General Public License v3.0
////Email: anmol.chachra@gmail.com
//////////////////////////////////////////////////////////////////////////

#include <iostream>
#include <fstream>
using namespace std;

////Sample Problem By Bo Zhu (bo.zhu@dartmouth.edu)
////Macros for sample problems
const int n=8;							////Grid size
const bool verbose=true;				////set false to turn off print for x and residual
const double tolerance=1e-3;			////Tolerance for the iterative solver

////Macros for multi-gpu version
const int num_gpus=4;                                   ////Number of GPUs
const int p=1;                                          ////Padding size
const int segment_n = n/2;                              ////segment size for each axis
const int sn = (segment_n + p*2) * (segment_n + p*2);   ////padded array size on a single GPU
#define GI(i,j) (i+p)*(segment_n + 2*p) + (j+p)         ////2D coordinate -> array index
#define GB(i,j) i<0||i>=segment_n||j<0||j>=segment_n    ////Check Boundary


//////////////////////////////////////////////////////////////////////////////////////////////
////Kernels and Device Functions
//////////////////////////////////////////////////////////////////////////////////////////////

////Code uses atomic add to calculate residue.
////Ref: https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html
////Note that any atomic operation can be implemented based on atomicCAS() (Compare And Swap).
////For example, atomicAdd() for double-precision floating-point numbers is not available on devices
////with compute capability lower than 6.0 but it can be implemented as follows:
#if __CUDA_ARCH__ < 600
__device__ double atomicCustomAdd(double* address, double val)
{
    unsigned long long int* address_as_ull =
                              (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;

    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
                        __double_as_longlong(val +
                               __longlong_as_double(assumed)));

    // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
    } while (assumed != old);

    return __longlong_as_double(old);
}
#endif

__global__ void Jacobi_GPU_Poorman(double* x, double* b, double* res)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int j = blockDim.y * blockIdx.y +  threadIdx.y;
    res[GI(i,j)] = (b[GI(i,j)]+x[GI(i-1,j)]+x[GI(i+1,j)]+x[GI(i,j-1)]+x[GI(i,j+1)])/4.0;
}

__global__ void Jacobi_Residue_GPU_Poorman(double* x, double* b, double* residual)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int j = blockDim.y * blockIdx.y +  threadIdx.y;

    double residue = pow(4.0*x[GI(i,j)]-x[GI(i-1,j)]-x[GI(i+1,j)]-x[GI(i,j-1)]-x[GI(i,j+1)]-b[GI(i,j)],2);
    atomicCustomAdd(residual, residue);
}

ofstream out;

//////////////////////////////////////////////////////////////////////////////////////////////
////Host Functions
//////////////////////////////////////////////////////////////////////////////////////////////

void JacobiMultiGPUSolver()
{
    double* h_x[num_gpus];
    double* h_b[num_gpus];

    double residual[num_gpus];
    double total_residual = 0.0;

    ////Initialize the data to be sent on GPUs - x and b
    for(int j=0; j<num_gpus; j++)
    {
        h_x[j] = (double*)malloc(sn*sizeof(double)); //sn is the padded array size on a single GPU - (segment_n + p*2) * (segment_n + p*2);
        memset(h_x[j],0x0000,sizeof(double)*sn);

        h_b[j] = (double*)malloc(sn*sizeof(double));
        memset(h_b[j],0x0000,sizeof(double)*sn);

        residual[j] = 0.0;
    }

    for(int j=0; j<num_gpus; j++)
    {
        for(int i=-1; i<=segment_n; i++)
        {
            for(int k=-1; k<=segment_n; k++)
            {
                h_b[j][GI(i,k)]=4.0; //Another way to fill data - 2D approach
            }
        }
    }

    ////Setting boundary condition for x
    ////Since our boundary conditions depends upon the full scaled indices - we need to create a mapping
    ////Similar to mapping we use in device functions [threads, blocks, grids] -- [idx, dim]
    for(int j=0; j<num_gpus; j++)
    {
        int col = j%2;
        int row = j/2;

        for(int i=-1;i<=segment_n;i++){
            for(int k=-1;k<=segment_n;k++){
                if(GB(i,k))
                {
                    int row_idx = row * (segment_n) + i; ////segment_n is the x/y axis size of the matrix on single GPU / dimension?
                    int col_idx = col * (segment_n) + k;
                    h_x[j][GI(i,k)]=(double)(row_idx*row_idx + col_idx*col_idx);	////set boundary condition for x
                }
            }
        }
    }


	double* d_x[num_gpus][2];                   ////maintaining a read write buffer for x on every gpu
	double* d_b[num_gpus];                      ////device buffer for b on every gpu
	double* d_residual[num_gpus];               ////residual calculation on every gpu

    int blockSize = segment_n / 8;              ////Increase accordingly, for grid 256, blockSize = 16 or 256 threads/block
    int iter_num = -1;                          ////One extra iteration for now
    int max_num = 1000;                         ////Maximum number of iterations to perform

    dim3 block_dim = dim3(blockSize, blockSize, 1);
    dim3 grid_dim = dim3(segment_n/blockSize, segment_n/blockSize);

    ////We need to create 4 events to get the correct timing as one may finish before other
    cudaEvent_t startA, startB, startC, startD, endA, endB, endC, endD;
    float gpu_time[num_gpus];

    ////No easier way to do this? cudaEvent_t does not allow array declaration :/
    cudaSetDevice(0);
    cudaEventCreate(&startA);
    cudaEventCreate(&endA);
    gpu_time[0] = 0.0f;
    cudaDeviceSynchronize();
    cudaEventRecord(startA);

    cudaSetDevice(1);
    cudaEventCreate(&startB);
    cudaEventCreate(&endB);
    gpu_time[1] = 0.0f;
    cudaDeviceSynchronize();
    cudaEventRecord(startB);

    cudaSetDevice(2);
    cudaEventCreate(&startC);
    cudaEventCreate(&endC);
    gpu_time[2] = 0.0f;
    cudaDeviceSynchronize();
    cudaEventRecord(startC);

    cudaSetDevice(3);
    cudaEventCreate(&startD);
    cudaEventCreate(&endD);
    gpu_time[3] = 0.0f;
    cudaDeviceSynchronize();
    cudaEventRecord(startD);

    ////Initializing the memory on all GPUS
    for(int j=0; j<num_gpus; j++)
    {
        cudaSetDevice(j);
        for(int i=0; i<2; i++) //read-write buffers for x
        {
            cudaMallocHost((void**)&d_x[j][i], sn*sizeof(double));
            cudaMemset(d_x[j][i], 0, sn*sizeof(double));
        }
        cudaMallocHost((void**)&d_b[j], sn*sizeof(double));
        cudaMemset(d_b[j], 0, sn*sizeof(double));
        cudaMallocHost((void**)&d_residual[j], sizeof(double));
    }

    //Copy data on all of the gpus
    for(int j=0; j<num_gpus; j++)
    {
        cudaSetDevice(j);
        for(int i=0; i<2; i++) //read-write buffers for x
        {
            cudaMemcpyAsync(d_x[j][i], h_x[j], sn*sizeof(double), cudaMemcpyHostToDevice);
        }
        cudaMemcpyAsync(d_b[j], h_b[j], sn*sizeof(double), cudaMemcpyHostToDevice);
    }

    //Running the iterations
    do
    {
        iter_num++;

        ////Reset residual
        total_residual = 0.0;
        for(int j=0; j<num_gpus; j++)
        {
            residual[j] = 0.0;
        }

        for(int j=0; j<num_gpus; j++)
        {
            cudaSetDevice(j);
            cudaMemcpyAsync(d_residual[j], &residual[j], sizeof(double), cudaMemcpyHostToDevice);
        }

//        //Synchronize data on all GPUs
//        for(int j=0; j<num_gpus; j++)
//        {
//            cudaSetDevice(j);
//            cudaDeviceSynchronize();
//        }

        ////Initializing "Halos"
        ////Exchange data on the boundary region

        for(int j=0; j<num_gpus-1; j++)
        {
            int boundary_idx = segment_n - 1;
            int boundary_r_idx = 0;
            int boundary_t_idx = 0;

            int halo_idx = segment_n + p;
            int halo_r_idx = 0;
            int halo_b_idx = (segment_n + p)*(segment_n + 2*p);
            int halo_t_idx = 0;

            ////Filling left-right halos
            int col=j%2;
            if(col == 0)
            {
                for(int i=-1; i<=segment_n; i++)
                {
                    cudaMemcpyAsync(d_x[j+1][0]+halo_r_idx + (i+1)*(segment_n + 2*p), d_x[j][0] + GI(i, boundary_idx), sizeof(double), cudaMemcpyDeviceToDevice);
                    cudaMemcpyAsync(d_x[j][0]+halo_idx + (i+1)*(segment_n + 2*p), d_x[j+1][0] + GI(i, boundary_r_idx), sizeof(double), cudaMemcpyDeviceToDevice);
                }
            }

            if(j<2)
            {
                cudaMemcpyAsync(d_x[j+2][0] + halo_t_idx, d_x[j][0] + GI(boundary_idx, -1), (segment_n + 2*p) * sizeof(double), cudaMemcpyDeviceToDevice);
                cudaMemcpyAsync(d_x[j][0] + halo_b_idx, d_x[j+2][0] + GI(boundary_t_idx, -1), (segment_n + 2*p) * sizeof(double), cudaMemcpyDeviceToDevice);
            }
        }

        ////Synchronize before the next step
        for(int j=0; j<num_gpus; j++)
        {
            cudaSetDevice(j);
            cudaDeviceSynchronize();
        }

        ////Compute the residual on every GPU
        ////Why compute residual first?
        for(int j=0; j<num_gpus; j++)
        {
            cudaSetDevice(j);
            Jacobi_Residue_GPU_Poorman<<<grid_dim, block_dim>>>(d_x[j][0], d_b[j], d_residual[j]);
            cudaMemcpyAsync(&residual[j], d_residual[j], sizeof(double), cudaMemcpyDeviceToHost);
        }

        ////Synchronize across all devices and add residue in total
        for(int j=0; j<num_gpus; j++)
        {
            cudaSetDevice(j);
            cudaDeviceSynchronize();
            total_residual+=residual[j];
        }

        if(total_residual <= tolerance)
        {
            break;
        }

        ////Compute halos for the next iteration
        for(int j=0; j<num_gpus; j++)
        {
            cudaSetDevice(j);
            Jacobi_GPU_Poorman<<<grid_dim, block_dim>>>(d_x[j][0], d_b[j], d_x[j][1]);
            cudaMemcpyAsync(d_x[j][0], d_x[j][1], sn*sizeof(double), cudaMemcpyDeviceToDevice);
        }

//        ////Synchronize across devices to make sure every device is finished.
//        for(int j=0; j<num_gpus; j++)
//        {
//            cudaSetDevice(j);
//            cudaDeviceSynchronize();
//        }

        if(verbose)cout<<"res: "<<total_residual<<endl;
    }
    while(iter_num!=max_num);

    float total_time = 0.0f;
    cudaSetDevice(0);
    cudaEventRecord(endA);
    cudaEventSynchronize(endA);
    cudaEventElapsedTime(&gpu_time[0], startA, endA);
    cudaEventDestroy(startA);
    cudaEventDestroy(endA);

    cudaSetDevice(1);
    cudaEventRecord(endB);
    cudaEventSynchronize(endB);
    cudaEventElapsedTime(&gpu_time[1], startB, endB);
    cudaEventDestroy(startB);
    cudaEventDestroy(endB);

    cudaSetDevice(2);
    cudaEventRecord(endC);
    cudaEventSynchronize(endC);
    cudaEventElapsedTime(&gpu_time[2], startC, endC);
    cudaEventDestroy(startC);
    cudaEventDestroy(endC);

    cudaSetDevice(3);
    cudaEventRecord(endD);
    cudaEventSynchronize(endD);
    cudaEventElapsedTime(&gpu_time[3], startD, endD);
    cudaEventDestroy(startD);
    cudaEventDestroy(endD);

    for(int j=0; j<num_gpus; j++)
    {
        if(gpu_time[j] > total_time)
        total_time = gpu_time[j];
    }

    if(verbose)
    {
        //Copy appropriate data on all of the gpus
        for(int j=0; j<num_gpus; j++)
        {
            cudaSetDevice(j);
            cudaMemcpyAsync(h_x[j], d_x[j][0], sn*sizeof(double), cudaMemcpyDeviceToHost);
        }
        double* x = new double[n*n];
        memset(x, 0x0000, n*n*sizeof(double));
        for(int j=0; j<num_gpus; j++)
        {
            int row = j/2;
            int col = j%2;

            for(int i=0; i<segment_n; i++)
            {
                for(int k=0; k<segment_n; k++)
                {
                    int i_idx = i + row * segment_n;
                    int j_idx = k + col * segment_n;
                    x[i_idx*n + j_idx] = h_x[j][GI(i, k)];
                }
            }
        }
        cout<<"\n\nx for your GPU solver:\n";
        for(int i=0;i<n;i++){
            for(int j=0;j<n;j++){
                int idx = i*n + j;
                cout<<x[idx]<<", ";
            }
            cout<<std::endl;
        }
    }

    printf("\nGPU runtime: %.4f ms\n",total_time);
	cout<<"Jacobi solver converges in "<<iter_num<<" iterations, with residual "<<total_residual<<endl;
	cout<<"\n\nresidual for your GPU solver: "<<total_residual<<endl;

	out<<"R0: "<<total_residual<<endl;
	out<<"T1: "<<total_time<<endl;

	for(int j=0; j<num_gpus; j++)
    {
        cudaSetDevice(j);
        for(int i=0; i<2; i++)
        {
            cudaFree(d_x[j][i]);
        }
        cudaFree(d_b[j]);
        cudaFree(d_residual[j]);
    }
}

///////////////////////////////////////////////////////////////////////////////////////////////
int main()
{
	JacobiMultiGPUSolver();
	return 0;
}
