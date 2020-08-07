#include <fstream>
using namespace std;

#define FLOAT_MAX 0x7FFFFFFF
#define REDUCE_BLOCK_WIDTH 256
#define BLOCK_WIDTH 16

// 计算各查询点与各参考点间的距离（共享内存优化算法）
//     输入：
//         k：维度；m：查询点数目；n：参考点数目；
//         searchPoints[m]：查询点数组；referencePoints[n]：参考点数组
//     输出：
//         res[m][n]各查询点与各参考点间的距离数组
__global__ void cuda_calc_distance(int k, int m, int n, float *searchPoints, float *referencePoints, float *res)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    __shared__ volatile float s_A[BLOCK_WIDTH][BLOCK_WIDTH];
    __shared__ volatile float s_B[BLOCK_WIDTH][BLOCK_WIDTH];

    float square_sum = 0.0f;
    for(int i = 0; i < k; i += BLOCK_WIDTH)
    {
        s_A[threadIdx.y][threadIdx.x] = (threadIdx.x + i < k && y < m) ?
            searchPoints[y * k + (threadIdx.x + i)] : 0.0f;
        
        s_B[threadIdx.y][threadIdx.x] = (threadIdx.y + i < k && x < n) ?
            referencePoints[x * k + (threadIdx.y + i)] : 0.0f;
        __syncthreads();
        
        for(int j = 0; j < BLOCK_WIDTH; j++)
        {
            float diff = s_A[threadIdx.y][j] - s_B[j][threadIdx.x];
            square_sum += diff * diff;
        }
        __syncthreads();
    }

    if(x < n && y < m)
        res[y * n + x] = square_sum;
}

// 规约求数组中的最小值下标。
// 用于从各查询点与各参考点间的距离数组中取得最近邻下标。
//     输入：
//         data：待求最小值下标的（二维）数组；dsize：数组高度；
//         block_vals：用于存放各block中规约值的数组；block_idxs：用于存放各block中规约得到的下标的数组；
//         pitch：内存对齐的宽度；block_count：同一行中到达某语句的block数，用于实现信号量。
//     输出：
//         result[dsize]数组各行中最小元素的下标
__global__ void min_index(const float *data, const int dsize, int *result, float *block_vals, int *block_idxs, size_t pitch, int *block_count)
{
    __shared__ volatile float s_vals[REDUCE_BLOCK_WIDTH];
    __shared__ volatile int s_idxs[REDUCE_BLOCK_WIDTH];
    __shared__ volatile int last_block;
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;

    const float *my_data = &data[y * dsize];
    float *my_block_vals = &block_vals[y * pitch];
    int *my_block_idxs = &block_idxs[y * pitch];
    int *my_block_count = &block_count[y];

    last_block = 0;
    float my_val = FLOAT_MAX;
    int my_idx = -1;

    // 从全局内存中找自己所在block负责的区域中对应自己的序号的元素的最小值
    for(int step = blockDim.x * gridDim.x; x < dsize; x += step)
    {
        if (my_data[x] < my_val)
        {
            my_val = my_data[x];
            my_idx = x;
        }
    }

    // 将自己找到的当前最小值加载至共享内存
    s_vals[threadIdx.x] = my_val;
    s_idxs[threadIdx.x] = my_idx;
    __syncthreads();

    // 手动循环展开，对block内部找到的最小值进行规约
    if (threadIdx.x < 128)
        if (s_vals[threadIdx.x] > s_vals[threadIdx.x + 128])
        {
            s_vals[threadIdx.x] = s_vals[threadIdx.x + 128];
            s_idxs[threadIdx.x] = s_idxs[threadIdx.x + 128];
        }
    __syncthreads();

    if (threadIdx.x < 64)
        if (s_vals[threadIdx.x] > s_vals[threadIdx.x + 64])
        {
            s_vals[threadIdx.x] = s_vals[threadIdx.x + 64];
            s_idxs[threadIdx.x] = s_idxs[threadIdx.x + 64];
        }
    __syncthreads();

    if (threadIdx.x < 32) // 32个线程以内的时候就内部不需要if判断16、8、4、2、1了，可以让一些线程做无用功
    {
        if (s_vals[threadIdx.x] > s_vals[threadIdx.x + 32])
        {
            s_vals[threadIdx.x] = s_vals[threadIdx.x + 32];
            s_idxs[threadIdx.x] = s_idxs[threadIdx.x + 32];
        }

        if (s_vals[threadIdx.x] > s_vals[threadIdx.x + 16])
        {
            s_vals[threadIdx.x] = s_vals[threadIdx.x + 16];
            s_idxs[threadIdx.x] = s_idxs[threadIdx.x + 16];
        }

        if (s_vals[threadIdx.x] > s_vals[threadIdx.x + 8])
        {
            s_vals[threadIdx.x] = s_vals[threadIdx.x + 8];
            s_idxs[threadIdx.x] = s_idxs[threadIdx.x + 8];
        }

        if (s_vals[threadIdx.x] > s_vals[threadIdx.x + 4])
        {
            s_vals[threadIdx.x] = s_vals[threadIdx.x + 4];
            s_idxs[threadIdx.x] = s_idxs[threadIdx.x + 4];
        }

        if (s_vals[threadIdx.x] > s_vals[threadIdx.x + 2])
        {
            s_vals[threadIdx.x] = s_vals[threadIdx.x + 2];
            s_idxs[threadIdx.x] = s_idxs[threadIdx.x + 2];
        }

        if (s_vals[threadIdx.x] > s_vals[threadIdx.x + 1])
        {
            s_vals[threadIdx.x] = s_vals[threadIdx.x + 1];
            s_idxs[threadIdx.x] = s_idxs[threadIdx.x + 1];
        }
    }

    // block间规约
    if (!threadIdx.x)
    {
        my_block_vals[blockIdx.x] = s_vals[0];
        my_block_idxs[blockIdx.x] = s_idxs[0];
        if (atomicAdd(my_block_count, 1) == gridDim.x - 1) // 相当于信号量
            last_block = 1;
    }
    __syncthreads();
    
    // 最后一个完成任务的块执行block间规约任务
    if (last_block)
    {
        // 这里不需要重置my_val和my_idx，因为如果在前面的任务中，该线程未参与前述动作，
        // 那么my_val和my_idx一定是最初值，即my_val = FLOAT_MAX，my_idx = -1；
        // 否则，该线程在前面的任务中找到了自己负责的区域的最小值和对应下标，这是真实存在的。
        for(x = threadIdx.x; x < gridDim.x; x += blockDim.x)
        {
            if (my_block_vals[x] < my_val)
            {
                my_val = my_block_vals[x];
                my_idx = my_block_idxs[x];
            }
        }
        
        // 将block间规约结果加载至共享内存
        s_vals[threadIdx.x] = my_val;
        s_idxs[threadIdx.x] = my_idx;
        __syncthreads();

        // 手动循环展开
        if (threadIdx.x < 128)
            if (s_vals[threadIdx.x] > s_vals[threadIdx.x + 128])
            {
                s_vals[threadIdx.x] = s_vals[threadIdx.x + 128];
                s_idxs[threadIdx.x] = s_idxs[threadIdx.x + 128];
            }
        __syncthreads();

        if (threadIdx.x < 64)
            if (s_vals[threadIdx.x] > s_vals[threadIdx.x + 64])
            {
                s_vals[threadIdx.x] = s_vals[threadIdx.x + 64];
                s_idxs[threadIdx.x] = s_idxs[threadIdx.x + 64];
            }
        __syncthreads();

        if (threadIdx.x < 32) // 32个线程以内的时候就不需要if了，可以让一些线程做无用功
        {
            if (s_vals[threadIdx.x] > s_vals[threadIdx.x + 32])
            {
                s_vals[threadIdx.x] = s_vals[threadIdx.x + 32];
                s_idxs[threadIdx.x] = s_idxs[threadIdx.x + 32];
            }

            if (s_vals[threadIdx.x] > s_vals[threadIdx.x + 16])
            {
                s_vals[threadIdx.x] = s_vals[threadIdx.x + 16];
                s_idxs[threadIdx.x] = s_idxs[threadIdx.x + 16];
            }

            if (s_vals[threadIdx.x] > s_vals[threadIdx.x + 8])
            {
                s_vals[threadIdx.x] = s_vals[threadIdx.x + 8];
                s_idxs[threadIdx.x] = s_idxs[threadIdx.x + 8];
            }

            if (s_vals[threadIdx.x] > s_vals[threadIdx.x + 4])
            {
                s_vals[threadIdx.x] = s_vals[threadIdx.x + 4];
                s_idxs[threadIdx.x] = s_idxs[threadIdx.x + 4];
            }

            if (s_vals[threadIdx.x] > s_vals[threadIdx.x + 2])
            {
                s_vals[threadIdx.x] = s_vals[threadIdx.x + 2];
                s_idxs[threadIdx.x] = s_idxs[threadIdx.x + 2];
            }

            if (s_vals[threadIdx.x] > s_vals[threadIdx.x + 1])
            {
                s_vals[threadIdx.x] = s_vals[threadIdx.x + 1];
                s_idxs[threadIdx.x] = s_idxs[threadIdx.x + 1];
            }
        }

        if (!threadIdx.x)
            result[y] = s_idxs[0];
    }
}

inline int divup(int a, int b)
{
    return (a + b - 1) / b;
}

// 输入：
//     head_index：自己负责的连续查询点集的起始序号
//     map_block_size：任务块中包含的连续查询点数
// 输出：
//     results[m]各查询点的最近邻数组
void cuda_callback(int head_index, int map_block_size, int *results)
{
    // 读入查询点数目、参考点数目和空间维度
    ifstream fin("input.txt");
    int m, n, k;
    fin >> m >> n >> k;
    fin.close();

    float *searchPoints = new float[map_block_size * k];
    float *referencePoints = new float[n * k];

    // 读入查询点数据
    fin.open("search_point.bin", ios::binary);
    fin.seekg(head_index * k * sizeof(float)); // 文件指针移动到自己被分配到的任务块处
    fin.read((char *)searchPoints, map_block_size * k * sizeof(float));
    fin.close();

    // 读入参考点数据
    fin.open("reference_points.bin", ios::binary);
    fin.read((char *)referencePoints, n * k * sizeof(float));
    fin.close();

    float *d_searchPoints, *d_referencePoints, *d_dis;

    // Allocate device memory and copy data from host to device
    cudaMalloc((void **)&d_searchPoints, map_block_size * k * sizeof(float));
    cudaMalloc((void **)&d_referencePoints, n * k * sizeof(float));
    cudaMalloc((void **)&d_dis, map_block_size * n * sizeof(float));
    cudaMemcpy(d_searchPoints, searchPoints, map_block_size * k * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_referencePoints, referencePoints, n * k * sizeof(float), cudaMemcpyHostToDevice);

    dim3 block(BLOCK_WIDTH, BLOCK_WIDTH);
    dim3 grid(divup(n, BLOCK_WIDTH), divup(map_block_size, BLOCK_WIDTH));
    cuda_calc_distance<<<grid, block>>>(k, map_block_size, n, d_searchPoints, d_referencePoints, d_dis);
    delete[] searchPoints;
    delete[] referencePoints;
    
    cudaDeviceSynchronize();

    cudaFree(d_referencePoints);
    cudaFree(d_searchPoints);

    int *d_res;
    cudaMalloc((void **)&d_res, map_block_size * sizeof(int));

    int grid_width = (n + (REDUCE_BLOCK_WIDTH - 1)) / REDUCE_BLOCK_WIDTH;

    size_t pitch; // 因为sizeof(float) == sizeof(int)，所以可以共用一个pitch
    float *d_block_vals;
    int *d_block_idxs;
    cudaMallocPitch(&d_block_vals, &pitch, grid_width * sizeof(float), map_block_size);
    cudaMallocPitch(&d_block_idxs, &pitch, grid_width * sizeof(int), map_block_size);
    pitch /= sizeof(int); // pitch默认单位是sizeof(char)，化为sizeof(int)

    int *d_block_count;
    cudaMalloc(&d_block_count, map_block_size * sizeof(float));
    cudaMemset(d_block_count, 0, map_block_size * sizeof(float));
    // 利用自己编写的kernel求各查询点的最近邻
    min_index<<<dim3(grid_width, map_block_size), REDUCE_BLOCK_WIDTH>>>(d_dis, n, d_res, d_block_vals, d_block_idxs, pitch, d_block_count);

    cudaMemcpy(results, d_res, map_block_size * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(d_block_count);
    cudaFree(d_block_idxs);
    cudaFree(d_block_vals);

    cudaFree(d_res);
    cudaFree(d_dis);
}