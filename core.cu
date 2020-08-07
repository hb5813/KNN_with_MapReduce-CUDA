#include <fstream>
using namespace std;

#define FLOAT_MAX 0x7FFFFFFF
#define REDUCE_BLOCK_WIDTH 256
#define BLOCK_WIDTH 16

// �������ѯ������ο����ľ��루�����ڴ��Ż��㷨��
//     ���룺
//         k��ά�ȣ�m����ѯ����Ŀ��n���ο�����Ŀ��
//         searchPoints[m]����ѯ�����飻referencePoints[n]���ο�������
//     �����
//         res[m][n]����ѯ������ο����ľ�������
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

// ��Լ�������е���Сֵ�±ꡣ
// ���ڴӸ���ѯ������ο����ľ���������ȡ��������±ꡣ
//     ���룺
//         data��������Сֵ�±�ģ���ά�����飻dsize������߶ȣ�
//         block_vals�����ڴ�Ÿ�block�й�Լֵ�����飻block_idxs�����ڴ�Ÿ�block�й�Լ�õ����±�����飻
//         pitch���ڴ����Ŀ�ȣ�block_count��ͬһ���е���ĳ����block��������ʵ���ź�����
//     �����
//         result[dsize]�����������СԪ�ص��±�
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

    // ��ȫ���ڴ������Լ�����block����������ж�Ӧ�Լ�����ŵ�Ԫ�ص���Сֵ
    for(int step = blockDim.x * gridDim.x; x < dsize; x += step)
    {
        if (my_data[x] < my_val)
        {
            my_val = my_data[x];
            my_idx = x;
        }
    }

    // ���Լ��ҵ��ĵ�ǰ��Сֵ�����������ڴ�
    s_vals[threadIdx.x] = my_val;
    s_idxs[threadIdx.x] = my_idx;
    __syncthreads();

    // �ֶ�ѭ��չ������block�ڲ��ҵ�����Сֵ���й�Լ
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

    if (threadIdx.x < 32) // 32���߳����ڵ�ʱ����ڲ�����Ҫif�ж�16��8��4��2��1�ˣ�������һЩ�߳������ù�
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

    // block���Լ
    if (!threadIdx.x)
    {
        my_block_vals[blockIdx.x] = s_vals[0];
        my_block_idxs[blockIdx.x] = s_idxs[0];
        if (atomicAdd(my_block_count, 1) == gridDim.x - 1) // �൱���ź���
            last_block = 1;
    }
    __syncthreads();
    
    // ���һ���������Ŀ�ִ��block���Լ����
    if (last_block)
    {
        // ���ﲻ��Ҫ����my_val��my_idx����Ϊ�����ǰ��������У����߳�δ����ǰ��������
        // ��ômy_val��my_idxһ�������ֵ����my_val = FLOAT_MAX��my_idx = -1��
        // ���򣬸��߳���ǰ����������ҵ����Լ�������������Сֵ�Ͷ�Ӧ�±꣬������ʵ���ڵġ�
        for(x = threadIdx.x; x < gridDim.x; x += blockDim.x)
        {
            if (my_block_vals[x] < my_val)
            {
                my_val = my_block_vals[x];
                my_idx = my_block_idxs[x];
            }
        }
        
        // ��block���Լ��������������ڴ�
        s_vals[threadIdx.x] = my_val;
        s_idxs[threadIdx.x] = my_idx;
        __syncthreads();

        // �ֶ�ѭ��չ��
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

        if (threadIdx.x < 32) // 32���߳����ڵ�ʱ��Ͳ���Ҫif�ˣ�������һЩ�߳������ù�
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

// ���룺
//     head_index���Լ������������ѯ�㼯����ʼ���
//     map_block_size��������а�����������ѯ����
// �����
//     results[m]����ѯ������������
void cuda_callback(int head_index, int map_block_size, int *results)
{
    // �����ѯ����Ŀ���ο�����Ŀ�Ϳռ�ά��
    ifstream fin("input.txt");
    int m, n, k;
    fin >> m >> n >> k;
    fin.close();

    float *searchPoints = new float[map_block_size * k];
    float *referencePoints = new float[n * k];

    // �����ѯ������
    fin.open("search_point.bin", ios::binary);
    fin.seekg(head_index * k * sizeof(float)); // �ļ�ָ���ƶ����Լ������䵽������鴦
    fin.read((char *)searchPoints, map_block_size * k * sizeof(float));
    fin.close();

    // ����ο�������
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

    size_t pitch; // ��Ϊsizeof(float) == sizeof(int)�����Կ��Թ���һ��pitch
    float *d_block_vals;
    int *d_block_idxs;
    cudaMallocPitch(&d_block_vals, &pitch, grid_width * sizeof(float), map_block_size);
    cudaMallocPitch(&d_block_idxs, &pitch, grid_width * sizeof(int), map_block_size);
    pitch /= sizeof(int); // pitchĬ�ϵ�λ��sizeof(char)����Ϊsizeof(int)

    int *d_block_count;
    cudaMalloc(&d_block_count, map_block_size * sizeof(float));
    cudaMemset(d_block_count, 0, map_block_size * sizeof(float));
    // �����Լ���д��kernel�����ѯ��������
    min_index<<<dim3(grid_width, map_block_size), REDUCE_BLOCK_WIDTH>>>(d_dis, n, d_res, d_block_vals, d_block_idxs, pitch, d_block_count);

    cudaMemcpy(results, d_res, map_block_size * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(d_block_count);
    cudaFree(d_block_idxs);
    cudaFree(d_block_vals);

    cudaFree(d_res);
    cudaFree(d_dis);
}