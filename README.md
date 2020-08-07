# 课程设计：KNN的MapReduce+CUDA实现

├── MapReduce.cpp       MapReduce主程序
├── cmp.cpp             正确性验证程序
├── core.cu             核心部分CUDA
├── data_gen.cpp        数据生成器
└── input.txt           输入数据格式

## 数据生成器

采用随机数生成的方法，按照查询点向量共m行输入，第i行表示编号为(i-1)的查询点，每行共有𝑘个单精度浮点数，为查询点向量在相应维度上的坐标分量，参考点向量同理的方式，便可生成输入数据。在这里我们将查询点数据保存为二进制格式"search_point.bin"，将参考点数据保存为"reference_point.bin"。具体代码如下所示：
```cpp
void data_gen(int m, int n, int k)
{
    ofstream fout;
    float *data;

    // 生成查询点
    data = new float[m * k];
    for(int i = 0; i < m * k; i++)
        data[i] = float(rand() % MAX_NUM);
    delete[] data;
    fout.open("search_point.bin", ios::binary);
    fout.write((const char *)data, m * k * sizeof(float));
    fout.close();

    // 生成参考点
    data = new float[n * k];
    for(int i = 0; i < n * k; i++)
        data[i] = float(rand() % MAX_NUM);
    delete[] data;
    fout.open("reference_point.bin", ios::binary);
    fout.write((const char *)data, n * k * sizeof(float));
    fout.close();
}

void data_gen(int m, int n, int k)
{
    ofstream fout;

    // 生成查询点
    fout.open("search_point.txt");
    for(int i = 0; i < m; i++)
    {
        fout << i << ' ';
        for(int j = 0; j < k - 1; j++)
            fout << (rand() % MAX_NUM) << ' ';
        fout << (rand() % MAX_NUM) << endl;
    }
    fout.close();

    // 生成参考点
    fout.open("reference_point.txt");
    for(int i = 0; i < m; i++)
    {
        fout << i << ' ';
        for(int j = 0; j < k - 1; j++)
            fout << (rand() % MAX_NUM) << ' ';
        fout << (rand() % MAX_NUM) << endl;
    }
    fout.close();
}
```

## MapReduce部分

map函数中，我们只需先读入查询点的数目，然后根据MAP_BLOCK_SIZE指明的块大小，循环分配m/MAP_BLOCK_SIZE个任务到context中即可，即将第i轮循环的循环计数器值作为第i任务块的序号，将其作为key，而value留空。从而每个任务块的key-value形式即为(i, "")：
```cpp
class MyMapper : public HadoopPipes::Mapper {
public:
    MyMapper(HadoopPipes::TaskContext &context) {}

    void map(HadoopPipes::MapContext& context)
    {
        // 读入查询点数目
        ifstream fin("input.txt");
        int m;
        fin >> m;
        fin.close();

        // 分配查询点
        for(int i = 0; i < m / MAP_BLOCK_SIZE; i++)
            context.emit(to_string(i), "");
    }
};
```

## MapReduce部分

### map部分
map函数中，我们只需先读入查询点的数目，然后根据MAP_BLOCK_SIZE指明的块大小，循环分配m/MAP_BLOCK_SIZE个任务到context中即可，即将第i轮循环的循环计数器值作为第i任务块的序号，将其作为key，而value留空。从而每个任务块的key-value形式即为(i, "")：
```cpp
class MyMapper : public HadoopPipes::Mapper {
public:
    MyMapper(HadoopPipes::TaskContext &context) {}

    void map(HadoopPipes::MapContext& context)
    {
        // 读入查询点数目
        ifstream fin("input.txt");
        int m;
        fin >> m;
        fin.close();

        // 分配查询点
        for(int i = 0; i < m / MAP_BLOCK_SIZE; i++)
            context.emit(to_string(i), "");
    }
};
```

### reduce部分
在reduce函数中，只需先从context中取得自己被分配到的任务块序号，然后乘上MAP_BLOCK_SIZE，便得到了自己负责的连续查询点集的起始序号，将这些信息传给CUDA函数，等待其找出各查询点的最近邻后，将结果以“(查询点序号, 最近邻序号)”的key-value对的形式输出。
```cpp
class MyReducer : public HadoopPipes::Reducer {
public:
    MyReducer(HadoopPipes::TaskContext &context) {}

    void reduce(HadoopPipes::ReduceContext &context)
    {
        // 自己被分配到的任务块序号
        int map_block_num = stoi(context.getInputKey());

        // 根据任务块序号计算得出自己负责的连续查询点集的起始序号
        int head_index = map_block_num * MAP_BLOCK_SIZE;

        // 用于记录最近邻的序号
        int nearest_neighbor[MAP_BLOCK_SIZE];

        // 利用CUDA计算当前任务块中所含查询点的最近邻
        cuda_callback(head_index, MAP_BLOCK_SIZE, nearest_neighbor);

        // 将MAP_BLOCK_SIZE对(查询点序号, 最近邻序号)写出
        for(int i = 0; j = head_index; i < MAP_BLOCK_SIZE; i++, j++)
            context.emit(to_string(j), to_string(nearest_neighbor[i]));
    }
};
```

## CUDA部分

### CUDA回调函数
这个部分主要完成与hadoop交互，及调用另外两个部分的函数的工作。
输入：
     head_index：自己负责的连续查询点集的起始序号
     map_block_size：任务块中包含的连续查询点数
 输出：
     results[m]各查询点的最近邻数组
```cpp
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
```

### 计算各查询点与各参考点间的距离的函数
计算各查询点与各参考点间的距离（共享内存优化算法）
     输入：
         k：维度；m：查询点数目；n：参考点数目；
         searchPoints[m]：查询点数组；referencePoints[n]：参考点数组
     输出：
         res[m][n]各查询点与各参考点间的距离数组
<br/>
对于矩阵乘法的优化方法很有可能可以应用到此处，又考虑到在一个16*16的block中，256个线程使用到的数据仅为16个参考点和16个查询点的数据，那么便可以将这些重复使用的数据加载入共享内存中，减少对全局内存的访问次数。这其实相当于分块矩阵乘法的shared memory优化。这样的优化，不仅解决了全局内存中参考点的访问效率底下的问题，还提高了对查询点的访问效率，可谓是一举两得。
```cpp
__global__ void cuda_calc_distance (int k, int m, int n, float *searchPoints, float *referencePoints, float *res)
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
```


### 规约求数组中的最小值下标的函数
```cpp
// 规约求数组中的最小值下标。
// 用于从各查询点与各参考点间的距离数组中取得最近邻下标。
//     输入：
//		data：待求最小值下标的（二维）数组；
//		dsize：数组高度；
//		n：参考点数目；
//	block_vals：用于存放各block中规约值的数组；
//	block_idxs：用于存放各block中规约得到的下标的数组；
//		pitch：内存对齐的宽度；
//		block_count：同行中到达某语句的block数，用于实现信号量。
//     输出：
//         result[dsize]数组各行中最小元素的下标
__global__ void min_index(const float *data, const int dsize, int *result, float *block_vals, int *block_idxs, size_t pitch, int *block_count)
```

该函数的规约过程为经典的CUDA求和规约过程的改进，主要分为自身规约、block内部规约、block间规约三个规约步骤。
整个规约过程的主体思想如下：
	规约步骤1：自身规约

1、从全局内存中找自己所在block负责的区域中对应自己的序号的元素的最小值
```cpp
for(int step = blockDim.x * gridDim.x; x < dsize; x += step)
{
    if (my_data[x] < my_val)
    {
        my_val = my_data[x];
        my_idx = x;
    }
}
```

2、将自己找到的当前最小值加载至共享内存
```cpp
    s_vals[threadIdx.x] = my_val;
    s_idxs[threadIdx.x] = my_idx;
    __syncthreads();
```

	规约步骤2：block内部规约 
1、手动循环展开，对block内部找到的最小值进行规约
```cpp
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
```

	规约步骤3：block间规约
1、各block中的0号线程将block内部的规约结果存入全局内存中的两个变量中
```cpp
    if (!threadIdx.x)
    {
        my_block_vals[blockIdx.x] = s_vals[0];
        my_block_idxs[blockIdx.x] = s_idxs[0];
        if (atomicAdd(my_block_count, 1) == gridDim.x - 1)
            last_block = 1;
    }
    __syncthreads();
```
注意，在block间规约的同时我们使用原子加法向全局内存中的全局变量my_block_count加1，最后一个执行该加法任务的线程将shared memory中的last_block标记为1，以表明自己是同一行中达到此处的最后一个block。因而，全局变量my_block_count实际上是实现信号量用的变量。
<br/>
2、最后一个完成任务的块执行block间规约任务
```cpp
    if (last_block)
    {
        for(x = threadIdx.x; x < gridDim.x; x += blockDim.x)
        {
            if (my_block_vals[x] < my_val)
            {
                my_val = my_block_vals[x];
                my_idx = my_block_idxs[x];
            }
        }
    ......未完......
```
注意此处不需要重置my_val和my_idx，因为如果在前面的任务中，该线程未参与前述动作，那么my_val和my_idx一定是最初值，即my_val = FLOAT_MAX，my_idx = -1；否则，该线程在前面的任务中找到了自己负责的区域的最小值和对应下标，这是真实存在的。
<br/>
3、将block间规约结果加载至共享内存
```cpp
        s_vals[threadIdx.x] = my_val;
        s_idxs[threadIdx.x] = my_idx;
        __syncthreads();
```
4、手动循环展开进行规约
```cpp
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
```
5、若自己是该行中最后一个达到的block中的0号线程，则将规约结果存入全局内存中的result变量中。
```cpp
        if (!threadIdx.x)
            result[y] = s_idxs[0];
```