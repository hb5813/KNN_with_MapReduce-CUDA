#include <string>
#include <iostream>
#include <fstream>
using namespace std;

#include "core.cu"
#include "Pipes.hh"
#include "TemplateFactory.hh"
#include "StringUtils.hh"

// sudo apt install libssl1.0-dev
// nvcc -o my_MapReduce my_MapReduce.cpp -I$HADOOP_HOME/include -L$HADOOP_HOME/lib/native -lhadooppipes -lhadooputils -lpthread -lcrypto -lssl -no-pie -O3
// hadoop pipes -D hadoop.pipes.java.recordreader=true -D hadoop.pipes.java.recordwriter=true -D mapred.job.name=test -input in.txt -output out -program my_MapReduce

#define MAP_BLOCK_SIZE 256 // 指明map每次分配任务的任务块中包含的连续查询点数

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

int main(int argc, char *argv[])
{
    return HadoopPipes::runTask(HadoopPipes::TemplateFactory<MyMapper, MyReducer>());  
}