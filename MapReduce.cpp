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

#define MAP_BLOCK_SIZE 256 // ָ��mapÿ�η��������������а�����������ѯ����

class MyMapper : public HadoopPipes::Mapper {
public:
    MyMapper(HadoopPipes::TaskContext &context) {}

    void map(HadoopPipes::MapContext& context)
    {
        // �����ѯ����Ŀ
        ifstream fin("input.txt");
        int m;
        fin >> m;
        fin.close();

        // �����ѯ��
        for(int i = 0; i < m / MAP_BLOCK_SIZE; i++)
            context.emit(to_string(i), "");
    }
};

class MyReducer : public HadoopPipes::Reducer {
public:
    MyReducer(HadoopPipes::TaskContext &context) {}

    void reduce(HadoopPipes::ReduceContext &context)
    {
        // �Լ������䵽����������
        int map_block_num = stoi(context.getInputKey());

        // �����������ż���ó��Լ������������ѯ�㼯����ʼ���
        int head_index = map_block_num * MAP_BLOCK_SIZE;

        // ���ڼ�¼����ڵ����
        int nearest_neighbor[MAP_BLOCK_SIZE];

        // ����CUDA���㵱ǰ�������������ѯ��������
        cuda_callback(head_index, MAP_BLOCK_SIZE, nearest_neighbor);

        // ��MAP_BLOCK_SIZE��(��ѯ�����, ��������)д��
        for(int i = 0; j = head_index; i < MAP_BLOCK_SIZE; i++, j++)
            context.emit(to_string(j), to_string(nearest_neighbor[i]));
    }
};

int main(int argc, char *argv[])
{
    return HadoopPipes::runTask(HadoopPipes::TemplateFactory<MyMapper, MyReducer>());  
}