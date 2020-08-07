#include <iostream>
#include <fstream>
#include <string>
#include <cstdlib>
using namespace std;

#define MAX_NUM 10 // 固定一个坐标分量的最大值，便于人眼查错

void data_gen(int m, int n, int k)
{
    ofstream fout;
    float *data;

    // 生成查询点
    data = new float[m * k];
    for(int i = 0; i < m * k; i++)
        data[i] = float(rand() % MAX_NUM);
    fout.open("search_point.bin", ios::binary);
    fout.write((const char *)data, m * k * sizeof(float));
    fout.close();
    delete[] data;

    // 生成参考点
    data = new float[n * k];
    for(int i = 0; i < n * k; i++)
        data[i] = float(rand() % MAX_NUM);
    fout.open("reference_point.bin", ios::binary);
    fout.write((const char *)data, n * k * sizeof(float));
    fout.close();
    delete[] data;
}

int main()
{
    srand(1000); // 固定一个种子，便于调试

    ifstream fin("input.txt");
    int m, n, k;
    fin >> m >> n >> k;
    fin.close();
    cout << m  << ' ' << n << ' ' << k << endl;

    data_gen(m, n, k);

    return 0;
}