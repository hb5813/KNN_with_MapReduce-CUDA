#include <iostream>
#include <fstream>
#include <string>
#include <cstdlib>
using namespace std;

#define MAX_NUM 10 // �̶�һ��������������ֵ���������۲��

void data_gen(int m, int n, int k)
{
    ofstream fout;
    float *data;

    // ���ɲ�ѯ��
    data = new float[m * k];
    for(int i = 0; i < m * k; i++)
        data[i] = float(rand() % MAX_NUM);
    fout.open("search_point.bin", ios::binary);
    fout.write((const char *)data, m * k * sizeof(float));
    fout.close();
    delete[] data;

    // ���ɲο���
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
    srand(1000); // �̶�һ�����ӣ����ڵ���

    ifstream fin("input.txt");
    int m, n, k;
    fin >> m >> n >> k;
    fin.close();
    cout << m  << ' ' << n << ' ' << k << endl;

    data_gen(m, n, k);

    return 0;
}