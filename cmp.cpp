#include <iostream>
#include <fstream>
using namespace std;

int m, n, k;

bool cmp(const string &search_point_file, const string &reference_point_file, const string &res_file)
{
    float *search_point = new float[m * k];
    float *reference_point = new float[n * k];

    ifstream fin;
    fin.open(search_point_file, ios::binary);
    fin.read((char *)search_point, m * k * sizeof(float));
    fin.close();

    fin.open(reference_point_file, ios::binary);
    fin.read((char *)reference_point, n * k * sizeof(float));
    fin.close();

    int minIndex;
    float minSquareSum, squareSum;

    fin.open(res_file);
    // 遍历每一个查询点
    for (int mInd = 0; mInd < m; mInd++)
    {
        minSquareSum = -1;
        // 遍历每一个参考点
        for (int nInd = 0; nInd < n; nInd++)
        {
            squareSum = 0;
            // 遍历坐标在每一维度的分量
            for (int kInd = 0; kInd < k; kInd++)
            {
                float diff = search_point[k * mInd + kInd] - reference_point[k * nInd + kInd];
                squareSum += diff * diff;
            }
            if (minSquareSum < 0 || squareSum < minSquareSum)
            {
                minSquareSum = squareSum;
                minIndex = nInd;
            }
        }

        int res_mInd, res_minIndex;
        fin >> res_mInd >> res_minIndex;
        if(res_mInd != mInd || res_minIndex != minIndex)
            return false;
    }
    fin.close();

    delete[] search_point;
    delete[] reference_point;

    return true;
}


void output(const string &search_point_file, const string &reference_point_file)
{
    float *search_point = new float[m * k];
    float *reference_point = new float[n * k];

    ifstream fin;
    fin.open(search_point_file, ios::binary);
    fin.read((char *)search_point, m * k * sizeof(float));
    fin.close();

    fin.open(reference_point_file, ios::binary);
    fin.read((char *)reference_point, n * k * sizeof(float));
    fin.close();

    int minIndex;
    float minSquareSum, squareSum;

    ofstream fout("cmp_res.txt");
    // 遍历每一个查询点
    for (int mInd = 0; mInd < m; mInd++)
    {
        minSquareSum = -1;
        // 遍历每一个参考点
        for (int nInd = 0; nInd < n; nInd++)
        {
            squareSum = 0;
            // 遍历坐标在每一维度的分量
            for (int kInd = 0; kInd < k; kInd++)
            {
                float diff = search_point[k * mInd + kInd] - reference_point[k * nInd + kInd];
                squareSum += diff * diff;
            }
            if (minSquareSum < 0 || squareSum < minSquareSum)
            {
                minSquareSum = squareSum;
                minIndex = nInd;
            }
        }
        fout << mInd << '\t' << minIndex << endl;
    }
    fout.close();

    delete[] search_point;
    delete[] reference_point;
}

int main()
{
    ifstream fin("input.txt");
    fin >> m >> n >> k;
    fin.close();

    // output("search_point.bin", "reference_point.bin");

    bool res = cmp("search_point.bin", "reference_point.bin", "cmp_res.txt");
    cout << (res ? "same" : "diff"); 

    return 0;
}