#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <dirent.h>

#include "net.h"

using namespace std;

void prettyMat(const ncnn::Mat &m) {
    for (int q = 0; q < m.c; q++) {
        const float *ptr = m.channel(q);
        for (int y = 0; y < m.h; y++) {
            for (int x = 0; x < m.w; x++) {
                printf("%f ", ptr[x]);
            }
            ptr += m.w;
            printf("\n");
        }
    }
}

/**
 * 计算预测的结果值
 * @param m
 */
void calculateIndex(const ncnn::Mat &m) {
    int index = 0;
    float maxValue = 0;
    for (int q = 0; q < m.c; q++) {
        const float *ptr = m.channel(q);
        for (int y = 0; y < m.h; y++) {
            for (int x = 0; x < m.w; x++) {
                if (ptr[x] > maxValue) {
                    maxValue = ptr[x];
                    index = x;
                }
            }
            ptr += m.w;
        }
        printf("index %d\n", index);
    }
}

/**
 * 打印原始图片的像素值
 * @param img
 */
void printImage(cv::Mat img) {
    int w = img.cols;
    int h = img.rows;
    int c = img.channels();
    for (int i = 0; i < c; i++) {
        for (int j = 0; j < h; j++) {
            for (int k = 0; k < w; k++) {
                uchar val = img.at<uchar>(j, k);
                printf("%d ", val);
            }
            printf("\n");
        }
    }
}

/**
 * 对图片进行推理并打印结果
 * @param imgPath
 */
void anticipate(const string &imgPath) {
    cv::Mat img = cv::imread(imgPath, cv::IMREAD_GRAYSCALE);
    printImage(img);
    printf("%s \n", imgPath.c_str());

    ncnn::Net net;
    net.load_param("./model/mnist-opt.param");
    net.load_model("./model/mnist-opt.bin");
    ncnn::Mat input = ncnn::Mat::from_pixels(img.data, ncnn::Mat::PIXEL_GRAY, img.cols, img.rows);
    //printf("w,h,c:   %d,%d,%d \n", input.w, input.h, input.c);
    const float mean_vals[3] = {1.f, 1.f, 1.f};
    const float norm_vals[3] = {1 / 255.f, 1 / 255.f, 1 / 255.f};
    input.substract_mean_normalize(mean_vals, norm_vals);
    //prettyMat(input);

    ncnn::Extractor extractor = net.create_extractor();
    extractor.input("input_0", input);
    ncnn::Mat output0;
    extractor.extract("output_0", output0);
    //printf("w,h,c:   %d,%d,%d \n", output0.w, output0.h, output0.c);
    //prettyMat(output0);
    calculateIndex(output0);
}

/**
 * 遍历image目录下的图片，并预测结果
 * @return
 */
int main() {
    string fileFolder = "./image/";
    string filePath;
    DIR *dir = opendir(fileFolder.c_str());
    dirent *dirTree;
    while ((dirTree = readdir(dir))) {
        if (dirTree->d_name[0] == '.' || strcmp(dirTree->d_name, "..") == 0) {
            continue;
        }
        filePath = fileFolder + dirTree->d_name;
        anticipate(filePath);
        printf("---------------------------------------------------\n");
    }
    return 0;
}