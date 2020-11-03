/**
 * @file Segmentation.h
 * @brief The segmentation header, which declares the methods to be used when segmenting DEMs.
 * @version 0.1
 * @date 2020-10-05
 * 
 * @copyright Copyright (c) 2020
 * 
 */

#ifndef SEGMENTATION_H
#define SEGMENTATION_H

class Segmentation
{
    private:
        cv::Mat image;
        std::string file;
        int min;
        int SLICSize;
        int localMaxWindowSize;
        cv::Mat getWatershedMask(cv::Mat& image);
        void display(cv::Mat& image,cv::Mat& boundaries,cv::Mat& slicBoundaries,cv::Mat& localMax);
        void recordTime(int duration, int durationSLIC, int durationWater);
    public:
        Segmentation(cv::Mat& image, std::string file, int min, int SLICSize, int localMaxWindowSize);
        ~Segmentation();
        void segment(bool test);      
        void evaluateSorensenDice(cv::Mat& image, cv::Mat& localMax, int numClusters);
        void evaluateSorensenDiceSLIC(cv::Mat& mask, std::set<int>& isTree, std::vector<cv::Point>* coord);
        void evaluateIOU(cv::Mat& mask);
        void evaluateIOUPerTree(cv::Mat& clusters, int numClusters);
        void evaluateIOUPerTreeSLIC(cv::Mat& clusters, std::set<int>& isTree, int numClusters);
        void evaluateCentre(cv::Mat& localMax);
};

#endif
