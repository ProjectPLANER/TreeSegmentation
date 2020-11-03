/**
 * @file Preprocessing.h
 * @brief The preprocessing header, which declares all methods requires to process DEMs before segmentaton can take place.
 * @version 0.1
 * @date 2020-10-05
 * 
 * @copyright Copyright (c) 2020
 * 
 */

#ifndef PREPROCESSING_H
#define PREPROCESSING_H

class Preprocessing
{
    private:
        void regulateImage(cv::Mat& image, int scaleFactor);
        void floodFillUtil(cv::Mat& image, int x, int y, int prevValue, int newValue);
        void regionGrowUtil(cv::Mat& image, int x, int y, uint8_t prevValue, uint8_t newValue);
    public:
        Preprocessing();
        ~Preprocessing();  
        void removeBoundary(cv::Mat& image, int min);
        void removeBoundary(cv::Mat& image, int limit, int newValue);
        cv::Mat createMask(cv::Mat& image, uint8_t limit);
        cv::Mat applyFilterGausian(cv::Mat& image);
        cv::Mat applyFilterBilateral(cv::Mat& image);
        cv::Mat applyThreshold(cv::Mat& image, int maxValue);
        cv::Mat applyThreshold(cv::Mat& image, int limit, int maxValue);
        cv::Mat applyDistanceTranform(cv::Mat& image);
        cv::Mat findLocalMax(cv::Mat& image, int windowSize);
        cv::Mat findLocalMax(cv::Mat& image);      
        void floodFill(cv::Mat& image, int x, int y, int newValue);        
        void regionGrow(cv::Mat& image, int x, int y, uint8_t newValue);
        float getVariance(std::vector<uint8_t> pixelValues);
        uint8_t getMax(std::vector<uint8_t> pixelValues);
        cv::Point getMaxPosition(std::vector<uint8_t> pixelValues, std::vector<cv::Point> pixelCoord);
        cv::Mat findEdges(cv::Mat& image);
        void findNeighbours(cv::Mat& image, std::vector<int>* neighbours);
        void sortCells(cv::Mat& image, std::vector<cv::Point>* clusters);
        void findCentroids(std::vector<cv::Point>* clusters, cv::Point* centroids, int numClusters);
};

#endif
