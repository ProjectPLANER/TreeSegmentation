/**
 * Lynolan Moodley
 * CSC4002W Project
 */

#ifndef PREPROCESSING_H
#define PREPROCESSING_H

class Preprocessing
{
    private:
        cv::Mat dem;
        std::vector<cv::Point2d> maskPoints;
        void regulateImage(cv::Mat& image, int num);
    public:
        Preprocessing();
        ~Preprocessing();
        void removeBoundary(cv::Mat& image, int min);
        void removeBoundary(cv::Mat& image, int t, int set);
        cv::Mat createMask(cv::Mat& image, uint8_t t, uint8_t set);
        cv::Mat applyFilter(cv::Mat& image);
        cv::Mat applyFilterGausian(cv::Mat& image);
        cv::Mat applyThreshold(cv::Mat& image);
        cv::Mat applyThreshold(cv::Mat& image, int t, int max);
        cv::Mat applyDistanceTranform(cv::Mat& image);
        cv::Mat findLocalMax(cv::Mat& image);
        void floodFillUtil(cv::Mat& image, int x, int y, int prevC, int newC);
        void floodFill(cv::Mat& image, int x, int y, int newC);
        void regionGrowUtil(cv::Mat& image, int x, int y, uint8_t prevC, uint8_t newC);
        void regionGrow(cv::Mat& image, int x, int y, uint8_t newC);
        float variance(std::vector<uint8_t>);
        uint8_t getMax(std::vector<uint8_t>);
        cv::Point getMaxPosition(std::vector<uint8_t>,std::vector<cv::Point>);
        cv::Mat findEdges(cv::Mat& image);
        void findNeighbours(cv::Mat& image, std::vector<int>* neighbours);
        void sortCells(cv::Mat& image, std::vector<cv::Point>* cells);
        void findCentroids(std::vector<cv::Point>* cells,cv::Point* centroids,int len);
};

#endif