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
        cv::Mat applyThreshold(cv::Mat& image);
        cv::Mat applyThreshold(cv::Mat& image, int t, int max);
        cv::Mat applyDistanceTranform(cv::Mat& image);
        cv::Mat findLocalMax(cv::Mat& image);
};

#endif