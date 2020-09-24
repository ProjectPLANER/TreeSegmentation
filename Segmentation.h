/**
 * Lynolan Moodley
 * CSC4002W Project
 */

#ifndef SEGMENTATION_H
#define SEGMENTATION_H

class Segmentation
{
    private:
        cv::Mat image;
        std::string file;
        cv::Mat getWatershedMask(cv::Mat& image);
        void display(cv::Mat& image,cv::Mat& boundaries,cv::Mat& slicBoundaries,cv::Mat& localMax);
    public:
        Segmentation(cv::Mat& image,std::string file);
        ~Segmentation();
        void segment();      
        void evaluate(cv::Mat& image, cv::Mat& SLICMask, cv::Mat& localMax, int numCells);
        void evaluate(cv::Mat& mask, std::set<int>& isTree, std::vector<cv::Point>* coord);
        void evaluate(cv::Mat& mask);
        void evaluatePer(cv::Mat& mask, int numCells);
};

#endif