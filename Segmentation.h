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
    public:
        Segmentation(cv::Mat& image,std::string file);
        ~Segmentation();
        void segment();
        void display(cv::Mat& image,cv::Mat& boundaries,cv::Mat& localMax);
};

#endif