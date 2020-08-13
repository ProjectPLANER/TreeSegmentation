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
    public:
        Segmentation(cv::Mat& image);
        ~Segmentation();
        void segment();
};

#endif