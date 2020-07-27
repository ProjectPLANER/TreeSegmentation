/**
 * Lynolan Moodley
 * CSC4002W_Project
 */

#ifndef WATERSHED
#define WATERSHED

class Watershed
{
    private:
        cv::Mat image;
    public:
        Watershed();
        ~Watershed();
        Watershed(cv::Mat& _image);
        void applyWatershed();
};

#endif