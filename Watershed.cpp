#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include "Watershed.h"
#include <unordered_map>
#include <iostream>

Watershed::Watershed() {}

Watershed::Watershed(cv::Mat& _image) {
    image = _image;
}

Watershed::~Watershed()
{

}

void Watershed::applyWatershed()
{
    cv::Mat boundary = cv::Mat::zeros(image.size(),CV_8U);

    for ( int i = 0; i < image.rows; i++ ) 
    {
        for ( int j = 0; j < image.cols; j++ ) 
        {
            if(image.at<float>(i,j) < 0)
            {
                image.at<float>(i,j) = 16.5745;
                boundary.at<uint8_t>(i,j) = 255;
            }
                
        }
    }

    /*std::unordered_map<float,int> umap;
    int count = 0;
    for ( int i = 0; i < image.rows; i++ ) 
    {
        for ( int j = 0; j < image.cols; j++ ) 
        {
            std::unordered_map<float,int>::const_iterator it = umap.find(image.at<float>(i,j));
            if(it == umap.end())
            {
                umap[image.at<float>(i,j)] = count;
                ++count;
            }
        }
    }
    std::cout << count << std::endl;*/

    cv::imwrite("img.tif",image); //=====fine
    cv::Mat kernel = (cv::Mat_<float>(3,3) <<
                  1,  1, 1,
                  1, -8, 1,
                  1,  1, 1); // an approximation of second derivative, a quite strong kernel
    // do the laplacian filtering as it is
    // well, we need to convert everything in something more deeper then CV_8U
    // because the kernel has some negative values,
    // and we can expect in general to have a Laplacian image with negative values
    // BUT a 8bits unsigned int (the one we are working with) can contain values from 0 to 255
    // so the possible negative number will be truncated
    cv::Mat normimg;
    cv::normalize(image, normimg, 0, 255, cv::NORM_MINMAX);
    cv::imwrite("norm.tif",normimg);
    cv::Mat imgLaplacian;
    cv::filter2D(image, imgLaplacian, CV_32F, kernel);
    //cv::Mat sharp;
    //image.convertTo(sharp, CV_32F);
    cv::Mat imgResult = normimg ;//- imgLaplacian;
    // convert back to 8bits gray scale

    cv::Mat bw;
    
    imgResult.convertTo(imgResult, CV_8U);
    bw = imgResult;
    cv::imwrite("bwbefore.tif",bw);
    cv::cvtColor(imgResult,imgResult,cv::COLOR_GRAY2BGR);
    //imgLaplacian.convertTo(imgLaplacian, CV_8UC3);
    // imshow( "Laplace Filtered Image", imgLaplacian );
    //imshow( "New Sharped Image", imgResult );
    // Create binary image from source image
    //cv::imwrite("imgres.tif",image);
    //std::cout << image.type() << " " << image.channels()  << " " << image.depth() << " " << CV_32S<< " " << CV_16F<< std::endl;
    //cv::Mat image_bw;
    //image.convertTo(image_bw,CV_8U);
    //std::cout << image_bw.type() << " " << image.channels()  << " " << image.depth() << " " << CV_32S<< " " << CV_16F<< std::endl;
    //cv::imwrite("outputlow.tif",image_bw);
    /*for ( int i = 0; i < image.rows; i++ ) 
    {
        for ( int j = 0; j < image.cols; j++ ) 
        {
            //std::cout << image.at<float>(i,j) << std::endl;
        }
    }*/

    //image_bw.con
    //cv::threshold(bw, bw, 40, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);
    //cv::imshow("Binary Image", bw);
    //cv::waitKey(0);
    uint8_t max = 0;
    uint8_t min = 255;
    for ( int i = 0; i < bw.rows; i++ ) 
    {
        for ( int j = 0; j < bw.cols; j++ ) 
        {
            if(bw.at<uint8_t>(i,j) != 0 && bw.at<uint8_t>(i,j) < min)
            {
                min = bw.at<uint8_t>(i,j);
            }
            if(bw.at<uint8_t>(i,j) > max)
            {
                max = bw.at<uint8_t>(i,j);
            }
        }
    }
    std::cout << unsigned(min) << " " << unsigned(max) << std::endl;

    /*uint8_t a = 0;
    uint8_t b = 255;
    for ( int i = 0; i < bw.rows; i++ ) 
    {
        for ( int j = 0; j < bw.cols; j++ ) 
        {
            if(bw.at<uint8_t>(i,j) <= min)
            {
                bw.at<uint8_t>(i,j) = 0;
            }
            if(bw.at<uint8_t>(i,j) != 0)
            {
                bw.at<uint8_t>(i,j) = (((b-a)*(bw.at<uint8_t>(i,j)-min)) / (max-min)) + a;
            }
            
        }
    }*/

    cv::imwrite("bw.tif",bw);
    // Perform the distance transform algorithm
    cv::Mat dist;
    cv::distanceTransform(bw, dist, cv::DIST_L2, 3);
    // Normalize the distance image for range = {0.0, 1.0}
    // so we can visualize and threshold it
    cv::normalize(dist, dist, 0, 1.0, cv::NORM_MINMAX);
    //imshow("Distance Transform Image", dist);
    // Threshold to obtain the peaks
    // This will be the markers for the foreground objects
    cv::imwrite("dist.tif",dist);
    cv::threshold(dist, dist, 0.4, 1.0, cv::THRESH_BINARY);
    // Dilate a bit the dist image
    cv::Mat kernel1 = cv::Mat::ones(3, 3, CV_8U);
    cv::dilate(dist, dist, kernel1);
    //cv::imshow("Peaks", dist);
    
    // Create the CV_8U version of the distance image
    // It is needed for findContours()
    cv::Mat dist_8u;
    dist.convertTo(dist_8u, CV_8U);
    // Find total markers
    std::vector<std::vector<cv::Point> > contours;
    cv::findContours(dist_8u, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    // Create the marker image for the watershed algorithm
    cv::Mat markers = cv::Mat::zeros(dist.size(), CV_32S);
    // Draw the foreground markers
    for (size_t i = 0; i < contours.size(); i++)
    {
        drawContours(markers, contours, static_cast<int>(i), cv::Scalar(static_cast<int>(i)+1), -1);
    }
    // Draw the background marker
    //cv::circle(markers, cv::Point(5,5), 3, cv::Scalar(255), -1); //===================================
    cv::imwrite("Markers.tif", markers*10000);
    // Perform the watershed algorithm
    //cv::Mat image3;
    //image.convertTo(image3,CV_8U);
    //cv::cvtColor(image3,image3,cv::COLOR_GRAY2BGR);


    std::vector<cv::Point> v = bhFindLocalMaximum(bw);
    cv::Mat newbw;
    newbw = bw.clone();
    for(int i = 0; i < v.size(); ++i)
    {
        if (!(v[i].x < 0 || v[i].y < 0))
        {
            newbw.at<uint8_t>(v[i].x,v[i].y) = 0;
        }
        
        
        //std::cout << v[i].x << " " << v[i].y << std::endl;
    }
    cv::imwrite("newbwmark.tif",newbw);

    cv::Mat maxims(bw.size(), bw.type()); // container for all local maximums
    cv::erode(bw, maxims, cv::Mat());
    //cv::erode(maxims, maxims, cv::Mat());
    cv::Mat thresh;
    cv::threshold(maxims,thresh,22,255,cv::THRESH_BINARY);
    cv::imwrite("thresh.tif",thresh);
    cv::Mat newThresh = cv::Mat::zeros(dist.size(), CV_32S);
    //thresh.convertTo(thresh,CV_32S);
    cv::threshold(thresh,newThresh,1,1,cv::THRESH_BINARY);
    newThresh.convertTo(newThresh,CV_32S);
    cv::imwrite("newthresh.tif",newThresh);
    //cv::Mat img; // your input image that you should fill with values
    //cv::Mat maxims(bw.size(), bw.type()); // container for all local maximums
    //cv::erode(bw, maxims, cv::Mat());
    //cv::imwrite("erode.tif",maxims);
    cv::Mat mininini = bw - maxims;
    //cv::imwrite("min.tif",mininini);
    //cv::bitwise_not(mininini,mininini);
    //cv::imwrite("min.tif",mininini);
    //mininini = 

    //cv::Mat notting;
    //cv::bitwise_not(bw,notting);
    //cv::Mat notting


    for (size_t i = 0; i < mininini.rows; i++)
    {
        for (size_t j = 0; j < mininini.cols; j++)
        {
            if (mininini.at<uint8_t>(i,j) > 3)
            {
                imgResult.at<cv::Vec3b>(i,j) = cv::Vec3b(0,0,255);
                
                
            }
        }
    }
    cv::imwrite("red.tif",imgResult);

    cv::watershed(imgResult, newThresh); //markers
    cv::imwrite("outputt.tif",newThresh);
    for (size_t i = 0; i < newThresh.rows; i++)
    {
        for (size_t j = 0; j < newThresh.cols; j++)
        {
            if (newThresh.at<int32_t>(i,j) == -1)
            {
                imgResult.at<cv::Vec3b>(i,j)[0] == 0;
                imgResult.at<cv::Vec3b>(i,j)[1] == 0;
                imgResult.at<cv::Vec3b>(i,j)[2] == 255;
            }
            
        }
    }
    cv::imshow("red",imgResult);
    cv::waitKey(0);
    cv::imwrite("output.tif",imgResult);
    

    //cv::Mat mark;
    //markers.convertTo(mark, CV_8U);
    //cv::bitwise_not(mark, mark);
    //    imshow("Markers_v2", mark); // uncomment this if you want to see how the mark
    // image looks like at that point
    // Generate random colors
    std::vector<cv::Vec3b> colors;
    for (size_t i = 0; i < contours.size(); i++)
    {
        int b = cv::theRNG().uniform(0, 256);
        int g = cv::theRNG().uniform(0, 256);
        int r = cv::theRNG().uniform(0, 256);
        colors.push_back(cv::Vec3b((uchar)b, (uchar)g, (uchar)r));
    }
    // Create the result image
    cv::Mat dst = cv::Mat::zeros(markers.size(), CV_8UC3);
    // Fill labeled objects with random colors
    for (int i = 0; i < markers.rows; i++)
    {
        for (int j = 0; j < markers.cols; j++)
        {
            int index = markers.at<int>(i,j);
            if (index > 0 && index <= static_cast<int>(contours.size()))
            {
                dst.at<cv::Vec3b>(i,j) = colors[index-1];
            }
        }
    }
    // Visualize the final image
    //cv::imwrite("output.tif",dst);
    //cv::imshow("Final Result", dst);
    //cv::waitKey(0);
}

std::vector<cv::Point> Watershed::bhContoursCenter(const std::vector<std::vector<cv::Point>>& contours,bool centerOfMass)
{
    int contourIdx = -1;
    std::vector<cv::Point> result;
    if (contourIdx > -1)
    {
        if (centerOfMass)
        {
            cv::Moments m = moments(contours[contourIdx],true);
            result.push_back( cv::Point(m.m10/m.m00, m.m01/m.m00));
        }
        else 
        {
            cv::Rect rct = boundingRect(contours[contourIdx]);
            result.push_back( cv::Point(rct.x + rct.width / 2 ,rct.y + rct.height / 2));
        }
    }
    else 
    {
        if (centerOfMass)
        {
            for (int i=0; i < contours.size();i++)
            {
                cv::Moments m = moments(contours[i],true);
                result.push_back( cv::Point(m.m10/m.m00, m.m01/m.m00));

            }
        }
        else 
        {

            for (int i=0; i < contours.size(); i++)
            {
                cv::Rect rct = boundingRect(contours[i]);
                result.push_back(cv::Point(rct.x + rct.width / 2 ,rct.y + rct.height / 2));
            }
        }
    }

    return result;
}


std::vector<cv::Point> Watershed::bhFindLocalMaximum(cv::Mat& src){
    //Mat src = _src.getMat();
    int neighbor = 2;
    cv::Mat peak_img = src.clone();
    cv::dilate(peak_img,peak_img,cv::Mat(),cv::Point(-1,-1),neighbor);
    peak_img = peak_img - src;



    cv::Mat flat_img ;
    cv::erode(src,flat_img,cv::Mat(),cv::Point(-1,-1),neighbor);
    flat_img = src - flat_img;


    cv::threshold(peak_img,peak_img,0,255,cv::THRESH_BINARY);
    cv::threshold(flat_img,flat_img,0,255,cv::THRESH_BINARY);
    cv::bitwise_not(flat_img,flat_img);

    peak_img.setTo(cv::Scalar::all(255),flat_img);
    bitwise_not(peak_img,peak_img);


    std::vector<std::vector<cv::Point>> contours;
    findContours(peak_img,contours,cv::RETR_EXTERNAL,cv::CHAIN_APPROX_SIMPLE);

    return bhContoursCenter(contours,true);
}

void Watershed::water(cv::InputArray _src, cv::InputOutputArray _markers)
{
    //cv::CV_INSTRUMENT_REGION();

    // Labels for pixels
    const int IN_QUEUE = -2; // Pixel visited
    const int WSHED = -1; // Pixel belongs to watershed

    // possible bit values = 2^8
    const int NQ = 4294967296;

    cv::Mat src = _src.getMat(), dst = _markers.getMat();
    cv::Size size = src.size();

    // Vector of every created node
    std::vector<WSNode> storage;
    int free_node = 0, node;
    // Priority queue of queues of nodes
    // from high priority (0) to low priority (255)
    WSQueue q[NQ];
    // Non-empty queue with highest priority
    int active_queue;
    int i, j;
    // Color differences
    int db, dg, dr;
    int subs_tab[513];

    // MAX(a,b) = b + MAX(a-b,0)
    #define ws_max(a,b) ((b) + subs_tab[(a)-(b)+NQ])
    // MIN(a,b) = a - MAX(a-b,0)
    #define ws_min(a,b) ((a) - subs_tab[(a)-(b)+NQ])

    // Create a new node with offsets mofs and iofs in queue idx
    #define ws_push(idx,mofs,iofs)          \
    {                                       \
        if( !free_node )                    \
            free_node = allocWSNodes( storage );\
        node = free_node;                   \
        free_node = storage[free_node].next;\
        storage[node].next = 0;             \
        storage[node].mask_ofs = mofs;      \
        storage[node].img_ofs = iofs;       \
        if( q[idx].last )                   \
            storage[q[idx].last].next=node; \
        else                                \
            q[idx].first = node;            \
        q[idx].last = node;                 \
    }

    // Get next node from queue idx
    #define ws_pop(idx,mofs,iofs)           \
    {                                       \
        node = q[idx].first;                \
        q[idx].first = storage[node].next;  \
        if( !storage[node].next )           \
            q[idx].last = 0;                \
        storage[node].next = free_node;     \
        free_node = node;                   \
        mofs = storage[node].mask_ofs;      \
        iofs = storage[node].img_ofs;       \
    }

    // Get highest absolute channel difference in diff
    #define c_diff(ptr1,ptr2,diff)           \
    {                                        \
        db = std::abs((ptr1)[0] - (ptr2)[0]);\
        dg = std::abs((ptr1)[1] - (ptr2)[1]);\
        dr = std::abs((ptr1)[2] - (ptr2)[2]);\
        diff = ws_max(db,dg);                \
        diff = ws_max(diff,dr);              \
        assert( 0 <= diff && diff <= 255 );  \
    }

    CV_Assert( src.type() == CV_8UC3 && dst.type() == CV_32SC1 );
    CV_Assert( src.size() == dst.size() );

    // Current pixel in input image
    const uchar* img = src.ptr();
    // Step size to next row in input image
    int istep = int(src.step/sizeof(img[0]));

    // Current pixel in mask image
    int* mask = dst.ptr<int>();
    // Step size to next row in mask image
    int mstep = int(dst.step / sizeof(mask[0]));

    for( i = 0; i < 256; i++ )
        subs_tab[i] = 0;
    for( i = 256; i <= 512; i++ )
        subs_tab[i] = i - 256;

    // draw a pixel-wide border of dummy "watershed" (i.e. boundary) pixels
    for( j = 0; j < size.width; j++ )
        mask[j] = mask[j + mstep*(size.height-1)] = WSHED;

    // initial phase: put all the neighbor pixels of each marker to the ordered queue -
    // determine the initial boundaries of the basins
    for( i = 1; i < size.height-1; i++ )
    {
        img += istep; mask += mstep;
        mask[0] = mask[size.width-1] = WSHED; // boundary pixels

        for( j = 1; j < size.width-1; j++ )
        {
            int* m = mask + j;
            if( m[0] < 0 ) m[0] = 0;
            if( m[0] == 0 && (m[-1] > 0 || m[1] > 0 || m[-mstep] > 0 || m[mstep] > 0) )
            {
                // Find smallest difference to adjacent markers
                const uchar* ptr = img + j*3;
                int idx = 256, t;
                if( m[-1] > 0 )
                    c_diff( ptr, ptr - 3, idx );
                if( m[1] > 0 )
                {
                    c_diff( ptr, ptr + 3, t );
                    idx = ws_min( idx, t );
                }
                if( m[-mstep] > 0 )
                {
                    c_diff( ptr, ptr - istep, t );
                    idx = ws_min( idx, t );
                }
                if( m[mstep] > 0 )
                {
                    c_diff( ptr, ptr + istep, t );
                    idx = ws_min( idx, t );
                }

                // Add to according queue
                assert( 0 <= idx && idx <= 255 );
                //ws_push( idx, i*mstep + j, i*istep + j*3 );
                m[0] = IN_QUEUE;
            }
        }
    }

    // find the first non-empty queue
    for( i = 0; i < NQ; i++ )
        if( q[i].first )
            break;

    // if there is no markers, exit immediately
    if( i == NQ )
        return;

    active_queue = i;
    img = src.ptr();
    mask = dst.ptr<int>();

    // recursively fill the basins
    for(;;)
    {
        int mofs, iofs;
        int lab = 0, t;
        int* m;
        const uchar* ptr;

        // Get non-empty queue with highest priority
        // Exit condition: empty priority queue
        if( q[active_queue].first == 0 )
        {
            for( i = active_queue+1; i < NQ; i++ )
                if( q[i].first )
                    break;
            if( i == NQ )
                break;
            active_queue = i;
        }

        // Get next node
        ws_pop( active_queue, mofs, iofs );

        // Calculate pointer to current pixel in input and marker image
        m = mask + mofs;
        ptr = img + iofs;

        // Check surrounding pixels for labels
        // to determine label for current pixel
        t = m[-1]; // Left
        if( t > 0 ) lab = t;
        t = m[1]; // Right
        if( t > 0 )
        {
            if( lab == 0 ) lab = t;
            else if( t != lab ) lab = WSHED;
        }
        t = m[-mstep]; // Top
        if( t > 0 )
        {
            if( lab == 0 ) lab = t;
            else if( t != lab ) lab = WSHED;
        }
        t = m[mstep]; // Bottom
        if( t > 0 )
        {
            if( lab == 0 ) lab = t;
            else if( t != lab ) lab = WSHED;
        }

        // Set label to current pixel in marker image
        assert( lab != 0 );
        m[0] = lab;

        if( lab == WSHED )
            continue;

        // Add adjacent, unlabeled pixels to corresponding queue
        if( m[-1] == 0 )
        {
            c_diff( ptr, ptr - 3, t );
            //ws_push( t, mofs - 1, iofs - 3 );
            active_queue = ws_min( active_queue, t );
            m[-1] = IN_QUEUE;
        }
        if( m[1] == 0 )
        {
            c_diff( ptr, ptr + 3, t );
            //ws_push( t, mofs + 1, iofs + 3 );
            active_queue = ws_min( active_queue, t );
            m[1] = IN_QUEUE;
        }
        if( m[-mstep] == 0 )
        {
            c_diff( ptr, ptr - istep, t );
            //ws_push( t, mofs - mstep, iofs - istep );
            active_queue = ws_min( active_queue, t );
            m[-mstep] = IN_QUEUE;
        }
        if( m[mstep] == 0 )
        {
            c_diff( ptr, ptr + istep, t );
            //ws_push( t, mofs + mstep, iofs + istep );
            active_queue = ws_min( active_queue, t );
            m[mstep] = IN_QUEUE;
        }
    }
}