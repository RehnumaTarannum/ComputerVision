#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include <stdio.h>
#include <stdlib.h>

using namespace cv;
using namespace std;

char* source_window = "Source image";
char* corners_window = "Corners detected";
Mat src,x2y2, xy,mtrace,src_gray, x_derivative, y_derivative,x2_derivative, y2_derivative,
    xy_derivative,x2g_derivative, y2g_derivative,xyg_derivative,dst,dst_norm, dst_norm_scaled;
int thresh = 83;

void onTrackbar( int, void* );

int main( int argc, char** argv )
{
    //Get Image
    src = imread( "/media/comp4102a-16-DVD/Sample/checker.jpg");

    //Create window to show source image
    namedWindow( source_window, CV_WINDOW_AUTOSIZE );

    //Create trackbar to change threshold
    createTrackbar( "Threshold", source_window, &thresh, 255, onTrackbar);

    //Show source image
    imshow( source_window, src );
    onTrackbar(thresh,0);

    waitKey(0);
    return(0);
}

void onTrackbar( int, void* ){
    cvtColor( src, src_gray, CV_BGR2GRAY );

    //Step one
    //to calculate x and y derivative of image we use Sobel function
    //Sobel( srcimage, dstimage, depthofimage -1 means same as input, xorder 1,yorder 0,kernelsize 3, BORDER_DEFAULT);
    Sobel(src_gray, x_derivative, CV_32FC1 , 1, 0, 3, BORDER_DEFAULT);
    Sobel(src_gray, y_derivative, CV_32FC1 , 0, 1, 3, BORDER_DEFAULT);

    //Step Two calculate other three images in M
    pow(x_derivative,2.0,x2_derivative);
    pow(y_derivative,2.0,y2_derivative);
    multiply(x_derivative,y_derivative,xy_derivative);

    //step three apply gaussain
    GaussianBlur(x2_derivative,x2g_derivative,Size(7,7),2.0,0.0,BORDER_DEFAULT);
    GaussianBlur(y2_derivative,y2g_derivative,Size(7,7),0.0,2.0,BORDER_DEFAULT);
    GaussianBlur(xy_derivative,xyg_derivative,Size(7,7),2.0,2.0,BORDER_DEFAULT);

    //forth step calculating R with k=0.04
    multiply(x2g_derivative,y2g_derivative,x2y2);
    multiply(xyg_derivative,xyg_derivative,xy);
    pow((x2g_derivative + y2g_derivative),2.0,mtrace);
    dst = (x2y2 - xy) - 0.04 * mtrace;

    //normalizing result from 0 to 255
    normalize( dst, dst_norm, 0, 255, NORM_MINMAX, CV_32FC1, Mat() );
    convertScaleAbs( dst_norm, dst_norm_scaled );

    // Drawing a circle around corners
    for( int j = 0; j < src_gray.rows ; j++ )
        { for( int i = 0; i < src_gray.cols; i++ )
            {
                if( (int) dst_norm.at<float>(j,i) > thresh )
                {
                    circle( src_gray, Point( i, j ), 5,  Scalar(255), 2, 8, 0 );
                }
            }
        }
  // Showing the result
  namedWindow( corners_window, CV_WINDOW_AUTOSIZE );
  imshow( corners_window, src_gray );
}
