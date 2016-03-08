#ifdef _CH_
#pragma package <opencv>
#endif

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include <iostream>
//#include <stdio.h>
#include <stdlib.h>
#include <string>
#include "opencv/cv.h"
#include <algorithm>
#include <vector>

#define WARPED_XSIZE 200
#define WARPED_YSIZE 300






#define VERY_LARGE_VALUE 100000

#define NO_MATCH    0
#define STOP_SIGN            1
#define SPEED_LIMIT_40_SIGN  2
#define SPEED_LIMIT_80_SIGN  3


using namespace cv;
using namespace std;

bool sortx (const Point & i,const Point & j) { return (i.x<j.x); }
bool sorty (const Point & i,const Point & j) { return (i.y<j.y); }
static double angle( Point pt1, Point pt2, Point pt0 )
{
    double dx1 = pt1.x - pt0.x;
    double dy1 = pt1.y - pt0.y;
    double dx2 = pt2.x - pt0.x;
    double dy2 = pt2.y - pt0.y;
    return (dx1*dx2 + dy1*dy2)/sqrt((dx1*dx1 + dy1*dy1)*(dx2*dx2 + dy2*dy2) + 1e-10);
}

vector<Point> sortV(vector<Point> point){
    vector<Point> sx, sy, sortedvector, upper, lower;
    sx = point;
    sy = point;

    sort(sx.begin(), sx.end(), sortx);
    sort(sy.begin(), sy.end(), sorty);

    for (std::vector<Point>::iterator it =sx.begin(); it<sx.end(); it++){
        int k=0;
        for (std::vector<Point>::iterator ij= sy.begin(); ij<sy.end(); ij++){
            if (it->x==ij->x){
            if (k>=2){
                lower.push_back(*it);
            }
            else{
                upper.push_back(*it);
            }
            }
            k++;
        }
    }

    sort(upper.begin(), upper.end(), sortx);
    sort(upper.begin(), upper.end(), sorty);
    sort(lower.begin(), lower.end(), sortx);
    sort(lower.begin(), lower.end(), sorty);
    sortedvector = upper;
    sortedvector.insert(sortedvector.end(), lower.begin(), lower.end());
    return sortedvector;
}


int main(int argc, char** argv)
{


    Mat src;
    Mat src_gray;
    Mat warped_result;
    Mat speed_80;
    Mat speed_40;
    int canny_thresh = 154;

    int sign_recog_result = NO_MATCH;
    speed_40 = imread("/media/comp4102a-16-DVD/Assignment2/assign2_16/Assignment2/speed_40.bmp",0);
    speed_80 = imread("/media/comp4102a-16-DVD/Assignment2/assign2_16/Assignment2/speed_80.bmp",0);

    // you run your program on these three examples (uncomment the two lines below)
    //string sign_name = "/media/comp4102a-16-DVD/Assignment2/assign2_16/Assignment2/stop4";
    //string sign_name = "/media/comp4102a-16-DVD/Assignment2/assign2_16/Assignment2/speedsign12";
    //string sign_name = "/media/comp4102a-16-DVD/Assignment2/assign2_16/Assignment2/speedsign3";
    string sign_name = "/media/comp4102a-16-DVD/Assignment2/assign2_16/Assignment2/speedsign4";
    string final_sign_input_name = sign_name + ".jpg";
    string final_sign_output_name = sign_name + "_result" + ".jpg";
    /// Load source image and convert it to gray
    src = imread(final_sign_input_name, 1);



    /// Convert image to gray and blur it
    cvtColor(src, src_gray, COLOR_BGR2GRAY);
    blur(src_gray, src_gray, Size(3, 3));
    warped_result = Mat(Size(WARPED_XSIZE, WARPED_YSIZE), src_gray.type());

    //----------------------begin of my code---------------------------//

    Mat c;
    vector<vector<Point> > contours;
    vector<vector<Point> > squares;
    Canny(src_gray, c , canny_thresh, canny_thresh*2, 3);
    findContours(src_gray,contours, RETR_EXTERNAL, CHAIN_APPROX_NONE, Point(0, 0));
    squares.clear();
    vector<Point> approx;
    int count=0;
    for( size_t i = 0; i < contours.size(); i++ )
    {
                    // approximate contour with accuracy proportional
                    // to the contour perimeter
                    approxPolyDP(Mat(contours[i]), approx, arcLength(Mat(contours[i]), true)*0.02, true);

                    // square contours should have 4 vertices after approximation
                    // relatively large area (to filter out noisy contours)
                    // and be convex.
                    // Note: absolute value of an area is used because
                    // area may be positive or negative - in accordance with the
                    // contour orientation
                    if( approx.size() == 4 &&
                        fabs(contourArea(Mat(approx))) > 1000 &&
                        isContourConvex(Mat(approx)) )
                    {
                        double maxCosine = 0;

                        for( int j = 2; j < 5; j++ )
                        {
                            // find the maximum cosine of the angle between joint edges
                            double cosine = fabs(angle(approx[j%4], approx[j-2], approx[j-1]));
                            maxCosine = MAX(maxCosine, cosine);
                        }

                        // if cosines of all angles are small
                        // (all angles are ~90 degree) then write quandrange
                        // vertices to resultant sequence
                        if( maxCosine < 0.3 )
                            squares.push_back(approx);
                        count=i;
                    }
                }


    vector<Point> p = sortV(squares[0]);

    Point2f inputQuad[4];

    Point2f outputQuad[4];
    Mat input, output;

    inputQuad[0] = p[0];
    inputQuad[1] = p[1];
    inputQuad[2] = p[3];
    inputQuad[3] = p[2];


    outputQuad[0] = Point2f( 0,0 );
    outputQuad[1] = Point2f( src_gray.cols-1,0);
    outputQuad[2] = Point2f( src_gray.cols-1,src_gray.rows-1);
    outputQuad[3] = Point2f( 0,src_gray.rows-1  );

    warped_result = getPerspectiveTransform(inputQuad, outputQuad);

    warpPerspective(src,output,warped_result,output.size() );






    resize(output, output, speed_40.size());




    double val = norm (output, NORM_L2);

    if(val<300000 && val>60000.6){sign_recog_result= SPEED_LIMIT_40_SIGN;}
    else if(val<300000 && val>50000.2){sign_recog_result= SPEED_LIMIT_80_SIGN; }
    else if(val<300000 && val>30000.5){sign_recog_result= STOP_SIGN; }
    else{sign_recog_result = NO_MATCH;}

    
    //--------------------end of my code-------------//
    
    // here you add the code to do the recognition, and set the variable
    // sign_recog_result to one of STOP_SIGN, SPEED_LIMIT_40_SIGN, SPEED_LIMIT_80_SIGN, or NO_MATCH

    string text;
    if (sign_recog_result == SPEED_LIMIT_40_SIGN) text = "Speed 40";
    else if (sign_recog_result == SPEED_LIMIT_80_SIGN) text = "Speed 80";
    else if (sign_recog_result == STOP_SIGN) text = "Stop";
    else if (sign_recog_result == NO_MATCH) text = "Fail";

    int fontFace = FONT_HERSHEY_SCRIPT_SIMPLEX;
    double fontScale = 2;
    int thickness = 3;
    cv::Point textOrg(10, 130);
    cv::putText(src, text, textOrg, fontFace, fontScale, Scalar::all(255), thickness, 8);

    /// Create Window
    char* source_window = "Result";
    namedWindow(source_window, WINDOW_AUTOSIZE);
    imshow(source_window, src);
    imwrite(final_sign_output_name, src);

    waitKey(0);

    return 0;
}

