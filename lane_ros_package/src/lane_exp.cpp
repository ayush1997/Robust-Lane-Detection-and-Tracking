
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <sys/types.h>
#include <dirent.h>
#include <errno.h>
#include <string>
#include <iostream>
#include <bits/stdc++.h>
#include <stdio.h>
#include <vector>
#include <math.h>
#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <opencv2/core/types_c.h>
#include <stdlib.h>
#include <fstream>
#include <string>
#include <sstream>
#include <iterator>
#include <time.h>
//Includes all the headers necessary to use the most common public pieces of the ROS system.
#include <ros/ros.h>
//Use image_transport for publishing and subscribing to images in ROS
#include <image_transport/image_transport.h>
//Use cv_bridge to convert between ROS and OpenCV Image formats
#include <cv_bridge/cv_bridge.h>
//Include some useful constants for image encoding. Refer to: http://www.ros.org/doc/api/sensor_msgs/html/namespacesensor__msgs_1_1image__encodings.html for more info.
#include <sensor_msgs/image_encodings.h>
//Include headers for OpenCV Image processing
#include <swarath_package/lane_detection.h>
#include <swarath_package/lstm_data.h>
#include "std_msgs/String.h"

#include "htlibaux.h"
#include "htlibhoughextended.h"

// #include <custom_msg/lstm_data.h>


#define WINDOW_NAME       "Hough Transformation"
#define THETA_GRANULARITY   (4*100)
#define RHO_GRANULARITY     1

using namespace std;
using namespace cv;
#define PI 3.1415926

//Store all constants for image encodings in the enc namespace to be used later.
namespace enc = sensor_msgs::image_encodings;
 
//Declare a string with the name of the window that we will create using OpenCV where processed images will be displayed.
static const char WINDOW[] = "LaneDetectorWindow";
ros::Publisher pub;
swarath_package::lane_detection ld;
 

int image_no = 1;

vector<double> road_grad;
vector<double>::iterator road_grad_i;

vector<double> white_grad;
vector<double>::iterator white_grad_i;

int road_grad_size,white_grad_size;


void removeClustersExt(vector<Point>& point1, vector<Point>& point2, vector<double>& slope,vector<float>& lane_rho,vector<float>& lane_theta)
{
  int i,j=0;
  Point temp;
  Point p1,p2,p3,p4;
  int t;
  double area,peri;
  /* Removing clusters of lines based on 
   * 1) Area/perimeter ratio of quadrilateral formed by two lines taken randomly
   * 2) Difference in slope of two lines
   */
  // cout <<point1.size();
  for(i=0;i<point1.size();i++)
  {
    for(j=i+1;j<point1.size();j++)
    {
      p1 = point1[i];
      p2 = point2[i];
      p3 = point1[j];
      p4 = point2[j];

      area = 0.5*((p1.x-p3.x)*(p2.y-p4.y)-(p2.x-p4.x)*(p1.y-p3.y));
      peri = norm(p1-p2) + norm(p2-p3) + norm(p3-p4) + norm(p4-p1);
      // if(abs(area)/peri<0.15 || (abs(area)/peri<1 && abs(atan(slope[i])-atan(slope[j])) < 15.0*PI/180))
      // cout <<  abs(atan(slope[i])-atan(slope[j])) ;
      // if((abs(area)/peri<0.15 || abs(area)/peri<1) || abs(atan(slope[i])-atan(slope[j])) <= 0.6*PI/180)
      if((abs(area)/peri<0.15 || abs(area)/peri<1) || abs(atan(slope[i])-atan(slope[j])) < 10*PI/180)

      {
        slope.erase(slope.begin()+j);
        point1.erase(point1.begin()+j);
        point2.erase(point2.begin()+j);

        lane_theta.erase(lane_theta.begin()+j);
        lane_rho.erase(lane_rho.begin()+j);


        j--;
      }
    }
  }
  //cout<<point1.size();
}

void hough_ext(Mat image, vector<Point>& point1, vector<Point>& point2,vector<float>& lane_rho,vector<float>& lane_theta){

  int const canny_threshold = 50;

  float const lines_rho = 1.0f;
  float const lines_theta = HT_CONSTANT_PIF / 180.0f;
  // float const lines_theta = HT_CONSTANT_PIF / 180.0f;
  int const lines_acc_thr = 15;
  // int const lines_acc_thr = 30;
  int const lines_max = 300;
  // int const lines_max = 80;

  cv::Mat input, bgr, hsv, gray;
  vector<cv::Mat> channels;

  // cv::Mat raw = cv::imread("/home/ayush/Documents/swarath_lane/swarath/src/lanedetection_shubhamIITK/src/hough_code/data/197.jpg");
  cv::Mat raw = image;


  if ((0 == raw.rows) || (0 == raw.cols))
  {
      fprintf(stderr, "File input.jpg NOT FOUND!\n");
      // return EXIT_FAILURE;
  }


  cv::cvtColor( raw, gray, CV_BGR2GRAY );

  int const rows = gray.rows;
  int const cols = gray.cols;


  /* Compute edge map and gradients. */
  cv::Mat edges, dx, dy;

  edges.create( rows, cols, CV_8UC1 );
  // cv::Canny( gray, edges, MAX(canny_threshold/2,1), canny_threshold, 3 );
  // cv::Canny( gray, edges, 500,1000,5);

  cv::Canny( gray, edges, 500,700,5);
  // cv::Canny( gray, edges, 500,1000,5);

  namedWindow( "canny", WINDOW_AUTOSIZE );   // Create a window for display.
  imshow( "canny", edges ); 

  // imwrite(pathc+std::to_string(image_no)+".jpg",edges);


  dx.create( rows, cols, CV_16SC1 );
  dy.create( rows, cols, CV_16SC1 );
  cv::Sobel( gray, dx, CV_16S, 1, 0, 5 );
  cv::Sobel( gray, dy, CV_16S, 0, 1, 5 );


  /* Compute Hough transform. */
  std::vector<cv::Vec2f> vec_lines_normal, vec_lines_extended;
  cv::Mat accum_normal, accum_extended;

  bool const rhextended =
    htHoughLinesExtended_inline(
                         edges,
                         dx,
                         dy,
                         vec_lines_extended,
                         lines_rho,
                         lines_theta,
                         lines_acc_thr,
                         lines_max,
                         accum_extended
                         );


  assert(true == rhextended);


  double min_slope = 0.3;
  // double min_slope = 0.46;
  // double max_slope = 11.43;
  double max_slope = 7.43;

  // vector<Point> point1;
  // vector<Point> point2;
  std::vector<double> slope;


  for( unsigned int i = 0; i < vec_lines_extended.size(); ++i ){

    float rho = vec_lines_extended[i][0], theta = vec_lines_extended[i][1];

    // cout<<rho;
    Point pt1, pt2;
    double a = cos(theta), b = sin(theta);
    double x0 = a*rho, y0 = b*rho;
    pt1.x = cvRound(x0 + 1000*(-b));
    pt1.y = cvRound(y0 + 1000*(a));
    pt2.x = cvRound(x0 - 1000*(-b));
    pt2.y = cvRound(y0 - 1000*(a));



    double m = (double)(pt2.y-pt1.y)/(pt2.x-pt1.x);
    if (!(abs(m) < min_slope)  && !(abs(m) > max_slope)){

      slope.push_back(m);
      point1.push_back(pt1);
      point2.push_back(pt2);
      lane_rho.push_back(rho);
      lane_theta.push_back(theta);
    }
     
  }

  removeClustersExt(point1, point2, slope,lane_rho,lane_theta);
}



void imageCallback(const sensor_msgs::ImageConstPtr& original_image)
{
    //Convert from the ROS image message to a CvImage suitable for working with OpenCV for processing

  
  clock_t tStart = clock();


  Mat org;
  // ros::Rate loop_rate(1);


  cv_bridge::CvImagePtr cv_ptr;
  try
  {
    //Always copy, returning a mutable CvImage
    //OpenCV expects color images to use BGR channel order.
    cv_ptr = cv_bridge::toCvCopy(original_image, enc::BGR8);
  }
  catch (cv_bridge::Exception& e)
  {
    //if there is an error during conversion, display it
    ROS_ERROR("swarath_package::main.cpp::cv_bridge exception: %s", e.what());
    return;
  }

  //After reading in the video stream and converting it to openCV format, it
    //performs img processing steps.


  sensor_msgs::ImagePtr im_msg = cv_bridge::CvImage(std_msgs::Header(), "bgr8", cv_ptr->image).toImageMsg();
  im_msg = cv_bridge::CvImage(std_msgs::Header(), "bgr8", cv_ptr->image).toImageMsg();
  swarath_package::lstm_data msg;
  msg.im = *im_msg;


  resize(cv_ptr->image,cv_ptr->image,Size(640,480));
  cout <<  cv_ptr->image.rows<<" "<<cv_ptr->image.cols<<endl;




  int bonnet = 430;// row value from where bonnet of car starts.
  // int bonnet = 870;// row value from where bonnet of car starts.

  int sky = 185;// row value where sky ends
  // int sky = 260;// row value where sky ends
  // int sky = 475;// row value where sky ends

  Mat org_before_roi = cv_ptr->image;

  // imshow("OriginalBeforeCrop", org_before_roi);


  // Rect roi(0,sky,cv_ptr->image.cols -20,bonnet - sky);
  Rect roi(0,sky,cv_ptr->image.cols,bonnet - sky);
  cv_ptr->image = cv_ptr->image(roi);


  vector<Point> point1;
  vector<Point> point2;
  vector<Point> point3;
  vector<Point> point4;
  vector<float> lane_rho;
  vector<float> lane_theta;

  org = cv_ptr->image;

      // hough(cv_ptr->image,point1,point2);
  hough_ext(cv_ptr->image,point1,point2,lane_rho,lane_theta);


  // Here we get points


  for(int i=0;i<point1.size();i++){
   
    line(cv_ptr->image,point1[i],point2[i],Scalar(0,255,0),1,1);
  
    }


  
  
  vector<int> global_points;


  // global_label.push_back(point1.size());

  for(int i=0;i<point1.size();i++){

    global_points.push_back(point1[i].x);
    global_points.push_back(point1[i].y);
    global_points.push_back(point2[i].x);
    global_points.push_back(point2[i].y);

  }


  image_no+=1;

  imshow("LaneDetectionAlgo", cv_ptr->image);
 
  
  msg.points = global_points;
  pub.publish(msg);
  // loop_rate.sleep();

  printf("Time taken: %.4fs\n", (double)(clock() - tStart)/CLOCKS_PER_SEC);

  //Add some delay in miliseconds. The function only works if there is at least one HighGUI window created and the window is active. If there are several HighGUI windows, any of them can be active
  cv::waitKey(1);



}


int main(int argc, char **argv)
  {
    
    ros::init(argc, argv, "Lane_DetectorNode");
    ros::NodeHandle nh;


    //Create an ImageTransport instance, initializing it with our NodeHandle.
    image_transport::ImageTransport it(nh);

    //OpenCV HighGUI call to create a display window on start-up.
    cv::namedWindow(WINDOW, CV_WINDOW_AUTOSIZE);


   
    image_transport::Subscriber sub = it.subscribe("camera/image_color", 1, imageCallback);

    // image_transport::Subscriber sub = it.subscribe("/clock", 1, imageCallback);

    // pub = nh.advertise<swarath_package::lane_detection>("Lane_DetectorTopic", 1);
    // pub = nh.advertise<swarath_package::lane_detection>("lstm", 1);
    pub = nh.advertise<swarath_package::lstm_data>("lstm", 1);

    //OpenCV HighGUI call to destroy a display window on shut-down.
    cv::destroyWindow(WINDOW);


    ros::spin();
    //ROS_INFO is the replacement for printf/cout.
    ROS_INFO("swarath_package::main.cpp::No error.");


    return 0;
  }