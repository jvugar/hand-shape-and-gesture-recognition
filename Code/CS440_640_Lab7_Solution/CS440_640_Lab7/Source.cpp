/*	CS440 P2
*	Artificial Intelligence Spring 2018
*	--------------
*	This program introduces the following concepts:
*		a) Reading a stream of images from a webcamera, and displaying the video
*		b) Skin color detection
*		c) horizontal and vertical projections to find bounding boxes of "skin-color blobs"
*		d) circularity of ”skin-color blobs”
*       e) tracking the position and orientation of moving objects
*	--------------
*/

//opencv libraries
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
//C++ standard libraries
#include <iostream>
#include <vector>

using namespace cv;
using namespace std;



/*
Function that detects whether a pixel belongs to the skin based on RGB values
@param src The source color image
@param dst The destination grayscale image where skin pixels are colored white and the rest are colored black
*/
void mySkinDetect(Mat& src, Mat& dst);

/*
Function that returns the maximum value of three ints
@params a b c ints to be compared
*/
int myMax(int a, int b, int c);

/*
Function that returns the minimum value of three ints
@params a b c ints to be compared
*/
int myMin(int a, int b, int c);

/*
Function that calculates the circularity of an image
@param dst The source image used to calculate circularity
*/
float circularity(Mat& dst);

/*
Function that calculates the center of mass of two successive images, and returns the direction of movement
@param first The first image
@param second The second image
*/
int direction(Mat& first, Mat& second);

/*
Function that does erosion and dilation to an image
@param dst The source image used to do erosion and dilation
*/
void erodil(Mat& dst);

/*
Function that draws a bounding box around the detected skin
@param dst The source image that contains the detected skin
*/
void boundingbox(Mat& dst);




int main(){



	// open the video camera no. 0
	VideoCapture cap(0);

	// if not successful, exit program
	if (!cap.isOpened())
	{
		cout << "Cannot open the video cam" << endl;
		return -1;
	}


	Mat frame0;

	// read a new frame from video
	bool bSuccess0 = cap.read(frame0);

	//if not successful, break loop
	if (!bSuccess0)
	{
		cout << "Cannot read a frame from video stream" << endl;
	}
	//create a window named Skin
	namedWindow("Skin", WINDOW_AUTOSIZE);


	//create a vector of matrixs and append two zero matrixs of same size as src mat
	vector<Mat> myMotionHistory;
	Mat fMH1, fMH2;
	fMH1 = Mat::zeros(frame0.rows, frame0.cols, CV_8UC1);
	fMH2 = fMH1.clone();
	myMotionHistory.push_back(fMH1);
	myMotionHistory.push_back(fMH2);

	while (1)
	{
		// read a new frame from video
		Mat frame;
		bool bSuccess = cap.read(frame);
		//if not successful, break loop
		if (!bSuccess)
		{
			cout << "Cannot read a frame from video stream" << endl;
			break;
		}

		// destination frame
		Mat frameDest;

		//returns a zero matrix of same size as src mat
		frameDest = Mat::zeros(frame.rows, frame.cols, CV_8UC1); 
		
		//do skin detection to the new frame, and put the result into frameDest
		mySkinDetect(frame, frameDest);

		//do erosion and dilation to frameDest
		erodil(frameDest);

		

		
	

		//make a copy of the new frame
		Mat frame1 = frame.clone();

		//erase the old frame, and append the frameDest into the vector
		myMotionHistory.erase(myMotionHistory.begin());
		myMotionHistory.push_back(frameDest);

		//do erosion and dilation to both images in vector
		erodil(myMotionHistory[0]);
		erodil(myMotionHistory[1]);

		//return the movement direction of sucessive images
		int result = direction(myMotionHistory[0], myMotionHistory[1]);

		//calculate the circularity of skin in frameDest
		float cr = circularity(frameDest);

		//if circularity is larger than the threshold and images are static, put text Fist in the new frame
		if (cr > 0.65 && result==3){
			putText(frame, "Fist", Point(100, 100), FONT_HERSHEY_PLAIN, 6, (200, 255, 155), 13, LINE_AA);
			//draw a bounding box around the skin in frameDest
			boundingbox(frameDest);
		}
		//show the frame image in Static window 
		imshow("Static", frame);

		//if result is horizontal movement, put text Wave Hand in frame1, if vertical movement, put text Raise Hand in frame1
		if (result == 1){
			putText(frame1, "Wave Hand", Point(100, 100), FONT_HERSHEY_PLAIN, 6, (200, 255, 155), 13, LINE_AA);
			//draw a bounding box around the skin in frameDest
			boundingbox(frameDest);
		}
		else if (result == 2){
			putText(frame1, "Raise Hand", Point(100, 100), FONT_HERSHEY_PLAIN, 6, (200, 255, 155), 13, LINE_AA);
			//draw a bounding box around the skin in frameDest
			boundingbox(frameDest);
		}
		else{}

		//show the frameDest image in Skin window
		imshow("Skin", frameDest);
		//show the frame1 image in Dynamic window
		imshow("Dynamic", frame1);

		//update frame0 to the newest frame
		frame0 = frame;
		//wait for 'esc' key press for 30ms. If 'esc' key is pressed, break loop
		if (waitKey(30) == 27)
		{
			cout << "esc key is pressed by user" << endl;
			break;
		}

	}
	cap.release();
	return 0;
}



int myMax(int a, int b, int c) {
	int m = a;
	(void)((m < b) && (m = b));
	(void)((m < c) && (m = c));
	return m;
}


int myMin(int a, int b, int c) {
	int m = a;
	(void)((m > b) && (m = b));
	(void)((m > c) && (m = c));
	return m;
}


void mySkinDetect(Mat& src, Mat& dst) {
	//Surveys of skin color modeling and detection techniques:
	//Vezhnevets, Vladimir, Vassili Sazonov, and Alla Andreeva. "A survey on pixel-based skin color detection techniques." Proc. Graphicon. Vol. 3. 2003.
	//Kakumanu, Praveen, Sokratis Makrogiannis, and Nikolaos Bourbakis. "A survey of skin-color modeling and detection methods." Pattern recognition 40.3 (2007): 1106-1122.
	for (int i = 0; i < src.rows; i++){
		for (int j = 0; j < src.cols; j++){
			//For each pixel, compute the average intensity of the 3 color channels
			//Vec3b is a vector of 3 uchar (unsigned character)
			Vec3b intensity = src.at<Vec3b>(i, j); 
			int B = intensity[0]; int G = intensity[1]; int R = intensity[2];
			//if condition is satisifid, make the corresponding pixel white
			if ((R > 95 && G > 40 && B > 20) && (myMax(R, G, B) - myMin(R, G, B) > 15) && (abs(R - G) > 15) && (R > G) && (R > B)){
				dst.at<uchar>(i, j) = 255;
			}
		}
	}
}







float circularity(Mat& dst){
	//find moment of the dst image.
	Moments mo;
	mo = moments(dst, true);
	//calculate a,b,c,h,Emin,Emax in formula provided in lecture
	float a = mo.mu20;
	float b = 2 * mo.mu11;
	float c = mo.mu02;
	float h = sqrt(pow(a - c, 2) + pow(b, 2));
	float Emin = (a + c) / 2 - pow(a - c, 2) / (2 * h) - pow(b, 2) / (2 * h);
	float Emax = (a + c) / 2 + pow(a - c, 2) / (2 * h) + pow(b, 2) / (2 * h);
	//calculate circularity and return
	float circularity = Emin / Emax;
	return circularity;

}


int direction(Mat& first, Mat& second){
	//find moment of first and second images.
	Moments mo1 = moments(first, true);
	Moments mo2 = moments(second, true);
	//calculate center of mass of both images
	float x1 = mo1.m10 / mo1.m00;
	float y1 = mo1.m01 / mo1.m00;
	float x2 = mo2.m10 / mo2.m00;
	float y2 = mo2.m01 / mo2.m00;

	//if move horizontally, return 1, if move vertically,return 2, if no move, return 3
	float absx = abs(x1 - x2);
	float absy = abs(y1 - y2);
	if (absx > 30 || absy > 30){
		if (absx > absy){
			return 1;
		}
		else{
			return 2;
		}
	}
	return 3;

}

void erodil(Mat& dst){
	//set pararmeters and do erosion and dilation
	int erosion_size = 3;
	int dilation_size = 3;
	Mat element = getStructuringElement(MORPH_RECT, Size(2 * erosion_size + 1, 2 * erosion_size + 1), Point(erosion_size, erosion_size));
	erode(dst, dst, element);
	dilate(dst, dst, element);
}

void boundingbox(Mat& dst){
	//horizontal histogram
	Mat horizontal(dst.cols, 1, CV_32S);
	horizontal = Scalar::all(0);
	//vertical histogram
	Mat vertical(dst.rows, 1, CV_32S);
	vertical = Scalar::all(0);
	//calculate projections to x and y axies  
	for (int i = 0; i < dst.cols; i++)
	{
		horizontal.at<int>(i, 0) = countNonZero(dst(Rect(i, 0, 1, dst.rows)));
	}
	for (int i = 0; i < dst.rows; i++)
	{
		vertical.at<int>(i, 0) = countNonZero(dst(Rect(0, i, dst.cols, 1)));
		vertical.at<int>(i, 0) = countNonZero(dst(Rect(0, i, dst.cols, 1)));
	}

	//find the points that define the rectangle around the skin
	int x1 = 0;
	int x2 = 0;
	int y1 = 0;
	int y2 = 0;

	
	for (int i = 0; i < horizontal.rows - 1; i++){
		if (horizontal.at<int>(i, 0)>25){
			x1 = i;
			break;
		}
	}
	for (int i = horizontal.rows - 1; i > 0; i--){
		if (horizontal.at<int>(i, 0) >25){
			x2 = i;
			break;
		}
	}
	for (int i = 0; i < vertical.rows - 1; i++){
		if (vertical.at<int>(i, 0) >25){
			y1 = i;
			break;
		}
	}
	for (int i = vertical.rows - 1; i > 0; i--){
		if (vertical.at<int>(i, 0) > 25){
			y2 = i;
			break;
		}
	}
	//draw the rectangle around the skin in dst
	Rect rec(x1, y1, x2, y2);
	rectangle(dst, rec, Scalar(255, 0, 0), 2,8,0);
}
	