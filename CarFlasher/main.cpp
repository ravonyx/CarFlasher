//Auteur : Manon Maillard

#include "stdafx.h"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/video/tracking.hpp>
#include <iostream>


cv::Mat nextInput;
std::vector<cv::Point2f> prevPoints;

cv::Mat prevInput;
std::vector<cv::Point2f> nextPoints;

cv::Rect roi;
cv::Point start(-1, -1);

void draw();
int video(char* videoname);
void detectPoints(cv::Mat & img);
void trackPoints();
std::vector<cv::Point2f> purgePoints(std::vector<cv::Point2f>& points, std::vector<uchar>& status);
void CallBackFunc(int event, int x, int y, int flags, void* userdata);
void updateROI();

int main()
{
	video("");
	return 0;
}

void draw()
{
	cv::Mat img = nextInput.clone();
	
	//draw prev points
	for (size_t i = 0; i < prevPoints.size(); i++)
	{
		cv::Point center(cvRound(prevPoints[i].x), cvRound(prevPoints[i].y));
		int radius = 5;
		cv::circle(img, center, radius, cv::Scalar(0, 255, 0), 2);
	}

	//draw next points
	for (size_t i = 0; i < nextPoints.size(); i++)
	{
		cv::Point center(cvRound(nextPoints[i].x), cvRound(nextPoints[i].y));
		int radius = 5;
		cv::circle(img, center, radius, cv::Scalar(255, 0, 0), 2);
	}

	//draw lines
	for (size_t i = 0; i < nextPoints.size(); i++)
	{
		cv::Point center(cvRound(nextPoints[i].x), cvRound(nextPoints[i].y));
		int radius = 5;
		cv::line(img, prevPoints[i], nextPoints[i], cv::Scalar(255, 255, 255), 2);
	}
	//draw rectangle
	cv::rectangle(img, roi, cv::Scalar(255, 255, 255));

	cv::imshow("input", img);
	cv::setMouseCallback("input", CallBackFunc, NULL);
}

int video(char* videoname) 
{
	cv::VideoCapture cap;
	if (videoname != "")
	{
		cap = cv::VideoCapture(videoname);
		if (!cap.isOpened())  // check if we succeeded
			return -1;
	}
	else
	{
		cap = cv::VideoCapture(0); // open the default camera
		if (!cap.isOpened())  // check if we succeeded
			return -1;
	}

	cap >> nextInput;
	while (!nextInput.empty())
	{
		trackPoints();
		updateROI();
		draw();
		cap >> nextInput;
		if (cv::waitKey(10) >= 0) break;
	}
}

void detectPoints(cv::Mat & img)
{
	int maxCorners = 155;
	double qualityLevel = 0.01;
	double minDistance = 10;
	int blockSize = 3;
	bool useHarrisDetector = false;
	double k = 0.04;

	cv::Mat gray;
	cv::cvtColor(img, gray, CV_BGR2GRAY);
	
	if (start.x >= 0 || roi.area() < 10)
		return;
	else
	{
		//creation mask
		cv::Mat mask(img.size(), CV_8UC1, cv::Scalar(0));
		mask(roi) = 1;

		//get points to track
		cv::goodFeaturesToTrack(gray, prevPoints, maxCorners, qualityLevel, minDistance, mask, blockSize, useHarrisDetector, k);
	}
}

void trackPoints() 
{
	int minPoints = 10;
	if (!prevInput.empty())
	{
		prevPoints = nextPoints;
		if (prevPoints.size() < minPoints)
			detectPoints(prevInput);
		if (prevPoints.size() < minPoints)
			return;
		std::vector<uchar> status(prevPoints.size());
		std::vector<float> err(prevPoints.size());
		cv::calcOpticalFlowPyrLK(prevInput, nextInput, prevPoints, nextPoints, status, err);
		purgePoints(prevPoints, status);
		purgePoints(nextPoints, status);
	}
	prevInput = nextInput.clone();
}

std::vector<cv::Point2f> purgePoints(std::vector<cv::Point2f>& points,
	std::vector<uchar>& status) 
{
	std::vector<cv::Point2f> result;
	for (int i = 0; i < points.size(); ++i) {
		if (status[i]>0)result.push_back(points[i]);
	}
	return result;
}

void CallBackFunc(int event, int x, int y, int flags, void* userdata)
{
	if (event == cv::EVENT_LBUTTONDOWN)
	{
		start = cv::Point(x, y);
		roi = cv::Rect();
		prevPoints.clear();
		nextPoints.clear();
	}
	else if (event == cv::EVENT_MOUSEMOVE)
	{
		if (start.x >= 0) 
		{
			cv::Point end(x, y);
			roi = cv::Rect(start, end);
		}
	}
	else if (event == cv::EVENT_LBUTTONUP) 
	{
		cv::Point end(x, y);
		roi = cv::Rect(start, end);
		start = cv::Point(-1, -1);
	}
}

void updateROI()
{
	if (start.x >= 0 || roi.area() < 10)
		return;
	else
		roi = cv::boundingRect(nextPoints);
}
