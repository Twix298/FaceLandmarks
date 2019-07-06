// Recognition.cpp : Этот файл содержит функцию "main". Здесь начинается и заканчивается выполнение программы.
//

#include <iostream>
#include <conio.h>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/face.hpp>
#include "drawLandmarks.hpp"

using namespace std;
using namespace cv;
using namespace cv::face;
int main()
{
	// Load Face Detector
	CascadeClassifier faceDetector("haarcascade_frontalface_alt2.xml");
	// Create an instance of Facemark
	Ptr<Facemark> facemark = FacemarkLBF::create();
	// Load landmark detector
	facemark->loadModel("lbfmodel.yaml");
	// Set up webcam for video capture

	VideoCapture cam(0);
	// Variable to store a video frame and its grayscale 
	Mat frame, gray;
	// Read a frame
	while (cam.read(frame))
	{
		// Find face
		vector<Rect> faces;
		// Convert frame to grayscale because
		// faceDetector requires grayscale image.
		cvtColor(frame, gray, COLOR_BGR2GRAY);
		// Detect faces
		faceDetector.detectMultiScale(gray, faces);
		// Variable for landmarks. 
		 // Landmarks for one face is a vector of points
		 // There can be more than one face in the image. Hence, we 
		 // use a vector of vector of points. 
		vector< vector<Point2f> > landmarks;
		// Run landmark detector
		bool success = facemark->fit(frame, faces, landmarks);
		if (success)
		{
			// If successful, render the landmarks on the face
			for (size_t i = 0; i < faces.size(); i++)
			{
				cv::rectangle(frame, faces[i], Scalar(0, 255, 0), 3);
			}
			for (int i = 0; i < landmarks.size(); i++)
			{
				drawLandmarks(frame, landmarks[i]);
				/*for (size_t j = 0; j < landmarks[i].size(); j++)
				 circle(frame, Point(landmarks[i][j].x, landmarks[i][j].y), 1, Scalar(255, 0, 0), 2);*/
			}
		}
		// Display results 
		imshow("Facial Landmark Detection", frame);
		// Exit loop if ESC is pressed
		if (waitKey(1) == 27) break;
	}
	return 0;
}

