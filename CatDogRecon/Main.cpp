#include <opencv\cv.h>
#include <opencv2\core\core_c.h>
#include <opencv2\highgui\highgui.hpp>
#include <opencv2\imgproc\imgproc.hpp>
#include <opencv2\features2d\features2d.hpp>
#include <opencv2\nonfree\nonfree.hpp>
#include <opencv2\nonfree\features2d.hpp>
#include <opencv2/ml/ml.hpp>

#include <iostream>
#include <fstream>
#include <string>
#include <limits>
#include <omp.h>
#include <Windows.h>

using namespace cv;
using namespace std;

const string ASSETS_PATH = "../Assets/";
const string TRAIN_CAT_PATH = ASSETS_PATH + "cat/";
const string TRAIN_DOG_PATH = ASSETS_PATH + "dog/";
const string TEST_PATH = ASSETS_PATH + "test/";
const int CAT_CLASS = 1;
const int DOG_CLASS = 2;
const int TRAIN_SIZE = 100; //12500

//Vocabulary
const int SURF_HESSIAN = 400;
const int CLUSTER_COUNT = 100; //1000?

//Data

/*
"FAST" – FastFeatureDetector
"STAR" – StarFeatureDetector
"SIFT" – SIFT (nonfree module)
"SURF" – SURF (nonfree module)
"ORB" – ORB
"BRISK" – BRISK
"MSER" – MSER
"GFTT" – GoodFeaturesToTrackDetector
"HARRIS" – GoodFeaturesToTrackDetector with Harris detector enabled
"Dense" – DenseFeatureDetector
"SimpleBlob" – SimpleBlobDetector
*/
const string DETECTOR_TYPE = "SURF";


/*
BruteForce (it uses L2 )
BruteForce-L1
BruteForce-Hamming
BruteForce-Hamming(2)
FlannBased
*/
const string MATCHER_TYPE = "BruteForce";


/*
"SIFT" – SIFT
"SURF" – SURF
"BRIEF" – BriefDescriptorExtractor
"BRISK" – BRISK
"ORB" – ORB
"FREAK" – FREAK
*/
const string EXTRACTOR_TYPE = "SURF";


Mat createVocabulary(vector<string> paths)
{
	//Extract train set descriptors and store them
	SurfFeatureDetector detector(SURF_HESSIAN);
	SurfDescriptorExtractor extractor;

	Mat trainDescriptors(1, extractor.descriptorSize(), extractor.descriptorType());

	//for each path, extract descriptors
	for (int path = 0; path < paths.size(); path++)
	{
		#pragma omp parallel for schedule(dynamic, 3)
		for (int i = 1; i <= TRAIN_SIZE; i++)
		{
			cout << i << endl;
			vector<KeyPoint> keypoints;
			Mat descriptors;
			Mat image = imread(paths[path] + to_string(i) + ".jpg", IMREAD_COLOR);
			detector.detect(image, keypoints);
			extractor.compute(image, keypoints, descriptors);

			#pragma omp critical
			{
				trainDescriptors.push_back(descriptors);
			}
		}
	}

	//Create a vocabulary from the stored descriptors
	BOWKMeansTrainer bowTrainer(CLUSTER_COUNT); //clusters
	bowTrainer.add(trainDescriptors);
	Mat vocabulary = bowTrainer.cluster();
	return vocabulary;
}

void addDataClass(string classPath, int classId, Mat vocabulary, map<int, Mat> &data)
{
	Ptr<FeatureDetector> detector = FeatureDetector::create(DETECTOR_TYPE);
	Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create(MATCHER_TYPE);
	Ptr<DescriptorExtractor> extractor = DescriptorExtractor::create(EXTRACTOR_TYPE);

	BOWImgDescriptorExtractor bowide(extractor, matcher);
	bowide.setVocabulary(vocabulary);

	#pragma omp parallel for schedule(dynamic, 3)
	for (int i = 1; i <= TRAIN_SIZE; i++)
	{
		cout << i << endl;
		vector<KeyPoint> keypoints;
		Mat responseHist;
		Mat image = imread(classPath + to_string(i) + ".jpg", IMREAD_COLOR);
		detector->detect(image, keypoints);
		bowide.compute(image, keypoints, responseHist);

		#pragma omp critical
		{
			if (data.count(classId) == 0)
			{
				data[classId].create(0, responseHist.cols, responseHist.type());
			}

			data[classId].push_back(responseHist);
		}
	}
}

CvNormalBayesClassifier computeBayesClassifier(map<int, Mat> data)
{	
	//prepare data
	Mat samples(0, CLUSTER_COUNT * data.size(), CV_32FC1); //why clustercount
	int label[4] = { 1, -1, -1, -1 };
	Mat labels(4, 1, CV_32SC1, label);

	map<int, Mat>::iterator it;
	for (it = data.begin(); it != data.end(); it++)
	{
		cout << "1";
		labels.push_back(it->first);
		cout << "2";
		samples.push_back(it->second);
		cout << "3";
	}

	//train
	CvNormalBayesClassifier classifier;
	classifier.train(samples, labels);
	classifier.save("../Assets/bayes.model");
	
	return classifier;
}

int main()
{
	double t;

	cout << "Creating vocabulary" << endl;
	t = GetTickCount();
	//Create a vocabulary from the whole dataset (cats AND dogs)
	vector<string> datasetPaths;
	datasetPaths.push_back(TRAIN_CAT_PATH);
	datasetPaths.push_back(TRAIN_DOG_PATH);
	Mat vocabulary = createVocabulary(datasetPaths);
	//----------------------------------------------------------
	cout << GetTickCount() - t << "ms" << endl;

	cout << "Processing data" << endl;
	t = GetTickCount();
	//Process data from vocabulary
	map<int, Mat> data;
	addDataClass(TRAIN_CAT_PATH, CAT_CLASS, vocabulary, data); //add cats
	addDataClass(TRAIN_DOG_PATH, DOG_CLASS, vocabulary, data); //add dogs
	//----------------------------
	cout << GetTickCount() - t << "ms" << endl;
	
	cout << "Training" << endl;
	t = GetTickCount();
	//Train with data
	CvNormalBayesClassifier classifier = computeBayesClassifier(data);
	//---------------
	cout << GetTickCount() - t << "ms" << endl;

	//Try
	cout << "Trying" << endl;
	for (int i = 1; i <= TRAIN_SIZE; i++)
	{
		Mat image = imread(TEST_PATH + to_string(i) + ".jpg", IMREAD_COLOR);

		int classId = classifier.predict(image);
		if (classId == CAT_CLASS) cout << "cat" << endl;
		if (classId == DOG_CLASS) cout << "dog" << endl;

		imshow("image", image);
		waitKey();
	}

	//gg
	cout << "finish" << endl;
	return 0;
}