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

typedef tuple<int, int> Result;
struct SortResults
{
	bool operator()(const Result& a, const Result& b)
	{
		if (get<0>(a) < get<0>(b))
			return true;

		return false;
	}
};

const string ASSETS_PATH = "../Assets/";
const string TRAIN_VOC_PATH = ASSETS_PATH + "vocabulary/";
const string TRAIN_CAT_PATH = ASSETS_PATH + "cat/";
const string TRAIN_DOG_PATH = ASSETS_PATH + "dog/";
const string TEST_PATH = ASSETS_PATH + "test/";
const int CAT_CLASS = 0;
const int DOG_CLASS = 1;
const int TRAIN_SIZE = 500; //12500
const int TEST_SIZE = 12500;

//Vocabulary
const int SURF_HESSIAN = 400;
const int CLUSTER_COUNT = 2000;

//Storage
const string VOCABULARY_PATH = ASSETS_PATH + "vocabulary.yml";
const string BAYES_PATH = ASSETS_PATH + "bayes.model";
const string SVM_PATH = ASSETS_PATH + "svm.model";
const string RESULTS_PATH = ASSETS_PATH + "results.csv";

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
	Ptr<FeatureDetector> detector = FeatureDetector::create(DETECTOR_TYPE);
	if (DETECTOR_TYPE == "SURF") detector->set("hessianThreshold", SURF_HESSIAN);

	Ptr<DescriptorExtractor> extractor = DescriptorExtractor::create(EXTRACTOR_TYPE);
	if (EXTRACTOR_TYPE == "SURF") extractor->set("hessianThreshold", SURF_HESSIAN);

	Mat trainDescriptors(1, extractor->descriptorSize(), extractor->descriptorType());

	//for each path, extract descriptors
	for (int path = 0; path < paths.size(); path++)
	{
		#pragma omp parallel for schedule(dynamic, 3)
		for (int i = 1; i <= TRAIN_SIZE; i++)
		{
			vector<KeyPoint> keypoints;
			Mat descriptors;
			Mat image = imread(paths[path] + to_string(i) + ".jpg", IMREAD_COLOR);
			detector->detect(image, keypoints);
			extractor->compute(image, keypoints, descriptors);

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

	FileStorage fs(VOCABULARY_PATH, FileStorage::WRITE);
	fs << "vocabulary" << vocabulary;
	fs.release();

	return vocabulary;
}

Mat loadVocabulary()
{
	Mat vocabulary;
	FileStorage fs(VOCABULARY_PATH, FileStorage::READ);

	fs["vocabulary"] >> vocabulary;
	fs.release();

	return vocabulary;
}

void addDataClass(string classPath, int classId, Mat vocabulary, map<int, Mat> &data)
{
	Ptr<FeatureDetector> detector = FeatureDetector::create(DETECTOR_TYPE);
	if (DETECTOR_TYPE == "SURF") detector->set("hessianThreshold", 100);

	Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create(MATCHER_TYPE);

	Ptr<DescriptorExtractor> extractor = DescriptorExtractor::create(EXTRACTOR_TYPE);
	if (EXTRACTOR_TYPE == "SURF") extractor->set("hessianThreshold", 100);

	BOWImgDescriptorExtractor bowide(extractor, matcher);
	bowide.setVocabulary(vocabulary);

	#pragma omp parallel for schedule(dynamic, 3)
	for (int i = 1; i <= TRAIN_SIZE; i++)
	{
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

//Train methods

Ptr<CvNormalBayesClassifier> computeBayesClassifier(map<int, Mat> data)
{
	//prepare data
	Mat samples;
	Mat labels;

	map<int, Mat>::iterator it;
	for (it = data.begin(); it != data.end(); it++)
	{
		int label = it->first;
		Mat sample = it->second;

		Mat labelMat = Mat(sample.rows, 1, sample.type(), Scalar(label));
		labels.push_back(labelMat);
		samples.push_back(sample);
	}

	//train
	Ptr<CvNormalBayesClassifier> classifier = new CvNormalBayesClassifier();
	classifier->train(samples, labels);
	classifier->save(BAYES_PATH.c_str());
	
	return classifier;
}

Ptr<CvSVM> computeSVMClassifier(map<int, Mat> data)
{
	//prepare data
	Mat samples;
	Mat labels;

	map<int, Mat>::iterator it;
	for (it = data.begin(); it != data.end(); it++)
	{
		int label = it->first;
		Mat sample = it->second;

		Mat labelMat = Mat(sample.rows, 1, sample.type(), Scalar(label));
		labels.push_back(labelMat);
		samples.push_back(sample);
	}

	//train
	Ptr<CvSVM> classifier = new CvSVM();
	classifier->train(samples, labels);
	classifier->save(SVM_PATH.c_str());

	return classifier;
}

//Test methods
void testBayesClassifier(Ptr<CvNormalBayesClassifier> classifier, Mat vocabulary)
{
	set<Result, SortResults> sortedResults;

	Ptr<FeatureDetector> detector = FeatureDetector::create(DETECTOR_TYPE);
	if (DETECTOR_TYPE == "SURF") detector->set("hessianThreshold", SURF_HESSIAN);

	Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create(MATCHER_TYPE);
	Ptr<DescriptorExtractor> extractor = DescriptorExtractor::create(EXTRACTOR_TYPE);
	if (EXTRACTOR_TYPE == "SURF") extractor->set("hessianThreshold", SURF_HESSIAN);

	BOWImgDescriptorExtractor bowide(extractor, matcher);
	bowide.setVocabulary(vocabulary);

	#pragma omp parallel for schedule(dynamic, 3)
	for (int i = 1; i <= TEST_SIZE; i++)
	{
		if (i % 500 == 0)
			cout << i << endl;

		Mat image = imread(TEST_PATH + to_string(i) + ".jpg", IMREAD_COLOR);
		vector<KeyPoint> kp;
		Mat hst;
		detector->detect(image, kp);
		bowide.compute(image, kp, hst);

		int classId;

		if (hst.rows == 0)
		{
			classId = 0;
		}
		else
		{
			classId = (int)classifier->predict(hst);
		}

		Result r(i, classId);

		#pragma omp critical
		{
			sortedResults.insert(r);
		}
	}

	string results = "id,label\n";
	set<Result>::iterator it;

	for (it = sortedResults.begin(); it != sortedResults.end(); it++)
	{
		Result r = *it;
		results += to_string(get<0>(r)) + "," + to_string(get<1>(r)) + "\n";
	}

	ofstream file(RESULTS_PATH);
	file << results;
}

void testSVMClassifier(Ptr<CvSVM> classifier, Mat vocabulary)
{
	set<Result, SortResults> sortedResults;

	Ptr<FeatureDetector> detector = FeatureDetector::create(DETECTOR_TYPE);
	if (DETECTOR_TYPE == "SURF") detector->set("hessianThreshold", SURF_HESSIAN);

	Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create(MATCHER_TYPE);
	Ptr<DescriptorExtractor> extractor = DescriptorExtractor::create(EXTRACTOR_TYPE);
	if (EXTRACTOR_TYPE == "SURF") extractor->set("hessianThreshold", SURF_HESSIAN);

	BOWImgDescriptorExtractor bowide(extractor, matcher);
	bowide.setVocabulary(vocabulary);

	#pragma omp parallel for schedule(dynamic, 3)
	for (int i = 1; i <= TEST_SIZE; i++)
	{
		if (i % 500 == 0)
			cout << i << endl;

		Mat image = imread(TEST_PATH + to_string(i) + ".jpg", IMREAD_COLOR);
		vector<KeyPoint> kp;
		Mat hst;
		detector->detect(image, kp);
		bowide.compute(image, kp, hst);

		int classId;

		if (hst.rows == 0)
		{
			classId = 0;
		}
		else
		{
			classId = (int)classifier->predict(hst);
		}

		Result r(i, classId);

		#pragma omp critical
		{
			sortedResults.insert(r);
		}
	}

	string results = "id,label\n";
	set<Result>::iterator it;

	for (it = sortedResults.begin(); it != sortedResults.end(); it++)
	{
		Result r = *it;
		results += to_string(get<0>(r)) + "," + to_string(get<1>(r)) + "\n";
	}

	ofstream file(RESULTS_PATH);
	file << results;
}

int main()
{
	initModule_nonfree();
	double t;

	cout << "Creating vocabulary" << endl;
	t = GetTickCount();
	//Create a vocabulary from the whole dataset (cats AND dogs)
	vector<string> datasetPaths;
	datasetPaths.push_back(TRAIN_VOC_PATH);
	//datasetPaths.push_back(TRAIN_CAT_PATH);
	//datasetPaths.push_back(TRAIN_DOG_PATH);
	Mat vocabulary = createVocabulary(datasetPaths);
//	Mat vocabulary = loadVocabulary();
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
	//Ptr<CvNormalBayesClassifier> classifier = computeBayesClassifier(data);
	Ptr<CvSVM> classifier = computeSVMClassifier(data);
	//---------------
	cout << GetTickCount() - t << "ms" << endl;

	cout << "Testing" << endl;
	t = GetTickCount();
	//Test classifier
	//testBayesClassifier(classifier, vocabulary);
	testSVMClassifier(classifier, vocabulary);
	//--------------
	cout << GetTickCount() - t << "ms" << endl;

	//gg
	cout << "finish" << endl;
	int a;
	cin >> a;

	return 0;
}