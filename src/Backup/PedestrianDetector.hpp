/*
 * PedestrianDetector.hpp
 *
 *  Created on: Mar 27, 2015
 *      Author: pistol
 */

#ifndef PEDESTRIANDETECTOR_HPP_
#define PEDESTRIANDETECTOR_HPP_

#include <cv.h>
#include <vector>

struct bbType
{
	double x1;
	double y2;
	double width3;
	double height4;
	double score5;
	double distance = -1;
	double angle;
};

struct LoadMatVariableType
{
	std::vector<int> matIndex;
};

struct ImagePyramid
{
	int nApprox;
	double nPerOct;
	double nOctUp;
	cv::Size2i pad; // OPS: cols x rows, width x height.
	cv::Size2i minDs; // OPS: cols x rows, width x height.
	std::vector<double> lambdas;
	double shrink;
	cv::Size2i sz; // OPS: cols x rows, width x height.
	int binSize;
};
struct PedOptions
{
	cv::Size2i modelDs; // OPS: cols x rows, width x height.
	cv::Size2i modelDsPad; // OPS: cols x rows, width x height.
	double stride;
	double cascThr;
	double cascCal;
};

struct PedModel
{
	std::vector<uint32_t> fids;
	std::vector<float> thrs;
	std::vector<uint32_t> child;
	std::vector<float> hs;
	std::vector<float> filter;
	std::vector<float> filterDim;
	cv::Mat kernel[10][4];
	int treeDepth;
	int nTreeNodes;
	int nTrees;
};

class PedestrianDetector {
private:
	const double PI  = 3.141592653589793238463;
	PedModel pedModel;
	ImagePyramid imagePyramid;

	PedOptions pedOptions;
	bool validMatFile = false;
	bool fastDetector;
	int nScales;
	cv::Size2i dimImage;
	cv::Mat scaleshw, scales;
	std::vector<int> isRealPyramid,isApproxPyramid,isN;
	double FOV_verticalRad;
	double FOV_horizontalRad;
	double angleTiltRad;
	double cameraHeight;
	bool cameraSettingsProvided = false;

	void loadMatFile(std::string strPedModel);
	void getScales(double nPerOct, double nOctUp, cv::Size2i minDsIn, double shrink, cv::Size2i szIn, cv::Mat &scalesMat, cv::Mat &scaleshwMat);
	void getPyramidIndexes(int nScales, int nApprox,std::vector<int> &isRealPyramid,std::vector<int> &isApproxPyramid,std::vector<int> &isN);
	std::vector<bbType> nmsMax(std::vector<bbType> bbs, float overlap);
	cv::Mat convert_OpenCV_2_PD(cv::Mat inputImage);
	cv::Mat convert_PD_2_OpenCV(cv::Mat image,bool toShowImageFormat);
	void DeterminePyramidParameters(cv::Size2i imageSize);
	void makeKernels();
public:


	PedestrianDetector (std::string);
	std::vector<bbType> pedDetector(cv::Mat inputImage);
	void setCameraSetup(double FOV_verticalDeg, double FOV_horizontalDeg, double angleTiltDegrees, double cameraHeightIn);
};

#endif /* PEDESTRIANDETECTOR_HPP_ */
