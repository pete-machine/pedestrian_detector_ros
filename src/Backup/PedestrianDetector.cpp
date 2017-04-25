/*
 * PedestrianDetector.cpp
 *
 *  Created on: Mar 27, 2015
 *      Author: pistol
 */


//#include "rgbConvertMex.hpp"
#include "mexFunctionsPiotrDollar.hpp"
#include <cv.h>
#include <highgui.h>
#include "matio.h"


//#include "gradientMex.hpp"
using namespace std;
using namespace cv;

//const char *file = "filename";
#define detOpt					0
#define detClf 					1
#define detInfo					2

#define detClf_featureIndex 	0
#define detClf_threshold 		1
#define detClf_child 			2
#define detClf_hs 				3
#define detClf_weight 			4
#define detClf_depth 			5
#define detClf_errs 			6
#define detClf_losses 			7
#define detClf_threeDepth		8

//#define detOpt_featureIndex 	0
#define detOptPyramid 			0
#define detOpt_filters 			1
#define detOpt_modelDs 			2
#define detOpt_modelDsPad		3
#define detOpt_stride 			5
#define detOpt_cascThr 			6
#define detOpt_cascCal 			7

//#define detOpt_featureIndex 	0
#define detOptPyramidChns 				0
#define detOptPyramid_nPerOct 			1
#define detOptPyramid_nOctUp 			2
#define detOptPyramid_nApprox 			3
#define detOptPyramid_lambdas 			4
#define detOptPyramid_pad	 			5
#define detOptPyramid_minDs 			6

#define detOptPyramidChns_shrink		0

PedestrianDetector::PedestrianDetector(std::string strPedModel){
	loadMatFile(strPedModel);

	dimImage.height = 0; // Initializing of dimImage-variable.
	dimImage.width = 0; // Initializing of dimImage-variable.
	if(pedModel.filterDim[0] == 0) { // The ACF-type detector
		fastDetector = true;
		imagePyramid.binSize = 4; // Bad-fix (to avoid loading it from the mat-file)
	}
	else { // The lcdf-type detector
		fastDetector = false;
		makeKernels();
		imagePyramid.binSize = 2; // Bad-fix (to avoid loading it from the mat-file)
	}
	//shrink = 4;
	//Size2i sz(inputImage.cols,inputImage.rows); // OPS: cols x rows, width x height.


}

void PedestrianDetector::makeKernels() {


	int nKernels = pedModel.filterDim[3]; // 4
	int nChannels = pedModel.filterDim[2]; // 10


	for(int n = 0; n<pedModel.filter.size(); n++) {
		cout << pedModel.filter[n] << ", ";
	}
	cout << endl;

	int countFilterCoef = 0;
	//for(int iFilterCoef = 0; iFilterCoef < pedModel.filter
	for(int iiChKernel = 0; iiChKernel < nKernels; iiChKernel++) {// Over 4 kernels
		for(int iChannel = 0; iChannel < nChannels; iChannel++) { // Over 10 scales
			Mat tmpKernel(Size(pedModel.filterDim[0],pedModel.filterDim[1]),CV_32F);
			for(int iiiCol = 0; iiiCol < pedModel.filterDim[1]; iiiCol++) {
				for(int iiiiRow = 0; iiiiRow < pedModel.filterDim[0]; iiiiRow++) {
					tmpKernel.at<float>(iiiiRow,iiiCol) = pedModel.filter[countFilterCoef++];
				}
			}
			pedModel.kernel[iChannel][iiChKernel] = tmpKernel;
		}
	}

//	for(int iChannel = 0; iChannel < nChannels; iChannel++) { // Over 10 scales
//		for(int iiChKernel = 0; iiChKernel < nKernels; iiChKernel++) {// Over 4 kernels
//			cout <<	pedModel.kernel[iChannel][iiChKernel] << endl;
//		}
//	}
}

void PedestrianDetector::DeterminePyramidParameters(cv::Size2i imageSize) {
	dimImage = imageSize;
	// Calculates scales and scaleshw.
	getScales(imagePyramid.nPerOct,imagePyramid.nOctUp,imagePyramid.minDs,imagePyramid.shrink,dimImage,scales,scaleshw);

	// Calculates isRealPyramid, isApproxPyramid and isN.
	nScales = scales.size[0];
	getPyramidIndexes(nScales, imagePyramid.nApprox,isRealPyramid,isApproxPyramid,isN);
}

// Creates a sorting function found in: http://stackoverflow.com/questions/1577475/c-sorting-and-keeping-track-of-indexes
//template <typename T>
//vector<size_t> sort_indexes(const vector<T> &v) {
//
//	// initialize original index locations
//	vector<size_t> idx(v.size());
//	for (size_t i = 0; i != idx.size(); ++i) idx[i] = i;
//
//	// sort indexes based on comparing values in v
//	sort(idx.begin(), idx.end(), [&v](size_t i1, size_t i2) {return v[i1] < v[i2];});
//
//	return idx;
//}

struct by_score {
    bool operator()(bbType const &a, bbType const &b) {
        return a.score5 > b.score5;
    }
};
void showDataFloat(string label,Mat image,int nDims){
	float* newJ = (float*)image.data;
	int nNumbers = (4000/nDims)*nDims;
	int nRows = image.size[0];
	int nCols = image.size[1];
	cout <<  label.data() << endl;;
	for (int nn = 0; nn < nDims;nn++)
	{
		cout  << "channel" << nn+1 << ": ";
		for (int n1 = 0; n1 < nNumbers; n1++)
		{
			cout << newJ[n1+nRows*nCols*nn] << " ";
		}
		cout << endl;
	}
	cout << endl;
}


void showOpenCVData(string label,Mat image,bool isChar){
	float* newJ = (float*)image.data;
	int nChannels = image.channels();
	int nNumbers = (100/nChannels)*nChannels;
	int nRows = image.size[0];
	int nCols = image.size[1];
	cout <<  label.data() << endl;;
//	for (int n = 0; n < 100; n++) {
//		cout << +image.data[n] << ", ";
//	}
	//for(int ch = image.channels()-1; ch >= 0; ch--)
	for (int iChannel = nChannels-1; iChannel >=0 ;iChannel--)
	{
		cout  << "channel" << nChannels-iChannel << ": ";
		for (int iiNumber = 0; iiNumber < nNumbers; iiNumber++)
		{
			if(isChar)
				cout << +image.data[iiNumber*nChannels*nCols + iChannel] << ",  "; // To show the same data in matlab data(1:10,1,channel)'
			else
				cout << (newJ[iiNumber*nChannels*nCols + iChannel]) << ",  "; // To show the same data in matlab data(1:10,1,channel)
		}
		cout << endl;
	}
	cout << endl;
}

void showDataChar(string label,Mat image,int nDims){
	uchar* newJ = (uchar*)image.data;
	int nNumbers = (500/nDims)*nDims;
	int nRows = image.size[0];
	int nCols = image.size[1];
	cout <<  label.data() << endl;;
	for (int nn = 0; nn < nDims;nn++)
	{
		cout  << "channel" << nn+1 << ": ";
		for (int n1 = 0; n1 < nNumbers; n1++)
		{
			cout << +newJ[n1+nRows*nCols*nn] << " ";
		}
		cout << endl;
	}
	cout << endl;
}

//void showDataCharFromOpenCV(string label,Mat image,int nDims){
//	uchar* newJ = (uchar*)image.data;
//	int nNumbers = (500/nDims)*nDims;
//	int nRows = image.size[0];
//	int nCols = image.size[1];
//	cout <<  label.data() << endl;;
//	for (int channel = 0; channel < nDims;channel++)
//	{
//		cout  << "channel" << channel+1 << ": ";
//		for (int row = 0; row < nNumbers; row++)
//		{
//			cout << +newJ[row+image.step+channel] << " ";
//		}
//		cout << endl;
//	}
//	cout << endl;
//}

// Rearrange the image values. Image is being converted from float to uint.
Mat PedestrianDetector::convert_PD_2_OpenCV(Mat image,bool isChar) {

	Mat remappedImage(image.size(), image.type());

	// Data is being converted from Piotr Dollars to the opencv way.


	int n = 0;
	if(isChar == true) {

	for (int row = 0; row < image.rows;row++)
		for(int col = 0; col < image.cols; col++)
			for(int ch = image.channels()-1; ch >= 0; ch--)
				remappedImage.data[n++] =  image.data[row + col*image.rows + ch*image.rows*image.cols]; //*multiplier

	}
	else {
		float* from = (float*)image.data; // Data is being pointed to a float* to display data.
		float* to = new float[remappedImage.cols*remappedImage.rows*remappedImage.channels()];
		for (int row = 0; row < image.rows;row++)
			for(int col = 0; col < image.cols; col++)
				for(int ch = image.channels()-1; ch >= 0; ch--)
					to[n++] =  from[row + col*image.rows + ch*image.rows*image.cols]; //*multiplier
		remappedImage.data = (uchar*) to;
	}

	return remappedImage;
}

Mat PedestrianDetector::convert_OpenCV_2_PD(Mat inputImage) {
	//Mat outputImage(inputImage.rows,inputImage.cols, CV_8UC3);
	Mat outputImage(inputImage.size(), inputImage.type());

	unsigned int nRows = inputImage.rows;
	unsigned int nCols = inputImage.cols;
	unsigned int nRowsCols = nRows*nCols;
	unsigned int n = 0;
	for (unsigned int row = 0; row < nRows;row++)
	{
		for(unsigned int col = 0; col < nCols; col++)
		{
			for(int ch = inputImage.channels()-1; ch >= 0; ch--)
			{
				outputImage.data[row + col*nRows + ch*nRowsCols] = inputImage.data[n++];
			}
		}
	}
	return outputImage;
}

//int get_nScales(double nPerOct, double nOctUp, Size2i minDsIn, double shrink, Size2i szIn){
//	double minDs[] = {double(minDsIn.height),double(minDsIn.width)};
//	double sz[] = {double(szIn.height),double(szIn.width)};
//	return int(trunc(nPerOct*(nOctUp+log2(min(sz[0]/minDs[0],sz[1]/minDs[1])))+1));
//}
void PedestrianDetector::getScales(double nPerOct, double nOctUp, Size2i minDsIn, double shrink, Size2i szIn, Mat &scalesMat, Mat &scaleshwMat){
	double d0, d1;
	double minDs[] = {double(minDsIn.height),double(minDsIn.width)};
	double sz[] = {double(szIn.height),double(szIn.width)};

	double nScales = trunc(nPerOct*(nOctUp+log2(min(sz[0]/minDs[0],sz[1]/minDs[1])))+1);
	vector<double> scales;
	for(int n = 0; n < nScales;n++)
	{
		scales.push_back(pow(2,-n/nPerOct+nOctUp));
	}


	if(sz[0]<sz[1]){
		d0=sz[0];
		d1=sz[1];
	}
	else{
		d0=sz[1];
		d1=sz[0];
	}

	for (int n = 0; n < nScales;n++)
	{
		double s,s0,s1,es0,es1;
		vector<double> es,ss;
		s=scales[n];
		s0=(round(d0*s/shrink)*shrink-0.25*shrink)/d0;
		s1=(round(d0*s/shrink)*shrink+0.25*shrink)/d0;
		float step = 0.01;
		int nSteps = int(1/step);
		for (int nn = 0; nn < nSteps;nn++)
		{
			ss.push_back((float)(nn)*step*(s1-s0)+s0);
			es0=d0*ss.back();
			es0=abs(es0-round(es0/shrink)*shrink);
			es1=d1*ss.back();
			es1=abs(es1-round(es1/shrink)*shrink);
			es.push_back(max(es0,es1));
		}
		int minIdx = distance(es.begin(),min_element(es.begin(), es.end()));
		scales[n] = ss[minIdx];
	}

	// Identical values are removed.
	vector<double>::iterator it2 = unique(scales.begin(), scales.end());   // 10 20 30 20 10 ?  ?  ?  ?
	scales.resize(distance(scales.begin(),it2));

	scaleshwMat = Mat(scales.size(),2,CV_32F);
	scalesMat = Mat(scales.size(),1,CV_32F);
	for(unsigned int n = 0; n<scales.size();n++ ){
		scalesMat.at<float>(n,0) = scales[n];
		scaleshwMat.at<float>(n,0) = round(sz[0]*scales[n]/shrink)*shrink/sz[0];
		scaleshwMat.at<float>(n,1) = round(sz[1]*scales[n]/shrink)*shrink/sz[1];
		//cout << scalesMat.at<float>(n,0) << " " << scaleshwMat.at<float>(n,0) << " " << scaleshwMat.at<float>(n,1) << endl;
	}
}

void PedestrianDetector::getPyramidIndexes(int nScales, int nApprox,vector<int> &isRealPyramid,vector<int> &isApproxPyramid,vector<int> &isN){
	vector<int> J;

	// Determine the index of real and approximated pyramids.
	// WARNING: Index are -1 compared to matlab. T
	for(int n = 0; n < nScales;n++)
	{
		if(n%(nApprox+1) == 0)
			isRealPyramid.push_back(n);
		else
			isApproxPyramid.push_back(n);
	}

	J.push_back(0);
	for(unsigned int n = 0; n < isRealPyramid.size()-1;n++)
		J.push_back(trunc((isRealPyramid[n]+isRealPyramid[n+1])/2));
	J.push_back(nScales);

	int nn = 0;
	for(int n = 0; n < nScales;n++){
		isN.push_back(isRealPyramid[nn]);
		if(n==J[nn+1])
			nn++;
	}

	//	for(unsigned int n = 0; n < isRealPyramid.size(); n++)
	//		cout << "isRealPyramid: " << isRealPyramid[n] << endl;
	//
	//	for(unsigned int n = 0; n < isApproxPyramid.size(); n++)
	//		cout << "isApproxPyramid: " << isApproxPyramid[n] << endl;
	//
	//	for(unsigned int n = 0; n < J.size(); n++)
	//		cout << "J: " << J[n] << endl;
	//
	//	for(unsigned int n = 0; n < isN.size(); n++)
	//		cout << "isN: " << isN[n] << endl;
}

vector<bbType> PedestrianDetector::nmsMax(vector<bbType> bbs, float overlap) {
	vector<double> sortArray2;
	int nBb = bbs.size();
	// METHOD 1
	vector<bbType> bbsSorted = bbs;
	std::sort(bbsSorted.begin(), bbsSorted.end(), by_score());

//	// METHOD 2
//	vector<bbType> bbsSorted;
//	for(unsigned int iBbs = 0; iBbs < bbs.size(); iBbs++){
//		sortArray2.push_back(bbs[iBbs].score5);
//	}
//
//	vector<size_t> indexValues = sort_indexes(sortArray2); // Ascending.
//
//
//	for(int iBbs = nBb-1; iBbs >= 0; iBbs--)
//	{
//		bbsSorted.push_back(bbs[indexValues[iBbs]]);
//	}


	vector<double> as, xs,xe,ys,ye;


	vector<char> kp(nBb,1);


	for(int iBbs = 0; iBbs < nBb; iBbs++)
	{
		//printf("n%d, x: %f, %f, %f, %f, %f, %d \n",iBbs,bbsSorted[iBbs].x1,bbsSorted[iBbs].y2, bbsSorted[iBbs].width3, bbsSorted[iBbs].height4, bbsSorted[iBbs].score5, kp[iBbs] );
		as.push_back(bbsSorted[iBbs].width3*bbsSorted[iBbs].height4); // Area
		xs.push_back(bbsSorted[iBbs].x1);					// x position (bottom)
		xe.push_back(bbsSorted[iBbs].x1+bbsSorted[iBbs].width3);		// x position (top)
		ys.push_back(bbsSorted[iBbs].y2);					// y position (bottom)
		ye.push_back(bbsSorted[iBbs].y2+bbsSorted[iBbs].height4);		// y position (top)
		//printf("n%d, x: %f, %f, %f, %f, %f\n",iBbs,as.back(),xs.back(), xe.back(), ys.back(), ye.back());
	}
	for(int i = 0; i<nBb; i++) {
		if(!kp[i])
			continue;
		for(int j = i+1; j < nBb; j++) {
			if(kp[j]==0)
				continue;

			double iw=min(xe[i],xe[j])-max(xs[i],xs[j]);
			if(iw<=0)
				continue;

			double ih = min(ye[i],ye[j])-max(ys[i],ys[j]);
			if(ih<=0)
				continue;

			double o = iw*ih;
			double u=min(as[i],as[j]);
			o = o/u;
			if(o>overlap)
				kp[j] = 0;
		}
	}

	vector<bbType> bbsTrue;
	for(int iBbs = 0; iBbs < nBb; iBbs++){
		if(kp[iBbs]==true)
			bbsTrue.push_back(bbsSorted[iBbs]);
	}

	//	cout << "true size: " << bbsTrue.size() << endl;
	//	for(int iBbs = 0; iBbs < bbsTrue.size(); iBbs++) {
	//		printf("n%d, x: %f, %f, %f, %f, %f \n",iBbs,bbsTrue[iBbs].x1,bbsTrue[iBbs].y2, bbsTrue[iBbs].width3, bbsTrue[iBbs].height4, bbsTrue[iBbs].score5 );
	//
	//	}

	return bbsTrue;
}

vector<bbType>  PedestrianDetector::pedDetector(cv::Mat inputImage){
	vector<bbType> bbs;
	Mat imageMATLAB,newImage,resampledImage2;
	Size2i sz1, cr, shrinkedSize;
	if(validMatFile) {
		///// Preprocessing: Initial transformation of the image. //////
		// Image is being converted from opencv to fit with Piotr Dollars code.
		//showOpenCVData("Test", inputImage,true);
		imageMATLAB = convert_OpenCV_2_PD(inputImage);

		//showDataChar("OpenCv to matlab print:",inputImage,inputImage.channels()); // Show result of resampled image.
		//showDataChar("PD to matlab print:",imageMATLAB,imageMATLAB.channels()); // Show result of resampled image.
		// PyramidParamters are only determined, when the image dimensions are changed.
		if(inputImage.rows != dimImage.height && inputImage.cols != dimImage.width) {
			dimImage.height =inputImage.rows;
			dimImage.width =inputImage.cols;
			DeterminePyramidParameters(dimImage);
		}

		// Transforms image to LUV-colorspace
		newImage = pdRgbConvert(imageMATLAB);

		//
		Mat allChannels[nScales][3];
		Mat allChannelsPad[nScales][3];



		///// Compute image pyramid for octaves (not approximated).
		for (unsigned int n = 0; n < isRealPyramid.size(); n++)
		{
			int nn = isRealPyramid[n];
			float s = scales.at<float>(nn,0);
			//Size2i sz1;
			sz1.width =round(dimImage.width*s/imagePyramid.shrink)*imagePyramid.shrink;
			sz1.height =round(dimImage.height*s/imagePyramid.shrink)*imagePyramid.shrink;
			//cout << "Width: " << sz1.width << ", Height: " << sz1.height << endl;
			// Rescales image to get even height and width
			resampledImage2 = pdImResample(newImage,sz1.height, sz1.width,1);

			// The image is cropped slightly to be "shrinkable" by 4.
			//Size2i cr;
			//Size2i shrinkedSize;
			cr.height = resampledImage2.size[0] % int(imagePyramid.shrink);
			cr.width = resampledImage2.size[1] % int(imagePyramid.shrink) ;
			shrinkedSize.height = resampledImage2.size[0]-cr.height;
			shrinkedSize.width = resampledImage2.size[1]-cr.width;
			if(cr.height+cr.width>0)
			{
				cv::Rect croppedImageSize(0,0, shrinkedSize.width, shrinkedSize.height);
				resampledImage2(croppedImageSize).copyTo(resampledImage2);
			}
			shrinkedSize.height=shrinkedSize.height/imagePyramid.shrink;
			shrinkedSize.width=shrinkedSize.width/imagePyramid.shrink;

			// Image is being filtered with a triangle filter
			Mat Filtered = pdConvConst(resampledImage2, (int)(1));
			// A shrinked version is used as a channel.
			allChannels[nn][0] = pdImResample(Filtered,shrinkedSize.height, shrinkedSize.width,1);

			// Magnitude and edge orientations.
			Mat outputMag[2] = {Mat(resampledImage2.rows,resampledImage2.cols, CV_32FC1),Mat(resampledImage2.rows,resampledImage2.cols, CV_32FC1)};
			pdGradientMag(Filtered,0,0,outputMag[0],outputMag[1]);

			// Magnitude is filtered
			Mat MagnitudeFiltered = pdConvConst(outputMag[0], (int)(5));

			// Magnitude is normalized (in some way)
			Mat GradientTmp = pdGradientMagNorm(outputMag[0], MagnitudeFiltered, 0.005);
			allChannels[nn][1] = pdImResample(GradientTmp,shrinkedSize.height, shrinkedSize.width,1);

			// The 6 gradient channels. (A bad fix is made around binSize)
			allChannels[nn][2] = pdGradientHist(GradientTmp, outputMag[1], imagePyramid.binSize, 6, 0, 0, 0.2f, false);

//					showDataFloat("Color Transformed:",newImage,newImage.channels()); // Show result of resampled image.
//					showDataFloat("Resampled Image:",resampledImage2,resampledImage2.channels()); // Show result of resampled image.
//					showDataFloat("TriFiltered image",Filtered,Filtered.channels()); // Show result of filtered image.
//					showDataFloat("M (not normalized)",outputMag[0],outputMag[0].channels()); // Show result of gradient image.
//					showDataFloat("Result of S (M filtered)",MagnitudeFiltered,MagnitudeFiltered.channels()); // Show result of filtered image.
//					showDataFloat("Result of M normalized before resampling",GradientTmp,GradientTmp.channels()); // Show result of filtered image.
//					showDataFloat("Result of M normalized after resampling",allChannels[nn][1],allChannels[nn][1].channels()); // Show result of filtered image.
//					showDataFloat("Result of O",outputMag[1],outputMag[1].channels()); // Show result of gradient image.
//					showDataFloat("Result of Histogram",allChannels[nn][2], 6); // Show result of filtered image.

		}
		//cout << "Size of isApproxPyramid: " << isApproxPyramid.size() << endl;

		// Compute/approximate the intermediate levels (between octaves) in image pyramid.
		for(unsigned int n = 0; n<isApproxPyramid.size();n++){
			int nn = isApproxPyramid[n]; // Equal to i
			int isReal = isN[nn];
			float s = scales.at<float>(nn,0);
			Size2i sz1;
			sz1.width =round(dimImage.width*s/imagePyramid.shrink);
			sz1.height =round(dimImage.height*s/imagePyramid.shrink);
			for(int nnn = 0; nnn < 3; nnn++){
				double ratio = pow(s/scales.at<float>(isReal,0),-imagePyramid.lambdas[nnn]);
				allChannels[nn][nnn]=pdImResample(allChannels[isReal][nnn],sz1.height,sz1.width,ratio);
			}
		}


		// The 10 channels are padded (3 LUV, 1 Mag, 6 Hist).
		Size2i pad2(imagePyramid.pad.width/imagePyramid.shrink,imagePyramid.pad.height/imagePyramid.shrink);
		const string paddingType[] = {"replicate","none","none"};
		for(unsigned int n = 0; n<nScales;n++)
			for(unsigned int nn = 0; nn<3;nn++)
				allChannelsPad[n][nn] = pdImPad(pdConvConst(allChannels[n][nn],(int)1),pad2,paddingType[nn]);


		if(fastDetector == 0) {
			int nKernels = 4;
			Mat allChannelsPadFiltered[nScales][10*nKernels];
			for(int iScales = 0; iScales<nScales;iScales++) {
				int channelCountTo40 = 0;
				int channelCountTo10 = 0;
				for(int iiTopChannels = 0; iiTopChannels < 3; iiTopChannels++) {

					// SOMETHING FUCKS UP FROM HERE!!!! (Probably)

					vector<Mat> singleChannels;
					Mat tmpMat = convert_PD_2_OpenCV(allChannelsPad[iScales][iiTopChannels],0);
					split(tmpMat,singleChannels);

					//showOpenCVData("Test", singleChannels[0],false);

					int nSingleChannels = singleChannels.size();
					for(int iiiChannels = 0; iiiChannels < nSingleChannels; iiiChannels++) {
						for(int iiiiFilters = 0; iiiiFilters < nKernels; iiiiFilters++) {
							Mat tmpOutput;

							//cout << singleChannels[iiiChannels] << endl;

							filter2D(singleChannels[iiiChannels],tmpOutput,-1,pedModel.kernel[channelCountTo10][iiiiFilters]);
							namedWindow("BoundingsBox", CV_WINDOW_AUTOSIZE);
							imshow("BoundingsBox",tmpOutput);
							cout << pedModel.kernel[channelCountTo10][iiiiFilters] << endl;
							showOpenCVData("Test", singleChannels[iiiChannels],false);
							showOpenCVData("Test", tmpOutput,false);
							waitKey(0);


							allChannelsPadFiltered[iScales][channelCountTo40] = convert_OpenCV_2_PD(tmpOutput);
							showDataFloat("Extreme test", allChannelsPadFiltered[iScales][channelCountTo40],1);
							channelCountTo40++;
						}
						channelCountTo10++;
					}
				}
			}
		}

		Size2f shift((pedOptions.modelDsPad.width-pedOptions.modelDs.width)/2 - imagePyramid.pad.width, (pedOptions.modelDsPad.height-pedOptions.modelDs.height)/2 - imagePyramid.pad.height);

		// Detection using a sliding window approach over the 10 channels.
		// The loop iterates over the image pyramid scales.
		for(int iScales = 0; iScales<nScales;iScales++)
		{
			// The image size of a particular scale.
			int nRows = allChannelsPad[iScales][0].size[0];
			int nCols = allChannelsPad[iScales][0].size[1];

			// nPixels in a channel.
			int nChannelPixels = nRows*nCols;
			int cData = 0;

			// All channels (for a particular scale) is put into the data-vector.
			float *data = new float[10*nChannelPixels];
			Mat channelScalei(nRows, nCols,CV_32FC(10));

			for (int iiChannels = 0; iiChannels < 3;iiChannels++)
			{
				float* tmpData = (float*)allChannelsPad[iScales][iiChannels].data;
				for (int iiiSubChannel= 0; iiiSubChannel < allChannelsPad[iScales][iiChannels].channels(); iiiSubChannel++)
					for (int iiiiPixels= 0; iiiiPixels < nChannelPixels; iiiiPixels++)
						data[cData++] = tmpData[iiiiPixels+nRows*nCols*iiiSubChannel];
			}
			channelScalei.data =  (uchar*)data;
//			Mat test = convert_PD_2_OpenCV(channelScalei,0);
//			Mat output;
			//filter2D(channelScalei,output,);
			//showDataFloat("Shows the 10 channels: ",channelScalei,channelScalei.channels()); // Show result of filtered image.
			//showDataFloat("Shows the 1 channels: ",allChannelsPad[0][0],allChannelsPad[0][0].channels()); // Show result of filtered image.


			//convert_PD_2_OpenCV(channelScalei,1);

			// Detection is performed.
			vector<bbType> bbsTmp = pfDetect(channelScalei, imagePyramid.shrink, pedOptions.modelDsPad.height, pedOptions.modelDsPad.width, pedOptions.stride,pedOptions.cascThr,nRows,nCols,10, PedestrianDetector::pedModel);
			bbType bbsTmpReal;
			delete data;
			// All boundings boxes are stored in bbs.
			for(unsigned int iBbs = 0; iBbs < bbsTmp.size(); iBbs++)
			{
				bbsTmpReal.x1 = ((bbsTmp[iBbs].x1+shift.width)/scaleshw.at<float>(iScales,1));
				bbsTmpReal.y2 = ((bbsTmp[iBbs].y2+shift.height)/scaleshw.at<float>(iScales,0));
				bbsTmpReal.width3 = pedOptions.modelDs.width/scales.at<float>(iScales);
				bbsTmpReal.height4 = pedOptions.modelDs.height/scales.at<float>(iScales);
				bbsTmpReal.score5 = bbsTmp[iBbs].score5;

				bbs.push_back(bbsTmpReal);
			}
		}

		// Non-maximum suppression is used to select the highest rated pedestrians.
		bbs = nmsMax(bbs, 0.65);


		// An estimate of the distance to the object is calculated using the camera setup.
			// Estimate is based on two assumptions: 1) The surface is flat. 2) The bottom of the bounding box is the bottom of the detected object.
		if(cameraSettingsProvided){
			double resolutionVertical = inputImage.rows;
			double resolutionHorisontal = inputImage.cols;
			//cout << resolutionVertical << endl;
			for (int n = 0; n < bbs.size();n++){
				double buttomRowPosition = bbs[n].y2+bbs[n].height4; // bbs(n,2)-bbs(n,4);
				double ColPosition = bbs[n].x1+bbs[n].width3/2; // bbs(n,2)-bbs(n,4);
				//cout << "buttomRow:" << buttomRowPosition << endl;
				//cout << "buttomRow:" << buttomRowPosition << endl;
				bbs[n].distance = tan(PI/2-(angleTiltRad+FOV_verticalRad/2) + FOV_verticalRad*(resolutionVertical-buttomRowPosition)/resolutionVertical)*cameraHeight;
				bbs[n].angle =((ColPosition-resolutionHorisontal/2)/resolutionHorisontal)*FOV_verticalRad;
				cout << "x1:" << bbs[n].x1 << ", y2:" << bbs[n].y2 << ", w3:" << bbs[n].width3 << ", h4:" << bbs[n].height4 << ", s5: " << bbs[n].score5 << ",a5: " << bbs[n].angle << endl;
				//cout << "Distance: " <<  bbs[n].distance << endl;
			}
		}

		//	cout << "The number of scales" << nScales << endl;
		//	// The data of all channels are displayed.
		//	for (int n = 0; n < nScales; n++)
		//	{
		//		cout << "------------------ Image index: " << n << "------------------" << endl;
		//		cout << "ImageDim: " << allChannelsPad[n][0].size[0] << "x" << allChannelsPad[n][0].size[1] << "x" << allChannelsPad[n][0].channels() << endl;
		//		cout << "ImageDim: " << allChannelsPad[n][1].size[0] << "x" << allChannelsPad[n][1].size[1] << "x" << allChannelsPad[n][1].channels() << endl;
		//		cout << "ImageDim: " << allChannelsPad[n][2].size[0] << "x" << allChannelsPad[n][2].size[1] << "x" << allChannelsPad[n][2].channels() << endl;
		//		showDataFloat("TriFiltered image",allChannelsPad[n][0],allChannelsPad[n][0].channels()); // Show result of filtered image.
		//		showDataFloat("Result of M normalized",allChannelsPad[n][1],allChannelsPad[n][1].channels()); // Show result of filtered image.
		//		showDataFloat("Result of Histogram",allChannelsPad[n][2], allChannelsPad[n][2].channels()); // Show result of filtered image.
		//	}


	}
	else
		printf("Invalid mat-file. The file-dir to a trained classifier (mat-file) from Piotr Dollars MATLAB framework is required to successfully detect pedestrians.");
	return bbs;
}

void PedestrianDetector::setCameraSetup(double FOV_verticalDeg, double FOV_horizontalDeg, double angleTiltDegrees, double cameraHeightIn) {
	FOV_verticalRad = FOV_verticalDeg*PI/180;
	FOV_horizontalRad = FOV_horizontalDeg*PI/180;
	angleTiltRad = angleTiltDegrees*PI/180;
	cameraHeight = cameraHeightIn;
	cameraSettingsProvided = true;
}

void PedestrianDetector::loadMatFile(std::string strPedModel){

	vector<LoadMatVariableType> loadMatTest;
	//LoadMatVariableType loadMatTest[18];

	int nLoadVariables = 18;
	int nSubVariables = 4;

	// load variables from mat-file: Detector.Opt.(modelDs,ModelDsPad,stride,cascThr,cascCal)
	static const int arr[18][4] = {
			{detOpt		, detOpt_modelDs		, -1					, -1},
			{detOpt		, detOpt_modelDsPad		, -1					, -1},
			{detOpt		, detOpt_stride			, -1					, -1},
			{detOpt		, detOpt_cascThr		, -1					, -1},
			{detOpt		, detOpt_cascCal		, -1					, -1},
			{detOpt		, detOpt_filters		, -1					, -1},
			{detOpt		, detOptPyramid			, detOptPyramid_nPerOct	, -1},
			{detOpt		, detOptPyramid			, detOptPyramid_nOctUp	, -1},
			{detOpt		, detOptPyramid			, detOptPyramid_nApprox	, -1},
			{detOpt		, detOptPyramid			, detOptPyramid_lambdas	, -1},
			{detOpt		, detOptPyramid			, detOptPyramid_pad		, -1},
			{detOpt		, detOptPyramid			, detOptPyramid_minDs	, -1},
			{detOpt		, detOptPyramid			, detOptPyramidChns		, detOptPyramidChns_shrink},
			{detClf		, detClf_featureIndex	, -1					, -1},
			{detClf		, detClf_threshold		, -1					, -1},
			{detClf		, detClf_child			, -1 					, -1},
			{detClf		, detClf_hs				, -1					, -1},
			{detClf		, detClf_threeDepth		, -1					, -1},
	};

	for(int n = 0; n < nLoadVariables; n++) {
		LoadMatVariableType singleTmp;
		for(int nn = 0; nn < nSubVariables; nn++) {
			if(arr[n][nn] != -1) {
				singleTmp.matIndex.push_back(arr[n][nn]);
				//cout << +arr[n][nn] << ", ";
			}
		}
		//cout << endl;
		loadMatTest.push_back(singleTmp);
	}
//	cout << "new line" << endl;
//	for(int n = 0; n < loadMatTest.size(); n++) {
//			for(int nn = 0; nn < loadMatTest[n].matIndex.size(); nn++) {
//					cout << +loadMatTest[n].matIndex[nn] << ", ";
//			}
//			cout << endl;
//		}
	//loadMatTest[0].matIndex = (arr, arr + sizeof(arr) / sizeof(arr[0]) );
	//loadMatTest[0].push_back(singleTmp);

//	int nLoadVariables = 18;
//	LoadMatVariableType loadMatTest[nLoadVariables];
//
//	// load variables from mat-file: Detector.Opt.(modelDs,ModelDsPad,stride,cascThr,cascCal)
//	loadMatTest[0].matIndex = {detOpt, detOpt_modelDs};
//	loadMatTest[1].matIndex = {detOpt, detOpt_modelDsPad};
//	loadMatTest[2].matIndex = {detOpt, detOpt_stride};
//	loadMatTest[3].matIndex = {detOpt, detOpt_cascThr};
//	loadMatTest[4].matIndex = {detOpt, detOpt_cascCal};
//	loadMatTest[16].matIndex = {detOpt, detOpt_filters};
//
//	// load variables from mat-file: Detector.Opt.pPyramid.(nPerOct,nOctUp,nApprox,lambdas,pad,minDs);
//	loadMatTest[5].matIndex = {detOpt, detOptPyramid,detOptPyramid_nPerOct};
//	loadMatTest[6].matIndex = {detOpt, detOptPyramid,detOptPyramid_nOctUp};
//	loadMatTest[7].matIndex = {detOpt, detOptPyramid,detOptPyramid_nApprox};
//	loadMatTest[8].matIndex = {detOpt, detOptPyramid,detOptPyramid_lambdas};
//	loadMatTest[9].matIndex = {detOpt, detOptPyramid,detOptPyramid_pad};
//	loadMatTest[10].matIndex = {detOpt, detOptPyramid,detOptPyramid_minDs};
//
//	// load variables from mat-file: Detector.Opt.pPyramid.pChns.(shrink);
//	loadMatTest[17].matIndex = {detOpt, detOptPyramid,detOptPyramidChns,detOptPyramidChns_shrink};
//
//	// load variables from mat-file: Detector.Clf.(featureIndex,treshold,child,hs,threeDepth)
//	loadMatTest[11].matIndex = {detClf,detClf_featureIndex};
//	loadMatTest[12].matIndex = {detClf,detClf_threshold};
//	loadMatTest[13].matIndex = {detClf,detClf_child};
//	loadMatTest[14].matIndex = {detClf,detClf_hs};
//	loadMatTest[15].matIndex = {detClf,detClf_threeDepth};

	mat_t *mat;
	matvar_t *matvar;
	mat = Mat_Open(strPedModel.data(),MAT_ACC_RDONLY);
	bool validFile = mat;
	validMatFile= validFile;

	if(validMatFile) {
		matvar = Mat_VarRead(mat, "detector");
		char * const * detectorFieldNames_1 = Mat_VarGetStructFieldnames(matvar); // get field names of detector. (opt, clf, info)

		// Load all variables one-by-one.
		for(int n = 0; n < nLoadVariables;n++) {
			int fieldNumber1 = loadMatTest[n].matIndex[0]; // Select level1 field: opts, clf or info
			matvar_t *matVarOpt = Mat_VarGetStructField(matvar, detectorFieldNames_1[fieldNumber1], MAT_BY_NAME, 0); // Pointer to either opts, clf or info.
			char * const *detectorFieldNames_2 = Mat_VarGetStructFieldnames(matVarOpt); // get field names.
			cout << detectorFieldNames_1[fieldNumber1] << "(" << fieldNumber1 << ")";
			// Load variables from Opt-field.
			if(fieldNumber1 == detOpt) {
				int fieldNumber2 = loadMatTest[n].matIndex[1]; // Select level2 field.
				matvar_t *matVarField2 = Mat_VarGetStructFieldByName(matVarOpt, detectorFieldNames_2[fieldNumber2], 0);
				cout << detectorFieldNames_2[fieldNumber2] << "(" << fieldNumber2 << ")";
				// Load variable from Opt.modelDs
				if(fieldNumber2 == detOpt_modelDs) {
					double *doubleVector = (double*) (matVarField2->data);
					pedOptions.modelDs.height = doubleVector[0];
					pedOptions.modelDs.width = doubleVector[1];
				}

				// Load variable from Opt.modelDsPad
				else if(fieldNumber2 == detOpt_modelDsPad) {
					double *doubleVector = (double*) (matVarField2->data);
					pedOptions.modelDsPad.height = doubleVector[0];
					pedOptions.modelDsPad.width = doubleVector[1];
				}

				// Load variable from Opt.strid
				else if(fieldNumber2 == detOpt_stride)
					pedOptions.stride  = *(double*) (matVarField2->data);

				// Load variable from Opt.cascThr
				else if(fieldNumber2 == detOpt_cascThr)
					pedOptions.cascThr = *(double*) (matVarField2->data);

				// Load variable from Opt.cascCal
				else if(fieldNumber2 == detOpt_cascCal)
					pedOptions.cascCal = *(double*) (matVarField2->data);

				else if(fieldNumber2 == detOpt_filters) {
					float* filter = (float*) (matVarField2->data);
					pedModel.filter.insert(pedModel.filter.end(),&filter[0],&filter[matVarField2->dims[0]*matVarField2->dims[1]*matVarField2->dims[2]*matVarField2->dims[3]]);
					pedModel.filterDim.push_back(matVarField2->dims[0]);
					pedModel.filterDim.push_back(matVarField2->dims[1]);
					pedModel.filterDim.push_back(matVarField2->dims[2]);
					pedModel.filterDim.push_back(matVarField2->dims[3]);

				}

				// Load all variables from Opt.pPyramid.
				else if(fieldNumber2 == detOptPyramid){
					int fieldNumber3 = loadMatTest[n].matIndex[2];
					matvar_t *matVarOptPyramid = Mat_VarGetStructField(matVarOpt, detectorFieldNames_2[detOptPyramid], MAT_BY_NAME, 0);
					char * const *detectorFieldNames_3 = Mat_VarGetStructFieldnames(matVarOptPyramid);
					matvar_t *matVarField3 = Mat_VarGetStructFieldByName(matVarOptPyramid, detectorFieldNames_3[fieldNumber3], 0);

					cout << detectorFieldNames_3[fieldNumber3] << "(" << fieldNumber3 << ")";
					// Load variable from Opt.pPyramid.nPerOct
					if(fieldNumber3 == detOptPyramid_nPerOct)
						imagePyramid.nPerOct = *(double*) (matVarField3->data);

					// Load variable from Opt.pPyramid.nOctUp
					else if(fieldNumber3 == detOptPyramid_nOctUp)
						imagePyramid.nOctUp = *(double*) (matVarField3->data);

					// Load variable from Opt.pPyramid.nApprox
					else if(fieldNumber3 == detOptPyramid_nApprox)
						imagePyramid.nApprox = *(double*) (matVarField3->data);

					// Load variable from Opt.pPyramid.lambdas
					else if(fieldNumber3 == detOptPyramid_lambdas) {
						double *doubleVector = (double*) (matVarField3->data);
						imagePyramid.lambdas.insert(imagePyramid.lambdas.end(),&doubleVector[0],&doubleVector[2]); // Three values only.
					}

					// Load variable from Opt.pPyramid.pad
					else if(fieldNumber3 == detOptPyramid_pad) {
						double *doubleVector = (double*) (matVarField3->data);
						imagePyramid.pad.height = doubleVector[0];
						imagePyramid.pad.width = doubleVector[1];
					}

					// Load variable from Opt.pPyramid.minDs
					else if(fieldNumber3 == detOptPyramid_minDs){
						double *doubleVector = (double*) (matVarField3->data);
						imagePyramid.minDs.height = doubleVector[0];
						imagePyramid.minDs.width = doubleVector[1];
					}

					// Load variable from Opt.pPyramid.pChns.
					else if(fieldNumber3 == detOptPyramidChns){
						//detOptPyramidChns_shrink
						int fieldNumber4 = loadMatTest[n].matIndex[3];
						matvar_t *matVarOptPyramidChns = Mat_VarGetStructField(matVarOptPyramid, detectorFieldNames_3[detOptPyramidChns], MAT_BY_NAME, 0);
						char * const *detectorFieldNames_4 = Mat_VarGetStructFieldnames(matVarOptPyramidChns);
						matvar_t *matVarField4 = Mat_VarGetStructFieldByName(matVarOptPyramidChns, detectorFieldNames_4[fieldNumber4], 0);

						cout << detectorFieldNames_4[fieldNumber4] << "(" << fieldNumber4 << ")";
						// Load variable from Opt.pPyramid.pChns.detOptPyramidChns_shrink
						if(fieldNumber4 == detOptPyramidChns_shrink) {
							imagePyramid.shrink = *(double*) (matVarField4->data);
						}
					}
				}
			}
			// Load variables from Clf-field.
			else if(fieldNumber1 == detClf) {
				int fieldNumber2 = loadMatTest[n].matIndex[1];
				matvar_t *matVarField2 = Mat_VarGetStructFieldByName(matVarOpt, detectorFieldNames_2[fieldNumber2], 0);

				cout << detectorFieldNames_2[fieldNumber2] << "(" << fieldNumber2 << ")";
				// Load variable from Clf.featureIndex.
				if(fieldNumber2 == detClf_featureIndex) {
					uint32_t* fids = (uint32_t*) (matVarField2->data);
					size_t* fidsSize = (size_t*) (matVarField2->dims);
					pedModel.fids.insert(pedModel.fids.end(),&fids[0],&fids[matVarField2->dims[0]*matVarField2->dims[1]]);
					pedModel.nTreeNodes = fidsSize[0];
					pedModel.nTrees = fidsSize[1];
				}

				// Load variable from Clf.threshold
				else if(fieldNumber2 == detClf_threshold) {
					float* thrs = (float*) (matVarField2->data);
					pedModel.thrs.insert(pedModel.thrs.end(),&thrs[0],&thrs[matVarField2->dims[0]*matVarField2->dims[1]]);
				}

				// Load variable from Clf.child
				else if(fieldNumber2 == detClf_child) {
					uint32_t* child = (uint32_t*) (matVarField2->data);
					pedModel.child.insert(pedModel.child.end(),&child[0],&child[matVarField2->dims[0]*matVarField2->dims[1]]);
				}

				// Load variable from Clf.hs
				else if(fieldNumber2 == detClf_hs) {
					float* hs = (float*) (matVarField2->data);
					pedModel.hs.insert(pedModel.hs.end(),&hs[0],&hs[matVarField2->dims[0]*matVarField2->dims[1]]);
				}

				// Load variable from Clf.threeDepth
				else if(fieldNumber2 == detClf_threeDepth)
					pedModel.treeDepth = *(int*) (matVarField2->data);
			}
			cout << endl;
		}

	}
	Mat_Close(mat);
}
