/*
 *  GPUSURFFeatureMatcher.cpp
 *  ExploringSfMWithOpenCV
 *
 *  Created by Roy Shilkrot on 6/13/12.
 *  The MIT License (MIT)
 *
 *  Copyright (c) 2013 Roy Shilkrot
 *
 *  Permission is hereby granted, free of charge, to any person obtaining a copy
 *  of this software and associated documentation files (the "Software"), to deal
 *  in the Software without restriction, including without limitation the rights
 *  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 *  copies of the Software, and to permit persons to whom the Software is
 *  furnished to do so, subject to the following conditions:
 *
 *  The above copyright notice and this permission notice shall be included in all
 *  copies or substantial portions of the Software.
 *
 *  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 *  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 *  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 *  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 *  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 *  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 *  SOFTWARE.
 *
 */

#include "GPUSURFFeatureMatcher.h"

#ifdef HAVE_OPENCV_GPU

#include "FindCameraMatrices.h"
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/video/tracking.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/nonfree/gpu.hpp>

#include <iostream>
#include <set>

using namespace std;
using namespace cv;
using namespace cv::gpu;

//c'tor
GPUSURFFeatureMatcher::GPUSURFFeatureMatcher(vector<cv::Mat>& imgs_, 
									   vector<std::vector<cv::KeyPoint> >& imgpts_) :
	imgpts(imgpts_),use_ratio_test(true)
{
	cv::gpu::printShortCudaDeviceInfo(cv::gpu::getDevice());

	extractor = new gpu::SURF_GPU();
	
	std::cout << " -------------------- extract feature points for all images (GPU) -------------------\n";
	
	imgpts.resize(imgs_.size());
	descriptors.resize(imgs_.size());

	CV_PROFILE("extract",
	for(int img_i=0;img_i<imgs_.size();img_i++) {
		GpuMat _m; _m.upload(imgs_[img_i]);
		(*extractor)(_m,GpuMat(),imgpts[img_i],descriptors[img_i]);
		cout << ".";
	}
	)
}	

void GPUSURFFeatureMatcher::MatchFeatures(int idx_i, int idx_j, vector<DMatch>* matches) {
	
#ifdef __SFM__DEBUG__
	Mat img_1; imgs[idx_i].download(img_1);
	Mat img_2; imgs[idx_j].download(img_2);
#endif
	const vector<KeyPoint>& imgpts1 = imgpts[idx_i];
	const vector<KeyPoint>& imgpts2 = imgpts[idx_j];
	const GpuMat& descriptors_1 = descriptors[idx_i];
	const GpuMat& descriptors_2 = descriptors[idx_j];
	
	std::vector< DMatch > good_matches_,very_good_matches_;
	std::vector<KeyPoint> keypoints_1, keypoints_2;
	
	//cout << "imgpts1 has " << imgpts1.size() << " points (descriptors " << descriptors_1.rows << ")" << endl;
	//cout << "imgpts2 has " << imgpts2.size() << " points (descriptors " << descriptors_2.rows << ")" << endl;
	
	keypoints_1 = imgpts1;
	keypoints_2 = imgpts2;
	
	if(descriptors_1.empty()) {
		CV_Error(0,"descriptors_1 is empty");
	}
	if(descriptors_2.empty()) {
		CV_Error(0,"descriptors_2 is empty");
	}
	
	//matching descriptor vectors using Brute Force matcher
	BruteForceMatcher_GPU<L2<float> > matcher;
	std::vector< DMatch > matches_;
	if (matches == NULL) {
		matches = &matches_;
	}
	if (matches->size() == 0) {
		cout << "match " << descriptors_1.rows << " vs. " << descriptors_2.rows << " ...";

		if(use_ratio_test) {
			vector<vector<DMatch> > knn_matches;
			GpuMat trainIdx,distance,allDist;
			CV_PROFILE("match", 
				matcher.knnMatchSingle(descriptors_1,descriptors_2,trainIdx,distance,allDist,2); 
				matcher.knnMatchDownload(trainIdx,distance,knn_matches);
			)

			(*matches).clear();

			//ratio test
			for(int i=0;i<knn_matches.size();i++) {
				if(knn_matches[i][0].distance / knn_matches[i][1].distance < 0.7) {
					(*matches).push_back(knn_matches[i][0]);
				}
			}
			cout << "kept " << (*matches).size() << " features after ratio test"<<endl;
		} else {
			CV_PROFILE("match",matcher.match( descriptors_1, descriptors_2, *matches );)
		}
	}
}

#endif
