/*
 *  IDistance.h
 *  SfMToyExample
 *
 *  Created by Roy Shilkrot on 4/15/12.
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

#pragma once

#define STRATEGY_USE_OPTICAL_FLOW		1
#define STRATEGY_USE_DENSE_OF			2
#define STRATEGY_USE_FEATURE_MATCH		4
#define STRATEGY_USE_HORIZ_DISPARITY	8

typedef std::vector<cv::Point3d> PointCloud;

class IDistance {
public:
	virtual void OnlyMatchFeatures() = 0;
	virtual void RecoverDepthFromImages() = 0;
	virtual PointCloud getPointCloud() = 0;
	virtual const std::vector<cv::Vec3b>& getPointCloudRGB() = 0;

	virtual ~IDistance() {}
};
