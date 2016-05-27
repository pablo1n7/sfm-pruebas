
#include <iostream>
#include <string.h>

#include "Distance.h"
#include "MultiCameraPnP.h"

#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>

using namespace std;

#include <opencv2/gpu/gpu.hpp>
using namespace cv;

std::vector<cv::Mat> images;
std::vector<std::string> images_names;
/*
void SORFilter() {
	
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_filtered (new pcl::PointCloud<pcl::PointXYZRGB>);
	
	std::cerr << "Cloud before SOR filtering: " << cloud->width * cloud->height << " data points" << std::endl;
	

	// Create the filtering object
	pcl::StatisticalOutlierRemoval<pcl::PointXYZRGB> sor;
	sor.setInputCloud (cloud);
	sor.setMeanK (50);
	sor.setStddevMulThresh (1.0);
	sor.filter (*cloud_filtered);
	
	std::cerr << "Cloud after SOR filtering: " << cloud_filtered->width * cloud_filtered->height << " data points " << std::endl;
	
	copyPointCloud(*cloud_filtered,*cloud);
	copyPointCloud(*cloud,*orig_cloud);
}
*/

void PopulatePCLPointCloud(const pcl::PointCloud<pcl::PointXYZRGB>::Ptr& mycloud,
						   const vector<cv::Point3d>& pointcloud, 
						   const std::vector<cv::Vec3b>& pointcloud_RGB,
						   bool write_to_file
						   )
	//Populate point cloud
{
	cout << "Creating point cloud...";
	double t = cv::getTickCount();
	
	for (unsigned int i=0; i<pointcloud.size(); i++) {
		// get the RGB color value for the point
		cv::Vec3b rgbv(255,255,255);
		if (pointcloud_RGB.size() > i) {
			rgbv = pointcloud_RGB[i];
		}

		// check for erroneous coordinates (NaN, Inf, etc.)
		if (pointcloud[i].x != pointcloud[i].x || 
			pointcloud[i].y != pointcloud[i].y || 
			pointcloud[i].z != pointcloud[i].z || 
#ifndef WIN32
			isnan(pointcloud[i].x) ||
			isnan(pointcloud[i].y) || 
			isnan(pointcloud[i].z) ||
#else
			_isnan(pointcloud[i].x) ||
			_isnan(pointcloud[i].y) || 
			_isnan(pointcloud[i].z) ||
#endif
			//fabsf(pointcloud[i].x) > 10.0 || 
			//fabsf(pointcloud[i].y) > 10.0 || 
			//fabsf(pointcloud[i].z) > 10.0
			false
			) 
		{
			continue;
		}
		
		pcl::PointXYZRGB pclp;
		
		// 3D coordinates
		pclp.x = pointcloud[i].x;
		pclp.y = pointcloud[i].y;
		pclp.z = pointcloud[i].z;
		
		
		// RGB color, needs to be represented as an integer
		uint32_t rgb = ((uint32_t)rgbv[2] << 16 | (uint32_t)rgbv[1] << 8 | (uint32_t)rgbv[0]);
		pclp.rgb = *reinterpret_cast<float*>(&rgb);
		/*
		*/
		mycloud->push_back(pclp);

	}
	
	mycloud->width = (uint32_t) mycloud->points.size(); // number of points
	mycloud->height = 1;								// a list, one row of data
	
	t = ((double)cv::getTickCount() - t)/cv::getTickFrequency();
	
	
	// write to file
	if (write_to_file) {
		//pcl::PLYWriter pw;
		//pw.write("pointcloud.ply",*mycloud);
		pcl::PCDWriter pw;
		pw.write("pointcloud.pcd",*mycloud);
		//pw.write(*name,*mycloud);
	}
	
	cout << "Done. (" << t <<"s)"<< endl;
}

int main(int argc, char const *argv[])
{
	if (argc < 2) {
		cerr << "reconstruccion3D image_folder" << endl;
		return 0;
	}

	double downscale_factor = 1.0;
	open_imgs_dir(argv[1],images,images_names,downscale_factor);
	if(images.size() == 0) { 
		cerr << "can't get image files" << endl;
		return 1;
	}

	cv::Ptr<MultiCameraPnP> distance = new MultiCameraPnP(images,images_names,string(argv[1]));
	distance->use_rich_features = true;
	distance->use_gpu = (cv::gpu::getCudaEnabledDeviceCount() > 0);

	distance->RecoverDepthFromImages();

	double scale_cameras_down = 1.0;
	{
		vector<cv::Point3d> cld = distance->getPointCloud();
		if (cld.size()==0) cld = distance->getPointCloudBeforeBA();
		cv::Mat_<double> cldm(cld.size(),3);
		for(unsigned int i=0;i<cld.size();i++) {
			cldm.row(i)(0) = cld[i].x;
			cldm.row(i)(1) = cld[i].y;
			cldm.row(i)(2) = cld[i].z;
		}
		cv::Mat_<double> mean;
		cv::PCA pca(cldm,mean,CV_PCA_DATA_AS_ROW);
		scale_cameras_down = pca.eigenvalues.at<double>(0) / 5.0;
		//if (scale_cameras_down > 1.0) {
		//	scale_cameras_down = 1.0/scale_cameras_down;
		//}
	}


	//pcl::io::savePCDFileASCII ("test_pcd.pcd", distance->getPointCloudRGB());
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud;
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud1;
	cloud1.reset(new pcl::PointCloud<pcl::PointXYZRGB>);
	cloud.reset(new pcl::PointCloud<pcl::PointXYZRGB>);

	//PopulatePCLPointCloud(cloud,distance->getPointCloud(),distance->getPointCloudRGB(),1);
	PopulatePCLPointCloud(cloud1,distance->getPointCloudBeforeBA(),distance->getPointCloudRGBBeforeBA(),1);

	return 0;
}

