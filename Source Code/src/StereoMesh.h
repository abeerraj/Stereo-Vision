#ifndef __STEREO_MESH_H
#define __STEREO_MESH_H
#include <VVRScene/canvas.h>
#include <VVRScene/mesh.h>
#include <VVRScene/settings.h>
#include <VVRScene/utils.h>
#include <MathGeoLib.h>
#include <opencv2/opencv.hpp>
#include <iostream>

#define DEFAULT 0.535353

using namespace cv;

namespace vvr {

	std::string stereo_dir = "../resources/images/";

	//Our Block Matching class
	StereoBM stereo_matcher(StereoBM::BASIC_PRESET, 16);

	//Stereo Matrices
	Mat leftImage;
	Mat rightImage;
	Mat trueDisp;
	Mat disp;
	Mat disp8; //8-bit disparity to display

	//Block Matching parameters;
	int bm_sad = 10; int bm_sad_max = 15; //*2 + 1
	int bm_num_disparities = 1; int bm_num_disparities_max = 10; //*16
	int bm_uniqueness = 0;	int bm_uniqueness_max = 100;

	//Stereo to 3D params  ( Z = f(B/d) ) (f = ResX / (2 * tan(FOV/2.0) ) (X = (V-Vc) * (Z/f))
	float stereo_baseline = 0.2; // 20cm
	float fieldOfView = 1.2; // rad
	float focal_length_constant = 1.0f / (2.0f * tan(fieldOfView / 2.0f));

	#define FLAG_SHOW_AXES       1
	#define FLAG_SHOW_WIRE       2
	#define FLAG_SHOW_SOLID      4
	#define FLAG_SHOW_NORMALS    8
	#define FLAG_SHOW_PLANE     16
	#define FLAG_SHOW_AABB      32

	void Task_1_CalculateDisparity(int, void*);
	void Task_2_CalculatePointCloud(const cv::Mat& disparity, std::vector<vec> &vertices);
	void Task_3_TriangulateMesh(const std::vector<vec> &vertices, vvr::Mesh*& mesh,float dist);
	void Task_2_pointclouds(const cv::Mat& disparity, std::vector<vec> &vertices, std::vector<std::vector<vec>> &m_pointclouds, std::vector<float> &distances);
	class Mesh3DScene : public vvr::Scene
	{
	public:
		Mesh3DScene();
		const char* getName() const { return "3D Scene"; }
		void resetFrustrum();
		void keyEvent(unsigned char key, bool up, int modif) override;
		virtual void mousePressed(int x, int y, int modif) override;

	private:
		void drawCloud(const vector<vec>& vertices,vvr::Colour);
		void draw() override;
		void reset() override;

	private:
		int m_style_flag;
		float m_plane_d;
		vvr::Canvas2D m_canvas;
		vvr::Colour m_obj_col;
		std::vector<vec> m_vertices;
		std::vector<std::vector<vec>> m_pointclouds ;
		std::vector<vvr::Mesh*> m_model ;
		std::vector< std::vector< int > > item;
		vvr::Mesh* modelo = nullptr;
		std::vector<float> distances;
		vvr::Mesh* modelaki;
	};
}

#endif // __VVR_LAB0_H