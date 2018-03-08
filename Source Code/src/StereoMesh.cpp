#include "StereoMesh.h"

using namespace cv;
using namespace vvr;
using namespace std;

#define SWITCH_TO_POINTCLOUD 1
int main(int argc, char* argv[]) {

	//////////////////////////////////////////////////////////////////////////
	// Part 1: Depth from Stereo
	//////////////////////////////////////////////////////////////////////////
	//*
	cv::namedWindow("Right Image");
	cv::namedWindow("Left Image");

	//Create Window and Trackbars
	cv::namedWindow("Disparity");
	cv::createTrackbar("SAD", "Disparity", &bm_sad, bm_sad_max, Task_1_CalculateDisparity);

	leftImage = imread(stereo_dir + "left.ppm", CV_LOAD_IMAGE_GRAYSCALE);
	rightImage = imread(stereo_dir + "right.ppm", CV_LOAD_IMAGE_GRAYSCALE);
	trueDisp = imread(stereo_dir + "truedisp.pgm", CV_LOAD_IMAGE_UNCHANGED);

	cv::imshow("True Disparity", trueDisp);

	//If not loaded, return
	if (leftImage.empty() || rightImage.empty()) {
		cout << "Image not Loaded: Incorrect filepath" << endl;
		system("pause");
		return -1;
	}

	cv::imshow("Right Image", rightImage);
	cv::imshow("Left Image", leftImage);
	Task_1_CalculateDisparity(0, (void*)0);

	while (true)
	{
		if (waitKey(10) == 'b')
			break;
	}

	//Clean up
	destroyAllWindows();

	// Start Scene Class
	try {
		return vvr::mainLoop(argc, argv, new Mesh3DScene);
	}
	catch (std::string exc) {
		cerr << exc << endl;
		return 1;
	}
	catch (...)
	{
		cerr << "Unknown exception" << endl;
		return 1;
	}
}



Mesh3DScene::Mesh3DScene()
{
	//! Load settings.
	vvr::Shape::DEF_LINE_WIDTH = 4;
	vvr::Shape::DEF_POINT_SIZE = 4;
	m_perspective_proj = true;
	m_bg_col = Colour("768E77");
	m_obj_col = Colour("454545");

	Task_2_CalculatePointCloud(trueDisp, m_vertices);

	Task_2_pointclouds(trueDisp, m_vertices, m_pointclouds, distances);
	echo(m_pointclouds.size());


	for (int i = 0; i < distances.size(); i++) {
		vvr::Mesh* model = nullptr;
		Task_3_TriangulateMesh(m_vertices, model,distances[i]);

		m_model.push_back(model);
		
	}
	reset();
}

void Mesh3DScene::reset()
{
	Scene::reset();

	//! Define what will be vissible by default
	m_style_flag = 0;
	m_style_flag |= FLAG_SHOW_SOLID;
	m_style_flag |= FLAG_SHOW_WIRE;
	m_style_flag |= FLAG_SHOW_AXES;

}

void Mesh3DScene::mousePressed(int x, int y, int modif)
{
	Ray myRay = this->unproject(x, y);
	float dist; // here the distance to the XY plane will be stored
	myRay.Intersects(Plane(vec(0, 0, 0), vec(0, 0, 1)), &dist);
	vec myPoint = myRay.GetPoint(dist); // Here is the XY point
	std::cout << myPoint << endl;

	// call superclass
	Scene::mousePressed(x, y, modif);
}

void Mesh3DScene::drawCloud(const vector<vec>& vertices, vvr::Colour color)
{
	for (auto &d : vertices)
		Point3D(d.x, d.y, d.z, color).draw();
}

void Mesh3DScene::resetFrustrum()
{
	Frustum res = this->getFrustum();
	res.SetFrame(vec(0, 0, 100), vec(0, 0, -1), vec(0, 1, 0));
	this->setFrustum(res);
	this->reset();
}

void Mesh3DScene::keyEvent(unsigned char key, bool up, int modif)
{
	Scene::keyEvent(key, up, modif);
	key = tolower(key);

	switch (key)
	{
	case 's': m_style_flag ^= FLAG_SHOW_SOLID; break;
	case 'w': m_style_flag ^= FLAG_SHOW_WIRE; break;
	case 'n': m_style_flag ^= FLAG_SHOW_NORMALS; break;
	case 'b': m_style_flag ^= FLAG_SHOW_AABB; break;
	case 'f': resetFrustrum();
	}
}

void Mesh3DScene::draw()
{
	//! Draw Point Cloud or Mesh


	if (SWITCH_TO_POINTCLOUD)
	{
		for (int i = 0; i < m_pointclouds.size(); i++) {
			vvr::Colour color(255, i * 35, 0);
			drawCloud(m_pointclouds[i], color);
		}
	}
	
	else {
		for (int i = 0; i < m_model.size(); i++) {
			vvr::Colour color(255, i * 19, i);
			if (m_style_flag & FLAG_SHOW_SOLID) m_model[i]->draw(color, SOLID);
			if (m_style_flag & FLAG_SHOW_WIRE) m_model[i]->draw(color, WIRE);
			if (m_style_flag & FLAG_SHOW_NORMALS) m_model[i]->draw(color, NORMALS);
			if (m_style_flag & FLAG_SHOW_AXES) m_model[i]->draw(color, AXES);
		}
	}
	
}

//! LAB Tasks

void vvr::Task_1_CalculateDisparity(int, void*)
{
	//Set state
	stereo_matcher = StereoBM(StereoBM::BASIC_PRESET, 16 * (bm_num_disparities + 1), 2 * (bm_sad + 2) + 1);
	stereo_matcher.state->uniquenessRatio = bm_uniqueness;

	//Calculate Disparity
	stereo_matcher(leftImage, rightImage, disp, CV_16S);

	/************************************************************************/
	/* Task 3: Use block-matching to compute disparity                      */
	/* 1) Go through each pixel
	/* 2) match block with next blocks of same row
	/* 3) Store the X-axis distance of best match in disp image
	*
	* HINTS:
	*
	* use Rect to extract 10x10 block from right image
	* use "matchTemplate()" to find best match
	* find max from matchTemplate result and store it to disp Mat
	/************************************************************************/
	//disp = cv::Mat(rightImage.rows, rightImage.cols, CV_16UC1);
	//cv::Rect roi(0, 0, 10, 10); //orismos tetragwnou
	//cv::Mat blockTemplate = rightImage(roi);
	//cv::Rect leftLine(0, 0, leftImage.cols, 10); //sthlh grammh
	//cv::Mat leftLineMat = leftImage(leftLine);
	//cv::Mat matchResults;
	//matchTemplate(leftLineMat, blockTemplate, matchResults, 5);
	//cv::Point result; // thelw to result.x to vazw sto disp
	//cv::minMaxLoc(matchResults, (double*)0, (double*)0, (Point*)0, &result);
	//disp.at<short>(0, 0) = result.x; //sthlh grammh
	/*
	int max = 0;
	for (int i = 0; i < leftImage.cols - 10; i++) {
	for (int j = 0; j < leftImage.rows - 10; j++) {

	cv::Rect roi(i, j, 10, 10);
	cv::Mat blockTemplate = rightImage(roi);
	cv::Rect leftLine(i, j, min(60, leftImage.cols - i), 10);
	cv::Mat leftLineMat = leftImage(leftLine);
	cv::Mat matchResults;
	matchTemplate(leftLineMat, blockTemplate, matchResults, 5);
	cv::Point result;
	cv::minMaxLoc(matchResults, (double*)0, (double*)0, (Point*)0, &result);
	disp.at<short>(j, i) = result.x;
	if (result.x > max) {
	max = result.x;
	}
	}
	}


	//Convert to 8-bit to display
	disp.convertTo(disp8, CV_8U, 255.0f / max);
	cv::imshow("Disparity", disp8);
	*/
}

void vvr::Task_2_CalculatePointCloud(const cv::Mat& disparity, std::vector<vec> &vertices)
{
	//!//////////////////////////////////////////////////////////////////////////////////
	//! TASK:
	//!
	//!  - Calculate 3D points from disparity and store them to vertices.
	//!  - Z = f*(B/d)
	//!  - f = ResX / (2 * tan(FOV/2.0)
	//!  - X = (V-Vo) * (Z/f) // V= pixel grammhs, Vo to kentro
	//!  - Y = (U-Uo) * (Z/f)
	//!
	//!//////////////////////////////////////////////////////////////////////////////////

	//...
	//...
	//...
	vertices.resize(disparity.cols*disparity.rows);
	float f_x = disparity.cols* focal_length_constant;
	float f_y = disparity.rows* focal_length_constant;
	float Vo = disparity.cols / 2;
	float Uo = disparity.rows / 2;
	for (int i = 0; i < disparity.cols; i++) {
		for (int j = 0; j < disparity.rows; j++) {
			float d = float(disparity.at<unsigned char>(j, i));
			if (d > 0.0)
			{
				float Z = -10000.*focal_length_constant * stereo_baseline / d;
				float X = -(i - Vo)* Z / f_x;
				float Y = (j - Uo)* Z / f_y;
				vertices[j*disparity.cols + i] = (vec(X, Y, Z));
			}
		}
	}
	// stub
}

void vvr::Task_2_pointclouds(const cv::Mat& disparity, vector<vec> &vertices, vector<vector<vec>> &m_pointclouds, std::vector<float> &distances) {
	
	std::vector<std::vector<vec>> dif_vert;
	float error = 0.01;
	int k = 0;
	std::vector<vec> clouds;
	std::vector<int> grammes;
	clouds.resize(vertices.size());
	grammes.resize(trueDisp.rows);
	for (int i = 0; i < 10; i++) {
		dif_vert.push_back(clouds);
	}

	distances.push_back(vertices[0].z);
	while (distances[0] < -1000) {
		k++;
		distances[0] = vertices[k].z;
	}
	for (int i = 0; i < vertices.size(); i++) {

		//if (i % trueDisp.rows == 0) {
		//	item[0].push_back(i);
		//}
		if (vertices[i].z > -1000) {

			//vector<float> dist_old = distances;
			for (int j = 0; j < distances.size(); j++) {
				
				if ((vertices[i].z > distances[j] - error) && (vertices[i].z < distances[j] + error)) {
					
					dif_vert[j][i] = vertices[i];
					
					break;
				}
				else if (j == distances.size() - 1) {
					distances.push_back(vertices[i].z);
					dif_vert[distances.size()-1][i] = vertices[i];

				}
			}

		}
	}
	echo(trueDisp.rows);
	for (int i = 0; i < dif_vert.size(); i++) {
		std::vector<vec> current_cloud;
		for (int j = 0; j < dif_vert[i].size(); j++) {
			if (dif_vert[i][j].z > -1000) {
				current_cloud.push_back(dif_vert[i][j]);
			}
		}
		if (current_cloud.size() > 10) {
			m_pointclouds.push_back(current_cloud);
		}
	}

}
void vvr::Task_3_TriangulateMesh(const std::vector<vec>& vertices, vvr::Mesh*& mesh, float dist)
{
	//!//////////////////////////////////////////////////////////////////////////////////
	//! TASK:
	//!
	//!  - Create 2 triangles for every 4-vertex block
	//!  - Create mesh "m_model" (dont forget new Mesh())
	//!
	//!//////////////////////////////////////////////////////////////////////////////////

	mesh = new Mesh();
	vector<vec> myVertices;

	std::vector<vec>& modelVerts = mesh->getVertices();
	std::vector<vvr::Triangle>& tris = mesh->getTriangles();

	for (int k = 0; k < vertices.size(); k++) {
		int y = k / trueDisp.cols;
		int x = k % trueDisp.cols; // to upoloipo
		float error = 0.0001;

		if ((x < trueDisp.cols - 1) && (y < trueDisp.rows - 1)) {
			if ((vertices[y*trueDisp.cols + x].z > dist - error) && (vertices[y*trueDisp.cols + x].z < dist + error)){
				//if ((vertices[y*trueDisp.cols + x].z > -1000) && (vertices[y*trueDisp.cols + x + 1].z > -1000) && (vertices[(y + 1)*trueDisp.cols + x].z > -100) && ((vertices[(y + 1)*trueDisp.cols + x + 1].z > -100))) {
					if ((vertices[y*trueDisp.cols + x].z > vertices[y*trueDisp.cols + x + 1].z - error) && (vertices[y*trueDisp.cols + x].z < vertices[y*trueDisp.cols + x + 1].z + error)) {
						if ((vertices[y*trueDisp.cols + x].z > vertices[(y + 1)*trueDisp.cols + x].z - error) && (vertices[y*trueDisp.cols + x].z < vertices[(y + 1)*trueDisp.cols + x].z + error)) {
							myVertices.push_back(vertices[y*trueDisp.cols + x]);
							myVertices.push_back(vertices[y*trueDisp.cols + x + 1]);
							myVertices.push_back(vertices[(y + 1)*trueDisp.cols + x]);


							if ((vertices[y*trueDisp.cols + x].z > vertices[(y + 1)*trueDisp.cols + x + 1].z - error) && (vertices[y*trueDisp.cols + x].z < vertices[(y + 1)*trueDisp.cols + x + 1].z + error)) {
								myVertices.push_back(vertices[y*trueDisp.cols + x + 1]);
								myVertices.push_back(vertices[(y + 1)*trueDisp.cols + x]);
								myVertices.push_back(vertices[(y + 1)*trueDisp.cols + x + 1]);

							}
						}
						if ((vertices[y*trueDisp.cols + x].z > vertices[(y + 1)*trueDisp.cols + x + 1].z - error) && (vertices[y*trueDisp.cols + x].z < vertices[(y + 1)*trueDisp.cols + x + 1].z + error)) {
							myVertices.push_back(vertices[y*trueDisp.cols + x]);
							myVertices.push_back(vertices[(y)*trueDisp.cols + x + 1]);
							myVertices.push_back(vertices[(y + 1)*trueDisp.cols + x + 1]);

						}
					}
				}
			//}
		}
	}

	

	for (auto& d : myVertices) modelVerts.push_back(d);
	for (int i = 0; i < modelVerts.size(); i = i + 3) {
		tris.push_back(vvr::Triangle(&modelVerts, i, i + 1, i + 2));
	}



}