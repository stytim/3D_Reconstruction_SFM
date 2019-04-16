#include <iostream>
#include "opencv2/core.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/xfeatures2d.hpp"
#include "opencv2/calib3d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <stdio.h>
#include <fstream>
#include <typeinfo>

using namespace std;

using namespace std;
using namespace cv;
using namespace cv::xfeatures2d;


void readme()
{
    std::cout << " Usage: ./SURF_FlannMatcher <img1> <img2>" << std::endl;
}

void extract_features(
                      vector<string>& image_names,
                      vector<vector<KeyPoint> >& key_points_for_all,
                      vector<Mat>& descriptor_for_all,
                      vector<vector<Vec3b> >& colors_for_all
                      )
{
    key_points_for_all.clear();
    descriptor_for_all.clear();
    int minHessian = 10;
    Mat image;
    
    // Read input images, extract and save features
    Ptr<SURF> detector = SURF::create();
    detector->setHessianThreshold(minHessian);
    
    for (auto it = image_names.begin(); it != image_names.end(); ++it)
    {
        image = imread(*it);
        if (image.empty()) continue;
        
        vector<KeyPoint> key_points;
        Mat descriptor;
        detector->detectAndCompute(image, Mat(), key_points, descriptor);
        
        // If an image contains supper less feature points, then take this image off
        if (key_points.size() <= 10) continue;
        
        key_points_for_all.push_back(key_points);
        descriptor_for_all.push_back(descriptor);
        // Save RGB information for features
        vector<Vec3b> colors(key_points.size());
        for (int i = 0; i < key_points.size(); ++i)
        {
            Point2f& p = key_points[i].pt;
            colors[i] = image.at<Vec3b>(p.y, p.x);
        }
        colors_for_all.push_back(colors);
    }
}

void match_features(Mat& query, Mat& train, vector<DMatch>& matches)
{
    vector<vector<DMatch> > knn_matches;
    BFMatcher matcher(NORM_L2);
    matcher.knnMatch(query, train, knn_matches, 2);
    
    // Get the shortest matching distance by Ratio Test
    float min_dist = FLT_MAX;
    for (int r = 0; r < knn_matches.size(); ++r)
    {
        // Ratio Test
        if (knn_matches[r][0].distance > 0.6*knn_matches[r][1].distance)
            continue;
        
        float dist = knn_matches[r][0].distance;
        if (dist < min_dist) min_dist = dist;
    }
    
    matches.clear();
    for (size_t r = 0; r < knn_matches.size(); ++r)
    {
        // Take off points that do not fit Ratio Test or the matching distance is to large
        if (
            knn_matches[r][0].distance > 0.6*knn_matches[r][1].distance ||
            knn_matches[r][0].distance > 5 * max(min_dist, 10.0f)
            )
            continue;
        
        // Save matching points
        matches.push_back(knn_matches[r][0]);
    }
    
}

bool find_transform(Mat& K, vector<Point2f>& p1, vector<Point2f>& p2, Mat& R, Mat& T, Mat& mask)
{
    // Based on the Calibration Matric, compute the Focal Lengths and Principle Point
    double focal_length = 0.5*(K.at<double>(0) + K.at<double>(4));
    Point2d principle_point(K.at<double>(2), K.at<double>(5));
    
    // Based on those matching features, to solve Essential Matrix. Using RANSAC to further take off those miss matching points
    Mat E = findEssentialMat(p1, p2, focal_length, principle_point, RANSAC, 0.999, 1.0, mask);
    if (E.empty()) return false;
    
    double feasible_count = countNonZero(mask);
    //cout << (int)feasible_count << " -in- " << p1.size() << endl;
    // By using RANSAC, if number of outliers is more than 50%, the result is unreliable
    if (feasible_count <= 10 || (feasible_count / p1.size()) < 0.6)
        return false;
    
    // Decompose Essential Matrix to get relative camera rotation and translation
    int pass_count = recoverPose(E, p1, p2, R, T, focal_length, principle_point, mask);
    
    // Make sure there has enough 3D points that will use for reconstraction
    if (((double)pass_count) / feasible_count < 0.6)
        return false;
    
    return true;
}

void get_matched_points(
                        vector<KeyPoint>& p1,
                        vector<KeyPoint>& p2,
                        vector<DMatch> matches,
                        vector<Point2f>& out_p1,
                        vector<Point2f>& out_p2
                        )
{
    out_p1.clear();
    out_p2.clear();
    for (int i = 0; i < matches.size(); ++i)
    {
        out_p1.push_back(p1[matches[i].queryIdx].pt);
        out_p2.push_back(p2[matches[i].trainIdx].pt);
    }
}

void get_matched_colors(
                        vector<Vec3b>& c1,
                        vector<Vec3b>& c2,
                        vector<DMatch> matches,
                        vector<Vec3b>& out_c1,
                        vector<Vec3b>& out_c2
                        )
{
    out_c1.clear();
    out_c2.clear();
    for (int i = 0; i < matches.size(); ++i)
    {
        out_c1.push_back(c1[matches[i].queryIdx]);
        out_c2.push_back(c2[matches[i].trainIdx]);
    }
}

void reconstruct(Mat& K, Mat& R, Mat& T, vector<Point2f>& p1, vector<Point2f>& p2, Mat& structure)
{
    // Set up Projection Matrix for triangulation
    Mat proj1(3, 4, CV_32FC1);
    Mat proj2(3, 4, CV_32FC1);
    
    proj1(Range(0, 3), Range(0, 3)) = Mat::eye(3, 3, CV_32FC1);
    proj1.col(3) = Mat::zeros(3, 1, CV_32FC1);
    
    R.convertTo(proj2(Range(0, 3), Range(0, 3)), CV_32FC1);
    T.convertTo(proj2.col(3), CV_32FC1);
    
    Mat fK;
    K.convertTo(fK, CV_32FC1);
    proj1 = fK*proj1;
    proj2 = fK*proj2;
    
    // Triangulation
    triangulatePoints(proj1, proj2, p1, p2, structure);
}

void makeup_points(vector<Point2f>& p1, Mat& mask)
{
    vector<Point2f> p1_copy = p1;
    p1.clear();
    
    for (int i = 0; i < mask.rows; ++i)
    {
        if (mask.at<uchar>(i) > 0)
            p1.push_back(p1_copy[i]);
    }
}

void makeup_colors(vector<Vec3b>& p1, Mat& mask)
{
    vector<Vec3b> p1_copy = p1;
    p1.clear();
    
    for (int i = 0; i < mask.rows; ++i)
    {
        if (mask.at<uchar>(i) > 0)
            p1.push_back(p1_copy[i]);
    }
}

// Save structure in yml file
void save_structure(string file_name, vector<Mat>& rotations, vector<Mat>& motions, Mat& structure, vector<Vec3b>& colors)
{
    int n = (int)rotations.size();
    
    FileStorage fs(file_name, FileStorage::WRITE);
    fs << "Camera Count" << n; // Number of camera poses
    fs << "Number of points are visible in each camera frame" << structure.cols;
    
    fs << "Rotation Matrices" << "[";
    for (size_t i = 0; i < n; ++i)
    {
        fs << rotations[i];
    }
    fs << "]";
    
    fs << "Translation vectors" << "[";
    for (size_t i = 0; i < n; ++i)
    {
        fs << motions[i];
    }
    fs << "]";
    
    fs << "Points" << "["; // 3D position of each feature point
    for (size_t i = 0; i < structure.cols; ++i)
    {
        Mat_<float> c = structure.col(i);
        c /= c(3);    // Get the real coordinate of the points
        fs << Point3f(c(0), c(1), c(2));
    }
    fs << "]";
    
    fs << "Colors" << "["; // RGB information about each feature
    for (size_t i = 0; i < colors.size(); ++i)
    {
        fs << colors[i];
    }
    fs << "]";
    
    fs.release();
}

void read_projective(vector<Mat>& projective, int num1, int num2){
    fstream fin;
    fin.open("projective.txt",ios::in);
    vector<string> data;
    char line[1000];
    while(fin.getline(line,sizeof(line),'\n')){
        string temp(line);
        data.push_back(temp);
    }
    
    vector<float> str_list1;
    vector<float> str_list2;
    
    
    do
    {
        int comma_n = 0;
        string tmp_s = "";
        comma_n = data[num1].find(" ");
        if( -1 == comma_n )
        {
            tmp_s = data[num1].substr( 0, data[num1].length() );
            str_list1.push_back( stof(tmp_s) );
            break;
        }
        tmp_s = data[num1].substr( 0, comma_n );
        data[num1].erase( 0, comma_n+1 );
        str_list1.push_back( stof(tmp_s) );
    }
    while(true);
    
    do
    {
        int comma_n = 0;
        string tmp_s = "";
        comma_n = data[num2].find(" ");
        if( -1 == comma_n )
        {
            tmp_s = data[num2].substr( 0, data[num2].length() );
            str_list2.push_back( stof(tmp_s) );
            break;
        }
        tmp_s = data[num2].substr( 0, comma_n );
        data[num2].erase( 0, comma_n+1 );
        str_list2.push_back( stof(tmp_s) );
    }
    while(true);

    Mat proj1 = (Mat_<float>(3,4)<< str_list1[0],   str_list1[1],   str_list1[2],    str_list1[3],
                 str_list1[4],   str_list1[5],  str_list1[6], str_list1[7],
                str_list1[8],  str_list1[9],  str_list1[10],  str_list1[11]);
    Mat proj2 = (Mat_<float>(3,4)<< str_list2[0],   str_list2[1],   str_list2[2],    str_list2[3],
                 str_list2[4],   str_list2[5],  str_list2[6], str_list2[7],
                 str_list2[8],  str_list2[9],  str_list2[10],  str_list2[11]);
    projective.push_back(proj1);
    projective.push_back(proj2);
}


int main(int argc, char** argv)
{
    if (argc < 4)
    {
        readme(); return -1;
    }
    
    // Calibration matrix
    Mat K(Matx33d(
                  39.4552, 0, 3.5547,
                  0, 28.1128, 13.1281,
                  0, 0, 0.0123));
    
    // Distortion Coeffs
    Mat disCoeffs = ((Mat_<float>(5,1)<< 0, 0,0,0,0));
    //-5.84026158e-01, 1.18702066e+00, 1.32601075e-02, 1.43725509e-02, -1.12817514e+00
    //vector<string> img_names = { img1, img2 };
    
    Mat img_1 = imread(argv[1], IMREAD_COLOR);
    Mat img_2 = imread(argv[2], IMREAD_COLOR);
    
    
    // Use intrinsic parameters and distortion coeffs to undistort two images first
    Mat undisImg1;
    Mat undisImg2;
    
    undistort(img_1, undisImg1, K, disCoeffs);
    undistort(img_2, undisImg2, K, disCoeffs);
    
    //imshow("Original Image1", img_1);
    //imshow("Undistorted Image1", undisImg1);
    
    //imshow("Original Image2", img_2);
    //imshow("Undistorted Image2", undisImg2);
    
    //imwrite("undistorted1.jpg", undisImg1);
    //imwrite("undistorted2.jpg", undisImg2);
    
    //string img1 = "undistorted1.jpg";
    //string img2 = "undistorted2.jpg";
    
    string img1 = argv[1];
    string img2 = argv[2];
    
    
    vector<string> img_names;
    img_names.push_back(img1);
    img_names.push_back(img2);
    
    vector<vector<KeyPoint> > key_points_for_all;
    vector<Mat> descriptor_for_all;
    vector<vector<Vec3b> > colors_for_all;
    vector<DMatch> matches12, matches21;
    
    
    // ***************************************************
    // ***************************************************
    // TASK 1: Extract image features and show correspondences between images
    
    // OUTPUT 3 WINDOWS:
    //      1. Display first image, indicating where matching features were found
    //      2. Display second image, indicating where matching features were found
    //
    
    // Extract features
    extract_features(img_names, key_points_for_all, descriptor_for_all, colors_for_all);
    // Matching features
    match_features(descriptor_for_all[0], descriptor_for_all[1], matches12);
    match_features(descriptor_for_all[1], descriptor_for_all[0], matches21);
    // Draw matches
    Mat img_matches;
    drawMatches(img_1, key_points_for_all[0], img_2, key_points_for_all[1], matches12, img_matches);
    // Show detected matches
    imwrite("Matches.jpg", img_matches);
    imshow("Matches", img_matches);
    waitKey(0);
    // ***************************************************
    // ***************************************************
    // TASK 2: Localize the camera in both frames
    
    // OUTPUT: Translation and Rotation of camera in frame 1
    // OUTPUT: Translation and Rotation of camera in frame 2
    
    // Compute Essentional Matrix, and get relative camera rotation and translation
    vector<Point2f> p1, p2, p3, p4;
    vector<Vec3b> c1, c2;
    Mat R12, T12, R21, T21;    // Relative camera rotation and translation
    Mat mask12, mask21;    // Used to determine whether it is matching points and miss matching points
    get_matched_points(key_points_for_all[0], key_points_for_all[1], matches12, p1, p2);
    get_matched_points(key_points_for_all[1], key_points_for_all[0], matches21, p3, p4);
    get_matched_colors(colors_for_all[0], colors_for_all[1], matches12, c1, c2);
    find_transform(K, p1, p2, R12, T12, mask12);
    find_transform(K, p3, p4, R21, T21, mask21);
    //cout << "Rotation of camera in frame 1 is:  " << endl << " " << R12 << endl << endl;
    //cout << "Translation of camera in frame 1 is: " << endl << " " << T12 << endl << endl;
    //cout << "Rotation of camera in frame 2 is:  " << endl << " " << R21 << endl << endl;
    //cout << "Translation of camera in frame 2 is: " << endl << " " << T21 << endl << endl;
    
    // ***************************************************
    // ***************************************************
    // TASK 3: Find the 3D positions of the matched features
    
    // OUTPUT: 3D position of each feature point
    // OUTPUT: How many shared points are visible in each camera frame
    // OUTPUT: The projection error in each camera frame
    
    // Triangulation
    Mat structure;
    vector<Point3f> objectPoints;
    vector<Point2f> reprojectionPoints12;
    vector<Point2f> reprojectionPoints21;
    Mat rvecs1, rvecs2;
    Mat tvecs1, tvecs2;
    double projectionError1;
    double projectionError2;
    makeup_points(p1, mask12);
    makeup_points(p2, mask12);
    //reconstruct(K, R21, T21, p1, p2, structure);
    
    //Mat proj1(3, 4, CV_32FC1);
    //Mat proj2(3, 4, CV_32FC1);
    vector<Mat> projective_for_all;
    read_projective(projective_for_all, atoi(argv[3]), atoi(argv[4]));
    
    
    triangulatePoints(projective_for_all[0], projective_for_all[1], p1, p2, structure);
    
    for (size_t i = 0; i < structure.cols; ++i)
    {
        Mat_<float> c = structure.col(i);
        c /= c(3);    // Get the real coordinate of the points
        objectPoints.push_back(Point3f(c(0), c(1), c(2)));
    }
    Mat c_cam = Mat(Size(3,3), CV_32FC1);
    Mat rotation_matrix = Mat(Size(3,3), CV_32FC1);
    tvecs1 = Mat(Size(4,1), CV_32FC1);
    decomposeProjectionMatrix(projective_for_all[0], c_cam, rotation_matrix, tvecs1);
    //solvePnP(objectPoints, p1, K, disCoeffs, rvecs1, tvecs1);
    //solvePnP(objectPoints, p2, K, disCoeffs, rvecs2, tvecs2);
    /*
    Mat rotation_matrix = Mat(Size(3,3), CV_64FC1);
    Rodrigues(rvecs1, rotation_matrix);
     */
    Mat homo_matrix = Mat::eye(4, 4, CV_32FC1);
     
    for (int i = 0; i < 3; i++){
        for (int j = 0; j < 3; j++){
            homo_matrix.at<float>(i, j) = rotation_matrix.at<float>(i, j);
        }
    }
    for (int i = 0; i < 3; i++){
        homo_matrix.at<float>(i, 3) = tvecs1.at<float>(i)/tvecs1.at<float>(3);
    }
    //cout<<rotation_matrix.type()<<endl;
    //cout<<rotation_matrix<<endl;
    //cout<<homo_matrix<<endl;
    
    FILE *f2;
    f2 = fopen("motion.txt", "a");
    if (f2 == NULL) {
        fprintf(stderr, "Can't open output file!\n");
        exit(1);
    }
    char buf[200];
    
    sprintf(buf, "%g %g %g %g %g %g %g %g %g %g %g %g %g %g %g %g\n", homo_matrix.at<float>(0, 0), homo_matrix.at<float>(0, 1), homo_matrix.at<float>(0, 2), homo_matrix.at<float>(0, 3), homo_matrix.at<float>(1, 0), homo_matrix.at<float>(1, 1), homo_matrix.at<float>(1, 2), homo_matrix.at<float>(1, 3), homo_matrix.at<float>(2, 0), homo_matrix.at<float>(2, 1), homo_matrix.at<float>(2, 2), homo_matrix.at<float>(2, 3), homo_matrix.at<float>(3, 0), homo_matrix.at<float>(3, 1), homo_matrix.at<float>(3, 2), homo_matrix.at<float>(3, 3));
    fprintf(f2, buf);
    fclose(f2);
    //cout<<"FUCK";
    
    /*
    // Compute reprojection error corresponding to each image frame
    projectPoints(objectPoints, rvecs1, tvecs1, K, disCoeffs, reprojectionPoints12);
    projectPoints(objectPoints, rvecs2, tvecs2, K, disCoeffs, reprojectionPoints21);
    projectionError1 = norm(p1, reprojectionPoints12, CV_L2);
    projectionError2 = norm(p2, reprojectionPoints21, CV_L2);
    projectionError1 = sqrt(projectionError1*projectionError1 / reprojectionPoints12.size());
    projectionError2 = sqrt(projectionError2*projectionError2 / reprojectionPoints21.size());
    //cout << "The projection error in frame 1 is : " << endl << " " << projectionError1 << endl;
    //cout << "The projection error in frame 2 is : " << endl << " " << projectionError2 << endl;
    */
    // Save all the data
    //vector<Mat> rotations;
    //vector<Mat> motions;
    //rotations.push_back(Mat::eye(3, 3, CV_64FC1));
    //rotations.push_back(R12);
    //motions.push_back(Mat::zeros(3, 1, CV_64FC1));
    //motions.push_back(T12);
    makeup_colors(c1, mask12);
    //save_structure("./output.yml", rotations, motions, structure, c1);
    
    //std::cout<<"WIDTH  ";
    //std::cout<<structure.cols<<std::endl;
    
    
    char buff[200];
    
    //// ***************************************************
    //// ***************************************************
    //// FINALLY: Modify this code to output the 3D points in a pointcloud PCD file
    
    FILE *fl;
    
    string output_name = "./pcd/";
    output_name = output_name.append(argv[5], 0, strlen(argv[5]));
    output_name = output_name.append(argv[3], 0, strlen(argv[3]));
    output_name = output_name.append(argv[4], 0, strlen(argv[4]));
    output_name = output_name + ".pcd";
    string text_out = "";
    text_out = text_out.append(output_name, 1, output_name.size()-1);
    cout<<text_out<<endl;
    
    
    fl = fopen(output_name.c_str(), "w");
    
    if (fl == NULL) {
        fprintf(stderr, "Can't open output file!\n");
        exit(1);
    }
    
    strcpy(buff, "# .PCD v.7 - Point Cloud Data file format\n");
    fprintf(fl, buff);
    strcpy(buff, "VERSION .7\n");
    fprintf(fl, buff);
    strcpy(buff, "FIELDS x y z rgb\n");
    fprintf(fl, buff);
    strcpy(buff, "SIZE 4 4 4 4\n");
    fprintf(fl, buff);
    strcpy(buff, "TYPE F F F F\n");
    fprintf(fl, buff);
    strcpy(buff, "COUNT 1 1 1 1\n");
    fprintf(fl, buff);
    sprintf(buff, "WIDTH %d\n", structure.cols);
    fprintf(fl, buff);
    strcpy(buff, "HEIGHT 1\n");
    fprintf(fl, buff);
    strcpy(buff, "VIEWPOINT 0 0 0 1 0 0 0\n");
    fprintf(fl, buff);
    sprintf(buff, "POINTS %d\n", structure.cols);
    fprintf(fl, buff);
    strcpy(buff, "DATA ascii\n");
    fprintf(fl, buff);
    
    for (int i = 0; i<structure.cols; i++) {
        uint8_t b = c1[i][0], g = c1[i][1], r = c1[i][2];
        uint32_t rgb = ((uint32_t)r << 16 | (uint32_t)g << 8 | (uint32_t)b);
        //cout<<*reinterpret_cast<float*>(&rgb)<<endl;
        // ......
        Mat_<float> c = structure.col(i);
        c /= c(3);    // Get the real coordinate of the points
        sprintf(buff, "%g %g %g %g\n", c(0), c(1), c(2), *reinterpret_cast<float*>(&rgb));
        // NOTE: if you don't have RGB information about each feature, just leave them white
    
        fprintf(fl, buff);
    }
    
    /*
    for (size_t i = 0; i < c1.size(); ++i)
    {
        //sprintf(buff,"%d 1/n", c1[i]);
        fprintf(fl, buff);
    }
    */
    fclose(fl);
    
    
    return 0;
}

