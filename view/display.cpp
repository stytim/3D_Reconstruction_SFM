#include <pcl/visualization/cloud_viewer.h>
#include <iostream>
#include <pcl/io/io.h>
#include <pcl/io/pcd_io.h>
#include <fstream>

int user_data;

void
viewerOneOff (pcl::visualization::PCLVisualizer& viewer)
{
    viewer.setBackgroundColor (1.0, 0.5, 1.0);
    pcl::PointXYZ o;
    o.x = 1.0;
    o.y = 0;
    o.z = 0;
    viewer.addSphere (o, 0.25, "sphere", 0);
    std::cout << "i only run once" << std::endl;
    
}

void
viewerPsycho (pcl::visualization::PCLVisualizer& viewer)
{
    static unsigned count = 0;
    std::stringstream ss;
    ss << "Once per viewer loop: " << count++;
    viewer.removeShape ("text", 0);
    viewer.addText (ss.str(), 200, 300, "text", 0);
    
    //FIXME: possible race condition here:
    user_data++;
}

int
main (int argc, char** argv)
{
    
    fstream fin;
    fin.open("/Users/weiwei/Documents/wei/Robotics_MS/Study/2017_Fall/Computer_Vision/project/build/tt.txt",ios::in);
    std::vector<std::string> data;
    char line[1000];
    while(fin.getline(line,sizeof(line),'\n')){
        std::string temp(line);
        std::string input;
        input = "/Users/weiwei/Documents/wei/Robotics_MS/Study/2017_Fall/Computer_Vision/project/build";
        input = input+temp;
        data.push_back(input);
    }
    
    fin.close();
    
    fstream fin2;
    fin2.open("/Users/weiwei/Documents/wei/Robotics_MS/Study/2017_Fall/Computer_Vision/project/build/motion.txt",ios::in);
    std::vector<std::string> motion;
    std::vector< std::vector<double> > str_list;
    cout<<"FUCKK";
    while(fin2.getline(line,sizeof(line),'\n')){
        std::string temp(line);
        motion.push_back(temp);
    }
    for (int i = 0; i < motion.size(); i++){
        std::vector<double> temp_list;
        str_list.push_back(temp_list);
        do
        {
            int comma_n = 0;
            std::string tmp_s = "";
            comma_n = motion[i].find(" ");
            if( -1 == comma_n )
            {
                tmp_s = motion[i].substr( 0, motion[i].length() );
                str_list[i].push_back( stod(tmp_s) );
                break;
            }
            tmp_s = motion[i].substr( 0, comma_n );
            motion[i].erase( 0, comma_n+1 );
            str_list[i].push_back( stod(tmp_s) );
        }while(true);
    }
    fin2.close();
    

    pcl::visualization::PCLVisualizer vizbunny;
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloudbunny (new pcl::PointCloud<pcl::PointXYZ>);
    pcl::io::loadPCDFile ("/Users/weiwei/Documents/wei/Robotics_MS/Study/2017_Fall/Computer_Vision/project/view/bunny.pcd", *cloudbunny);
    vizbunny.addPointCloud<pcl::PointXYZ> (cloudbunny, "sample cloud");
    vizbunny.setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 15, "sample cloud");
    
    pcl::visualization::PCLVisualizer viz0;
    std::vector <pcl::PointCloud<pcl::PointXYZRGBA>::Ptr> all_cloud;
    for(int i = 0; i < data.size(); i++ ){
        pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZRGBA>);
        all_cloud.push_back(cloud);
    }
    for(int i = 0; i < data.size(); i++ ){
        pcl::io::loadPCDFile (data[i], *all_cloud[i]);
        viz0.addPointCloud<pcl::PointXYZRGBA> (all_cloud[i], std::to_string(i));
        viz0.setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 5, std::to_string(i));
        Eigen::Affine3f A = Eigen::Affine3f::Identity();
        A(0,0) = str_list[i][0];
        A(0,1) = str_list[i][1];
        A(0,2) = str_list[i][2];
        A(0,3) = str_list[i][3];
        A(1,0) = str_list[i][4];
        A(1,1) = str_list[i][5];
        A(1,2) = str_list[i][6];
        A(1,3) = str_list[i][7];
        A(2,0) = str_list[i][8];
        A(2,1) = str_list[i][9];
        A(2,2) = str_list[i][10];
        A(2,3) = str_list[i][11];
        A(3,0) = str_list[i][12];
        A(3,1) = str_list[i][13];
        A(3,2) = str_list[i][14];
        A(3,3) = str_list[i][15];
        cout<<A.matrix()<<endl;
        std::string name = "coor";
        name = name+std::to_string(i);
        viz0.addCoordinateSystem (0.08, A, name, 0);
        viz0.spinOnce ();
        if (i == 100){
        viz0.spin();
        std::cin.get();
            
        }
    }
    
    //==========
    /*
    pcl::visualization::PCLVisualizer viz0;
    std::string input1;
    input1 = "/Users/weiwei/Documents/wei/Robotics_MS/Study/2017_Fall/Computer_Vision/project/";
    input1 = input1.append(argv[1], 0, strlen(argv[1]));
    
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud1 (new pcl::PointCloud<pcl::PointXYZRGB>);
    
    pcl::io::loadPCDFile (input1, *cloud1);
    
    viz0.addPointCloud<pcl::PointXYZRGB> (cloud1, "cloud3");
     */
    //==========

    
    viz0.addCoordinateSystem (0.08, "cloud", 0);

    viz0.spin();
    return 0;
}
