#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <k4a/k4a.h>
#include <k4abt.h>

#include <unistd.h>  
#include <sys/socket.h> 
#include <netinet/in.h> 
#include <string.h>

#include <ros/ros.h>
#include <sensor_msgs/image_encodings.h>
#include <sensor_msgs/distortion_models.h>
#include <image_transport/image_transport.h>
using namespace ros;
using namespace sensor_msgs;
using namespace image_transport;
using namespace std; 
#define PORT 8080
using DepthPixel = uint16_t;
struct BgraPixel
{
    uint8_t Blue;
    uint8_t Green;
    uint8_t Red;
    uint8_t Alpha;
};

#define VERIFY(result, error)                                                                            \
    if(result != K4A_RESULT_SUCCEEDED)                                                                   \
    {                                                                                                    \
        printf("%s \n - (File: %s, Function: %s, Line: %d)\n", error, __FILE__, __FUNCTION__, __LINE__); \
        exit(1);                                                                                         \
    }                                                                                                    \

void camera_info_publish(k4a_calibration_t sensor_calibration, ros::Publisher rgb_raw_camerainfo_publisher_)
{
    CameraInfo camera_info;
    camera_info.header.frame_id = "azure/camera_info";
    camera_info.width = sensor_calibration.color_camera_calibration.resolution_width;
    camera_info.height = sensor_calibration.color_camera_calibration.resolution_height;
    camera_info.distortion_model = sensor_msgs::distortion_models::RATIONAL_POLYNOMIAL;

    k4a_calibration_intrinsic_parameters_t parameters = sensor_calibration.color_camera_calibration.intrinsics.parameters;
    camera_info.D = {parameters.param.k1, parameters.param.k2, parameters.param.p1, parameters.param.p2, parameters.param.k3, parameters.param.k4, parameters.param.k5, parameters.param.k6};
    camera_info.K = {parameters.param.fx,  0.0f,                   parameters.param.cx, 
                    0.0f,                  parameters.param.fy,   parameters.param.cy, 
                    0.0f,                  0.0,                    1.0f};
    camera_info.P = {parameters.param.fx,  0.0f,                   parameters.param.cx,   0.0f,
                    0.0f,                  parameters.param.fy,   parameters.param.cy,   0.0f,
                    0.0f,                  0.0,                    1.0f,                   0.0f};
    camera_info.R = {1.0f, 0.0f, 0.0f, 
                    0.0f, 1.0f, 0.0f, 
                    0.0f, 0.0f, 1.0f};

    camera_info.header.stamp = Time::now();
    rgb_raw_camerainfo_publisher_.publish(camera_info);

}

void depth_publish(k4a_image_t depth_image, image_transport::Publisher depth_raw_publisher_)
{
    ImagePtr depth_image_ros(new Image);
    size_t depth_source_size = static_cast<size_t>(k4a_image_get_width_pixels(depth_image) * k4a_image_get_height_pixels(depth_image) * sizeof(DepthPixel));
    // Access the depth image as an array of uint16 pixels
    DepthPixel* depth_frame_buffer = reinterpret_cast<DepthPixel *>(k4a_image_get_buffer(depth_image));
    size_t depth_pixel_count = depth_source_size / sizeof(DepthPixel);

    // Build the ROS message
    depth_image_ros->height = k4a_image_get_height_pixels(depth_image);
    depth_image_ros->width = k4a_image_get_width_pixels(depth_image);
    depth_image_ros->encoding = sensor_msgs::image_encodings::TYPE_32FC1;
    depth_image_ros->is_bigendian = false;
    depth_image_ros->step = k4a_image_get_width_pixels(depth_image) * sizeof(float);

    // Enlarge the data buffer in the ROS message to hold the frame
    depth_image_ros->data.resize(depth_image_ros->height * depth_image_ros->step);

    float* depth_image_data = reinterpret_cast<float*>(&depth_image_ros->data[0]);

    // Copy the depth pixels into the ROS message, transforming them to floats at the same time
    // TODO: can this be done faster?
    for (size_t i = 0; i < depth_pixel_count; i++)
    {
        depth_image_data[i] = static_cast<float>(depth_frame_buffer[i]);

        if (depth_image_data[i] <= 0.f)
        {
            depth_image_data[i] = std::numeric_limits<float>::quiet_NaN();
        }
        else
        {
            depth_image_data[i] /= 1000.0f;
        }
    }
    depth_image_ros->header.stamp = Time::now();
    depth_image_ros->header.frame_id = "azure/depth_image";
    depth_raw_publisher_.publish(depth_image_ros);

}

void image_publish(k4a_image_t  color_image, image_transport::Publisher rgb_raw_publisher_)
{
    ImagePtr rgb_image(new Image);

    size_t color_image_size = static_cast<size_t>(k4a_image_get_width_pixels(color_image) * k4a_image_get_height_pixels(color_image) * sizeof(BgraPixel));
    if (k4a_image_get_size(color_image) != color_image_size)
    {
        ROS_WARN("Invalid k4a_bgra_frame returned from K4A");
        return;
    }
    
    
    // Build the ROS message
    rgb_image->height = k4a_image_get_height_pixels(color_image);
    rgb_image->width = k4a_image_get_width_pixels(color_image);
    rgb_image->encoding = sensor_msgs::image_encodings::BGRA8;
    rgb_image->is_bigendian = false;
    rgb_image->step = k4a_image_get_width_pixels(color_image) * sizeof(BgraPixel);

    // Enlarge the data buffer in the ROS message to hold the frame
    rgb_image->data.resize(rgb_image->height * rgb_image->step);

    ROS_ASSERT_MSG(color_image_size == rgb_image->height * rgb_image->step, "Pixel buffer and ROS message buffer sizes are different");

    uint8_t *rgb_image_data = reinterpret_cast<uint8_t *>(&rgb_image->data[0]);
    uint8_t *bgra_frame_buffer = k4a_image_get_buffer(color_image);
    // Copy memory from the Azure Kinect buffer into the ROS buffer
    memcpy(rgb_image_data, bgra_frame_buffer, rgb_image->height * rgb_image->step);

    rgb_image->header.stamp = Time::now();
    rgb_image->header.frame_id = "azure/color_image";
    
    rgb_raw_publisher_.publish(rgb_image);

}

int main(int argc, char** argv)
{
    ros::init(argc, argv, "my_node_name", ros::init_options::NoSigintHandler);
    ros::NodeHandle nh;
    image_transport::Publisher rgb_raw_publisher_;
    image_transport::Publisher depth_raw_publisher_;
    image_transport::ImageTransport image_transport_(nh);
    rgb_raw_publisher_ = image_transport_.advertise("azure/color_image", 1);
    depth_raw_publisher_ = image_transport_.advertise("azure/depth_image", 1);
    ros::Publisher rgb_raw_camerainfo_publisher_;
    rgb_raw_camerainfo_publisher_ = nh.advertise<CameraInfo>("azure/camera_info", 1);

    int server_fd, new_socket, valread; 
    struct sockaddr_in address; 
    int opt = 1; 
    int addrlen = sizeof(address); 
    char buffer[1024] = {0};
    int coordi_num = 2*3*26+3;
       
    // Creating socket file descriptor 
    if ((server_fd = socket(AF_INET, SOCK_STREAM, 0)) == 0) 
    { 
        perror("socket failed"); 
        exit(EXIT_FAILURE); 
    } 
       
    // Forcefully attaching socket to the port 8080 
    if (setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR | SO_REUSEPORT, 
                                                  &opt, sizeof(opt))) 
    { 
        perror("setsockopt"); 
        exit(EXIT_FAILURE); 
    } 
    address.sin_family = AF_INET; 
    address.sin_addr.s_addr = INADDR_ANY; 
    address.sin_port = htons( PORT ); 

    if (bind(server_fd, (struct sockaddr *)&address,  
                                 sizeof(address))<0) 
    { 
        perror("bind failed"); 
        exit(EXIT_FAILURE); 
    } 
    if (listen(server_fd, 3) < 0) 
    { 
        perror("listen"); 
        exit(EXIT_FAILURE); 
    } 

    if ((new_socket = accept(server_fd, (struct sockaddr *)&address,  
                    (socklen_t*)&addrlen))<0) 
    { 
        perror("accept"); 
        exit(EXIT_FAILURE); 
    } 

    k4a_device_t device = NULL;
    VERIFY(k4a_device_open(0, &device), "Open K4A Device failed!");

    // Start camera. Make sure depth camera is enabled.
    k4a_device_configuration_t deviceConfig = K4A_DEVICE_CONFIG_INIT_DISABLE_ALL;
    deviceConfig.depth_mode = K4A_DEPTH_MODE_NFOV_UNBINNED;
    deviceConfig.synchronized_images_only = true;
    deviceConfig.color_resolution = K4A_COLOR_RESOLUTION_720P;
    deviceConfig.color_format = K4A_IMAGE_FORMAT_COLOR_BGRA32;
    deviceConfig.depth_delay_off_color_usec = 0;
    deviceConfig.wired_sync_mode = K4A_WIRED_SYNC_MODE_STANDALONE;
    deviceConfig.subordinate_delay_off_master_usec = 0;
    deviceConfig.disable_streaming_indicator = false;
    VERIFY(k4a_device_start_cameras(device, &deviceConfig), "Start K4A cameras failed!");

    k4a_calibration_t sensor_calibration;
    VERIFY(k4a_device_get_calibration(device, deviceConfig.depth_mode, deviceConfig.color_resolution, &sensor_calibration),
        "Get depth camera calibration failed!");

    k4abt_tracker_t tracker = NULL;
    k4abt_tracker_configuration_t tracker_config = K4ABT_TRACKER_CONFIG_DEFAULT;
    VERIFY(k4abt_tracker_create(&sensor_calibration, tracker_config, &tracker), "Body tracker initialization failed!");
    k4a_transformation_t transformation_handle = k4a_transformation_create(&sensor_calibration);
    int frame_count = 0;
    do
    {
        k4a_capture_t sensor_capture;
        k4a_wait_result_t get_capture_result = k4a_device_get_capture(device, &sensor_capture, K4A_WAIT_INFINITE);
        if (get_capture_result == K4A_WAIT_RESULT_SUCCEEDED)
        {
            // k4a_image_t color_image = k4a_capture_get_color_image(sensor_capture);
            // image_publish(color_image, rgb_raw_publisher_);
            k4a_image_t color_image = k4a_capture_get_color_image(sensor_capture);
            k4a_image_t depth_image = k4a_capture_get_depth_image(sensor_capture);
            if ((color_image != NULL) && (depth_image != NULL))
            {
                // printf(" | Color16 res:%4dx%4d stride:%5d\n",
                //         k4a_image_get_height_pixels(color_image),
                //         k4a_image_get_width_pixels(color_image),
                //         k4a_image_get_stride_bytes(color_image));
                
                image_publish(color_image, rgb_raw_publisher_);
                camera_info_publish(sensor_calibration, rgb_raw_camerainfo_publisher_);
                k4a_image_release(color_image);
            }
            if (depth_image != NULL)
            {
                // printf(" | Depth16 res:%4dx%4d stride:%5d\n",
                //         k4a_image_get_height_pixels(depth_image),
                //         k4a_image_get_width_pixels(depth_image),
                //         k4a_image_get_stride_bytes(depth_image));

                
                k4a_image_t  transformed_depth_image;

                if(! transformation_handle)
                {
                    ROS_WARN("Invalid transformation handle");
                    return -1;
                }

                k4a_result_t result = k4a_image_create(K4A_IMAGE_FORMAT_DEPTH16, k4a_image_get_width_pixels(color_image), k4a_image_get_height_pixels(color_image),
                                                    k4a_image_get_width_pixels(color_image) * sizeof(uint16_t), & transformed_depth_image);
                if(result == K4A_RESULT_FAILED)
                {
                    ROS_WARN("Transformed imaged creation fail\n");
                    // return -1;
                }
                result = k4a_transformation_depth_image_to_color_camera(transformation_handle, depth_image, transformed_depth_image);	
                if(result == K4A_RESULT_FAILED)
                {
                    ROS_WARN("Transformed imaged tranformation fail\n");
                    // k4a_image_release(depth_image);
                    // k4a_image_release(transformed_depth_image);
                    // return -1;
                }
                if(result == K4A_RESULT_SUCCEEDED)
                {
                    // printf(" | Transformed Depth16 res:%4dx%4d stride:%5d\n",
                    //                 k4a_image_get_height_pixels(transformed_depth_image),
                    //                 k4a_image_get_width_pixels(transformed_depth_image),
                    //                 k4a_image_get_stride_bytes(transformed_depth_image));
                    depth_publish(transformed_depth_image, depth_raw_publisher_);
                }
                // k4a_image_release(depth_image);
                k4a_image_release(transformed_depth_image);
                k4a_image_release(depth_image);
            }
            frame_count++;
            k4a_wait_result_t queue_capture_result = k4abt_tracker_enqueue_capture(tracker, sensor_capture, K4A_WAIT_INFINITE);
            k4a_capture_release(sensor_capture); // Remember to release the sensor capture once you finish using it
            if (queue_capture_result == K4A_WAIT_RESULT_TIMEOUT)
            {
                // It should never hit timeout when K4A_WAIT_INFINITE is set.
                printf("Error! Add capture to tracker process queue timeout!\n");
                break;
            }
            else if (queue_capture_result == K4A_WAIT_RESULT_FAILED)
            {
                printf("Error! Add capture to tracker process queue failed!\n");
                break;
            }

            k4abt_frame_t body_frame = NULL;
            k4a_wait_result_t pop_frame_result = k4abt_tracker_pop_result(tracker, &body_frame, K4A_WAIT_INFINITE);
            if (pop_frame_result == K4A_WAIT_RESULT_SUCCEEDED)
            {
                // Successfully popped the body tracking result. Start your processing
                float coordinates[coordi_num] = {0.0};
                size_t num_bodies = k4abt_frame_get_num_bodies(body_frame);
                printf("%zu bodies are detected!\n", num_bodies);
                if(num_bodies==0)
                {
                    printf("original coodinate sent\n");
                    if (send(new_socket, &coordinates, coordi_num * sizeof(float), 0) < 0) { 
                        puts("Send failed"); 
                        return 1; 
                    }
                }
                for (size_t i = 0; i < num_bodies; i++)
                {
                    if(i==0)
                    {
                        coordinates[0] = 1;
                    }
                    else if(i==1)
                    {
                        coordinates[1] = 1;
                    }
                    k4abt_skeleton_t skeleton;
                    k4abt_frame_get_body_skeleton(body_frame, i, &skeleton);
                    for(int j = 0; j < K4ABT_JOINT_COUNT; j++)
                    {
                        k4abt_joint_t joints = skeleton.joints[j];
                        k4a_float3_t position = joints.position;
                        k4a_float3_t::_xyz xyz = position.xyz;
                        printf("(x,y,z): (%.3f, %.3f, %.3f)\n", 
                                xyz.x, xyz.y, xyz.z);
                        coordinates[3+26*3*i+3*j] = xyz.x;
                        coordinates[3+26*3*i+3*j+1] = xyz.y;
                        coordinates[3+26*3*i+3*j+2] = xyz.z;
                    }
                    
                    // valread = read( new_socket , buffer, 1024); 
                    // printf("%s\n",buffer );
                    
                }
                printf("body coodinate sent\n");
                if (send(new_socket, &coordinates, coordi_num * sizeof(float), 0) < 0) { 
                    puts("Send failed"); 
                    return 1; 
                }
                k4abt_frame_release(body_frame); // Remember to release the body frame once you finish using it
            }
            else if (pop_frame_result == K4A_WAIT_RESULT_TIMEOUT)
            {
                //  It should never hit timeout when K4A_WAIT_INFINITE is set.
                printf("Error! Pop body frame result timeout!\n");
                break;
            }
            else
            {
                printf("Pop body frame result failed!\n");
                break;
            }
        }
        else if (get_capture_result == K4A_WAIT_RESULT_TIMEOUT)
        {
            // It should never hit time out when K4A_WAIT_INFINITE is set.
            printf("Error! Get depth frame time out!\n");
            break;
        }
        else
        {
            printf("Get depth capture returned error: %d\n", get_capture_result);
            break;
        }

    } while (frame_count < 10000);

    printf("Finished body tracking processing!\n");

    k4abt_tracker_shutdown(tracker);
    k4abt_tracker_destroy(tracker);
    k4a_device_stop_cameras(device);
    k4a_device_close(device);

    return 0;
}
