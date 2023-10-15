#!/usr/bin/env python3

'''
Run as:
# check model path line ~30is
rosrun depthai_publisher dai_publisher_yolov5_runner
'''
############################### ############################### Libraries ###############################
from pathlib import Path
import threading
import csv
import argparse
import time
import sys
import json     # Yolo conf use json files
import cv2
import numpy as np
import depthai as dai
import rospy
from sensor_msgs.msg import CompressedImage, Image, CameraInfo
from cv_bridge import CvBridge, CvBridgeError
from depthai_publisher.msg import target_msg
from geometry_msgs.msg import Pose, PoseStamped
import math
############################### ############################### Parameters ###############################
# Global variables to deal with pipeline creation
pipeline = None
cam_source = 'rgb' #'rgb', 'left', 'right'
cam=None
# sync outputs
syncNN = True
# model path
modelsPath = "/home/uavteam2/catkin_ws/src/depthai_publisher-yolo_detector/src/depthai_publisher/models"
# modelName = 'exp31Yolov5_ov21.4_6sh'
modelName = 'best_openvino_2022.1_6shave'
# confJson = 'exp31Yolov5.json'
confJson = 'best.json'

################################  Yolo Config File
# parse config
configPath = Path(f'{modelsPath}/{modelName}/{confJson}')
if not configPath.exists():
    raise ValueError("Path {} does not exist!".format(configPath))

with configPath.open() as f:
    config = json.load(f)
nnConfig = config.get("nn_config", {})

# Extract metadata
metadata = nnConfig.get("NN_specific_metadata", {})
classes = metadata.get("classes", {})
coordinates = metadata.get("coordinates", {})
anchors = metadata.get("anchors", {})
anchorMasks = metadata.get("anchor_masks", {})
iouThreshold = metadata.get("iou_threshold", {})
confidenceThreshold = metadata.get("confidence_threshold", {})
# Parse labels
nnMappings = config.get("mappings", {})
labels = nnMappings.get("labels", {})

class RunningAverage:
    def __init__(self, window_size):
        self.window_size = window_size
        self.samples = []
        self.sum = 0.0

    def add_sample(self, sample):
        self.samples.append(sample)
        self.sum += sample

        if len(self.samples) > self.window_size:
            removed_sample = self.samples.pop(0)
            self.sum -= removed_sample

    def get_average(self):
        if len(self.samples) == 0:
            return 0.0
        return self.sum / len(self.samples)

class DepthaiCamera():
    # res = [416, 416]
    fps = 20.0

    pub_topic = '/depthai_node/image/compressed'
    #pub_topic_raw = '/depthai_node/image/raw'
    pub_topic_detect = '/depthai_node/detection/compressed'
    pub_topic_cam_inf = '/depthai_node/camera/camera_info'
    pub_topic_detections = '/depthai_node/target_detections'

    def __init__(self):
        self.pipeline = dai.Pipeline()
        self.sub_pose = rospy.Subscriber("/mavros/local_position/pose", PoseStamped, self.callback_pose)
        self.current_location = Pose()
         # Input image size
        if "input_size" in nnConfig:
            self.nn_shape_w, self.nn_shape_h = tuple(map(int, nnConfig.get("input_size").split('x')))

        # Pulbish ros image data
        self.pub_image = rospy.Publisher(self.pub_topic, CompressedImage, queue_size=10)
        #self.pub_image_raw = rospy.Publisher(self.pub_topic_raw, Image, queue_size=10)
        self.pub_image_detect = rospy.Publisher(self.pub_topic_detect, CompressedImage, queue_size=10)
        self.pub_target_detect = rospy.Publisher(self.pub_topic_detections, target_msg, queue_size=10)
        # Create a publisher for the CameraInfo topic
        self.pub_cam_inf = rospy.Publisher(self.pub_topic_cam_inf, CameraInfo, queue_size=10)
        # Create a timer for the callback
        self.timer = rospy.Timer(rospy.Duration(1.0 / 10), self.publish_camera_info, oneshot=False)

        rospy.loginfo("Publishing images to rostopic: {}".format(self.pub_topic))

        self.br = CvBridge()

        rospy.on_shutdown(lambda: self.shutdown())

    def publish_camera_info(self, timer=None):
        # Create a publisher for the CameraInfo topic

        # Create a CameraInfo message
        camera_info_msg = CameraInfo()
        camera_info_msg.header.frame_id = "camera_frame"
        camera_info_msg.height = self.nn_shape_h # Set the height of the camera image
        camera_info_msg.width = self.nn_shape_w  # Set the width of the camera image

        # Set the camera intrinsic matrix (fx, fy, cx, cy)
        camera_info_msg.K = [615.381, 0.0, 320.0, 0.0, 615.381, 240.0, 0.0, 0.0, 1.0]
        # Set the distortion parameters (k1, k2, p1, p2, k3)
        camera_info_msg.D = [-0.10818, 0.12793, 0.00000, 0.00000, -0.04204]
        # Set the rectification matrix (identity matrix)
        camera_info_msg.R = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        # Set the projection matrix (P)
        camera_info_msg.P = [615.381, 0.0, 320.0, 0.0, 0.0, 615.381, 240.0, 0.0, 0.0, 0.0, 1.0, 0.0]
        # Set the distortion model
        camera_info_msg.distortion_model = "plumb_bob"
        # Set the timestamp
        camera_info_msg.header.stamp = rospy.Time.now()

        self.pub_cam_inf.publish(camera_info_msg)  # Publish the camera info message

    def rgb_camera(self):
        cam_rgb = self.pipeline.createColorCamera()
        cam_rgb.setPreviewSize(self.res[0], self.res[1])
        cam_rgb.setInterleaved(False)
        cam_rgb.setFps(self.fps)

        # Def xout / xin
        ctrl_in = self.pipeline.createXLinkIn()
        ctrl_in.setStreamName("cam_ctrl")
        ctrl_in.out.link(cam_rgb.inputControl)

        xout_rgb = self.pipeline.createXLinkOut()
        xout_rgb.setStreamName("video")

        cam_rgb.preview.link(xout_rgb.input)

    def run(self):
        #self.rgb_camera()
        ############################### Run Model ###############################
        # Pipeline defined, now the device is assigned and pipeline is started
        pipeline = None
        # Get argument first
        # Model parameters
        modelPathName = f'{modelsPath}/{modelName}/{modelName}.blob'
        print(metadata)
        nnPath = str((Path(__file__).parent / Path(modelPathName)).resolve().absolute())
        print(nnPath)

        pipeline = self.createPipeline(nnPath)

        with dai.Device() as device:
            cams = device.getConnectedCameras()
            depth_enabled = dai.CameraBoardSocket.LEFT in cams and dai.CameraBoardSocket.RIGHT in cams
            if cam_source != "rgb" and not depth_enabled:
                raise RuntimeError("Unable to run the experiment on {} camera! Available cameras: {}".format(cam_source, cams))
            device.startPipeline(pipeline)

            # Output queues will be used to get the rgb frames and nn data from the outputs defined above
            q_nn_input = device.getOutputQueue(name="nn_input", maxSize=4, blocking=False)
            q_nn = device.getOutputQueue(name="nn", maxSize=4, blocking=False)

            frame = None
            detections = []
            start_time = time.time()
            counter = 0
            fps = 0
            
            olor2 = (255, 255, 255)
            layer_info_printed = False
            dims = None

            ############################################################################################################
            #                                           DETECTION VARIABLES                                            #
            averaging_length = 25       # the length of the averaging array. longer will be slower but more accurate
            detection_threshold = 0.7   # the minimum confidence needed from depthAI to add a detection to the array
            message_threshold = 0.5     # the minimum average value before sending message to ROS

            depthAI_backpack_target_ID = 2
            depthAI_head_target_ID = 3
            depthAI_shirt_target_ID = 5
            depthAI_arm_target_ID = 0
            depthAI_pants_target_ID = 4

            ROS_backpack_target_ID = 1.0
            ROS_human_target_ID = 2.0
            ############################################################################################################ 

            backpack_detections = RunningAverage(averaging_length)
            human_detections = RunningAverage(averaging_length)

            backpack_x = RunningAverage(10)
            backpack_y = RunningAverage(10)
            human_x = RunningAverage(10)
            human_y = RunningAverage(10)

            sent_backpack = False
            sent_human = False

            while True:
                found_classes = []
                # instead of get (blocking) used tryGet (nonblocking) which will return the available data or None otherwise
                inRgb = q_nn_input.get()
                inDet = q_nn.get()

                if inRgb is not None:
                    frame = inRgb.getCvFrame()
                else:
                    print("Cam Image empty, trying again...")
                    continue

                # create temporary values to add to array
                backpack_temp = 0
                human_temp = 0
                
                if inDet is not None:
                    detections = inDet.detections
                    #print(detections)
                    for detection in detections:
                        #print(detection)
                        #print("{},{},{},{},{},{},{}".format(detection.label,labels[detection.label],detection.confidence,detection.xmin, detection.ymin, detection.xmax, detection.ymax))
                        if detection.confidence > detection_threshold:
                            # add to found classes
                            found_classes.append(detection.label)
                            # populate temp arrays and location averaging
                            # backpack
                            if detection.label == depthAI_backpack_target_ID:
                                backpack_temp = backpack_temp + 1
                                (x,y) = self.find_centre(detection.xmin, detection.ymin, detection.xmax, detection.ymax)
                                backpack_x.add_sample(x)
                                backpack_y.add_sample(y)
                            # human. the more elements of the human detected, the faster the array fills
                            if detection.label == depthAI_head_target_ID:
                                human_temp = human_temp + 1
                                (x,y) = self.find_centre(detection.xmin, detection.ymin, detection.xmax, detection.ymax)
                                human_x.add_sample(x)
                                human_y.add_sample(y)
                            if detection.label == depthAI_shirt_target_ID:
                                human_temp = human_temp + 1
                                (x,y) = self.find_centre(detection.xmin, detection.ymin, detection.xmax, detection.ymax)
                                human_x.add_sample(x)
                                human_y.add_sample(y)
                            if detection.label == depthAI_arm_target_ID:
                                human_temp = human_temp + 1
                                (x,y) = self.find_centre(detection.xmin, detection.ymin, detection.xmax, detection.ymax)
                                human_x.add_sample(x)
                                human_y.add_sample(y)
                            if detection.label == depthAI_pants_target_ID:
                                human_temp = human_temp + 1
                                (x,y) = self.find_centre(detection.xmin, detection.ymin, detection.xmax, detection.ymax)
                                human_x.add_sample(x)
                                human_y.add_sample(y)
                        # print(dai.ImgDetection.getData(detection))

                    found_classes = np.unique(found_classes)
                    # print(found_classes)
                    overlay = self.show_yolo(frame, detections)
                else:
                    print("Detection empty, trying again...")
                    continue

                # append temporary values to detection arrays
                backpack_detections.add_sample(backpack_temp)
                human_detections.add_sample(human_temp)

                # calculate array averages
                backpack_detections_avg = backpack_detections.get_average()
                human_detections_avg = human_detections.get_average()

                # send ROS messages
                if (backpack_detections_avg > message_threshold):
                    if sent_backpack == False:
                        (targX, targY) = self.world_transform(backpack_x.get_average(), backpack_y.get_average())
                        self.publish_detection_msg(targX, targY, ROS_backpack_target_ID, 1000.0)
                        sent_backpack = True
                if (human_detections_avg > message_threshold):
                    if sent_human == False:
                        (targX, targY) = self.world_transform(human_x.get_average(), human_y.get_average())
                        self.publish_detection_msg(targX, targY, ROS_human_target_ID, 1000.0)
                        sent_human = True

                if frame is not None:
                    cv2.putText(overlay, "NN fps: {:.2f}".format(fps), (2, overlay.shape[0] - 4), cv2.FONT_HERSHEY_TRIPLEX, 0.4, (255, 0, 255))
                    cv2.putText(overlay, "Found classes {}".format(found_classes), (2, 10), cv2.FONT_HERSHEY_TRIPLEX, 0.4, (255, 0, 255))
                    # cv2.imshow("nn_output_yolo", overlay)
                    self.publish_to_ros(frame)
                    self.publish_detect_to_ros(overlay)
                    self.publish_camera_info()

                ## Function to compute FPS
                counter+=1
                if (time.time() - start_time) > 1 :
                    fps = counter / (time.time() - start_time)

                    counter = 0
                    start_time = time.time()


            # with dai.Device(self.pipeline) as device:
            #     video = device.getOutputQueue(name="video", maxSize=1, blocking=False)

            #     while True:
            #         frame = video.get().getCvFrame()

            #         self.publish_to_ros(frame)
            #         self.publish_camera_info()

    # calculates the centre of a bounding box
    def find_centre(self, x_min, y_min, x_max, y_max):
        return ((x_min + x_max) / 2, (y_min + y_max) / 2)

    def remap(self, val, in_min, in_max, out_min, out_max):
        return (val - in_min) * (out_max - out_min) / (in_max - in_min) + out_min

    # does the world transform from camera -> drone -> world frame
    def world_transform(self,camX,camY):
        # debugging values
        #q0 = 0
        #q1 = 0
        #q2 = 0
        #q3 = 1
        #siny_cosp = 2 * (q3 * q2 + q0 * q1)
        #cosy_cosp = 1 - 2 * (q1 * q1 + q2 * q2)

        # drone position
        realWorldX = self.current_location.position.x
        realWorldY = self.current_location.position.y
        realWorldZ = self.current_location.position.z
        # camera field of view
        hfov = 78.0
        vfov = 78.0
        # distance on floor given cameras height
        x_floor_max = realWorldZ * math.tan(math.radians(vfov/2))
        y_floor_max = realWorldZ * math.tan(math.radians(hfov/2))
        # camera to drone frame transformation
        droneX = self.remap(camX, 0, 1, -x_floor_max, x_floor_max)
        droneY = self.remap(camY, 0, 1, y_floor_max, -y_floor_max)
        # perform quarterion to euler angle transformation for z axis
        siny_cosp = 2 * (self.current_location.orientation.w * self.current_location.orientation.z + self.current_location.orientation.x * self.current_location.orientation.y)
        cosy_cosp = 1 - 2 * (self.current_location.orientation.y * self.current_location.orientation.y + self.current_location.orientation.z * self.current_location.orientation.z)
        # calculate Z axis yaw angle
        yaw = -math.atan2(siny_cosp, cosy_cosp)
        # rotate target position according to z axis yaw
        tempX = droneX * math.cos(yaw) + droneY * math.sin(yaw)
        tempY = - droneX * math.sin(yaw) + droneY * math.cos(yaw)
        # rotate 90 degrees for weird X / Y axis configuration in flight area
        worldX = tempX * math.cos(math.radians(90)) + tempY * math.sin(math.radians(90))
        worldY = - tempX * math.sin(math.radians(90)) + tempY * math.cos(math.radians(90))
        # find the new target position
        targetX = realWorldX + worldX
        targetY = realWorldY + worldY
        # print values
        if(0):
            print("CAM X: " + str(camX) + " CAM Y: " + str(camY))
            print("DRONE X: " + str(droneX) + " DRONE Y:" + str(droneY))
            print("yaw: " + str(yaw) + " (" + str(math.degrees(yaw))+  ")")
            print("floor_max_x: " + str(x_floor_max))
            print("rotated X: " + str(worldX) + " rotated Y: " + str(worldY))
            print("drone world pos X: " + str(realWorldX) + " drone world pos Y: " + str(realWorldY))
            print("target X: " + str(targetX) + " target Y: " + str(targetY))

        return (targetX, targetY)



    def publish_detection_msg(self, X, Y, ID, aruco_number):
        msg_out = target_msg()
        msg_out.x = X
        msg_out.y = Y
        msg_out.id = ID
        msg_out.aruco_number = aruco_number
        self.pub_target_detect.publish(msg_out)
        target_str = ""
        if ID == 1.0:
            target_str = "'backpack'"
        if ID == 2.0:
            target_str = "'human'"
        if ID == 3.0:
            target_str = "'aruco'"
        print("Publishing target " + target_str + " at " + str(X) + ", " + str(Y))

    def publish_to_ros(self, frame):
        msg_out = CompressedImage()
        msg_out.header.stamp = rospy.Time.now()
        msg_out.format = "jpeg"
        msg_out.header.frame_id = "home"
        msg_out.data = np.array(cv2.imencode('.jpg', frame)[1]).tostring()
        self.pub_image.publish(msg_out)
        # Publish image raw
        #msg_img_raw = self.br.cv2_to_imgmsg(frame, encoding="bgr8")
        #self.pub_image_raw.publish(msg_img_raw)

    def publish_detect_to_ros(self, frame):
        msg_out = CompressedImage()
        msg_out.header.stamp = rospy.Time.now()
        msg_out.format = "jpeg"
        msg_out.header.frame_id = "home"
        msg_out.data = np.array(cv2.imencode('.jpg', frame)[1]).tostring()
        self.pub_image_detect.publish(msg_out)
        
    ############################### ############################### Functions ###############################
    ######### Functions for Yolo Decoding
    # nn data, being the bounding box locations, are in <0..1> range - they need to be normalized with frame width/height
    def frameNorm(self, frame, bbox):
        normVals = np.full(len(bbox), frame.shape[0])
        normVals[::2] = frame.shape[1]
        return (np.clip(np.array(bbox), 0, 1) * normVals).astype(int)

    def show_yolo(self, frame, detections):
        color = (255, 0, 0)
        # Both YoloDetectionNetwork and MobileNetDetectionNetwork output this message. This message contains a list of detections, which contains label, confidence, and the bounding box information (xmin, ymin, xmax, ymax).
        overlay =  frame.copy()
        for detection in detections:
            bbox = self.frameNorm(overlay, (detection.xmin, detection.ymin, detection.xmax, detection.ymax))
            cv2.putText(overlay, labels[detection.label], (bbox[0] + 10, bbox[1] + 20), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
            cv2.putText(overlay, f"{int(detection.confidence * 100)}%", (bbox[0] + 10, bbox[1] + 40), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
            cv2.rectangle(overlay, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
            #cv2.circle(overlay, ((bbox[0] + bbox[2]) / 2, (bbox[1]+ bbox[3]) / 2), 10, color, 2)
        return overlay

    # Start defining a pipeline
    def createPipeline(self, nnPath):

        pipeline = dai.Pipeline()

        # pipeline.setOpenVINOVersion(version = dai.OpenVINO.VERSION_2021_4)
        # pipeline.setOpenVINOVersion(version=dai.OpenVINO.Version.VERSION_2021_4)
        pipeline.setOpenVINOVersion(version=dai.OpenVINO.Version.VERSION_2022_1)

        # Define a neural network that will make predictions based on the source frames
        detection_nn = pipeline.create(dai.node.YoloDetectionNetwork)
        # Network specific settings
        detection_nn.setConfidenceThreshold(confidenceThreshold)
        detection_nn.setNumClasses(classes)
        detection_nn.setCoordinateSize(coordinates)
        detection_nn.setAnchors(anchors)
        detection_nn.setAnchorMasks(anchorMasks)
        detection_nn.setIouThreshold(iouThreshold)
        # generic nn configs
        detection_nn.setBlobPath(nnPath)
        detection_nn.setNumPoolFrames(4)
        detection_nn.input.setBlocking(False)
        detection_nn.setNumInferenceThreads(2)

        # Define a source - color camera
        if cam_source == 'rgb':
            cam = pipeline.create(dai.node.ColorCamera)
            cam.setPreviewSize(self.nn_shape_w,self.nn_shape_h)
            cam.setInterleaved(False)
            cam.preview.link(detection_nn.input)
            cam.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
            cam.setFps(10)
            print("Using RGB camera...")
        elif cam_source == 'left':
            cam = pipeline.create(dai.node.MonoCamera)
            cam.setBoardSocket(dai.CameraBoardSocket.LEFT)
            print("Using BW Left cam")
        elif cam_source == 'right':
            cam = pipeline.create(dai.node.MonoCamera)
            cam.setBoardSocket(dai.CameraBoardSocket.RIGHT)
            print("Using BW Rigth cam")

        if cam_source != 'rgb':
            manip = pipeline.create(dai.node.ImageManip)
            manip.setResize(self.nn_shape_w,self.nn_shape_h)
            manip.setKeepAspectRatio(True)
            # manip.setFrameType(dai.RawImgFrame.Type.BGR888p)
            manip.setFrameType(dai.RawImgFrame.Type.RGB888p)
            cam.out.link(manip.inputImage)
            manip.out.link(detection_nn.input)

        # Create outputs
        xout_rgb = pipeline.create(dai.node.XLinkOut)
        xout_rgb.setStreamName("nn_input")
        xout_rgb.input.setBlocking(False)

        detection_nn.passthrough.link(xout_rgb.input)

        xinDet = pipeline.create(dai.node.XLinkOut)
        xinDet.setStreamName("nn")
        xinDet.input.setBlocking(False)

        detection_nn.out.link(xinDet.input)

        return pipeline


    def shutdown(self):
        cv2.destroyAllWindows()

    def callback_pose(self, msg_in):
        # Store the current position at all times so it can be accessed later
        #rospy.loginfo("Updating Pose UAV...")
        self.current_location = msg_in.pose


#### Main code that creates a depthaiCamera class and run it.
def main():
    rospy.init_node('depthai_node')
    dai_cam = DepthaiCamera()

    while not rospy.is_shutdown():
        dai_cam.run()

    dai_cam.shutdown()
