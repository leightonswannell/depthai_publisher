#!/usr/bin/env python3

import cv2

import rospy
from sensor_msgs.msg import CompressedImage
from cv_bridge import CvBridge, CvBridgeError
from depthai_publisher.msg import target_msg
import numpy as np
from geometry_msgs.msg import Pose, PoseStamped
import math

class ArucoDetector():
    aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_5X5_100)
    aruco_params = cv2.aruco.DetectorParameters_create()

    frame_sub_topic = '/depthai_node/image/compressed'
    pub_topic_detections = '/depthai_node/target_detections'

    found_arucos = [1000.0]

    def __init__(self):
        self.time_finished_processing = rospy.Time(0)
        self.aruco_pub = rospy.Publisher(
            '/processed_aruco/image/compressed', CompressedImage, queue_size=10)
        self.sub_pose = rospy.Subscriber("/mavros/local_position/pose", PoseStamped, self.callback_pose)
        self.current_location = Pose()
        self.br = CvBridge()
        self.pub_target_detect = rospy.Publisher(self.pub_topic_detections, target_msg, queue_size=10)
        if not rospy.is_shutdown():
            self.frame_sub = rospy.Subscriber(
                self.frame_sub_topic, CompressedImage, self.img_callback)

    def img_callback(self, msg_in):
        if msg_in.header.stamp > self.time_finished_processing:
            try:
                frame = self.br.compressed_imgmsg_to_cv2(msg_in)
            except CvBridgeError as e:
                rospy.logerr(e)
            

            aruco = self.find_aruco(frame)
            self.publish_to_ros(aruco)
            self.time_finished_processing = rospy.Time.now()

        # cv2.imshow('aruco', aruco)
        # cv2.waitKey(1)

    def find_aruco(self, frame):
        (corners, ids, _) = cv2.aruco.detectMarkers(
            frame, self.aruco_dict, parameters=self.aruco_params)

        if len(corners) > 0:
            ids = ids.flatten()

            for (marker_corner, marker_ID) in zip(corners, ids):
                corners = marker_corner.reshape((4, 2))
                (top_left, top_right, bottom_right, bottom_left) = corners

                top_right = (int(top_right[0]), int(top_right[1]))
                bottom_right = (int(bottom_right[0]), int(bottom_right[1]))
                bottom_left = (int(bottom_left[0]), int(bottom_left[1]))
                top_left = (int(top_left[0]), int(top_left[1]))

                cv2.line(frame, top_left, top_right, (0, 255, 0), 2)
                cv2.line(frame, top_right, bottom_right, (0, 255, 0), 2)
                cv2.line(frame, bottom_right, bottom_left, (0, 255, 0), 2)
                cv2.line(frame, bottom_left, top_left, (0, 255, 0), 2)

                #rospy.loginfo("Aruco detected, ID: {}".format(marker_ID))
            
                cv2.putText(frame, str(
                    marker_ID), (top_left[0], top_right[1] - 15), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 0), 2)
            
                fid = float(marker_ID)
                if fid in self.found_arucos:
                    # skip
                    continue
                else:
                    # add aruco ID to list already published
                    self.found_arucos.append(fid)
                    # find the center of the aruco using bounding boxes
                    (boxX,boxY) = self.find_aruco_centre(bottom_left,bottom_right,top_left,top_right)
                    # perform world transform to find location of aruco
                    (targX, targY) = self.world_transform(boxX, boxY)
                    # publish message to ROS network
                    self.publish_detection_msg(targX, targY, 3.0, fid)
        return frame

    def publish_to_ros(self, frame):
        msg_out = CompressedImage()
        msg_out.header.stamp = rospy.Time.now()
        msg_out.format = "jpeg"
        msg_out.data = np.array(cv2.imencode('.jpg', frame)[1]).tostring()

        self.aruco_pub.publish(msg_out)

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
            target_str = "'aruco' (" + str(aruco_number) + ")"
        print("Publishing target " + target_str + " at " + str(X) + ", " + str(Y))
    
    def callback_pose(self, msg_in):
        # Store the current position at all times so it can be accessed later
        #rospy.loginfo("Updating Pose UAV...")
        self.current_location = msg_in.pose

    # calculates the centre of a bounding box
    def find_aruco_centre(self, bottom_left, bottom_right, top_left, top_right):
        x = (bottom_left[0] + bottom_right[0] + top_left[0] + top_right[0]) / 4
        y = (bottom_left[1] + bottom_right[1] + top_left[1] + top_right[1]) / 4
        x = x / 416.0
        y = y / 416.0
        return (x , y)

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


def main():
    rospy.init_node('EGB349_vision', anonymous=True)
    rospy.loginfo("Processing images...")

    aruco_detect = ArucoDetector()

    rospy.spin()
