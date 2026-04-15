import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import os

from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy


class TestVideoPublisher(Node):
    def __init__(self):
        super().__init__('test_video_publisher')

        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )
        

        self.pub = self.create_publisher(Image, '/camera/image_raw', sensor_qos)
        # self.pub = self.create_publisher(Image, '/camera/image_raw', 10)
        self.bridge = CvBridge()

        video_path = './sample.mp4'
        if not os.path.exists(video_path):
            self.get_logger().error(f"Video not found: {video_path}")
            raise FileNotFoundError(video_path)

        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise RuntimeError("Failed to open video file")

        # ---- Get real FPS ----
        fps = self.cap.get(cv2.CAP_PROP_FPS)

        # Safety check (OpenCV sometimes returns 0 or invalid values)
        if fps is None or fps <= 0 or fps > 120:
            self.get_logger().warn(f"Invalid FPS ({fps}), defaulting to 30")
            fps = 30.0

        self.fps = fps
        self.frame_interval = 1.0 / fps

        self.get_logger().info(f"Video FPS: {self.fps}")

        # Timer runs fast; we control FPS manually
        self.timer = self.create_timer(0.001, self.publish_frame)

        self.last_time = self.get_clock().now()

    def publish_frame(self):
        now = self.get_clock().now()
        elapsed = (now - self.last_time).nanoseconds * 1e-9

        # enforce real FPS timing
        if elapsed < self.frame_interval:
            return

        self.last_time = now

        ret, frame = self.cap.read()

        # loop video when finished
        if not ret:
            self.get_logger().info("Restarting video")
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            return

        msg = self.bridge.cv2_to_imgmsg(frame, encoding='bgr8')
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'camera_frame'

        self.pub.publish(msg)
        self.get_logger().info('Published frame')


def main(args=None):
    rclpy.init(args=args)
    node = TestVideoPublisher()
    rclpy.spin(node)
    node.cap.release()
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()