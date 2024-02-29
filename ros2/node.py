import sys

import rclpy
from rclpy.node import Node


class NodeBase(Node):

    @classmethod
    def main(cls):
        rclpy.init(args=sys.argv)
        node = cls()
        node.info(f"Successful initialization.")
        rclpy.spin(node)
        rclpy.shutdown()

    def __init__(self, name="demo"):
        super().__init__(name)
        # log methods
        self.info = self.get_logger().info
        self.warn = self.get_logger().warn
        self.error = self.get_logger().error
        self.debug = self.get_logger().debug
