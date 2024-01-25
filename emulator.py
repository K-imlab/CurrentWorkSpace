
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import random
from functools import partial
import time

class RandomHexPublisher(Node):
    def __init__(self):
        super().__init__('machineEmulator')
        self.publisher = self.create_publisher(String, 'canTXToMgr', 10)
        
        target_ids = ['0CF00400', '18FEE500', '19FF9E21', '19FF9A21']
        periods = [0.02, 1, 0.1, 0.1]
        self.st_time = time.time()
        self.declared_timer = []
        for can_id, period in zip(target_ids, periods):
            timer = self.create_timer(period, partial(self.timer_callback, additional_arg=can_id))
            self.declared_timer.append(timer)

    def timer_callback(self, additional_arg):
        header, payload = self.generate_random_hex()
        self.publish_random_hex(header, payload , additional_arg)

    def generate_random_hex(self):
        t = time.time() - self.st_time
        header = f"m 2 ced "
        payload = ""
        for i in range(8):
            random_value = random.randint(0, 2**8 - 1)
            hex_value = format(random_value, '02X')  # 16진수로 변환
            payload += f" {hex_value}"
        payload += "\n"
        return header, payload

    def publish_random_hex(self, header, payload, can_id):
        msg = String()
        msg.data = header + can_id + payload
        self.publisher.publish(msg)
        '''data: 'm 2 ced 19FF9A21 10 96 DD 33 C0 CF C2 7E

        '
        ---'''
        self.get_logger().info(f'{msg.data}')

def main(args=None):
    rclpy.init(args=args)
    node = RandomHexPublisher()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
