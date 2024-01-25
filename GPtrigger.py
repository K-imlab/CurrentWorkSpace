import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import random
from functools import partial
import time


triggers = {'ResetCMD_EO': ['18EF4A28', 516281, 4, 1, 1, '18E8FF4A'], 
            'ResetCMD_HO': ['18EF4A28', 516282, 4, 2, 1, '18E8FF4A'],
            'replOPS_EO' : ['18EF4A28', 517482, 4, 3, 2, '18E8FF4A'], 
            'replOPS_HO' : ['18EF4A28', 517481, 4, 5, 2, '18E8FF4A'],
            'changeVG_HO': ['18EF4A28', 517479, 5, 1, 4, '18E8FF4A'],
            }
keys = list(triggers.keys())

def byte_pack(SPN, FMI, OC, CM):
    byte1 =    (SPN & 0xFF) >> (0*8) 
    byte2 =  (SPN & 0xFF00) >> (1*8) 
    msb_SPN = (SPN & 0xFF0000) >> (2*8)
    byte3 = (FMI << 3) + msb_SPN
    byte4 = (CM << 7) + OC
    return f" {byte1:02X} {byte2:02X} {byte3:02X} {byte4:02X} FF FF FF FF"

def main(args=None):
    rclpy.init(args=args)
    node = rclpy.create_node('GP')
    publisher = node.create_publisher(String, 'canTXToMgr', 10)
    while True:
        try:
            print("\n","="*100)
            print('select trigger :')
            for i, key in enumerate(keys):
                print('\t', i, key)
            num = input('type number :')
            name = keys[int(num)]
            if name == 'changeVG_HO':
                print("select VG: ")
                print("\t0. BACK\n\t1. VG32\n\t2. VG46\n\t3. VG68")
                value = input("select: ")
                value = int(value)
                assert(value < 4)
                if value == 0:
                    continue
                else:
                    can_id = triggers[name][0]
                    SPN = triggers[name][1]
                    FMI = value-1
                    ACK = triggers[name][-1]
            
            elif (name == 'ResetCMD_EO') or (name == 'ResetCMD_HO') or (name == 'replOPS_EO') or (name == 'replOPS_HO'):
                header = 'm 2 ced '
                can_id = triggers[name][0]
                SPN = triggers[name][1]
                ACK = triggers[name][-1]
                FMI = 1
            
            payload = byte_pack(SPN, FMI, 0, 0)
            msg_data = header + can_id + payload
            print(msg_data)

            msg = String()
            msg.data = msg_data
            publisher.publish(msg)

        except Exception as e:
            print(e, "please select again")    


if __name__ == '__main__':
    main()