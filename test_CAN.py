import os
import sys
import can
import cantools
import logging

logger = logging.getLogger()
logger.setLevel(logging.INFO)
log_handler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
log_handler.setFormatter(formatter)
logger.addHandler(log_handler)

def info(message: str):
    logger.info(message)

can_interface = 'can1'
filters = [
    {"can_id": 0x18FEEE3F, "can_mask": 0x1FFFFFFF, "extended": True },
    {"can_id": 0x18FF313F, "can_mask": 0x1FFFFFFF, "extended": True },
    {"can_id": 0x1CFA673F, "can_mask": 0x1FFFFFFF, "extended": True },
    {"can_id": 0x1CFD083F, "can_mask": 0x1FFFFFFF, "extended": True },
    {"can_id": 0x18FEEEAE, "can_mask": 0x1FFFFFFF, "extended": True },
    {"can_id": 0x18FF31AE, "can_mask": 0x1FFFFFFF, "extended": True },
    {"can_id": 0x1CFA67AE, "can_mask": 0x1FFFFFFF, "extended": True },
    {"can_id": 0x1CFD08AE, "can_mask": 0x1FFFFFFF, "extended": True }
]

db = cantools.database.load_file('/usr/local/bin/VSS_J1939.dbc')
bus = can.interface.Bus(channel=can_interface, bustype='socketcan', can_filters=filters)


if __name__ == '__main__':
    while True:
        try:
            ops = bus.recv()
            decode_messages = db.decode_message(ops.arbitration_id, ops.data)
            print(decode_messages)
        except Exception as e :
            continue
