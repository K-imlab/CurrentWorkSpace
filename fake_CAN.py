import mat73
import glob
import numpy as np
import pandas as pd
import cantools
import can
import time
import threading
from canlib import canlib
from canlib import Frame
import struct


def open_channel(channel):
    ch = canlib.openChannel(channel, canlib.canOPEN_ACCEPT_VIRTUAL)
    ch.setBusOutputControl(canlib.canDRIVER_NORMAL)
    ch.setBusParams(canlib.canBITRATE_1M)
    ch.busOn()
    return ch


def close_channel(ch):
    ch.busOff()
    ch.close()


def send_(ch0, id, bytes, dlc):
    # send a CAN frame without using kvadblib
    frame = Frame(id_=id, data=bytes, dlc=dlc, flags=canlib.MessageFlag.EXT)
    ch0.write(frame)
    # ch0.write(id_=id, msg=bytes, flag=1, dlc=8)


class FakeSender:
    def __init__(self, ):
        self.signals = None
        path = "D:/Project/OilPhm/samples/*.mat"
        self.mat_paths = glob.glob(path)
        self.hydraulic_cols = ['ECU_EEC1_EngineSpeed', 'ECU_EEC1_ActualEnginePercentTorque',
                          'ECU_HOURS_EngineTotalHoursOfOperation', 'PGN64776_3F_Diel_const', 'PGN64776_3F_Dyn_Visco',
                          'PGN64776_3F_Density', 'PGN65262_3F_Oil_Temp', 'PGN65329_3F_Status']
        self.start_time = 10e7
        self.ch0 = open_channel(0)
        # self.ch1 = open_channel(1)
        self._load_data()
        self.can_ids = {0x1CFD083F: ['PGN64776_3F_Diel_const', 'PGN64776_3F_Dyn_Visco', 'PGN64776_3F_Density'],
                        0x18FEEE3F: ['PGN65262_3F_Oil_Temp'],
                        0x18FF313F: ['PGN65329_3F_Status'],
                        0x0CF00400: ['ECU_EEC1_EngineSpeed'],
                        0x18FEE500: ['ECU_HOURS_EngineTotalHoursOfOperation']}

    def _load_data(self):
        self.signals = {key: pd.Series(dtype=np.float16) for key in self.hydraulic_cols}
        for p in self.mat_paths:
            m = mat73.loadmat(p)

            for col in self.hydraulic_cols:
                ts_tmp = pd.Series(m[col][:, 1], index=m[col][:, 0])
                self.signals[col] = pd.concat([self.signals[col], ts_tmp])

        for col in self.hydraulic_cols:
            self.signals[col].sort_index(inplace=True)
            if self.start_time > self.signals[col].index[0]:
                self.start_time = self.signals[col].index[0]

        for col in self.hydraulic_cols:
            self.signals[col].index = (self.signals[col].index - self.start_time) * 24 * 60 * 60
        print("")
        return

    def run(self, can_id, speed):
        s_time = time.time()
        first = True
        for col in self.can_ids[can_id]:
            if first:
                message = pd.DataFrame(self.signals[col])
                first = False
            else:
                message = pd.concat([message, self.signals[col]], axis=1)
        for i, data in enumerate(message.values):
            while True:
                current = (time.time()-s_time) * speed
                if current >= message.index[i]:
                    byte_data = np.array(data, dtype=np.float16).tobytes()
                    dlc = len(byte_data)
                    send_(self.ch0, can_id, byte_data, dlc)
                    break
        return

    def send_on_demand(self):
        event_can_ids = {0x18EF4A28: ['OilChange', 'SensorChange'],
                         0x18EFFFFF: ['HydraulicVG', 'EngineVG'],
                         0x18FFB334: ['KeyOff']}
        while True:
            print(event_can_ids)
            name = input("trigger : ")
            if name == "HydraulicVG":
                print("select VG:")
                print("\t1. VG32\n\t2. VG46\n\t3. VG68")
                value = input("select: ")
                value = int(value)
            else:
                value = 1

            data_field = []
            id_ = 0
            triggered = False
            for key, trigger_types in zip(event_can_ids.keys(), event_can_ids.values()):
                for trigger_type in trigger_types:
                    if trigger_type == name:
                        id_ = key
                        data_field.append(value)
                        triggered = True
                    else:
                        data_field.append(0)
                if triggered:
                    triggers = data_field
                    v = np.array(triggers, dtype=np.float16).tobytes()
                    dlc = len(v)
                    print(id_, triggers)
                    send_(self.ch0, id_, v, dlc)
                    triggered = False
                data_field = []


if __name__ == "__main__":
    sender = FakeSender()

    thread = threading.Thread(target=sender.send_on_demand, args=())
    threads = [thread]
    thread.start()
    for can_id in sender.can_ids.keys():
        thread = threading.Thread(target=sender.run, args=(can_id, 50))
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()
