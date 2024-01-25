import queue
import threading
import time
import sys

from event_handlers import *
from funcs import *
import time
st_time = time.time()

can0_event = [
]
can0_Eg = [
    {"can_id": 0x0CF00400, "can_mask": 0x1FFFFFFF, "extended": True },
    {"can_id": 0x18FEE500, "can_mask": 0x1FFFFFFF, "extended": True },
    {"can_id": 0x18FFB334, "can_mask": 0x1FFFFFFF, "extended": True},
]
can1_event = [
    {"can_id": 0x18EF4A28, "can_mask": 0x1FFFFFFF, "extended": True},
    {"can_id": 0x18EFFFFF, "can_mask": 0x1FFFFFFF, "extended": True},
]
can1_3F = [
    {"can_id": 0x18FEEE3F, "can_mask": 0x1FFFFFFF, "extended": True },
    {"can_id": 0x18FF313F, "can_mask": 0x1FFFFFFF, "extended": True },
    {"can_id": 0x1CFA673F, "can_mask": 0x1FFFFFFF, "extended": True },
    {"can_id": 0x1CFD083F, "can_mask": 0x1FFFFFFF, "extended": True }
]
can1_AE = [
    {"can_id": 0x18FEEEAE, "can_mask": 0x1FFFFFFF, "extended": True },
    {"can_id": 0x18FF31AE, "can_mask": 0x1FFFFFFF, "extended": True },
    {"can_id": 0x1CFA67AE, "can_mask": 0x1FFFFFFF, "extended": True },
    {"can_id": 0x1CFD08AE, "can_mask": 0x1FFFFFFF, "extended": True }
]


class VolatileQueue:
    def __init__(self):
        self.queue = None
        self.lock = threading.Lock()
    
    def put(self, item):
        with self.lock:
            self.queue = item

    def get(self):
        with self.lock:
            out = self.queue
            self.queue = None
            return out


class EventConsumer:
    """ Event 가 발생하면 이벤트 항목을 Queue에 저장해 놓고, event에 해당하는 Handler를 호출하는 역할"""
    def __init__(self, event_queue: queue.Queue):
        self.event_queue = event_queue
        self.handlers = {}

    def add_event_handler(self, event_type, handler):
        if event_type not in self.handlers:
            self.handlers[event_type] = []
        self.handlers[event_type].append(handler)

    def remove_event_handler(self, event_type, handler):
        if event_type in self.handlers and handler in self.handlers[event_type]:
            self.handlers[event_type].remove(handler)

    def run(self):
        while True:
            try:
                event = self.event_queue.get()
                if event is not None:
                    event_type = event['type']
                    data = event['data']

                    if event_type in self.handlers:
                        for handler in self.handlers[event_type]:
                            out = handler(data)
            except Exception as e:
                print("in event_consumer", e)
                pass


class MonitoringSystem:
    def __init__(self, rq, event_que, trigger_que, name):
        self.name = name
        self.rq = rq
        self.event_que = event_que
        self.trigger_que = trigger_que
        self.buffer = Buffer()
        self.sensor = Sensor()

        self.sender = Sender()

        self.monitor_dielec = RealTimeMonitoring(percent_margin=2, method=DIELECTRIC_METHOD, VG=VG46)
        self.monitor_visco = RealTimeMonitoring(percent_margin=20, method=VISCOSITY_METHOD, VG=VG46)

        self.oil_change_count = 0
        self.sensor_repl_count = 0
        self.trigger_q = []
        self.viscosity_grade = VG46  # VG46
        self.use_reference_formula = False
        self.prev_warning_level = 0
    
    def emit_event(self, event_type, data=None):
        event = {'type': event_type, 'data': data}
        self.event_que.put(event)
    
    def _check_activation(self, data) -> bool:
        # activation_mode = (data['engine_speed'] > 1500) & (data['StMsgCode'] < 1) & (data['OilAvrgTmp'] > 40) & (data['kine_viscosity'] < 50)
        activation_mode = True
        return activation_mode
    
    def long_term_events(self):
        self.trigger_q = list(set(self.trigger_q))
        if TRIG_OIL_CHANGED in self.trigger_q and TRIG_SENSOR_REPLACED in self.trigger_q:
            self.trigger_q.remove(TRIG_SENSOR_REPLACED)
        self.emit_event('buffer_is_full', {'data': self.buffer, 'formula': (
            self.monitor_dielec.coefficient, self.monitor_visco.coefficient)})
        for trigger in self.trigger_q:
            if trigger == TRIG_NONE:
                pass
            elif trigger == TRIG_OIL_CHANGED:
                result, done = offset_reset(self.buffer, self.oil_change_count, self.monitor_dielec.coefficient, self.monitor_visco.coefficient)
                print("\nin long term events oil changed : ", self.oil_change_count, self.trigger_q)
                self.oil_change_count += 1
                if self.oil_change_count >= 8 or done == 1:
                    # update formula
                    # FIXME Address D에서 5회의 값을 읽어와서 수식 업데이트
                    # self.monitor_dielec.update(new_coefficient)
                    # self.monitor_visco.update(new_coefficient)
                    print('\n update formula')

                    self.trigger_q.remove(TRIG_OIL_CHANGED)
                    self.oil_change_count = 0
                    self.use_reference_formula = False
                    if FIGURE:
                        fig.show()

            elif trigger == TRIG_SENSOR_REPLACED:
                result, done = offset_reset(self.buffer, self.sensor_repl_count)
                print("\nin long term events sensor replaced : ", self.sensor_repl_count, self.trigger_q)
                self.sensor_repl_count += 1
                if self.sensor_repl_count >= 8 or done == 1:
                    # FIXME Address D에서 5회의 값을 읽어와서 수식 업데이트 Address C 에서 기존 값과 비교
                    # self.monitor_dielec.update(new_coefficient)
                    # self.monitor_visco.update(new_coefficient)
                    print('\n update formula')
                    self.trigger_q.remove(TRIG_SENSOR_REPLACED)
                    self.sensor_repl_count = 0
                    self.use_reference_formula = False

            elif trigger == TRIG_ZERO_OH:
                result, done = offset_reset(self.buffer, self.oil_change_count)
                print("\nin long term events new vehicle : ", self.oil_change_count, self.trigger_q)
                self.oil_change_count += 1
                if self.oil_change_count >= 8 or done == 1:
                    # update formula
                    # FIXME Address D에서 5회의 값을 읽어와서 수식 업데이트
                    # self.monitor_dielec.update(new_coefficient)
                    # self.monitor_visco.update(new_coefficient)
                    print('\n update formula')
                    self.trigger_q.remove(TRIG_OIL_CHANGED)
                    self.oil_change_count = 0
                    self.use_reference_formula = False

            elif trigger == TRIG_CHANGE_OIL_TYPE:
                self.trigger_q.remove(TRIG_CHANGE_OIL_TYPE)
                # FIXME 점도등급 트리거 -> 오일 교환 순서로 발생하면 문제없음, but 오일교환 -> 점도등급 순서로 발생하면 이 과정이 30분 이내 이루어져야함, 둘다 발생해야 진행되는 것으로
            else:
                #  정의 되지 않은 트리거
                raise NotImplementedError

    def right_after_trig(self, trig, trig_value):
        if trig == TRIG_OIL_CHANGED:
            self.buffer.reset()
            self.use_reference_formula = True

        elif trig == TRIG_SENSOR_REPLACED:
            self.emit_event('sensor_replaced', {'data': self.buffer})
            self.buffer.reset()
            self.use_reference_formula = True

        elif trig == TRIG_ZERO_OH:
            self.buffer.reset()
            self.use_reference_formula = True

        elif trig == TRIG_CHANGE_OIL_TYPE:
            self.viscosity_grade = trig_value
            # self.buffer.reset()
            # load new oil reference formula
        elif trig == TRIG_KEY_OFF:
            self.buffer.save()
        logger.info(trig)

    def run(self):
        prev_trigger_mode = TRIG_NONE
        while True:
            try:
                trigger = self.trigger_que.get()
                if trigger is not None:
                    trigger_mode, trigger_value = catch_trigger(trigger, self.name)
                    if trigger_mode != TRIG_NONE and prev_trigger_mode != trigger_mode:
                    # 감지 즉시 실행할 것
                        self.trigger_q.append(trigger_mode)
                        self.right_after_trig(trigger_mode, trigger_value)
                    prev_trigger_mode = trigger_mode
            except Exception as e:
                print(e)
                pass


            if self.rq.queue is None:
                continue

            OilMachineData, message_id = self.rq.get()
            sensor_malfunc, msg = self.sensor.check_malfunction(OilMachineData, message_id)
            if sensor_malfunc:
                self.emit_event("sensor_not_respond", {'msg': msg})
            OilMachineData['kine_viscosity'] = OilMachineData['OilVcsty'] / (OilMachineData['Oildensity'] + sys.float_info.epsilon)
            act_mode = self._check_activation(OilMachineData)
            if not act_mode:
                continue
            else:
                is_buffer_full = self.buffer.append_data(OilMachineData, time.time())

                '''버퍼가 가득 찰 때 할 일'''
                if is_buffer_full:
                    self.long_term_events()
                    self.buffer.reset()

                '''이상 감지 로직'''
                if self.use_reference_formula:
                    monitor_visco = RealTimeMonitoring(percent_margin=20, method=VISCOSITY_METHOD, VG=self.viscosity_grade)
                    res_visco_t, abnormal_visco, low_thresh_visco, high_thresh_visco = monitor_visco.monitor(
                        OilMachineData['kine_viscosity'], OilMachineData['OilAvrgTmp'])
                    warning_level = bool(abnormal_visco)
                    low_thresh_dielec = -1
                    high_thresh_dielec = -1
                else:
                    res_dielec_t, abnormal_dielec, low_thresh_dielec, high_thresh_dielec = self.monitor_dielec.monitor(
                        OilMachineData['Oildieleccst'], OilMachineData['OilAvrgTmp'])
                    res_visco_t, abnormal_visco, low_thresh_visco, high_thresh_visco = self.monitor_visco.monitor(
                        OilMachineData['kine_viscosity'], OilMachineData['OilAvrgTmp'])
                    warning_level = abnormal_visco or abnormal_dielec

                if warning_level and self.prev_warning_level != warning_level:
                    self.emit_event('abnormal', {'data': self.buffer, 'formula': (self.monitor_dielec.coefficient, self.monitor_visco.coefficient)})
                    self.buffer.reset()
                
                self.prev_warning_level = warning_level

                self.sender.send_warning_popup(warning_level, self.name)
                self.sender.cansend_EVtest(OilMachineData['kine_viscosity'], low_thresh_dielec, high_thresh_dielec,
                         low_thresh_visco, high_thresh_visco, self.name)
                print(f"{self.name} [Warning Gauge] {self.monitor_dielec.gauge}, {self.monitor_visco.gauge} [BUFFER]{self.buffer.count} [[DIELEC]{low_thresh_dielec: .3f}<{OilMachineData['Oildieleccst']: .3f}<{high_thresh_dielec: .3f}] [[VISCO]{low_thresh_visco: .3f}<{OilMachineData['kine_viscosity']: .3f}<{high_thresh_visco: .3f}] [TRIG]{self.trigger_q}")
                    

class MultiBus:
    def __init__(self, device, dataCAN, eventCAN, rq_AE, rq_3F, q, trig_q, name):
        self.recv_bus = Receiver(device, dataCAN, eventCAN)
        self.que = q
        self.ready_que_AE = rq_AE
        self.ready_que_3F = rq_3F
        self.trigger_que = trig_q
        self.name = name
        self.prev_key = 'Crank or Engine Running'

    def run(self):
        while True:
            message_id = self.recv_bus.receive()
            try:
                GPevent = self.recv_bus.events.pop()
                self.trigger_que.put(GPevent)
            except IndexError:
                pass

            if self.name == 'can1':
                dicAE = self.recv_bus.data_SA[0xAE]
                readyAE = not sum([x is None for x in dicAE.values()])
                dic3F = self.recv_bus.data_SA[0x3F]
                ready3F = not sum([x is None for x in dic3F.values()])
                if (readyAE or ready3F) and self.que.queue is not None:
                    dic00 = self.que.get()
                    if readyAE:
                        dicAE.update(dic00)
                        self.recv_bus.init_data(0xAE)
                        self.ready_que_AE.put([dicAE, message_id])
                    if ready3F:
                        dic3F.update(dic00)
                        self.recv_bus.init_data(0x3F)
                        self.ready_que_3F.put([dic3F, message_id])

            elif self.name == 'can0':
                dic00 = self.recv_bus.data_SA[0x00]
                self.que.put(dic00)

                periodic_machine_status = self.recv_bus.data_SA[0x34]
                key = periodic_machine_status['KeyStChange']
                if (key == 'Key Off') and (self.prev_key != key):
                    print("Key OFF")
                    self.trigger_que.put({'SPN': 0, 'FMI': 0})
                self.prev_key = key


if __name__ == "__main__":

    ready_que_AE = VolatileQueue()
    ready_que_3F = VolatileQueue()
    que = VolatileQueue()
    event_que = VolatileQueue()
    trigger_que = VolatileQueue()  # seems to need two que (only key off trigger), cause two thead get and discard

    MB1 = MultiBus('can1', can1_3F+can1_AE, can1_event, ready_que_AE, ready_que_3F, que, trigger_que, 'can1')
    MB2 = MultiBus('can1', can0_Eg, can0_event, ready_que_AE, ready_que_3F, que, trigger_que, 'can0')
    MS_AE = MonitoringSystem(ready_que_AE, event_que, trigger_que, "EngineOil")
    MS_3F = MonitoringSystem(ready_que_3F, event_que, trigger_que, "HydraulicOil")
    EC = EventConsumer(event_que)
    
    # 이벤트 핸들러 등록
    EC.add_event_handler('buffer_is_full', handle_buffer_is_full)
    EC.add_event_handler('sensor_replaced', handle_sensor_replaced)
    EC.add_event_handler('abnormal', handle_abnormal)
    EC.add_event_handler('sensor_not_respond', handle_sensor_not_respond)

    bus1_thread = threading.Thread(target=MB1.run, args=(), name='bus1')
    bus1_thread.start()
    bus2_thread = threading.Thread(target=MB2.run, args=(), name='bus2')
    bus2_thread.start()
    MS_thread = threading.Thread(target=MS_AE.run, args=(), name='MS_AE')
    MS_thread.start()
    MS_thread = threading.Thread(target=MS_3F.run, args=(), name='MS_3F')
    MS_thread.start()
    EC_thread = threading.Thread(target=EC.run, args=(), name='EC')
    EC_thread.start()

    bus1_thread.join()
    bus2_thread.join()
    MS_thread.join()
    EC_thread.join()
    