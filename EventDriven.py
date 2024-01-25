import queue
import threading
import time
import sys

from event_handlers import *
from funcs import *


class EventConsumer:
    """ Event 가 발생하면 이벤트 항목을 Queue에 저장해 놓고, event에 해당하는 Handler를 호출하는 역할"""
    def __init__(self, event_queue: queue.Queue, params_queue: queue.Queue):
        self.event_queue = event_queue
        self.params_queue = params_queue
        self.handlers = {}

    def add_event_handler(self, event_type, handler):
        if event_type not in self.handlers:
            self.handlers[event_type] = []
        self.handlers[event_type].append(handler)

    def remove_event_handler(self, event_type, handler):
        if event_type in self.handlers and handler in self.handlers[event_type]:
            self.handlers[event_type].remove(handler)

    def process_events(self):
        while True:
            try:
                event = self.event_queue.get()
                event_type = event['type']
                data = event['data']

                if event_type in self.handlers:
                    for handler in self.handlers[event_type]:
                        out = handler(data)
                        self.params_queue.put(out)
            except queue.Empty:
                pass


class MonitoringSystem:
    """Event Producer 임. 즉 모니터링을 진행하면서 Trigger 가 발생되면 emit_event() 매서드를 통해 EventConsumer에게 알려줌"""
    def __init__(self, event_queue: queue.Queue, params_queue: queue.Queue):
        self.event_queue = event_queue
        self.params_queue = params_queue
        # self.receiver = Receiver()
        self.buffer = Buffer()
        self.sensor = Sensor()

        can0_event = [
        ]
        can0_Eg = [
            {"can_id": 0x0CF00400, "can_mask": 0x1FFFFFFF, "extended": True },
            {"can_id": 0x18FEE500, "can_mask": 0x1FFFFFFF, "extended": True },
        ]
        can1_event = [
            {"can_id": 0x18EF4A28, "can_mask": 0x1FFFFFFF, "extended": True},
            {"can_id": 0x18EFFFFF, "can_mask": 0x1FFFFFFF, "extended": True},
            {"can_id": 0x18FFB334, "can_mask": 0x1FFFFFFF, "extended": True},
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

        self.recv_bus1 = Receiver('can1', can1_AE+can1_3F, can1_event)
        self.recv_bus2 = Receiver('can1', can0_Eg, can0_event)

        self.monitor_dielec = RealTimeMonitoring(percent_margin=2, method=DIELECTRIC_METHOD, VG=VG46)
        self.monitor_visco = RealTimeMonitoring(percent_margin=20, method=VISCOSITY_METHOD, VG=VG46)
        # self.monitor_visco_vogel = RealTimeMonitoring(percent_margin=20, method='vogel', VG=VG46)

        self.oil_change_count = 0
        self.sensor_repl_count = 0
        self.trigger_q = []
        self.viscosity_grade = VG46  # VG46
        self.use_reference_formula = False
        self.prev_warning_level = 0

    def emit_event(self, event_type, data=None):
        event = {'type': event_type, 'data': data}
        self.event_queue.put(event)

    def get_data(self):
        """
        data: (type dict) signal name을 key, 한 시점의 signal 값을 value로 가지는 딕셔너리
                초기 value는 모든 key에 대해 None 값을 가지고 있음
                CAN or 기타 통신을 통해 값이 업데이트 되도록 receiver.receive() 가 구현됨
                EX) data = {'engine_speed': 1800, 'viscosity': 21.222, 'density': 2.34}
        updated_signal_name: (type string) 직전에 업데이트된 시그널의 이름 EX) 'engine_speed', VSS naming rule을 따라 작성
        ready: data 의 값들이 모두 None이 아닌 경우; 모든 값이 적어도 한번 업데이트 되면 True
                None이 하나라도 있으면 False
        :return:
        """
        try:
            self.recv_bus1.receive()
            self.recv_bus2.receive()

            message_id, data, events = self.receiver.receive()
            ready = not sum([x is None for x in data.values()])
        except Exception as e:
            ready = False
            data = None
            message_id = None
            events = None

        return ready, data, message_id, events

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

    def process_run(self):
        prev_trigger_mode = TRIG_NONE
        while True:
            '''Event Consumer로 부터 실행된 Handler의 결과 값 (params_queue)을 가져옴'''
            if not self.params_queue.empty():
                params_set = self.params_queue.get(timeout=1)
            else:
                pass

            '''ready 될때 까지 반복 업데이트, ready 되면 필터링, 조건을 만족할 때 까지 반복'''
            ready, data, sig_id, event_q = self.get_data()
            if data is None:
                continue
            '''트리거 감지'''
            trigger_mode, trigger_value = catch_trigger(event_q)
            if trigger_mode != TRIG_NONE and prev_trigger_mode != trigger_mode:
                # 감지 즉시 실행할 것
                self.right_after_trig(trigger_mode, trigger_value)
                self.trigger_q.append(trigger_mode)
            prev_trigger_mode = trigger_mode

            sensor_malfunc, msg = self.sensor.check_malfunction(data, sig_id)
            if sensor_malfunc:
                self.emit_event("sensor_not_respond", {'msg': msg})

            if not ready or data is None:
                continue
            elif ready and data is not None:
                data['kine_viscosity'] = data['OilVcsty'] / (data['Oildensity'] + sys.float_info.epsilon)
                act_mode = self._check_activation(data)
                if not act_mode:
                    continue
                else:
                    is_buffer_full = self.buffer.append_data(data, time.time())
                    self.receiver.init_data()  # 모든 시그널이 한번씩 채워진 후 다시 초기화

                    '''버퍼가 가득 찰 때 할 일'''
                    if is_buffer_full:
                        self.long_term_events()
                        self.buffer.reset()

                    '''이상 감지 로직'''
                    if self.use_reference_formula:
                        monitor_visco = RealTimeMonitoring(percent_margin=20, method=VISCOSITY_METHOD, VG=self.viscosity_grade)
                        res_visco_t, abnormal_visco, low_thresh_visco, high_thresh_visco = monitor_visco.monitor(
                            data['kine_viscosity'], data['OilAvrgTmp'])
                        warning_level = bool(abnormal_visco)
                        low_thresh_dielec = -1
                        high_thresh_dielec = -1
                    else:
                        res_dielec_t, abnormal_dielec, low_thresh_dielec, high_thresh_dielec = self.monitor_dielec.monitor(
                            data['Oildieleccst'], data['OilAvrgTmp'])
                        res_visco_t, abnormal_visco, low_thresh_visco, high_thresh_visco = self.monitor_visco.monitor(
                            data['kine_viscosity'], data['OilAvrgTmp'])
                        warning_level = abnormal_visco + abnormal_dielec
                    if warning_level and self.prev_warning_level != warning_level:
                        self.emit_event('abnormal', {'data': self.buffer, 'warning_level': warning_level, 'formula': (
                            self.monitor_dielec.coefficient, self.monitor_visco.coefficient)})
                        self.buffer.reset()

                    self.prev_warning_level = warning_level

                    print(f"\r[BUFFER]{self.buffer.count} [[DIELEC]{low_thresh_dielec: .3f}<{data['Oildieleccst']: .3f}<{high_thresh_dielec: .3f}] [[VISCO]{low_thresh_visco: .3f}<{data['kine_viscosity']: .3f}<{high_thresh_visco: .3f}] [TRIG]{self.trigger_q}", end="")
            else:
                # something wrong, throw error
                raise('Error! while making data')


if __name__ == "__main__":
    # Event Driven System 인스턴스 생성
    if LOG:
        logger.info("START PROGRAM")

    hyd_event_que = queue.Queue()  # 공유메모리  event_que = [{"event": 260buffer, }]
    params_que = queue.Queue()
    hyd_consumer = EventConsumer(hyd_event_que, params_que)
    hyd_monitor_system = MonitoringSystem(hyd_event_que, params_que)

    # 이벤트 핸들러 등록
    hyd_consumer.add_event_handler('buffer_is_full', handle_buffer_is_full)
    hyd_consumer.add_event_handler('sensor_replaced', handle_sensor_replaced)
    hyd_consumer.add_event_handler('abnormal', handle_abnormal)
    hyd_consumer.add_event_handler('sensor_not_respond', handle_sensor_not_respond)

    # 이벤트 생성 및 처리 루프 실행
    if LOG:
        logger.info("START THREAD")

    hyd_monitoring_thread = threading.Thread(target=hyd_monitor_system.process_run, args=(), name='hyd_monitor')
    hyd_monitoring_thread.start()
    hyd_consumer_thread = threading.Thread(target=hyd_consumer.process_events, args=(), name='hyd_consumer')
    hyd_consumer_thread.start()

    hyd_monitoring_thread.join()
    hyd_consumer_thread.join()
    
