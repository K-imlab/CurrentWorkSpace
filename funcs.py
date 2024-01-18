import numpy as np
import pickle
from scipy import interpolate
from scipy.optimize import curve_fit
import uncertainties.unumpy as unp
import uncertainties as unc
from canlib import canlib
from canlib import Frame

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from logging.handlers import RotatingFileHandler
import logging


def set_logger():
    LOG_FILE = 'logfile.txt'
    LOG_MAX_SIZE = 1024*100
    LOG_BACKUP_COUNT = 5
    logger = logging.getLogger('LogOilPHM')
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(threadName)s - %(funcName)s - %(message)s')
    file_handler = RotatingFileHandler(LOG_FILE, maxBytes=LOG_MAX_SIZE, backupCount=LOG_BACKUP_COUNT)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger


LOG = True
if LOG:
    logger = set_logger()

FIGURE = True
if FIGURE:
    fig = make_subplots(rows=2, cols=2)

TRIG_NONE = 0
TRIG_ZERO_OH = 2
TRIG_OIL_CHANGED = 3
TRIG_SENSOR_REPLACED = 4
TRIG_CHANGE_OIL_TYPE = 5
TRIG_KEY_OFF = 6

SENSOR_Hz = 1/30
CAN_Hz = 100

VG32 = 1
VG46 = 2
VG68 = 3

DIELECTRIC_METHOD = 'linear'
VISCOSITY_METHOD = 'vogel'  # 'power'

MAX_BUFFER_SIZE = 360

ABNORMAL_TOLERANCE = 60
SENSOR_OUT_TOLERANCE = 120
SENSOR_RESPONSE_TOLERANCE = CAN_Hz * 300

COLORS = ['rgb(255,0,0)', 'rgb(255,165,0)', 'rgb(255,255,0)', 'rgb(0,128,0)', 'rgb(0,0,255)']

HYDRAULIC = {'oil': 'hydraulic', 'PERCENT_MARGIN_DIELECTRIC': 2, 'PERCENT_MARGIN_VISCOSITY': 20,
             'ADDRESS_AB': './EEPROM/hyd_buffer', 'ADDRESS_C': './EEPROM/hyd_previous_sensor',
             'ADDRESS_D': './EEPROM/hyd_formula'}
ENGINE = {'oil': 'engine', 'PERCENT_MARGIN_DIELECTRIC': 3, 'PERCENT_MARGIN_VISCOSITY': 20,
          'ADDRESS_AB': './EEPROM/hyd_buffer', 'ADDRESS_C': './EEPROM/hyd_previous_sensor',
          'ADDRESS_D': './EEPROM/hyd_formula'}


class Receiver:
    def __init__(self):
        # variables
        self.ch = self._open_channel(0)
        self.can_ids = {0x1CFD083F: ['dielec', 'visco', 'density'],
                        0x18FEEE3F: ['temp'],
                        0x18FF313F: ['status'],
                        0x0CF00400: ['engine_speed'],
                        0x18FEE500: ['OH']}
        l = []
        for v in list(self.can_ids.values()):
            l += v
        self.columns = l
        self.periodic_data = {key: None for key in self.columns}
        self.event_can_ids = {0x18EF4A28: ['OilChange', 'SensorChange'],
                              0x18EFFFFF: ['HydraulicVG', 'EngineVG'],
                              0x18FFB334: ['KeyOff']}
        self.event_q = []

    def _open_channel(self, channel):
        ch = canlib.openChannel(channel, canlib.canOPEN_ACCEPT_VIRTUAL)
        ch.setBusOutputControl(canlib.canDRIVER_NORMAL)
        ch.setBusParams(canlib.canBITRATE_1M)
        ch.busOn()
        return ch

    def close_channel(self):
        self.ch.busOff()
        self.ch.close()

    def receive(self):
        d = self.ch.read(timeout=100)  # VIRTUAL can
        v = np.frombuffer(d.data, dtype=np.float16)
        # [dielec, visco, density]
        # 0xFFEA3849 -> [dielec, visco, density]

        if d.id in self.can_ids.keys():
            for i, names in enumerate(self.can_ids[d.id]):
                self.periodic_data[names] = v[i]

        elif d.id in self.event_can_ids.keys():
            for i, name in enumerate(self.event_can_ids[d.id]):
                if v[i] != 0:
                    self.event_q.append({"event": name, "value": v[i]})
            print("\nrecv from GP ", d.id, v, self.periodic_data, self.event_q)
        else:
            pass

        return d.id, self.periodic_data, self.event_q

    def parsing(self, msg, st_bit, len, byte_number):  # FIXME 추후 구현할 것
        bytes = msg[byte_number]
        bytes & 0b00000000
        return

    def init_data(self):
        self.periodic_data = {key: None for key in self.columns}


class Buffer:
    def __init__(self):
        # constant
        maxlen = MAX_BUFFER_SIZE
        self.maxlen = maxlen
        # variable
        self.q_dielec = [0.]*maxlen
        self.q_k_visco = [0.]*maxlen
        self.q_temper = [0.]*maxlen
        self.count = 0

    def append_data(self, data, unix_time):
        dielectric = data['dielec']
        kine_viscosity = data['kine_viscosity']
        temperature = data['temp']

        self.q_dielec.pop(self.count)
        self.q_dielec.insert(self.count, dielectric)
        self.q_k_visco.pop(self.count)
        self.q_k_visco.insert(self.count, kine_viscosity)
        self.q_temper.pop(self.count)
        self.q_temper.insert(self.count, temperature)

        self.count += 1

        if self.maxlen <= self.count:
            is_buffer_full = True
        else:
            is_buffer_full = False

        return is_buffer_full

    def reload_data(self):
        with open(ADDRESS_AB, 'rb') as f:
            buffer = pickle.load(f)
        self.q_dielec, self.q_k_visco, self.q_temper, self.count = buffer

    def save(self):
        with open(ADDRESS_AB, 'wb') as f:
            pickle.dump([self.q_dielec, self.q_k_visco, self.q_temper, self.count], f)

    def save_others(self):
        # FIXME 연산 값을 저장, 어떤 계수가 될지는 미정
        with open(ADDRESS_C, 'wb') as f:
            pickle.dump([self.q_dielec, self.q_k_visco, self.q_temper, self.count], f)

    def reset(self):
        self.count = 0
        pass

    def values(self):
        return self.q_dielec, self.q_k_visco, self.q_temper

    def sorted_values(self):
        if self.count >= self.maxlen:
            return self.values()
        else:
            sorted_dielec = self.q_dielec[self.count:] + self.q_dielec[:self.count]
            sorted_k_visco = self.q_k_visco[self.count:] + self.q_k_visco[:self.count]
            sorted_temper = self.q_temper[self.count:] + self.q_temper[:self.count]
            return sorted_dielec, sorted_k_visco, sorted_temper


class RealTimeMonitoring:
    def __init__(self, percent_margin, method, VG):
        self.percent_margin = percent_margin
        self.tolerance = ABNORMAL_TOLERANCE
        self.gauge = 0
        self.reference_formula, self.coefficient = self.table_method_VG(method, VG)

    def table_method_VG(self, method, VG):
        # FIXME 레퍼런스 수식 및 계수 획득
        if method == "power":
            coefficient = (36702.869720, -1.814116)
            reference_formula = POWER
        elif method == "vogel":
            coefficient = (0.372215, 539.303946, 200.533208)
            reference_formula = VOGEL
        else:
            coefficient = (-0.00220675981728211, 2.3298374998223057)
            reference_formula = LINEAR
        return reference_formula, coefficient

    def monitor(self, value, temperature):
        lt, ut = self._get_threshold(temperature)
        if (lt <= value) and (value <= ut):
            anomaly = False
        else:
            anomaly = True
        self.gauge = int(self.gauge * anomaly) + int(anomaly)
        if self.gauge >= self.tolerance:
            ABNORMAL = True
        else:
            ABNORMAL = False
        return anomaly, ABNORMAL, lt, ut

    def update(self, new_coefficient):
        self.coefficient = new_coefficient

    def _get_threshold(self, temperature):
        offset = self.reference_formula(temperature, *self.coefficient)
        lower_threshold = offset*(1 - (self.percent_margin / 100))
        upper_threshold = offset*(1 + (self.percent_margin / 100))
        return lower_threshold, upper_threshold


def LINEAR(x, a_Di, b_Di):
    return a_Di * x + b_Di


def LINEAR_CONSTANT(a_Di, b_Di):
    def linear_(x, c_Di):
        return a_Di * x + b_Di + c_Di
    return linear_


def POWER(x, a_Kr, b_Kr):
    return a_Kr * (x ** b_Kr)


def POWER_CONSTANT(a_Kr, b_Kr):
    def power_(x, c_Kr, d_Kr):
        return a_Kr * ((x-c_Kr) ** b_Kr) + d_Kr
    return power_


def VOGEL(x, a_Kr, b_Kr, c_Kr):
    x = x + 273
    return a_Kr * np.exp(b_Kr/(x-c_Kr))


def VOGEL_CONSTANT(a_Kr, b_Kr, c_Kr):
    def vogel_(x, d_Kr, e_Kr):
        x + 273
        return a_Kr * np.exp(b_Kr / ((x-d_Kr) - c_Kr)) + e_Kr
    return vogel_


def analysis(function, temperature, value):
    popt, pcov = curve_fit(function, temperature, value, maxfev=1000000)
    px = np.linspace(40, 80, 200)
    py = function(px, *popt)
    nom = unp.nominal_values(py)
    trend = function(60, *popt)
    nom_Threshold = unp.nominal_values(trend)
    return py, nom_Threshold, popt, trend


def analysis_vogel(function, temperature, value):
    popt, pcov = curve_fit(function, temperature, value, maxfev=1000000)
    px = np.linspace(40, 80, 200)
    py = function(px, *popt)
    nom = unp.nominal_values(py)
    trend = function(60, *popt)
    nom_Threshold = unp.nominal_values(trend)
    return nom, nom_Threshold, popt, trend


class Sensor:
    def __init__(self):
        self.prev_sensor_out = 0
        self.sensor_out_gauge = 0
        self.duration_from_latest_response = 0
        self.prev_sensor_no_response = 0

    def catch_sensor_out(self, data, sig_id):
        if sig_id == 0x18FF313F:  # FIXME can id 외부 관리
            non_working = bool(data['status'])  # sensor status 프로토콜이 미정
            self.sensor_out_gauge = int(self.sensor_out_gauge * non_working) + int(non_working)

            if self.sensor_out_gauge >= SENSOR_OUT_TOLERANCE:
                sensor_out = True
            else:
                sensor_out = False
        else:
            sensor_out = False

        return sensor_out

    def catch_sensor_no_response(self, data, sig_id):
        sensor_data = [data['visco'], data['density'], data['dielec'], data['temp'], data['status']]
        try:
            check_response = sum(sensor_data)
            self.duration_from_latest_response = 0
        except TypeError:
            pass

        if sig_id == 0x0CF00400:  # engine speed  보단 timestamp로 판단하는게 좋아보임
            self.duration_from_latest_response += 1
            if self.duration_from_latest_response >= SENSOR_RESPONSE_TOLERANCE:
                sensor_no_response = True
            else:
                sensor_no_response = False
        else:
            sensor_no_response = self.prev_sensor_no_response

        return sensor_no_response

    def check_malfunction(self, data, sig_id):
        SPN = 0x7EA62
        MSG = None
        malfunction = False
        sensor_out = self.catch_sensor_out(data, sig_id)
        sensor_no_response = self.catch_sensor_no_response(data, sig_id)
        if sensor_out and self.prev_sensor_out != sensor_out:  # 0 -> 1
            FMI = 11
            MSG = 0xFFFF62EAEBFF
            malfunction = True

        if sensor_no_response and self.prev_sensor_no_response != sensor_no_response:
            FMI = 12
            MSG = 0xFFFF62EAECFF
            malfunction = True

        self.prev_sensor_out = sensor_out
        self.prev_sensor_no_response = sensor_no_response
        return malfunction, MSG
