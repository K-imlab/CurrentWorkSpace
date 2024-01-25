from funcs import *
if FIGURE:
    import plotly.graph_objs as go


def offset_reset(data, count, origin_coef_dielec, origin_coef_visco):
    xs = np.linspace(40, 80, 200)
    if count < 3:
        mean = None
        done = 0
        if FIGURE and count == 0:
            y_dielec = LINEAR(xs, *origin_coef_dielec)
            y_visco = VOGEL(xs, *origin_coef_visco)
            fig.add_trace(go.Scatter(x=xs, y=y_dielec, mode='lines', name='origin',
                                     line=dict(color='rgb(0,0,0)')), row=1, col=1)
            fig.add_trace(go.Scatter(x=xs, y=y_visco, mode='lines', name='origin',
                                     line=dict(color='rgb(0,0,0)')), row=1, col=2)
            fig.add_trace(go.Scatter(x=xs, y=y_visco*1.2, mode='lines', name='upper threshold',
                                     line=dict(color='rgb(0,0,0)', dash='dash')), row=1, col=2)
            fig.add_trace(go.Scatter(x=xs, y=y_visco * 0.8, mode='lines', name='lower threshold',
                                     line=dict(color='rgb(0,0,0)', dash='dash')), row=1, col=2)
            fig.add_trace(go.Scatter(x=xs, y=y_dielec * 1.02, mode='lines', name='upper threshold',
                                     line=dict(color='rgb(0,0,0)', dash='dash')), row=1, col=1)
            fig.add_trace(go.Scatter(x=xs, y=y_dielec * 0.98, mode='lines', name='lower threshold',
                                     line=dict(color='rgb(0,0,0)', dash='dash')), row=1, col=1)
    elif count < 8:
        dielec, kine_visco, temper = data.values()
        nom_dielec, _, popt_dielec, _ = analysis(LINEAR_CONSTANT(*origin_coef_dielec), temper, dielec)
        if VISCOSITY_METHOD == "vogel":
            nom_visco, _, popt_visco, _ = analysis_vogel(VOGEL_CONSTANT(*origin_coef_visco), temper, kine_visco)
        else:
            nom_visco, _, popt_visco, _ = analysis(POWER_CONSTANT(*origin_coef_visco), temper, kine_visco)
        # FIXME save at Address D 저장할게 nom? 계수?
        done = -1
        mean = nom_visco  # FIXME 5회 연산 옵티마이즈
        if FIGURE:
            fig.add_trace(go.Scatter(x=xs, y=nom_dielec, mode='lines', name=f'count{count}',
                                     line=dict(color=COLORS[count - 3])), row=1, col=1)
            fig.add_trace(go.Scatter(x=temper, y=dielec, mode='markers', name=f'count{count}',
                                     marker=dict(color=COLORS[count - 3], opacity=0.5)), row=1, col=1)
            fig.add_trace(go.Scatter(x=xs, y=nom_visco, mode='lines', name=f'count{count}',
                                     line=dict(color=COLORS[count - 3])), row=1, col=2)
            fig.add_trace(go.Scatter(x=temper, y=kine_visco, mode='markers', name=f'count{count}',
                                     marker=dict(color=COLORS[count - 3], opacity=0.5)), row=1, col=2)
    else:
        done = 1
        mean = None
    if LOG:
        logger.info("")
    return mean, done


def handle_buffer_is_full(data):

    print("\nHANDLER buffer_is_full")
    dielec, kine_visco, temper = data['data'].values()
    coefficient_linear, coefficient_visco = data['formula']
    
    nom, _, _, trend_linear = analysis(LINEAR_CONSTANT(*coefficient_linear), temper, dielec)
    if VISCOSITY_METHOD == "vogel":
        nom, _, _, trend_visco = analysis_vogel(VOGEL_CONSTANT(*coefficient_visco), temper, kine_visco)
    else:
        nom, _, _, trend_visco = analysis(POWER_CONSTANT(*coefficient_visco), temper, kine_visco)

    # print("\n trend", trend_linear, trend_visco)
    # FIXME server로 값을 보냄
    if LOG:
        logger.info("")


def handle_sensor_replaced(data):
    buffer = data['data']
    buffer.save_others()
    if LOG:
        logger.info("")
    return


def handle_abnormal(data):
    '''1. Send to GP/Server buffer sorted by OH,'''
    dielec, kine_visco, temper = data['data'].sorted_values()

    '''2. 60도에 해당하는 회귀 결과 값을 연산하여 추세선 그래프에도 추가'''
    dielec, kine_visco, temper = data['data'].values()
    coefficient_linear, coefficient_visco = data['formula']
    nom, _, _, trend_linear = analysis(LINEAR_CONSTANT(*coefficient_linear), temper, dielec)
    if VISCOSITY_METHOD == "vogel":
        nom, _, _, trend_visco = analysis_vogel(VOGEL_CONSTANT(*coefficient_visco), temper, kine_visco)
    else:
        nom, _, _, trend_visco = analysis(POWER_CONSTANT(*coefficient_visco), temper, kine_visco)

    trend = (trend_linear, trend_visco)
    if LOG:
        logger.info("")
    return


def handle_sensor_not_respond(data):
    msg = data['msg']
    DM_CAN_ID = 0x18FECA4A
    print("\n Send DM code ! sensor out")
    # FIXME can tx
    if LOG:
        logger.info("")
    return


def catch_trigger(event_q, engine_or_hyd):
    fmi = event_q['FMI']
    if event_q['SPN'] == 516282:
        if engine_or_hyd == 'HydraulicOil':
            out = TRIG_OIL_CHANGED
            print("catch Hydraulic Oil Change")
        else:
            out = TRIG_NONE

    elif event_q['SPN'] == 516281:
        if engine_or_hyd == 'EngineOil':
            out = TRIG_OIL_CHANGED
            print("catch Engine Oil Change")
        else:
            out = TRIG_NONE

    elif event_q['SPN'] == 517481:
        if engine_or_hyd == 'HydraulicOil':
            out = TRIG_SENSOR_REPLACED
            print("catch OPSensor Engine REPLACED")
        else:
            out = TRIG_NONE

    elif event_q['SPN'] == 517482:
        if engine_or_hyd == 'EngineOil':
            out = TRIG_SENSOR_REPLACED
            print("catch OPSensor Hydraulic REPLACED")
        else:
            out = TRIG_NONE

    elif event_q['SPN'] == 517479:
        if engine_or_hyd == 'HydraulicOil':
            out = TRIG_CHANGE_OIL_TYPE
            print("catch select Oil Viscosity Grade")
        else:
            out = TRIG_NONE

    elif event_q['SPN'] == 0:
        out = TRIG_KEY_OFF
        print("catch key off")
    else:
        out = TRIG_NONE

    return out, fmi

