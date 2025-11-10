# main.py (versão corrigida / mais robusta)
import time
import math
import numpy as np
from coppeliasim_zmqremoteapi_client import RemoteAPIClient
from occupancy_grid import OccupancyGrid
from visualize_map import visualize_and_save_map

def filter_ranges(arr, min_valid=0.01, max_valid=30.0):
    a = np.array(arr, dtype=float)
    # descarta zeros, negative, nan, inf e muito longe
    mask = np.isfinite(a) & (a >= min_valid) & (a <= max_valid)
    return a[mask]

def center_window_min(ranges, window_frac=0.1):
    n = len(ranges)
    if n == 0:
        return float('inf')
    center = n // 2
    w = max(1, int(n * window_frac / 2))
    start = max(0, center - w)
    end = min(n, center + w + 1)
    return float(np.min(ranges[start:end]))

def choose_turn_direction(ranges):
    # escolhe virar para o lado com maior distância média
    n = len(ranges)
    if n == 0:
        return 1.0
    left = ranges[:n//3]
    right = ranges[-(n//3):]
    mean_left = np.mean(left) if len(left) else 0
    mean_right = np.mean(right) if len(right) else 0
    return -1.0 if mean_left > mean_right else 1.0

def main():
    client = RemoteAPIClient()
    sim = client.getObject('sim')

    # restart sim (silencioso)
    try:
        sim.stopSimulation()
    except Exception:
        pass
    time.sleep(0.2)
    sim.startSimulation()
    print("Simulação iniciada.")

    # nomes (use os mesmos que você já confirmou)
    KOBUKI_NAME = '/kobuki'
    LEFT_MOTOR_NAME = '/kobuki_leftMotor'
    RIGHT_MOTOR_NAME = '/kobuki_rightMotor'
    # sinal do hokuyo (você já usa esses sinais)
    RANGES_SIGNAL_NAME = 'hokuyo_range_data'
    ANGLES_SIGNAL_NAME = 'hokuyo_angle_data'

    try:
        kobuki_handle = sim.getObject(KOBUKI_NAME)
        left_motor_handle = sim.getObject(LEFT_MOTOR_NAME)
        right_motor_handle = sim.getObject(RIGHT_MOTOR_NAME)
        if kobuki_handle == -1 or left_motor_handle == -1 or right_motor_handle == -1:
            raise RuntimeError("Algum handle retornou -1")
    except Exception as e:
        print("Erro obtendo handles:", e)
        try: sim.stopSimulation()
        except: pass
        return

    # parâmetros
    L, r = 0.230, 0.035
    MAP_X_MIN, MAP_X_MAX, MAP_Y_MIN, MAP_Y_MAX = -5, 5, -5, 5
    MAP_RESOLUTION = 0.05
    NOISE_DISTANCE_STD, NOISE_ANGLE_STD = 0.02, np.deg2rad(0.5)

    # controle
    SIMULATION_DURATION = 5.0        # tempo total da exploração (s) - aumentei
    DATA_COLLECT_INTERVAL = 0.5      # coleta dados a cada 0.5s
    TIME_STEP = 0.05
    FORWARD_SPEED = 0.35             # m/s (ajuste para seu robô)
    TURN_SPEED = 0.8                 # rad/s
    SAFE_DISTANCE = 0.45             # m
    MAX_WHEEL_VEL = 4.0              # rad/s (limite para evitar valores absurdos)
    WHEEL_BASE = 0.230               # distância entre rodas (L)

    collected_data = []
    last_collect_t = -9999
    start_t = 0

    print("Iniciando exploração por {:.1f}s ...".format(SIMULATION_DURATION))
    try:
        x = 1
        while x - start_t > 0:
            x = x+ 1
            t0 = time.time()

            ranges_signal = sim.getStringSignal(RANGES_SIGNAL_NAME)
            if not ranges_signal:
                # sem sinal -> espera curto e continua
                # print("Nenhum sinal de ranges ainda.")
                time.sleep(0.02)
                continue

            ranges = sim.unpackFloatTable(ranges_signal)
            ranges = np.array(ranges, dtype=float)

            # Filtra leituras inválidas antes de tomar decisões
            valid = filter_ranges(ranges, min_valid=0.01, max_valid=30.0)
            if len(valid) == 0:
                # se tudo inválido, assume livre por segurança (ou pode parar)
                min_front = float('inf')
            else:
                # usa janela central do array original (não do 'valid') para manter correspondência de ângulos
                min_front = center_window_min(ranges, window_frac=0.08)

            # política de movimento
            linear = 0.0
            angular = 0.0
            if min_front > SAFE_DISTANCE:
                linear = FORWARD_SPEED
                angular = 0.0
            else:
                # virar direção com mais espaço lateral
                dir_sign = choose_turn_direction(ranges)
                linear = 0.0
                angular = TURN_SPEED * dir_sign

            # converte linear+angular para velocidades de roda (vr, vl)
            # v = (vr + vl)/2 ; omega = (vr - vl)/L => solve:
            vr = linear + (angular * WHEEL_BASE / 2.0)
            vl = linear - (angular * WHEEL_BASE / 2.0)

            # limita velocidades (e inverte se necessário)
            vr = float(np.clip(vr, -MAX_WHEEL_VEL, MAX_WHEEL_VEL))
            vl = float(np.clip(vl, -MAX_WHEEL_VEL, MAX_WHEEL_VEL))

            # enviar para os motores
            sim.setJointTargetVelocity(left_motor_handle, vl)
            sim.setJointTargetVelocity(right_motor_handle, vr)

            # debug (imprime a cada N iterações / 0.5s)
            if time.time() - last_collect_t >= DATA_COLLECT_INTERVAL:
                print(f"[t={int(time.time()-start_t)}s] min_front={min_front:.3f} vl={vl:.3f} vr={vr:.3f} linear={linear:.2f} angular={angular:.2f}")

            # coleta de dados para mapeamento
            if time.time() - last_collect_t >= DATA_COLLECT_INTERVAL:
                angles_signal = sim.getStringSignal(ANGLES_SIGNAL_NAME)
                if angles_signal:
                    angles = sim.unpackFloatTable(angles_signal)
                    # somente aceite se o tamanho bater
                    if len(angles) == len(ranges):
                        # transforma e guarda
                        robot_pos = sim.getObjectPosition(kobuki_handle, -1)
                        robot_orient = sim.getObjectOrientation(kobuki_handle, -1)
                        pose = (robot_pos[0], robot_pos[1], robot_orient[2])
                        collected_data.append({'pose': pose, 'ranges': ranges.tolist(), 'angles': angles})
                        last_collect_t = time.time()

            # pequeno sleep para respeitar loop
            elapsed = time.time() - t0
            if elapsed < TIME_STEP:
                time.sleep(TIME_STEP - elapsed)

    except Exception as e:
        print("Erro durante exploração:", e)
    finally:
        # parar motores
        try:
            sim.setJointTargetVelocity(left_motor_handle, 0.0)
            sim.setJointTargetVelocity(right_motor_handle, 0.0)
        except Exception:
            pass
        print("Exploração finalizada. Pontos coletados:", len(collected_data))

    # processamento do mapa (batch)
    print("Iniciando processamento do mapa...")
    occupancy_map = OccupancyGrid(MAP_X_MIN, MAP_X_MAX, MAP_Y_MIN, MAP_Y_MAX, MAP_RESOLUTION)

    for i, dp in enumerate(collected_data):
        pose = dp['pose']
        ranges = np.array(dp['ranges'], dtype=float)
        angles = np.array(dp['angles'], dtype=float)

        # filtra pares inválidos (mesma regra)
        mask = np.isfinite(ranges) & (ranges > 0.01) & (ranges < 30.0)
        ranges = ranges[mask]
        angles = angles[mask]

        # adiciona ruído (como você fazia)
        dx = ranges * np.cos(angles) + np.random.normal(0, NOISE_DISTANCE_STD, size=len(ranges))
        dy = ranges * np.sin(angles) + np.random.normal(0, NOISE_DISTANCE_STD, size=len(ranges))
        pts_rel = np.vstack([dx, dy]).T

        x, y, theta = pose
        c = math.cos(theta); s = math.sin(theta)
        R = np.array([[c, -s],[s, c]])
        pts_world = (R.dot(pts_rel.T)).T + np.array([x, y])

        occupancy_map.update(pose, pts_world)

    prob_map = occupancy_map.get_probability_map()
    visualize_and_save_map(prob_map, "occupancy_map.png")
    print("Mapa salvo como 'occupancy_map.png'")

    try:
        sim.stopSimulation()
    except Exception:
        pass

if __name__ == "__main__":
    main()
