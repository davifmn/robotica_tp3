import time
import numpy as np
from coppeliasim_zmqremoteapi_client import RemoteAPIClient
from occupancy_grid import OccupancyGrid
from visualize_map import visualize_and_save_map

def main():
    # --- Conexão e Configuração da Simulação Síncrona ---
    client = RemoteAPIClient()
    sim = client.getObject('sim')
    
    sim.stopSimulation()
    time.sleep(1)
    
    # Habilita o modo síncrono (stepping), que foi bem-sucedido no teste.
    sim.setStepping(True)
    
    sim.startSimulation()
    print("Simulação iniciada em modo síncrono.")

    # --- Obtenção dos Handles ---
    # Nomes confirmados pelo sucesso do teste.py
    KOBUKI_NAME = '/kobuki'
    LEFT_MOTOR_NAME = '/kobuki_leftMotor'
    RIGHT_MOTOR_NAME = '/kobuki_rightMotor'
    LASER_NAME = '/kobuki/fastHokuyo'

    try:
        kobuki_handle = sim.getObject(KOBUKI_NAME)
        left_motor_handle = sim.getObject(LEFT_MOTOR_NAME)
        right_motor_handle = sim.getObject(RIGHT_MOTOR_NAME)
        laser_handle = sim.getObject(LASER_NAME)
        if any(h == -1 for h in [kobuki_handle, left_motor_handle, right_motor_handle, laser_handle]):
            raise ValueError("Um ou mais handles de objeto não foram encontrados.")
    except Exception as e:
        print(f"ERRO CRÍTICO ao obter handles: {e}")
        sim.stopSimulation()
        return

    print("Handles dos objetos obtidos com sucesso.")

    # --- Parâmetros ---
    MAP_X_MIN, MAP_X_MAX, MAP_Y_MIN, MAP_Y_MAX = -5, 5, -5, 5
    MAP_RESOLUTION = 0.05
    NOISE_DISTANCE_STD, NOISE_ANGLE_STD = 0.02, np.deg2rad(0.5)
    
    # --- Loop Síncrono de Navegação e Coleta ---
    simulation_duration = 120 # 2 minutos para um bom mapeamento
    print(f"Iniciando navegação por {simulation_duration} segundos de tempo de simulação...")
    
    start_sim_time = sim.getSimulationTime()
    collected_data = []
    
    try:
        while sim.getSimulationTime() - start_sim_time < simulation_duration:
            # Leitura dos Sensores
            ranges_signal = sim.getStringSignal('hokuyo_range_data')
            angles_signal = sim.getStringSignal('hokuyo_angle_data')
            
            if not ranges_signal or not angles_signal:
                sim.step() # Avança a simulação mesmo sem sinal
                continue

            ranges = sim.unpackFloatTable(ranges_signal)
            angles = sim.unpackFloatTable(angles_signal)
            
            # Lógica de Navegação: Apenas o cone frontal importa para desviar
            front_ranges = [dist for angle, dist in zip(angles, ranges) if -np.pi/4 < angle < np.pi/4]
            min_front_dist = min(front_ranges) if front_ranges else 1.0

            # Controle dos Motores
            v_left, v_right = (1.5, 1.5) if min_front_dist > 0.6 else (-0.6, 0.6)
            sim.setJointTargetVelocity(left_motor_handle, v_left)
            sim.setJointTargetVelocity(right_motor_handle, v_right)

            # Coleta de Dados para o mapa (a cada passo)
            robot_pos = sim.getObjectPosition(kobuki_handle, -1)
            robot_orient_euler = sim.getObjectOrientation(kobuki_handle, -1)
            robot_pose = (robot_pos[0], robot_pos[1], robot_orient_euler[2])
            collected_data.append({'pose': robot_pose, 'ranges': ranges, 'angles': angles})

            # Avança a simulação em um passo
            sim.step()

    except Exception as e:
        print(f"Ocorreu um erro durante a navegação: {e}")
    finally:
        sim.stopSimulation()
        print(f"Navegação concluída. {len(collected_data)} pontos de dados coletados.")

    # --- Mapeamento em Lote (após a simulação) ---
    if not collected_data:
        print("Nenhum dado foi coletado. O mapa não será gerado.")
        return
        
    print("Iniciando processamento do mapa. Isso pode levar alguns instantes...")
    occupancy_map = OccupancyGrid(MAP_X_MIN, MAP_X_MAX, MAP_Y_MIN, MAP_Y_MAX, MAP_RESOLUTION)
    
    data_to_process = collected_data[::5] # Otimização: processa 1 a cada 5 pontos de dados
    total_to_process = len(data_to_process)
    
    for i, data_point in enumerate(data_to_process):
        if (i+1) % 50 == 0: # Imprime o progresso a cada 50 pontos processados
            print(f"Processando... {i+1}/{total_to_process}")
        
        robot_pose = data_point['pose']
        points_relative = []
        for angle, dist in zip(data_point['angles'], data_point['ranges']):
            noisy_dist = dist + np.random.normal(0, NOISE_DISTANCE_STD)
            noisy_angle = angle + np.random.normal(0, NOISE_ANGLE_STD)
            points_relative.append([noisy_dist * np.cos(noisy_angle), noisy_dist * np.sin(noisy_angle)])
        
        x, y, theta = robot_pose
        rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta),  np.cos(theta)]])
        points_world = np.dot(np.array(points_relative), rotation_matrix.T) + np.array([x, y])
        occupancy_map.update(robot_pose, points_world)

    # --- Finalização e Salvamento do Mapa ---
    print("Mapeamento concluído. Salvando a imagem.")
    prob_map = occupancy_map.get_probability_map()
    visualize_and_save_map(prob_map, "occupancy_map.png")
    print("Mapa salvo como 'occupancy_map.png'")

if __name__ == '__main__':
    main()