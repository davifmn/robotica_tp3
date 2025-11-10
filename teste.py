import time
import numpy as np
from coppeliasim_zmqremoteapi_client import RemoteAPIClient
# Importando os outros arquivos conforme solicitado
from occupancy_grid import OccupancyGrid
from visualize_map import visualize_and_save_map

def main():
    # --- Conexão e Configuração da Simulação Síncrona ---
    client = RemoteAPIClient()
    sim = client.getObject('sim')
    
    # Garante que qualquer simulação anterior seja interrompida
    sim.stopSimulation()
    time.sleep(1)
    
    # Habilita o modo síncrono (stepping mode). O script Python agora controla cada passo da simulação.
    sim.setStepping(True)
    
    # Inicia a simulação
    sim.startSimulation()
    print("Simulação iniciada em modo síncrono para teste de movimento.")
    
    # Avança um passo para garantir que a simulação está rodando antes do loop
    sim.step()

    # --- Obtenção dos Handles ---
    KOBUKI_NAME = '/kobuki'
    LEFT_MOTOR_NAME = '/kobuki_leftMotor'
    RIGHT_MOTOR_NAME = '/kobuki_rightMotor'
    LASER_NAME = '/fastHokuyo'

    try:
        kobuki_handle = sim.getObject(KOBUKI_NAME)
        left_motor_handle = sim.getObject(LEFT_MOTOR_NAME)
        right_motor_handle = sim.getObject(RIGHT_MOTOR_NAME)
        laser_handle = sim.getObject(KOBUKI_NAME + LASER_NAME)
        
        if kobuki_handle == -1: raise ValueError(f"Robô '{KOBUKI_NAME}' não encontrado.")
        if left_motor_handle == -1: raise ValueError(f"Motor '{LEFT_MOTOR_NAME}' não encontrado.")
        if right_motor_handle == -1: raise ValueError(f"Motor '{RIGHT_MOTOR_NAME}' não encontrado.")
        if laser_handle == -1: raise ValueError(f"Laser '{KOBUKI_NAME + LASER_NAME}' não encontrado.")

    except Exception as e:
        print(f"ERRO CRÍTICO ao obter handles de objeto: {e}")
        sim.stopSimulation()
        return

    print("Handles dos objetos obtidos com sucesso.")

    # --- Parâmetros ---
    simulation_duration = 10  # segundos
    forward_speed = 1.5       # Velocidade para andar em linha reta
    MAP_X_MIN, MAP_X_MAX, MAP_Y_MIN, MAP_Y_MAX = -5, 5, -5, 5
    MAP_RESOLUTION = 0.05
    NOISE_DISTANCE_STD, NOISE_ANGLE_STD = 0.02, np.deg2rad(0.5)
    DATA_COLLECT_INTERVAL = 1.0 # Coleta dados a cada 1.0 segundo de simulação

    # --- Loop de Movimento e Coleta de Dados ---
    print(f"Iniciando movimento e coleta de dados por {simulation_duration} segundos...")
    start_sim_time = sim.getSimulationTime()
    last_data_collect_time = start_sim_time
    collected_data = []
    
    try:
        # Loop principal baseado no tempo da simulação
        while sim.getSimulationTime() - start_sim_time < simulation_duration:
            current_sim_time = sim.getSimulationTime()
            
            # Define a mesma velocidade para ambos os motores para andar em linha reta
            sim.setJointTargetVelocity(left_motor_handle, forward_speed)
            sim.setJointTargetVelocity(right_motor_handle, forward_speed)

            # Coleta de Dados (em intervalos de tempo de SIMULAÇÃO)
            if current_sim_time - last_data_collect_time > DATA_COLLECT_INTERVAL:
                print(f"Coletando dados... Tempo de simulação: {int(current_sim_time)}s")
                last_data_collect_time = current_sim_time
                
                robot_pos = sim.getObjectPosition(kobuki_handle, -1)
                robot_orient_euler = sim.getObjectOrientation(kobuki_handle, -1)
                robot_pose = (robot_pos[0], robot_pos[1], robot_orient_euler[2])
                
                ranges_signal = sim.getStringSignal('hokuyo_range_data')
                angles_signal = sim.getStringSignal('hokuyo_angle_data')
                
                if ranges_signal and angles_signal:
                    ranges = sim.unpackFloatTable(ranges_signal)
                    angles = sim.unpackFloatTable(angles_signal)
                    collected_data.append({'pose': robot_pose, 'ranges': ranges, 'angles': angles})
            
            # Comando essencial no modo síncrono: avança a simulação em um passo
            sim.step()

    except Exception as e:
        print(f"Ocorreu um erro durante o loop de movimento: {e}")
    finally:
        # --- Finalização do Movimento ---
        print("Tempo de teste concluído.")
        sim.setJointTargetVelocity(left_motor_handle, 0)
        sim.setJointTargetVelocity(right_motor_handle, 0)
        print(f"Navegação concluída. {len(collected_data)} pontos de dados coletados.")

    # --- Mapeamento em Lote (Batch Processing) ---
    if collected_data:
        print("Iniciando processamento do mapa. Isso pode levar alguns instantes...")
        occupancy_map = OccupancyGrid(MAP_X_MIN, MAP_X_MAX, MAP_Y_MIN, MAP_Y_MAX, MAP_RESOLUTION)
        
        for i, data_point in enumerate(collected_data):
            print(f"Processando data point {i+1}/{len(collected_data)}...")
            
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

        # --- Geração e Salvamento do Mapa ---
        print("Mapeamento concluído. Gerando e salvando a imagem.")
        prob_map = occupancy_map.get_probability_map()
        visualize_and_save_map(prob_map, "occupancy_map_teste.png")
        print("Mapa salvo como 'occupancy_map_teste.png'")

    # --- Finalização da Simulação ---
    sim.stopSimulation()
    print("Simulação finalizada.")

if __name__ == '__main__':
    main()