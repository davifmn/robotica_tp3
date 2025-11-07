#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Integração com CoppeliaSim (Remote API 'sim.py'):
- Conecta no servidor (porta 19999 por padrão)
- Lê pose do robô (localização conhecida assumida)
- Lê varredura do laser (Hokuyo) via string signal (se disponível)
- Injeta ruído nas leituras (configurável)
- Atualiza Occupancy Grid
- Estratégia simples de navegação para explorar (wander + desvio de obstáculo)
- Salva periódicamente o grid em .npz para pós-processamento
"""

import os
import time
import math
import argparse
import numpy as np

# sim.py deve estar disponível no PYTHONPATH ou no mesmo diretório
try:
    import sim
except Exception as e:
    raise SystemExit(
        "Erro ao importar 'sim.py' (Remote API). "
        "Certifique-se que sim.py e a lib remota estão no PYTHONPATH ou na mesma pasta."
    ) from e

from occupancy_grid import OccupancyGrid


def v_omega_to_wheels(v: float, w: float, L: float, r: float) -> tuple[float, float]:
    # Cinemática diferencial
    wd = (v + 0.5 * L * w) / r
    we = (v - 0.5 * L * w) / r
    return wd, we


def _normalize_angle(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


class HokuyoReader:
    """
    Lê dados do Hokuyo publicados como string signal de floats empacotados.
    Convenções comuns em CoppeliaSim (pode variar por cena):
      - 'fastHokuyo_data' ou 'hokuyoData'
    O sinal geralmente contém pares (x,y) no frame do scanner.
    """
    def __init__(self, client_id: int, signals=("fastHokuyo_data", "hokuyoData", "scanData")):
        self.client_id = client_id
        self.signals = signals
        self.active_signal = None
        # inicializa streaming
        for s in self.signals:
            rc, _ = sim.simxGetStringSignal(self.client_id, s, sim.simx_opmode_streaming)
        # tenta detectar qual está ativo
        t0 = time.time()
        while time.time() - t0 < 1.0:
            for s in self.signals:
                rc, data = sim.simxGetStringSignal(self.client_id, s, sim.simx_opmode_buffer)
                if rc == sim.simx_return_ok and data:
                    self.active_signal = s
                    return
            time.sleep(0.01)

    def read_scan(self, max_range: float) -> tuple[np.ndarray, np.ndarray]:
        if self.active_signal is None:
            return np.array([]), np.array([])
        rc, data = sim.simxGetStringSignal(self.client_id, self.active_signal, sim.simx_opmode_buffer)
        if rc != sim.simx_return_ok or not data:
            return np.array([]), np.array([])
        floats = sim.simxUnpackFloats(data)
        if len(floats) < 2:
            return np.array([]), np.array([])
        # Converte pares (x, y) em (r, a)
        pts = np.array(floats, dtype=np.float32).reshape(-1, 2)
        xs, ys = pts[:, 0], pts[:, 1]
        ranges = np.hypot(xs, ys)
        angles = np.arctan2(ys, xs)
        # filtra pontos além do max_range (considera sem hit)
        mask = ranges <= max_range * 1.05
        return ranges[mask], angles[mask]


def try_get_handle(client_id: int, name: str) -> int:
    rc, handle = sim.simxGetObjectHandle(client_id, name, sim.simx_opmode_oneshot_wait)
    if rc != sim.simx_return_ok:
        raise RuntimeError(f"Falha ao obter handle de '{name}', rc={rc}")
    return handle


def main():
    parser = argparse.ArgumentParser(description="Exploração e mapeamento (Occupancy Grid) em CoppeliaSim.")
    # Grid
    parser.add_argument("--width", type=float, default=10.0, help="Largura do grid [m]")
    parser.add_argument("--height", type=float, default=10.0, help="Altura do grid [m]")
    parser.add_argument("--resolution", type=float, default=0.1, help="Tamanho da célula [m/cell]")
    # Sim
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=19999)
    parser.add_argument("--start-sim", action="store_true", help="Tenta iniciar a simulação via Remote API")
    # Robot e motores (default: Pioneer). Ajuste se usar Kobuki.
    parser.add_argument("--robot", type=str, default="Pioneer_p3dx", help="Nome do objeto base do robô")
    parser.add_argument("--left-motor", type=str, default="Pioneer_p3dx_leftMotor", help="Nome do motor esquerdo")
    parser.add_argument("--right-motor", type=str, default="Pioneer_p3dx_rightMotor", help="Nome do motor direito")
    # Cinemática (Kobuki: L=0.230, r=0.035)
    parser.add_argument("--L", type=float, default=0.230, help="Distância entre rodas [m]")
    parser.add_argument("--r", type=float, default=0.035, help="Raio da roda [m]")
    # Laser
    parser.add_argument("--max-range", type=float, default=5.0, help="Alcance máximo do laser [m]")
    parser.add_argument("--noise-range-std", type=float, default=0.0, help="Desvio padrão do ruído no range [m]")
    parser.add_argument("--noise-angle-std", type=float, default=0.0, help="Desvio padrão do ruído no ângulo [rad]")
    # Navegação
    parser.add_argument("--duration", type=float, default=60.0, help="Duração da exploração [s]")
    parser.add_argument("--v-forward", type=float, default=0.15, help="Velocidade linear [m/s]")
    parser.add_argument("--w-turn", type=float, default=0.6, help="Velocidade angular para desvio [rad/s]")
    parser.add_argument("--obs-dist", type=float, default=0.6, help="Distância de segurança para obstáculo [m]")
    # Saída
    parser.add_argument("--save-dir", type=str, default="maps", help="Diretório para salvar mapas (.npz)")
    parser.add_argument("--save-interval", type=float, default=10.0, help="Intervalo para salvar [s]")

    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    # Conexão
    print("Conectando ao CoppeliaSim...")
    sim.simxFinish(-1)
    client_id = sim.simxStart(args.host, args.port, True, True, 5000, 5)
    if client_id == -1:
        raise SystemExit("Falha na conexão com o servidor Remote API.")

    try:
        print("Conexão estabelecida.")

        if args.start_sim:
            sim.simxStartSimulation(client_id, sim.simx_opmode_oneshot_wait)

        # Handles
        robot_h = try_get_handle(client_id, args.robot)
        lm_h = try_get_handle(client_id, args.left_motor)
        rm_h = try_get_handle(client_id, args.right_motor)

        # Inicializa streams de pose
        sim.simxGetObjectPosition(client_id, robot_h, -1, sim.simx_opmode_streaming)
        sim.simxGetObjectOrientation(client_id, robot_h, -1, sim.simx_opmode_streaming)

        # Leitor de laser
        hokuyo = HokuyoReader(client_id)

        # Grid
        grid = OccupancyGrid(args.width, args.height, args.resolution)

        # Loop
        t0 = time.time()
        t_last_save = t0

        # Inicializa velocidades
        sim.simxSetJointTargetVelocity(client_id, rm_h, 0.0, sim.simx_opmode_streaming)
        sim.simxSetJointTargetVelocity(client_id, lm_h, 0.0, sim.simx_opmode_streaming)

        while True:
            now = time.time()
            if now - t0 >= args.duration:
                print("Tempo de exploração esgotado.")
                break

            # Pose do robô (buffer após streaming)
            rc_a, ang = sim.simxGetObjectOrientation(client_id, robot_h, -1, sim.simx_opmode_buffer)
            rc_p, pos = sim.simxGetObjectPosition(client_id, robot_h, -1, sim.simx_opmode_buffer)
            if rc_a != sim.simx_return_ok or rc_p != sim.simx_return_ok:
                time.sleep(0.01)
                continue
            x, y, th = float(pos[0]), float(pos[1]), float(ang[2])

            # Scan do laser
            ranges, angles = hokuyo.read_scan(args.max_range)

            # Injeta ruído (se configurado)
            if ranges.size > 0:
                if args.noise_range_std > 0.0:
                    ranges = ranges + np.random.normal(0.0, args.noise_range_std, size=ranges.shape)
                    ranges = np.clip(ranges, 0.0, args.max_range)
                if args.noise_angle_std > 0.0:
                    angles = angles + np.random.normal(0.0, args.noise_angle_std, size=angles.shape)
                    angles = np.vectorize(_normalize_angle)(angles)

                # Atualiza o grid com o scan
                grid.update_with_scan((x, y, th), ranges, angles, args.max_range)

            # Controle simples: desvio de obstáculo baseado no mínimo range
            v_cmd = args.v_forward
            w_cmd = 0.0
            if ranges.size > 0:
                rmin = float(np.min(ranges))
                if rmin < args.obs_dist:
                    v_cmd = 0.0
                    # Decide virar para o lado com maior abertura
                    left_open = np.mean(ranges[angles > 0]) if np.any(angles > 0) else rmin
                    right_open = np.mean(ranges[angles < 0]) if np.any(angles < 0) else rmin
                    w_cmd = args.w_turn if left_open >= right_open else -args.w_turn

            wd, we = v_omega_to_wheels(v_cmd, w_cmd, args.L, args.r)
            sim.simxSetJointTargetVelocity(client_id, rm_h, wd, sim.simx_opmode_streaming)
            sim.simxSetJointTargetVelocity(client_id, lm_h, we, sim.simx_opmode_streaming)

            # Salvamento periódico
            if now - t_last_save >= args.save_interval:
                stamp = time.strftime("%Y%m%d_%H%M%S")
                out_path = os.path.join(args.save_dir, f"grid_{stamp}_res{args.resolution:.3f}.npz")
                grid.save_npz(out_path)
                print(f"Mapa salvo: {out_path}")
                t_last_save = now

            time.sleep(0.03)

        # Parada
        sim.simxSetJointTargetVelocity(client_id, rm_h, 0.0, sim.simx_opmode_streaming)
        sim.simxSetJointTargetVelocity(client_id, lm_h, 0.0, sim.simx_opmode_streaming)
        sim.simxGetPingTime(client_id)

        # Salva ao final
        stamp = time.strftime("%Y%m%d_%H%M%S")
        final_path = os.path.join(args.save_dir, f"grid_final_{stamp}_res{args.resolution:.3f}.npz")
        grid.save_npz(final_path)
        print(f"Mapa final salvo: {final_path}")

    finally:
        sim.simxFinish(client_id)
        print("Conexão encerrada.")


if __name__ == "__main__":
    main()
