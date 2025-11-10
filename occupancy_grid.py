import numpy as np

class OccupancyGrid:
    def __init__(self, x_min, x_max, y_min, y_max, resolution):
        """
        Inicializa a grade de ocupação.

        Args:
            x_min (float): Coordenada X mínima do mapa.
            x_max (float): Coordenada X máxima do mapa.
            y_min (float): Coordenada Y mínima do mapa.
            y_max (float): Coordenada Y máxima do mapa.
            resolution (float): Resolução da grade (tamanho de cada célula).
        """
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max
        self.resolution = resolution

        # Calcula o tamanho da grade em número de células
        self.x_size = int(np.ceil((x_max - x_min) / resolution))
        self.y_size = int(np.ceil((y_max - y_min) / resolution))

        # A grade armazena a log-probabilidade (log-odds) de ocupação
        # Inicializa com 0, que corresponde a uma probabilidade de 0.5
        self.grid = np.zeros((self.x_size, self.y_size))
        
        # Log-odds para ocupado e livre
        self.l_occ = np.log(0.85 / (1 - 0.85)) # Probabilidade de 85% para ocupado
        self.l_free = np.log(0.15 / (1 - 0.15)) # Probabilidade de 15% para livre
        
        # Limites para a log-probabilidade para evitar valores infinitos
        self.l_max = 5
        self.l_min = -5

    def to_grid_coords(self, x, y):
        """Converte coordenadas do mundo para coordenadas da grade."""
        i = int((x - self.x_min) / self.resolution)
        j = int((y - self.y_min) / self.resolution)
        return i, j

    def is_in_bounds(self, i, j):
        """Verifica se as coordenadas da grade estão dentro dos limites."""
        return 0 <= i < self.x_size and 0 <= j < self.y_size

    def update(self, robot_pose, laser_readings):
        """
        Atualiza a grade de ocupação com base em uma nova leitura do laser.

        Args:
            robot_pose (tuple): Posição (x, y, theta) do robô.
            laser_readings (list): Lista de pontos (x, y) detectados pelo laser no referencial do mundo.
        """
        rx, ry, r_theta = robot_pose
        
        # Converte a posição do robô para coordenadas da grade
        robot_i, robot_j = self.to_grid_coords(rx, ry)

        # Atualiza as células ocupadas
        for point_x, point_y in laser_readings:
            # Ponto final do raio laser
            end_i, end_j = self.to_grid_coords(point_x, point_y)
            if self.is_in_bounds(end_i, end_j):
                self.grid[end_i, end_j] = min(self.l_max, self.grid[end_i, end_j] + self.l_occ)

            # Células livres ao longo do raio do laser (algoritmo de Bresenham)
            self.bresenham_line(robot_i, robot_j, end_i, end_j)
            
    def bresenham_line(self, x0, y0, x1, y1):
        """

        Traça uma linha entre dois pontos na grade e marca as células como livres.
        Implementação do algoritmo de Bresenham.
        """
        dx = abs(x1 - x0)
        dy = -abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx + dy
        
        while True:
            # Marca a célula atual como livre
            if self.is_in_bounds(x0, y0):
                self.grid[x0, y0] = max(self.l_min, self.grid[x0, y0] + self.l_free)

            if x0 == x1 and y0 == y1:
                break
                
            e2 = 2 * err
            if e2 >= dy:
                err += dy
                x0 += sx
            if e2 <= dx:
                err += dx
                y0 += sy

    def get_probability_map(self):
        """Converte a grade de log-odds para um mapa de probabilidade."""
        # p = 1 - 1 / (1 + exp(l))
        return 1 - 1 / (1 + np.exp(self.grid))
