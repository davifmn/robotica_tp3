#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import numpy as np
from dataclasses import dataclass


def _logit(p: float) -> float:
    p = min(max(p, 1e-6), 1 - 1e-6)
    return math.log(p / (1.0 - p))


def _sigmoid(l: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-l))


@dataclass
class GridSpec:
    width_m: float
    height_m: float
    resolution: float  # meters per cell


class OccupancyGrid:
    """
    Occupancy Grid com log-odds, atualização via modelo inverso do sensor:
    - Células ao longo do feixe: livres
    - Endpoint (se bateu em obstáculo) : ocupada
    """
    def __init__(
        self,
        width_m: float,
        height_m: float,
        resolution: float,
        p0: float = 0.5,
        p_occ: float = 0.7,
        p_free: float = 0.3,
        l_min: float = -5.0,
        l_max: float = 5.0,
    ):
        self.spec = GridSpec(width_m, height_m, resolution)
        self.nx = int(round(width_m / resolution))
        self.ny = int(round(height_m / resolution))
        self.x_min = -width_m / 2.0
        self.y_min = -height_m / 2.0

        self.l0 = _logit(p0)
        self.l_occ = _logit(p_occ)
        self.l_free = _logit(p_free)
        self.l_min = l_min
        self.l_max = l_max

        self.logodds = np.full((self.ny, self.nx), self.l0, dtype=np.float32)

    # Conversões
    def world_to_grid(self, x: float, y: float) -> tuple[int, int]:
        i = int((x - self.x_min) / self.spec.resolution)
        j = int((y - self.y_min) / self.spec.resolution)
        return i, j  # i -> coluna (x), j -> linha (y)

    def grid_to_world(self, i: int, j: int) -> tuple[float, float]:
        x = self.x_min + (i + 0.5) * self.spec.resolution
        y = self.y_min + (j + 0.5) * self.spec.resolution
        return x, y

    def clamp(self):
        np.clip(self.logodds, self.l_min, self.l_max, out=self.logodds)

    def get_probabilities(self) -> np.ndarray:
        return _sigmoid(self.logodds)

    # Ray tracing (Bresenham)
    @staticmethod
    def _bresenham(i0: int, j0: int, i1: int, j1: int):
        """Gera células (i, j) ao longo do segmento, incluindo início e fim."""
        di = abs(i1 - i0)
        dj = abs(j1 - j0)
        si = 1 if i0 < i1 else -1
        sj = 1 if j0 < j1 else -1
        err = di - dj
        i, j = i0, j0
        while True:
            yield i, j
            if i == i1 and j == j1:
                break
            e2 = 2 * err
            if e2 > -dj:
                err -= dj
                i += si
            if e2 < di:
                err += di
                j += sj

    def _in_bounds(self, i: int, j: int) -> bool:
        return 0 <= i < self.nx and 0 <= j < self.ny

    def update_with_scan(
        self,
        robot_pose: tuple[float, float, float],
        ranges: np.ndarray,
        angles: np.ndarray,
        max_range: float,
        mark_endpoints_as_occ: bool = True,
    ):
        """
        Atualiza o grid com um conjunto de feixes (ranges, angles), onde:
        - robot_pose = (x, y, theta) no mundo.
        - angles são relativos ao robô; endpoint = pose + range*[cos(theta+angle), sin(theta+angle)].
        """
        x_r, y_r, th_r = robot_pose
        i_r, j_r = self.world_to_grid(x_r, y_r)
        if not self._in_bounds(i_r, j_r):
            # Robô fora do grid
            return

        for r, a in zip(ranges, angles):
            # Ponto final do feixe (limitado ao max_range)
            rr = min(max(r, 0.0), max_range)
            x_end = x_r + rr * math.cos(th_r + a)
            y_end = y_r + rr * math.sin(th_r + a)
            i_end, j_end = self.world_to_grid(x_end, y_end)

            # Percorre células do robô até endpoint
            ray_cells = list(self._bresenham(i_r, j_r, i_end, j_end))
            if not ray_cells:
                continue

            # Marca células livres ao longo do caminho (exceto endpoint)
            for (ii, jj) in ray_cells[:-1]:
                if self._in_bounds(ii, jj):
                    self.logodds[jj, ii] += self.l_free

            # Marca endpoint como ocupado se realmente houve retorno (< max_range)
            if mark_endpoints_as_occ and (r < max_range * 0.995):
                ii, jj = ray_cells[-1]
                if self._in_bounds(ii, jj):
                    self.logodds[jj, ii] += self.l_occ

        self.clamp()

    # Persistência
    def save_npz(self, path: str):
        np.savez_compressed(
            path,
            logodds=self.logodds,
            width_m=self.spec.width_m,
            height_m=self.spec.height_m,
            resolution=self.spec.resolution,
            x_min=self.x_min,
            y_min=self.y_min,
            l0=self.l0,
            l_occ=self.l_occ,
            l_free=self.l_free,
            l_min=self.l_min,
            l_max=self.l_max,
        )

    @staticmethod
    def load_npz(path: str) -> "OccupancyGrid":
        data = np.load(path)
        grid = OccupancyGrid(
            float(data["width_m"]),
            float(data["height_m"]),
            float(data["resolution"]),
        )
        grid.logodds = data["logodds"]
        grid.x_min = float(data["x_min"])
        grid.y_min = float(data["y_min"])
        grid.l0 = float(data["l0"])
        grid.l_occ = float(data["l_occ"])
        grid.l_free = float(data["l_free"])
        grid.l_min = float(data["l_min"])
        grid.l_max = float(data["l_max"])
        return grid

    # Gera imagem em escala de cinza (uint8): 0=preto (ocupado), 255=branco (livre)
    def to_grayscale_image(self) -> np.ndarray:
        p = self.get_probabilities()
        # Mais escuro = maior probabilidade de ocupação
        img = (1.0 - p) * 255.0
        return np.clip(img, 0, 255).astype(np.uint8)
