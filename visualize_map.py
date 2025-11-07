#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Carrega um Occupancy Grid salvo (.npz) e exporta uma imagem PNG em escala de cinza,
onde "mais escuro = maior probabilidade de ocupação".
Também pode mostrar na tela com matplotlib.

Uso:
  python visualize_map.py --input maps/grid_final_...npz --output mapa.png --show
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt

from occupancy_grid import OccupancyGrid


def main():
    parser = argparse.ArgumentParser(description="Visualização de Occupancy Grid (.npz) -> PNG")
    parser.add_argument("--input", type=str, required=True, help="Caminho do arquivo .npz salvo")
    parser.add_argument("--output", type=str, default="", help="Caminho do PNG de saída (opcional)")
    parser.add_argument("--show", action="store_true", help="Mostra janela interativa com matplotlib")
    args = parser.parse_args()

    if not os.path.isfile(args.input):
        raise SystemExit(f"Arquivo não encontrado: {args.input}")

    grid = OccupancyGrid.load_npz(args.input)
    img = grid.to_grayscale_image()

    if args.output:
        plt.imsave(args.output, img, cmap="gray", vmin=0, vmax=255)
        print(f"Imagem salva em: {args.output}")

    if args.show or not args.output:
        plt.figure(figsize=(6, 6))
        # cmap=gray já mostra 0=preto (ocupado), 255=branco (livre)
        plt.imshow(img, cmap="gray", origin="lower",
                   extent=[grid.x_min, grid.x_min + grid.spec.width_m,
                           grid.y_min, grid.y_min + grid.spec.height_m])
        plt.colorbar(label="Intensidade (0=ocupado, 255=livre)")
        plt.title(f"Occupancy Grid (res={grid.spec.resolution} m)")
        plt.xlabel("x [m]")
        plt.ylabel("y [m]")
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()
