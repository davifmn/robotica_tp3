import matplotlib.pyplot as plt
import numpy as np

def visualize_and_save_map(prob_map, filename):
    """
    Visualiza e salva a grade de ocupação como uma imagem.

    Args:
        prob_map (np.ndarray): O mapa de probabilidade (0 a 1).
        filename (str): O nome do arquivo para salvar a imagem.
    """
    # Inverte as cores: ocupado (alta prob) -> preto, livre (baixa prob) -> branco
    # A probabilidade vai de 0 (livre) a 1 (ocupado).
    # O mapa de cores 'gray' mapeia 0 para preto e 1 para branco.
    # Queremos o contrário, então usamos 1 - prob_map.
    
    plt.figure(figsize=(10, 10))
    
    # Usamos np.flipud para que a origem (0,0) do mapa fique no canto inferior esquerdo
    # Usamos .T (transposto) porque o matplotlib plota (linha, coluna) como (y, x)
    plt.imshow(1 - prob_map.T, cmap='gray', origin='lower')
    
    plt.title("Mapa de Grade de Ocupação")
    plt.xlabel("Células da Grade (X)")
    plt.ylabel("Células da Grade (Y)")
    plt.grid(False) # Desativa a grade do matplotlib
    
    # Salva a imagem
    plt.savefig(filename, bbox_inches='tight', pad_inches=0.1)
    plt.close()

if __name__ == '__main__':
    # Exemplo de uso (para teste)
    # Cria um mapa de teste simples
    test_map = np.zeros((100, 100))
    test_map[20:40, 30:70] = 0.9  # Área ocupada
    test_map[60:80, 10:30] = 0.8  # Outra área ocupada
    test_map = np.clip(test_map + np.random.rand(100, 100) * 0.2, 0, 1) # Adiciona ruído

    visualize_and_save_map(test_map, "test_map.png")
    print("Mapa de teste salvo como 'test_map.png'")