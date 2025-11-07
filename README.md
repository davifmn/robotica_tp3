Como executar testes de resolução (exemplos):

Cenário estático, sem ruído:
0.01 m/célula:
python sim_integration.py --resolution 0.01 --duration 60 --start-sim
0.1 m/célula:
python sim_integration.py --resolution 0.1 --duration 60 --start-sim
0.5 m/célula:
python sim_integration.py --resolution 0.5 --duration 60 --start-sim
Visualização:
python visualize_map.py --input maps/grid_final_YYYYmmdd_HHMMSS_res0.100.npz --output mapa_res010.png --show


Para usar o Kobuki:
Ajuste os nomes dos motores com --left-motor e --right-motor para os nomes usados na sua cena do CoppeliaSim.
Confirme L=0.230 e r=0.035 (já padrão nos argumentos).
Certificar de que o laser Hokuyo da cena publique um string signal suportado
