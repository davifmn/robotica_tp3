import time
from coppeliasim_zmqremoteapi_client import RemoteAPIClient

# lista de nomes comuns/variantes para tentar encontrar o Kobuki e os motores/laser
COMMON_BASE_NAMES = [
    "/kobuki", "kobuki", "/Turtlebot2", "turtlebot2", "kobuki_base", "base_link"
]
COMMON_LEFT_NAMES = [
    "/kobuki/wheel_left_joint", "wheel_left_joint", "wheel_left", "motor_left",
    "/Turtlebot2/motor_left", "left_wheel", "left_wheel_joint"
]
COMMON_RIGHT_NAMES = [
    "/kobuki/wheel_right_joint", "wheel_right_joint", "wheel_right", "motor_right",
    "/Turtlebot2/motor_right", "right_wheel", "right_wheel_joint"
]
COMMON_LASER_NAMES = [
    "/kobuki/fastHokuyo", "fastHokuyo", "hokuyo", "Hokuyo", "/kobuki/hokuyo_link",
    "laser", "hokuyo_link", "/Turtlebot2/Hokuyo"
]

def try_get_alias_or_name(sim, handle):
    """Tenta obter o nome/alias de um handle usando métodos diferentes."""
    name = None
    try:
        name = sim.getObjectAlias(handle, 0)
    except Exception:
        pass
    if not name:
        try:
            name = sim.getObjectName(handle)
        except Exception:
            pass
    return name

def safe_get_objects_in_tree(sim):
    """Tenta chamar getObjectsInTree com algumas assinaturas possíveis e retorna lista de handles."""
    attempts = []
    # tentativa 1: usar constantes do sim (se existirem)
    try:
        handles = sim.getObjectsInTree(sim.handle_scene, sim.handle_all, 0)
        attempts.append(("handle_scene/handle_all/0", handles))
    except Exception as e:
        attempts.append(("handle_scene/handle_all/0 raised", e))

    # tentativa 2: sem o terceiro argumento
    try:
        handles = sim.getObjectsInTree(sim.handle_scene, sim.handle_all)
        attempts.append(("handle_scene/handle_all", handles))
    except Exception as e:
        attempts.append(("handle_scene/handle_all raised", e))

    # tentativa 3: tentar obter handle da cena explicitamente (algumas API wrappers precisam)
    try:
        scene_handle = sim.getObjectHandle(sim.handle_scene)
        handles = sim.getObjectsInTree(scene_handle, sim.handle_all, 0)
        attempts.append(("getObjectHandle(handle_scene)", handles))
    except Exception as e:
        attempts.append(("getObjectHandle(handle_scene) raised", e))

    # escolher o primeiro que retornou uma lista
    for tag, result in attempts:
        if isinstance(result, (list, tuple)):
            return result
    # se nada funcionou, retornar lista vazia e imprimir as tentativas para debug
    print("Nenhuma chamada getObjectsInTree funcionou. Detalhes das tentativas:")
    for tag, result in attempts:
        print(f"  - {tag}: {type(result)} -> {result}")
    return []

def find_handles_by_iteration(sim):
    handles = safe_get_objects_in_tree(sim)
    print(f"Total de handles obtidos: {len(handles)}")
    names = []
    for i, h in enumerate(handles):
        try:
            name = try_get_alias_or_name(sim, h)
            if not name:
                # por fim tenta sim.getObjectAlias com flag 1/2 se disponível
                try:
                    name = sim.getObjectAlias(h, 1)
                except Exception:
                    pass
        except Exception:
            name = None
        names.append((h, name))
    # imprimir os primeiros 200 nomes (ou menos)
    print("\n--- Primeiros objetos encontrados (handle, nome) ---")
    for h, name in names[:200]:
        print(f"{h} -> {name}")
    return names

def try_common_names(sim):
    found = {}
    def try_list(lst, label):
        for candidate in lst:
            try:
                h = sim.getObjectHandle(candidate)
                found[label] = (candidate, h)
                print(f"Encontrado {label}: '{candidate}' -> handle {h}")
                return
            except Exception:
                # não encontrou esse nome, continua
                pass
        print(f"Não encontrou {label} entre as variantes testadas.")
    print("\n--- Testando nomes comuns diretamente com getObjectHandle ---")
    try_list(COMMON_BASE_NAMES, "base")
    try_list(COMMON_LEFT_NAMES, "left_motor")
    try_list(COMMON_RIGHT_NAMES, "right_motor")
    try_list(COMMON_LASER_NAMES, "laser")
    return found

def main():
    client = RemoteAPIClient()
    sim = client.getObject('sim')

    # garantir sim parada antes de iniciar para evitar comportamento estranho
    try:
        sim.stopSimulation()
    except Exception:
        pass
    time.sleep(0.5)

    # start (algumas cenas precisam estar rodando para criar objetos dinamicamente)
    try:
        sim.startSimulation()
        print("Simulação iniciada.")
    except Exception as e:
        print("Não foi possível iniciar a simulação:", e)

    # busca por iteração
    names = find_handles_by_iteration(sim)

    # busca por nomes comuns
    found_common = try_common_names(sim)

    # se encontrou handles via nomes comuns, testar movimento (com segurança)
    if "left_motor" in found_common and "right_motor" in found_common:
        left_h = found_common["left_motor"][1]
        right_h = found_common["right_motor"][1]
        try:
            print("\nTestando movimento rápido por 1.5s...")
            sim.setJointTargetVelocity(left_h, 3.0)
            sim.setJointTargetVelocity(right_h, 3.0)
            time.sleep(1.5)
            sim.setJointTargetVelocity(left_h, 0)
            sim.setJointTargetVelocity(right_h, 0)
            print("Movimento de teste concluído.")
        except Exception as e:
            print("Erro ao tentar aplicar velocidade nos joints:", e)
    else:
        print("\nNão foi possível localizar ambos os motores via nomes comuns; veja a lista impressa acima para procurar nomes manualmente na hierarquia da cena.")

    # parar sim ao final
    try:
        sim.stopSimulation()
    except Exception:
        pass
    print("Fim do debug.")

if __name__ == "__main__":
    main()
