from nes_py.wrappers import JoypadSpace
import gym_tetris
from gym_tetris.actions import MOVEMENT
from piezas import pieza_clase
import random
from DQN import DQNAgent
from datetime import datetime
from statistics import mean, median
from board import board_prop,eliminar_linea,board_peso_max
import numpy as np

# Dado el estado del tablero actual (es decir, las piezas ya colocadas),
# y la pieza a colocar calcula todos los estados posibles simulando la colocacion de dicha
# pieza en todas las posiciones posibles y todos los giros posibles.

def posibles_estados(board,nom_piez):
    posiciones=[]
    piezas=[]
    estados=[]
    eliminados=[]
    pieza=pieza_clase()
    pieza.asignar_info(nom_piez)
    giros_pieza=pieza.get_giro()
    x=pieza.get_x()
    x_min=-min(pieza.x)+0
    x_max=-max(pieza.x)+10
    for i in range(x_min,x_max):
        board_aux=np.copy(board)
        board_aux=pieza.colocar_pieza(board_aux,i)
        estado=board_prop(board_aux)
        board_aux=eliminar_linea(board_aux,estado[0])
        estado=board_prop(board_aux)
        posiciones.append(i)
        piezas.append(nom_piez)
        estados.append(estado)

    for pieza_nom in giros_pieza:
        pieza.asignar_info(pieza_nom)
        x=pieza.get_x()
        x_min=-min(pieza.x)+0
        x_max=-max(pieza.x)+10
        for i in range(x_min,x_max):
            board_aux=np.copy(board)
            board_aux=pieza.colocar_pieza(board_aux,i)
            estado=board_prop(board_aux)
            board_aux=eliminar_linea(board_aux,estado[0])
            estado=board_prop(board_aux)
            posiciones.append(i)
            piezas.append(pieza_nom)
            estados.append(estado)


    return posiciones,estados,piezas,giros_pieza 

# Obtenemos el mejor estado calculando la diferencia entre el estado actual
# y los posibles estados

def best_estado(estado_actual,estados,i):
    premio_max=-1000
    k=0
    k_max=0
    estado_max=[]
    for estado in estados:
        premio=(estado_actual[0]-estado[0])*i[0]+(estado_actual[1]-estado[1])*i[1]+(estado_actual[2]-estado[2])*i[2]+(estado_actual[3]-estado[3])*i[3]
        if premio>premio_max:
            premio_max=premio
            estado_max=estado[:]
            k_max=k
        k=k+1
    return estado_max,k_max

# Cambiando el peso de los valores del estado (4 en total) obtenemos 4 mejores estados.

def best_estados(estado_actual,estados):
    mejores_estados=[]
    ks=[]
    valores=[[1.5,1,1,1],[1,1.5,1,1],[1,1,1.5,1],[1,1,1,1.5]]
    for i in valores:
        estado,k=best_estado(estado_actual,estados,i)
        mejores_estados.append(estado)
        ks.append(k)
    return mejores_estados,ks

env = gym_tetris.make('TetrisA-v0')
env = JoypadSpace(env, MOVEMENT)

episodios = 5000               # Numero de partidas a realizar
max_steps = None               # Numero de pasos a realizar en cada partida 

entrenar_cada = 1                # Numero que indica cada cuantos episodios entrenar 
file = open("resultados_Combine.txt", "w")


agent = DQNAgent()

puntuacion_max=0
for episodio in range(0,episodios):
    state = env.reset()
    terminado = False
    pos=5
    piez=pieza_clase()
    informacion=env.get_info()
    antiguo_statistics=informacion['statistics']

    estado=[0,0,0,0]
    pieza_colocada=True
    u=-1
    ant_nom_piez=''
    lineas_completadas=0
    eliminado=False
    
    # Empieza un episodio
    for i in range(0,100000):
        
        #Si la partida se ha acabado salimos del episodio

        if terminado:
            break
 
        env.render()
        informacion=env.get_info()
        nom_piez=informacion['current_piece']
        board=env.board()
     
        # Si acabamos de empezar o acabamos de colocar una pieza
        # calculamos cual es la posicion de la siguiente pieza 
        if pieza_colocada:
            pieza_colocada=False
            pos=5
            u=-1
            ant_nom_piez=''
            estado=board_prop(board)  
            posiciones,estados,piezas,giros=posibles_estados(board,nom_piez)
            mejores_estados,mejores_k=best_estados(estado,estados)[:]
            mejor_estado,j=agent.el_mejor_estado(mejores_estados)[:]     # En este caso hacemos calcular al agente el mejor 
                                                                    # estado entre 4 estados
            k=mejores_k[j]
            if k<0:
                k=0
                for estado_i in estados:
                    if estado_i==mejor_estado:
                        break
                    k=k+1
            pos_objetivo=posiciones[k]
            pieza=piezas[k]

            

       # Giramos la pieza hasta que tengamos el giro que queramos
        if nom_piez!=pieza and not terminado:
            state, reward, terminado, info = env.step(1)  
            nom_piez=informacion['current_piece']
            accion=0
        # Movemos la pieza hasta que este en la posicion que queramos
        elif pos>pos_objetivo and not terminado:
            state, reward, terminado, info = env.step(6)
            pos=pos-1
            accion=0
        elif pos<pos_objetivo and not terminado:
            state, reward, terminado, info = env.step(3)
            pos=pos+1
            accion=0
        # Una vez que esta donde queremos lo movemos rapidamente hacia abajo para
        # su colocacion
        elif not terminado:
            state, reward, terminado, info = env.step(9)
            accion=9
        else:
            accion=0
        if not terminado:
            state, reward, terminado, info = env.step(accion)
        env.render()
        informacion=env.get_info()

        # Comprobamos si la pieza ya ha sido colocada, y en caso afirmativo calculamos la recompensa
        # y añadimos los datos a la  memoria

        if antiguo_statistics!=informacion['statistics']:

            premio=0
            informacion=env.get_info()
            antiguo_statistics=informacion['statistics']
            board=env.board()
            mejor_estado=board_prop(board)[:]
            if lineas_completadas<informacion['number_of_lines'] and not terminado:
               premio=premio+40*(-lineas_completadas+informacion['number_of_lines'])
               
               lineas_completadas=informacion['number_of_lines']
               state, reward, terminado, info = env.step(0)
               agent.add_memoria(estado, mejor_estado, premio, terminado)

            agent.add_memoria(estado, mejor_estado, premio, terminado)
            estado=mejor_estado[:]
            pieza_colocada=True
            eliminado=False

    puntuacion=informacion['score']
    file.write(str(puntuacion) + ",")

    if puntuacion>puntuacion_max:
        puntuacion_max=puntuacion
        agent.modelo.save('modelo_max.modelo')
    if episodio % entrenar_cada== 0:
        agent.train(batch_size=128, epochs=1)

file.close()
env.close()

