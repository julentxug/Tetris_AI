from nes_py.wrappers import JoypadSpace
import gym_tetris
from gym_tetris.actions import MOVEMENT
from piezas_2 import pieza_clase
import random
from DQN2 import DQNAgent
from datetime import datetime
from statistics import mean, median
from board import board_prop,eliminar_linea,board_peso_max
import numpy as np

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
        board_aux,elimado=eliminar_linea(board_aux,estado[0])
        estado=board_prop(board_aux)
        posiciones.append(i)
        piezas.append(nom_piez)
        estados.append(estado)
        eliminados.append(eliminado)
    for pieza_nom in giros_pieza:
        pieza.asignar_info(pieza_nom)
        x=pieza.get_x()
        x_min=-min(pieza.x)+0
        x_max=-max(pieza.x)+10
        for i in range(x_min,x_max):
            board_aux=np.copy(board)
            board_aux=pieza.colocar_pieza(board_aux,i)
            estado=board_prop(board_aux)
            board_aux,elimado=eliminar_linea(board_aux,estado[0])
            estado=board_prop(board_aux)
            posiciones.append(i)
            piezas.append(pieza_nom)
            estados.append(estado)
            eliminados.append(eliminado)

    return posiciones,estados,piezas,giros_pieza,eliminados  


env = gym_tetris.make('TetrisA-v0')
env = JoypadSpace(env, MOVEMENT)

episodes = 5000
max_steps = None
epsilon_stop_episode =2000
mem_size = 5000
discount = 0.95
batch_size = 128
epochs = 1
render_every = 50
log_every = 50
#replay_start_size = 2000
train_every = 1
n_neurons = [64, 64]
render_delay = None
activation = ['relu']

agent = DQNAgent(4)

puntuacion_max=0
for episode in range(0,episodes):
    state = env.reset()
    done = False
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
    for i in range(0,100000):
        
        if done:
            informacion=env.get_info()
            antiguo_statistics=informacion['statistics']
            pos=5
            u=0
            break
 
        env.render()
        informacion=env.get_info()
        nom_piez=informacion['current_piece']
        board=env.board()
        if pieza_colocada:
            pieza_colocada=False
            pos=5
            u=-1
            ant_nom_piez=''
            #estado=board_prop(board)  
            posiciones,estados,piezas,giros,eliminados=posibles_estados(board,nom_piez)
            mejor_estado,k=agent.best_state(estados)[:]
            if k<0:
                k=0
                for estado_i in estados:
                    if estado_i==mejor_estado:
                        break
                    k=k+1
            pos_objetivo=posiciones[k]
            pieza=piezas[k]
            eliminado=eliminados[k]
            

        if nom_piez!=pieza and not done:
            state, reward, done, info = env.step(1)  
            nom_piez=informacion['current_piece']
            accion=0
        elif pos>pos_objetivo and not done:
            state, reward, done, info = env.step(6)
            pos=pos-1
            accion=0
        elif pos<pos_objetivo and not done:
            state, reward, done, info = env.step(3)
            pos=pos+1
            accion=0
        elif not done and not eliminado:
            state, reward, done, info = env.step(9)
            accion=9
        else:
            accion=0
        if not done:
            state, reward, done, info = env.step(accion)
        env.render()
        informacion=env.get_info()
        if antiguo_statistics!=informacion['statistics']:
            premio=estado[0]-mejor_estado[0]+estado[1]-mejor_estado[1]+estado[2]-mejor_estado[2]+estado[3]-mejor_estado[3]
            informacion=env.get_info()
            antiguo_statistics=informacion['statistics']
            board=env.board()
            mejor_estado=board_prop(board)[:]
            if lineas_completadas<informacion['number_of_lines'] and not done:
               premio=premio+40*(-lineas_completadas+informacion['number_of_lines'])
               
               lineas_completadas=informacion['number_of_lines']
               state, reward, done, info = env.step(0)
               agent.add_to_memory(estado, mejor_estado, premio, done)

            agent.add_to_memory(estado, mejor_estado, premio, done)
            estado=mejor_estado[:]
            pieza_colocada=True
            eliminado=False

    puntuacion=informacion['score']
    print(puntuacion)
    #if puntuacion>puntuacion_max*0.5:
    if puntuacion>puntuacion_max:
        puntuacion_max=puntuacion
        agent.model.save('modelo_max.model')
    if episode % train_every == 0:
        agent.train(batch_size, epochs)


env.close()

