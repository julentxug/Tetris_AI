import numpy as np

def board_prop(board):
    pesos=board_pesos(board)
    peso_max=board_peso_max(pesos)
    diferencia=board_diferencia_pesos(pesos)
    agujeros=board_agujeros(board)
    sum_pesos=np.sum(pesos)
    return[float(peso_max), float(diferencia), float(agujeros), float(sum_pesos)]


def board_pesos(board):
    pesos=[]
    for x in range(0,10):
        peso=0
        k=0
        for y in board:
            if y[x]!=239:
                peso=19-k
                break
            k=k+1
        pesos.append(peso)
    return pesos


def board_peso_max(pesos):
    peso_max=0
    for peso in pesos:
       if peso>peso_max:
           peso_max=peso
    return peso_max

def board_diferencia_pesos(pesos):
    dif_pesos=0
    for i in range(0,9):
        dif_pesos=abs(pesos[i]-pesos[i+1])+dif_pesos
    return dif_pesos

def board_peso_pos(board,pos):
    peso=0
    for y in range (0,20):
        if board[y][pos]!=239:
            peso=20-y
            break
    return peso

def linea_completada(linea):
    for i in linea:
        if i==239:
            return False
    return True

def eliminar_linea(board,peso):
    k=0
    eliminado=False
    while k<4 and (peso-k)>0:
       if linea_completada(board[19-(peso-k)]):
           np.delete(board,19-(peso-k))
           eliminado=True
       k=k+1
    return board,eliminado

def board_agujeros(board):
    agujeros=0
    for x in range(0,10):
        agujero=0
        for y in range(1,20):
            if board[y][x]==239 and board[y-1][x]!=239:
                agujero=agujero+1
        agujeros=agujeros+agujero
    return agujeros
