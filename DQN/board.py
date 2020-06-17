import numpy as np

# Obtenemos las diferentes propiedades del tablero y las devolvemos en una lista,
# la cual sera el estado del tablero

def board_prop(board):
    pesos=board_pesos(board)
    peso_max=board_peso_max(pesos)
    #casillas_para_completar=casillas_completar(board,19-peso_max)
    diferencia=board_diferencia_pesos(pesos)
    agujeros=board_agujeros(board)
    sum_pesos=np.sum(pesos)
    return[peso_max,diferencia,sum_pesos,agujeros]


# Calculamos el peso de cada columna, es decir, la altura maxima que tienen las casillas
# llenas(con pieza colocada) 

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

# Calculamos la altura máxima de todo el tablero

def board_peso_max(pesos):
    peso_max=0
    for peso in pesos:
       if peso>peso_max:
           peso_max=peso
    return peso_max

def board_peso_min(pesos):
    peso_min=21
    for peso in pesos:
       if peso<peso_min:
           peso_min=peso
    return peso_min

# Deolvemos el peso maximo de la columna indicada

def board_peso_pos(board,pos):
    peso=0
    for y in range (0,20):
        if board[y][pos]!=239:
            peso=20-y
            break
    return peso

# Calculamos las diferencias totales que hay entre los pesos de las lineas. 
# Es decir: diferencia(columna1,columna2)+diferencia(columna2,columna3)+...

def board_diferencia_pesos(pesos):
    dif_pesos=0
    for i in range(0,9):
        dif_pesos=abs(pesos[i]-pesos[i+1])+dif_pesos
    return dif_pesos

#Comprobamos si una linea ha sido completada

def linea_completada(linea):
    for i in linea:
        if i==239:
            return False
    return True

# Simulamos la eliminacion de una linea cuando esta es completada

def eliminar_linea(board,peso):
    k=0
    eliminado=False
    while k<4 and (peso-k)>0:
       if linea_completada(board[19-(peso-k)]):
           np.delete(board,19-(peso-k))
           eliminado=True
       k=k+1
    return board

# Calculamos el numero de agujeros que hay en el tablero

def board_agujeros(board):
    agujeros=0
    for x in range(0,10):
        agujero=0
        for y in range(1,20):
            if board[y][x]==239 and board[y-1][x]!=239:
                agujero=agujero+1
        agujeros=agujeros+agujero
    return agujeros

def casillas_completar(board,x):
    cont=0
    for i in range(0,10):
        if board[x][i]==239:
            cont=cont+1
    return cont

def casillas_completar_totales(board,peso_max):
    total=0
    for y in range (19-peso_max,20):
        total=total+casillas_completar(board,y)
    return total


