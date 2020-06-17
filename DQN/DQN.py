from keras.models import Sequential, save_model, load_model
from keras.layers import Dense
from collections import deque
import numpy as np
import random


class DQNAgent:

    '''
    Argumentos:
        tam_estado (int): Tamaño del estado
        tam_memoria (int): Tamaño de la memoria donde se guardan los movimientos y sus recompensas
        descuento (float): La importancia de los premios futuros respecto a los actuales
        epsilon (float): El valor que indica la probabilidad de realizar un movimiento aleatorio
        epsilon_min (float): El valor minimo que puede alcanzar epsilon
        epsilon_stop_episodio (int): El numero de episodio en el que epsilon se deja de usar
        neuronas (list(int)): Lista con el numero de neuronas
        activaciones (list): Lista con las activaciones a usar en la red neuronal
        perdida (obj): Funcion de perdida
        optimizador (obj): optimizador usado
        tam_inicio_entr: Tamaño minimo de la memoria alcanzado para poder entrenar
    '''

    def __init__(self, tam_estado=4, tam_memoria=10000, descuento=0.95,
                 epsilon=1, epsilon_min=0.05, epsilon_stop_episodio=200,
                 neuronas=[32,32], activaciones=['relu', 'relu', 'linear'],
                 perdida='mse', optimizador='adam', tam_inicio_entr=None):

        assert len(activaciones) == len(neuronas) + 1

        self.tam_estado = tam_estado
        self.memoria = deque(maxlen=tam_memoria)
        self.descuento = descuento
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = (self.epsilon - self.epsilon_min) / (epsilon_stop_episodio)
        self.neuronas = neuronas
        self.activaciones = activaciones
        self.perdida = perdida
        self.optimizador = optimizador
        if not tam_inicio_entr:
            tam_inicio_entr = tam_memoria/10
        self.tam_inicio_entr = tam_inicio_entr
        self.modelo = self._crear_modelo()

    # Creamos el modelo de la red neuronal

    def _crear_modelo(self):
        
        modelo = Sequential()
        modelo.add(Dense(self.neuronas[0], input_dim=self.tam_estado, activation=self.activaciones[0]))

        for i in range(1, len(self.neuronas)):
            modelo.add(Dense(self.neuronas[i], activation=self.activaciones[i]))

        modelo.add(Dense(1, activation=self.activaciones[-1]))

        modelo.compile(loss=self.perdida, optimizer=self.optimizador)
        
        return modelo

    # Añadimos la jugada a la memoria
    
    def add_memoria(self, estado_actual, siguiente_estado, premio, terminado):

        self.memoria.append((estado_actual, siguiente_estado, premio, terminado))


    # Predice la puntuacion del estado dado.

    def pred_estado(self, estado):

        return self.modelo.predict(estado)[0]




    # Dada una lista de estados devuelve el mejor de ellos


    def el_mejor_estado(self, estados):

        max_valor = None
        mejor_estado = None
        k=-1

        if random.random() <= self.epsilon:
            return random.choice(list(estados)),k,0

        else:
            for estado in estados:
                k=k+1
                valor = self.pred_estado(np.reshape(estado, [1, self.tam_estado]))
                if not max_valor or valor > max_valor:
                    max_valor = valor
                    mejor_estado = estado
                    mejor_k=k
                
        
        return mejor_estado,mejor_k,max_valor


    # Entrenamos al agente

    def train(self, batch_size, epochs, puntuacion, q_actual):

        n = len(self.memoria)

        if n >= self.tam_inicio_entr and n >= batch_size:

            batch = random.sample(self.memoria, batch_size)

            # Obtenemos la puntuacion esperada de los siguientes estados
            siguientes_estados = np.array([x[1] for x in batch])
            siguientes_qs = [x[0] for x in self.modelo.predict(siguientes_estados)]

            x = []
            y = []

            # Construimos una estructura xy para luego poder amoldarla a la red
            for i, (estado, _, premio, terminado) in enumerate(batch):
                if not terminado:

                    new_q = premio + self.descuento * siguientes_qs[i]
                else:
                    new_q = premio
                if new_q>0:
                    new_q=new_q*(puntuacion+1)


                y.append(new_q)

                x.append(estado)



            self.modelo.fit(np.array(x), np.array(y), batch_size=batch_size, epochs=epochs, verbose=0)


            if self.epsilon > self.epsilon_min:
                self.epsilon -= self.epsilon_decay
