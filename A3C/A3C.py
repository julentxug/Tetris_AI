import os


from nes_py.wrappers import JoypadSpace
import gym_tetris
from gym_tetris.actions import MOVEMENT
from board import board_prop,eliminar_linea,board_peso_max
import threading
import gym
import multiprocessing
import numpy as np
from queue import Queue
import argparse
import matplotlib.pyplot as plt


import tensorflow as tf
from tensorflow.python import keras
from tensorflow.python.keras import layers

tf.enable_eager_execution()
file = open("resultados_A3C.txt", "w")
episodios=100



class ActorCriticModel(keras.Model):
  def __init__(self, state_size, action_size):
    super(ActorCriticModel, self).__init__()
    self.state_size = state_size
    self.action_size = action_size
    self.dense1 = layers.Dense(40, activation='relu')
    self.policy_logits = layers.Dense(action_size)
    self.dense2 = layers.Dense(40, activation='relu')
    self.values = layers.Dense(1)

  def call(self, inputs):
    # Forward pass
    x = self.dense1(inputs)
    logits = self.policy_logits(inputs)
    v1 = self.dense2(inputs)
    values = self.values(v1)
    return logits, values

# Funcion para guardar los resultados obtenidos y ver el progreso del entrenamiento

def record(episode,
           episode_reward,
           worker_idx,
           global_ep_reward,
           result_queue,
           total_loss,
           num_steps):
  
  if global_ep_reward == 0:
    global_ep_reward = episode_reward
  else:
    global_ep_reward = global_ep_reward * 0.99 + episode_reward * 0.01
  
  file.write(str(episode_reward) + ",")
  if episode % 10==0:
    result_queue.put(episode_reward)
  return global_ep_reward



# Nuestro agente principal, el que tiene el modelo global y a los agentes trabajadores

class MasterAgent():
  def __init__(self):


    self.save_dir = '/'

    env = gym_tetris.make('TetrisA-v0')
    self.state_size = 4
    self.action_size = 40
    self.opt = tf.train.AdamOptimizer(0.001, use_locking=True)
    print(self.state_size, self.action_size)

    self.global_model = ActorCriticModel(self.state_size, self.action_size)  # global network
    self.global_model(tf.convert_to_tensor(np.random.random((1, self.state_size)), dtype=tf.float32))

  def train(self):
    
    res_queue = Queue()

    workers = [Worker(self.state_size,
                      self.action_size,
                      self.global_model,
                      self.opt, res_queue,
                      i, game_name='Tetris',
                      save_dir=self.save_dir) for i in range(multiprocessing.cpu_count())]

    for i, worker in enumerate(workers):
      print("Starting worker {}".format(i))
      worker.start()

    moving_average_rewards = []  # record episode reward to plot
    while True:
      reward = res_queue.get()
      if reward is not None:
        moving_average_rewards.append(reward)
      else:
        break
    [w.join() for w in workers]

    plt.plot(moving_average_rewards)
    plt.ylabel('Moving average ep reward')
    plt.xlabel('Step')
    plt.savefig(os.path.join(self.save_dir,
                             '{} Moving Average.png'.format('CartPole-v0')))
    plt.show()

  def play(self):
    env = gym_tetris.make('TetrisA-v0')
    env = JoypadSpace(env, MOVEMENT)
    state = env.reset()
    model = self.global_model
    model_path = os.path.join(self.save_dir, 'model_{}.h5'.format('Tetris'))
    print('Loading model from: {}'.format(model_path))
    model.load_weights(model_path)
    done = False
    step_counter = 0
    reward_sum = 0
    pieza_colocada=True
    informacion=env.get_info()
    antiguo_statistics=informacion['statistics']
    state=[0,0,0,0]
    while not done:
      env.render()
      if pieza_colocada:
        pieza_colocada=False
        pos=5
        giro=0
        u=-1
        state=[state]
        policy, value = model(tf.convert_to_tensor(state, dtype=tf.float32))
        policy = tf.nn.softmax(policy)
        action = np.argmax(policy)
        pos_objetivo=action  % 10
        giro_objetivo=action // 10
      if (giro % giro_objetivo)!=0 and not done:
        state, reward, done, info = env.step(1)  
        accion=0
        giro=giro+1
      elif pos>pos_objetivo and not done:
        state, reward, done, info = env.step(6)
        pos=pos-1
        accion=0
      elif pos<pos_objetivo and not done:
        state, reward, done, info = env.step(3)
        pos=pos+1
        accion=0
      elif not done and not pieza_colocada:
        state, reward, done, info = env.step(9)
        accion=9
      else:
        accion=0
      if not done:
        state, reward, done, info = env.step(accion)
      env.render() 
      informacion=env.get_info()
      if antiguo_statistics!=informacion['statistics']:
        antiguo_statistics=informacion['statistics']
        step_counter += 1


    env.close()

# La memoria donde iremos guardando los resultados

class Memory:
  def __init__(self):
    self.states = []
    self.actions = []
    self.rewards = []

  def store(self, state, action, reward):
    self.states.append(state)
    self.actions.append(action)
    self.rewards.append(reward)

  def clear(self):
    self.states = []
    self.actions = []
    self.rewards = []


# Los trabajadores, los cuales trabajaran simultaneamente y mandaran los resultaos al modelo global

class Worker(threading.Thread):

  global_episode = 0

  global_moving_average_reward = 0
  best_score = 0
  save_lock = threading.Lock()

  def __init__(self,
               state_size,
               action_size,
               global_model,
               opt,
               result_queue,
               idx,
               game_name='Tetris',
               save_dir='/tmp'):
    super(Worker, self).__init__()
    self.state_size = state_size
    self.action_size = action_size
    self.result_queue = result_queue
    self.global_model = global_model
    self.opt = opt
    self.local_model = ActorCriticModel(self.state_size, self.action_size)
    self.worker_idx = idx
    self.env = gym_tetris.make('TetrisA-v0')
    self.env = JoypadSpace(self.env, MOVEMENT)
    self.save_dir = save_dir
    self.ep_loss = 0.0
    self.game_name='Tetris'

  def run(self):
    total_step = 1
    mem = Memory()
    
    while Worker.global_episode < episodios:
      self.env.reset()
      estado =[0.,0.,0.,0.]
      mem.clear()
      ep_reward = 0.
      ep_steps = 0
      self.ep_loss = 0
      informacion=self.env.get_info()
      antiguo_statistics=informacion['statistics']
      time_count = 0
      
      done = False
      pieza_colocada=True

      while not done:
        
        # Si hemos colocado la pieza calculamos la posicion y el giro de la proxima pieza
        if pieza_colocada:
          pieza_colocada=False
          pos=5
          giro=1
          u=-1
          ant_nom_piez=''
          estado=[estado]

          logits, _ = self.local_model(
             tf.convert_to_tensor(estado,
                                 dtype=tf.float32))

          probs = tf.nn.softmax(logits)

          prob=probs[0][39]
          probs=np.delete(probs[0],39)
          suma=np.sum(probs)
          probs=np.insert(probs,39,abs(1-suma))

          action = np.random.choice(self.action_size, p=probs)
          pos_objetivo=action  % 10
          giro_objetivo=(action // 10)+1


        # Colocamos la pieza donde hemos calculado girandola y moviendola
        if (giro % giro_objetivo)!=0 and not done:
          state, reward, done, info = self.env.step(1)  
          accion=0
          giro=giro+1
        elif pos>pos_objetivo and not done:
          state, reward, done, info = self.env.step(6)
          pos=pos-1
          accion=0
        elif pos<pos_objetivo and not done:
          state, reward, done, info = self.env.step(3)
          pos=pos+1
          accion=0
        elif not done and not pieza_colocada:
          state, reward, done, info = self.env.step(9)
          accion=9
        else:
          accion=0
        if not done:
          new_state, reward, done, info = self.env.step(accion)


        
        informacion=self.env.get_info()
        
        # Si la pieza ha sido colocada calculamos las ganancias del movimiento

        if antiguo_statistics!=informacion['statistics']:
          antiguo_statistics=informacion['statistics']
          ep_reward_new=informacion['score']
          reward=ep_reward_new-ep_reward
          board=self.env.board()
          nuevo_estado=board_prop(board)[:]
          pieza_colocada=True
          k=1
          if nuevo_estado[0]>18:
            done=True
          

          ep_reward = ep_reward_new
          

          mem.store(estado[0], action, reward)

          # Calculamos el gradiente local usando la perdida calculada de nuestra partida actual y 
          # nuestro modelo
          
          if time_count == 10 or done:

            with tf.GradientTape() as tape:
              total_loss = self.compute_loss(done,
                                           nuevo_estado,
                                           mem,
                                           0.99)
            self.ep_loss += total_loss

            grads = tape.gradient(total_loss, self.local_model.trainable_weights)

            self.opt.apply_gradients(zip(grads,
                                       self.global_model.trainable_weights))

            self.local_model.set_weights(self.global_model.get_weights())

            mem.clear()
            time_count = 0

            if done:  
              Worker.global_moving_average_reward = \
              record(Worker.global_episode, ep_reward, self.worker_idx,
                     Worker.global_moving_average_reward, self.result_queue,
                     self.ep_loss, ep_steps)


              if ep_reward > Worker.best_score:
                with Worker.save_lock:
       
                  self.global_model.save_weights(
                    os.path.join(self.save_dir,
                                 'model_{}.h5'.format(self.game_name))
                )
                  Worker.best_score = ep_reward
              Worker.global_episode += 1
          ep_steps += 1

          time_count += 1
          estado = nuevo_estado

          total_step += 1
    self.result_queue.put(None)

  # Calculamos la perdida
 
  def compute_loss(self,
                   done,
                   nuevo_estado,
                   memory,
                   gamma=0.99):
    if done:
      reward_sum = 0.  # terminal
    else:
      nuevo_estado=[nuevo_estado]
      reward_sum = self.local_model(
          tf.convert_to_tensor(nuevo_estado,
                               dtype=tf.float32))[-1].numpy()[0]


    discounted_rewards = []
    for reward in memory.rewards[::-1]: 
      reward_sum = reward + gamma * reward_sum
      discounted_rewards.append(reward_sum)
    discounted_rewards.reverse()

    logits, values = self.local_model(
        tf.convert_to_tensor(np.vstack(memory.states),
                             dtype=tf.float32))

    advantage = tf.convert_to_tensor(np.array(discounted_rewards)[:, None],
                            dtype=tf.float32) - values

    value_loss = advantage ** 2


    policy = tf.nn.softmax(logits)
    entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=policy, logits=logits)

    policy_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=memory.actions,
                                                                 logits=logits)
    policy_loss *= tf.stop_gradient(advantage)
    policy_loss -= 0.01 * entropy
    total_loss = tf.reduce_mean((0.5 * value_loss + policy_loss))
    return total_loss


if __name__ == '__main__':

  master = MasterAgent()
  master.train()
  file.close()
  master.play()


