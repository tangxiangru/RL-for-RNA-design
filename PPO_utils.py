import bisect
import copy 
import os 
from collections import deque, Counter
import random
import logging
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import editdistance
import sys
import RNA
from typing import Dict, List, Tuple
import time 

# import path 
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

from utils.sequence_utils import translate_one_hot_to_string,generate_random_mutant
from utils.sequence_utils import translate_string_to_one_hot, translate_one_hot_to_string
from models.Theoretical_models import *
from models.Noise_wrapper import *
from exploration_strategies.CE import *
from utils.landscape_utils import *
from models.RNA_landscapes import *
from models.Multi_dimensional_model import *

import tensorflow as tf
from tf_agents.drivers import dynamic_step_driver
from tf_agents.metrics import tf_metrics
from tf_agents.agents import tf_agent
from tf_agents.policies import random_tf_policy
from tf_agents.agents.ppo import ppo_policy, ppo_agent, ppo_utils
from tf_agents.environments import py_environment, tf_py_environment
from tf_agents.environments.utils import validate_py_environment
from tf_agents.drivers import dynamic_episode_driver
from tf_agents.networks import network, normal_projection_network
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.utils import common
from tf_agents.trajectories import time_step as ts
from tf_agents.specs import array_spec

from tf_agents.environments import parallel_py_environment
from tf_agents.eval import metric_utils

def renormalize_moves(one_hot_input, rewards_output):
    """ensures that staying in place gives no reward"""
    zero_current_state = (one_hot_input - 1) * (-1)
    return np.multiply(rewards_output, zero_current_state)

def walk_away_renormalize_moves(one_hot_input, one_hot_wt, rewards_output):
    """ensures that moving toward wt is also not useful"""
    zero_current_state=(one_hot_input-1)*-1
    zero_wt=((one_hot_wt-1)*-1)
    zero_conservative_moves=np.multiply(zero_wt,zero_current_state)
    return np.multiply(rewards_output,zero_conservative_moves)

def get_all_singles_fitness(model,sequence,alphabet):
    prob_singles=np.zeros((len(alphabet),len(sequence)))
    for i in range(len(sequence)):
        for j in range(len(alphabet)):
            putative_seq=sequence[:i]+alphabet[j]+sequence[i+1:]
           # print (putative_seq)
            prob_singles[j][i]=model.get_fitness(putative_seq)
    return prob_singles

def get_all_mutants(sequence):
    mutants = []
    for i in range(sequence.shape[0]):
        for j in range(sequence.shape[1]):
            putative_seq = sequence.copy()
            putative_seq[:, j] = 0
            putative_seq[i, j] = 1
            mutants.append(putative_seq)
    return np.array(mutants)

def sample_greedy(matrix):
    i,j=matrix.shape
    max_arg=np.argmax(matrix)
    y=max_arg%j
    x=int(max_arg/j)
    output=np.zeros((i,j))
    output[x][y]=matrix[x][y]
    return output

def sample_multi_greedy(matrix):
    n = 5 # the number of base positions to greedily change
    max_args = np.argpartition(matrix.flatten(), -n)[-n:]
    i,j=matrix.shape
    output=np.zeros((i,j))
    for max_arg in max_args:
        y=max_arg%j
        x=int(max_arg/j)
        output[x][y]=matrix[x][y]
    return output

def sample_random(matrix):
    i,j=matrix.shape
    non_zero_moves=np.nonzero(matrix)
   # print (non_zero_moves)
    k=len(non_zero_moves)
    l=len(non_zero_moves[0])
    if k!=0 and l!=0:
        rand_arg=random.choice([[non_zero_moves[alph][pos] for alph in range(k)] for pos in range(l)])
    else:
        rand_arg=[random.randint(0,i-1),random.randint(0,j-1)]
    #print (rand_arg)
    y=rand_arg[1]
    x=rand_arg[0]
    output=np.zeros((i,j))
    output[x][y] = 1
    return output   

def action_to_scalar(matrix):
    matrix = matrix.ravel()
    for i in range(len(matrix)):
        if matrix[i] != 0:
            return i
    
def construct_mutant_from_sample(pwm_sample, one_hot_base):
    one_hot = np.zeros(one_hot_base.shape)
    one_hot += one_hot_base
    nonzero = np.nonzero(pwm_sample)
    nonzero = list(zip(nonzero[0], nonzero[1]))
    for nz in nonzero: # this can be problematic for non-positive fitnesses
        i, j = nz
        one_hot[:,j]=0
        one_hot[i,j]=1
    return one_hot

def best_predicted_new_gen(actor, genotypes, alphabet, pop_size):
    mutants = get_all_mutants(genotypes)
    one_hot_mutants = np.array([translate_string_to_one_hot(mutant, alphabet) for mutant in mutants])
    torch_one_hot_mutants = torch.from_numpy(np.expand_dims(one_hot_mutants, axis=0)).float()
    predictions = actor(torch_one_hot_mutants)
    predictions = predictions.detach().numpy()
    best_pred_ind = predictions.argsort()[-pop_size:]
    return mutants[best_pred_ind]

def make_one_hot_train_test(genotypes, model, alphabet):
    genotypes_one_hot = np.array([translate_string_to_one_hot(genotype, alphabet) for genotype in genotypes])
    genotype_fitnesses = []
    for genotype in genotypes:
        genotype_fitnesses.append(model.get_fitness(genotype))
    genotype_fitnesses = np.array(genotype_fitnesses)

    return genotypes_one_hot, genotype_fitnesses

class PPO_RL_Agent(ppo_agent.PPOAgent):
    def __init__(self, 
                 env_load_fn,
                 actor_net,
                 value_net,
                 alphabet,
                 num_parallel_environments=30,
                 num_epochs=50,
                 experiment_batch_size=500,
                 collect_episodes_per_iteration=30,
                 num_eval_episodes=30,
                 num_environment_steps=10000000,
                 replay_buffer_capacity=1000+1,
                 train_checkpoint_interval=500,
                 policy_checkpoint_interval=500,
                 eval_interval=500,
                 log_interval=50,
                 optimizer=tf.compat.v1.train.AdamOptimizer(learning_rate=1e-5),
                 train_dir='train'
                ):
        self.env_load_fn = env_load_fn
        self.global_step = tf.compat.v1.train.get_or_create_global_step()
        self.train_checkpoint_interval = train_checkpoint_interval
        self.policy_checkpoint_interval = policy_checkpoint_interval
        self.eval_interval = eval_interval 
        self.num_eval_episodes = num_eval_episodes 
        self.eval_metrics = [
            tf_metrics.AverageReturnMetric(buffer_size=num_eval_episodes),
            tf_metrics.AverageEpisodeLengthMetric(buffer_size=num_eval_episodes)
        ]
        self.log_interval = log_interval 
        
        # environment 
        self.alphabet = alphabet 
        self.experiment_batch_size = experiment_batch_size 
        self.eval_tf_env = tf_py_environment.TFPyEnvironment(env_load_fn())
        self.tf_env = tf_py_environment.TFPyEnvironment(
            parallel_py_environment.ParallelPyEnvironment(
                [lambda: env_load_fn()] * num_parallel_environments))
        self.num_eval_episodes = num_eval_episodes 
        
        # agent object 
        self.tf_agent = ppo_agent.PPOAgent(
            self.tf_env.time_step_spec(),
            self.tf_env.action_spec(),
            optimizer,
            actor_net=actor_net,
            value_net=value_net,
            num_epochs=num_epochs,
            summarize_grads_and_vars=False,
            train_step_counter=self.global_step
        )
        self.tf_agent.initialize()
        
        self.eval_policy = self.tf_agent.policy
        collect_policy = self.tf_agent.collect_policy
        
        # other utilities we need for training 
        self.num_environment_steps = num_environment_steps 
        self.environment_steps_metric = tf_metrics.EnvironmentSteps()
        self.step_metrics = [
            tf_metrics.NumberOfEpisodes(),
            self.environment_steps_metric,
        ]

        self.train_metrics = self.step_metrics + [
            tf_metrics.AverageReturnMetric(
                batch_size=num_parallel_environments),
            tf_metrics.AverageEpisodeLengthMetric(
                batch_size=num_parallel_environments),
        ]

        self.replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
            self.tf_agent.collect_data_spec,
            batch_size=num_parallel_environments,
            max_length=replay_buffer_capacity)

        self.collect_driver = dynamic_episode_driver.DynamicEpisodeDriver(
            self.tf_env,
            collect_policy,
            observers=[self.replay_buffer.add_batch] + self.train_metrics,
            num_episodes=collect_episodes_per_iteration)
        
    def train_step(self):
        trajectories = self.replay_buffer.gather_all()
        return self.tf_agent.train(experience=trajectories)
    
    def run_train(self):
        self.collect_driver.run = common.function(self.collect_driver.run, autograph=False)
        self.tf_agent.train = common.function(self.tf_agent.train, autograph=False)
        train_step = common.function(self.train_step)
        collect_time = 0
        train_time = 0
        timed_at_step = self.global_step.numpy()
        
        while self.environment_steps_metric.result() < self.num_environment_steps:
            global_step_val = self.global_step.numpy()
            if global_step_val % self.eval_interval == 0:
                metric_utils.eager_compute(
                    self.eval_metrics,
                    self.eval_tf_env,
                    self.eval_policy,
                    num_episodes=self.num_eval_episodes,
                    train_step=self.global_step
                )
            start_time = time.time()
            self.collect_driver.run()
            collect_time += time.time() - start_time

            start_time = time.time()
            total_loss, _ = train_step()
            self.replay_buffer.clear()
            train_time += time.time() - start_time

            for train_metric in self.train_metrics:
                train_metric.tf_summaries(
                    train_step=self.global_step, step_metrics=self.step_metrics)

        if global_step_val % self.log_interval == 0:
            logging.info('step = %d, loss = %f', global_step_val, total_loss)
            steps_per_sec = (
                (global_step_val - timed_at_step) / (collect_time + train_time))
            logging.info('%.3f steps/sec', steps_per_sec)
            logging.info('collect_time = {}, train_time = {}'.format(
                collect_time, train_time))
        with tf.compat.v2.summary.record_if(True):
            tf.compat.v2.summary.scalar(
              name='global_steps_per_sec', data=steps_per_sec, step=self.global_step)
            with tf.compat.v2.summary.record_if(True):
                tf.compat.v2.summary.scalar(
                  name='global_steps_per_sec', data=steps_per_sec, step=self.global_step)

            timed_at_step = global_step_val
            collect_time = 0
            train_time = 0

        # One final eval before exiting.
        print('EXITED')
        metric_utils.eager_compute(
            self.eval_metrics,
            self.eval_tf_env,
            self.eval_policy,
            num_episodes=self.num_eval_episodes,
            train_step=self.global_step
        )
        
    def pick_action(self):
        steps = 0
        seq_and_fitness = []
        tf_env_temp = tf_py_environment.TFPyEnvironment(self.env_load_fn())
        while steps < self.experiment_batch_size:
            if tf_env_temp.current_time_step().is_last():
                tf_env_temp.reset()
            else:
                action = self.tf_agent.policy.action(tf_env_temp.current_time_step()).action
                next_time_step = self.tf_env.step(action)
                reward, obs = next_time_step.reward.numpy()[0], next_time_step.observation.numpy()[0]
                seq = translate_one_hot_to_string(obs, self.alphabet)
                seq_and_fitness.append((reward, seq))
                steps += 1
                
        return seq_and_fitness