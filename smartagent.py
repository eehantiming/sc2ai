''' 
trains a RL zerg agent using basic qlearning.
state space defined in current_state
action space defined in smart_actions
saves trained q table (pd dataframe) in a pickle file. use watchsmartagent.py to play agent
use readqtable.py (select 1) to read values of q table
'''

import os.path

import random
import math

import numpy as np
import pandas as pd
import pickle

from pysc2.agents import base_agent
from pysc2.lib import actions
from pysc2.lib import features, units

_NO_OP = actions.FUNCTIONS.no_op.id
_SELECT_POINT = actions.FUNCTIONS.select_point.id
_BUILD_SUPPLY_DEPOT = actions.FUNCTIONS.Build_SupplyDepot_screen.id
_BUILD_BARRACKS = actions.FUNCTIONS.Build_Barracks_screen.id
_TRAIN_MARINE = actions.FUNCTIONS.Train_Marine_quick.id
_SELECT_ARMY = actions.FUNCTIONS.select_army.id
_ATTACK_MINIMAP = actions.FUNCTIONS.Attack_minimap.id

_PLAYER_RELATIVE = features.SCREEN_FEATURES.player_relative.index
_UNIT_TYPE = features.SCREEN_FEATURES.unit_type.index
_PLAYER_ID = features.SCREEN_FEATURES.player_id.index

_PLAYER_SELF = 1

_TERRAN_COMMANDCENTER = 18
_TERRAN_SCV = 45 
_TERRAN_SUPPLY_DEPOT = 19
_TERRAN_BARRACKS = 21

_NOT_QUEUED = [0]
_SELECTSINGLE = [0]
_QUEUED = [1]

ACTION_DO_NOTHING = 'donothing'
ACTION_ATTACK_SCV = 'attackwithscv'
ACTION_BUILD_SUPPLY_DEPOT = 'buildsupplydepot'
ACTION_BUILD_BARRACKS = 'buildbarracks'
ACTION_BUILD_MARINE = 'buildmarine'
ACTION_ATTACK_MARINE = 'attackwitharmy'

smart_actions = [
    ACTION_ATTACK_SCV,
    ACTION_DO_NOTHING,
    ACTION_BUILD_SUPPLY_DEPOT,
    ACTION_BUILD_BARRACKS,
    ACTION_BUILD_MARINE,
    ACTION_ATTACK_MARINE,
]

EPSILON_INITIAL = 0.2 # always decay from 1 to 0.01
EPSILON_DECAY = 0.999 #decay to 0.1: {0.9:22, 0.95:45, 0.98:114, 0.99:229, 0.995:459, 0.998:1150}
GAMMA_INITIAL = 0.99 # always decay from 0.99 to 0.999
GAMMA_DECAY = 0.99668 #0.1 ** (math.log(EPSILON_DECAY)/math.log(0.01)) #half life :{0.99: 24.63s, 0.999: 247.43}

KILL_UNIT_REWARD = 0.5
KILL_BUILDING_REWARD = 1.0
MINERAL_REWARD = 0.1
KILLED_REWARD = -0.3

# Stolen from https://github.com/MorvanZhou/Reinforcement-learning-with-tensorflow
class QLearningTable:
    def __init__(self, actions, learning_rate=0.01, reward_decay=GAMMA_INITIAL, e_greedy=EPSILON_INITIAL): #half life of 24.6s for step_mul 8
        self.actions = actions  # a list
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon = e_greedy
        self.q_table = pd.DataFrame(columns=self.actions, dtype=np.float64)
        self.no_of_epis = 0

    def choose_action(self, observation):
        self.check_state_exist(observation) 
        
        if np.random.uniform() > self.epsilon:
            # choose best action
            state_action = self.q_table.loc[observation] # one row turns into series (columns)
            # some actions have the same value, this shuffles the series
            state_action = state_action.reindex(np.random.permutation(state_action.index))
            
            action = state_action.idxmax() #returns an integer which is the argmax
        else:
            # choose random action
            action = np.random.choice(self.actions)
            
        return action

    def learn(self, s, a, r, s_):
        self.check_state_exist(s_)
        self.check_state_exist(s)
        
        q_predict = self.q_table.loc[s, a]
        q_target = r + self.gamma * self.q_table.loc[s_].max()
        
        # update
        self.q_table.ix[s, a] += self.lr * (q_target - q_predict)

    def check_state_exist(self, state):
        if state not in self.q_table.index:
            # append new state to q table
            self.q_table = self.q_table.append(pd.Series([-1] + [0] * (len(self.actions)-1), index=self.q_table.columns, name=state))

class SmartAgent(base_agent.BaseAgent):
    def __init__(self): #only happens once when script is ran.
        super(SmartAgent, self).__init__()

        if os.path.exists('savedqtable.pickle'): #continued training
            with open('savedqtable.pickle', 'rb') as file:
                self.qlearn = pickle.load(file)
            self.qlearn.epsilon = EPSILON_INITIAL # train from full explore if 1
            self.qlearn.gamma = GAMMA_INITIAL
            with open('savedqtable.pickle', 'wb') as file:
                pickle.dump(self.qlearn, file)            
        else: #new training
            self.qlearn = QLearningTable(actions=list(range(len(smart_actions))))

        self.previous_killed_unit_score = 0
        self.previous_killed_building_score = 0
        self.previous_spent_minerals = 0
        
        self.previous_action = None
        self.previous_state = None

    def get_units_by_type(self, obs, unit_type):
        return [unit for unit in obs.observation.feature_units if unit.unit_type == unit_type]

    def transformLocation(self, x, x_distance, y, y_distance):
        if not self.base_top_left:
            return [x - x_distance, y - y_distance]
        return [x + x_distance, y + y_distance]

    def clip(self, stuff): #screen 84, minimap 64
        stuff.x = sorted((0,stuff.x,83))[1]
        stuff.y = sorted((0,stuff.y,83))[1]
        return stuff
        
    def step(self, obs): #repeats when epi ends and new window opens
        super(SmartAgent, self).step(obs)

        if obs.first():
            if os.path.exists('savedqtable.pickle'):
                with open('savedqtable.pickle', 'rb') as file:
                    self.qlearn = pickle.load(file)

            self.qlearn.epsilon = max(self.qlearn.epsilon * EPSILON_DECAY, 0.01) #10% in 459 episodes
            self.qlearn.gamma = min(1 - GAMMA_DECAY * (1 - self.qlearn.gamma), 0.999)
            print(f'\t\t\tepsilon: {self.qlearn.epsilon}\t gamma: {self.qlearn.gamma}'.upper())

            player_y, player_x = (obs.observation['feature_minimap'][_PLAYER_RELATIVE] == _PLAYER_SELF).nonzero()
            self.base_top_left = 1 if player_y.any() and player_y.mean() <= 31 else 0

            self.flag_attack = False
            self.flag_attack_2 = False
            self.flag_attack_3 = False
            self.flag_builddepot = False
            self.flag_buildbarracks = False
            self.flag_trainmarine = False
        
        if obs.last():
            with open('savedqtable.pickle', 'wb') as file:
                self.qlearn.no_of_epis += 1
                pickle.dump(self.qlearn, file)

        depots = self.get_units_by_type(obs,units.Terran.SupplyDepot)
        supply_depot_count = len(depots)

        barracks = self.get_units_by_type(obs,units.Terran.Barracks)
        barracks_count = len(barracks)

        if self.flag_builddepot:
            if actions.FUNCTIONS.Build_SupplyDepot_screen.id in obs.observation.available_actions:
                self.flag_builddepot = False
                if len(self.get_units_by_type(obs, units.Terran.CommandCenter))>0:
                    cc = self.get_units_by_type(obs, units.Terran.CommandCenter)[0]
                    if supply_depot_count == 0:
                        target = self.transformLocation(cc.x, 10, cc.y, 15)
                    else: target = self.transformLocation(cc.x, 15, cc.y, 15)
                else: target = [random.randint(0, 63), random.randint(0,63)]
                return actions.FUNCTIONS.Build_SupplyDepot_screen('now', target)

        if self.flag_buildbarracks:
            if actions.FUNCTIONS.Build_Barracks_screen.id in obs.observation.available_actions:
                self.flag_buildbarracks = False
                if len(self.get_units_by_type(obs, units.Terran.CommandCenter))>0:
                    cc = self.get_units_by_type(obs, units.Terran.CommandCenter)[0]
                    if barracks_count == 0:
                        target = self.transformLocation(cc.x, 15, cc.y, -15)
                    else: target = self.transformLocation(cc.x, 15, cc.y, 0)
                else: target = [random.randint(0, 63), random.randint(0,63)]
                return actions.FUNCTIONS.Build_Barracks_screen('now', target)

        if self.flag_attack_3:
            if actions.FUNCTIONS.Attack_minimap.id in obs.observation.available_actions:
                self.flag_attack_3 = False
                if self.base_top_left:
                    return actions.FunctionCall(_ATTACK_MINIMAP, [_QUEUED, [36, 41]])
                return actions.FunctionCall(_ATTACK_MINIMAP, [_QUEUED, [20, 30]])

        if self.flag_attack_2:
            if actions.FUNCTIONS.Attack_minimap.id in obs.observation.available_actions:
                self.flag_attack_2 = False
                self.flag_attack_3 = True
                if self.base_top_left:
                    return actions.FunctionCall(_ATTACK_MINIMAP, [_QUEUED, [41, 47]])
                return actions.FunctionCall(_ATTACK_MINIMAP, [_QUEUED, [15, 24]])

        if self.flag_attack:
            if actions.FUNCTIONS.Attack_minimap.id in obs.observation.available_actions:
                self.flag_attack = False
                self.flag_attack_2 = True
                if self.base_top_left:
                    return actions.FunctionCall(_ATTACK_MINIMAP, [_QUEUED, [15, 47]])
                return actions.FunctionCall(_ATTACK_MINIMAP, [_QUEUED, [41, 24]])

        if self.flag_trainmarine:
            if actions.FUNCTIONS.Train_Marine_quick.id in obs.observation.available_actions:
                self.flag_trainmarine = False
                return actions.FunctionCall(_TRAIN_MARINE, [_QUEUED])

        unit_type = obs.observation['feature_screen'][_UNIT_TYPE]
        
        army_supply = obs.observation['player'][5]
        minerals_bank = obs.observation['player'][1]//50
        
        killed_unit_score = obs.observation['score_cumulative'][5]
        killed_building_score = obs.observation['score_cumulative'][6]
        spent_minerals = obs.observation['score_cumulative'][11]

        current_state = [
            supply_depot_count,
            barracks_count,
            army_supply,
            minerals_bank
        ]
        
        if self.previous_action is not None:
            reward = 0
                
            if killed_unit_score > self.previous_killed_unit_score:
                reward += KILL_UNIT_REWARD
                    
            if killed_building_score > self.previous_killed_building_score:
                reward += KILL_BUILDING_REWARD
            
            if spent_minerals > self.previous_spent_minerals:
                reward += MINERAL_REWARD

            self.qlearn.learn(str(self.previous_state), self.previous_action, reward, str(current_state))
        
        rl_action = self.qlearn.choose_action(str(current_state))
        smart_action = smart_actions[rl_action]
        # print('action chosen is ', smart_action)
        # print('unit score is ', unit_score)
        
        self.previous_killed_unit_score = killed_unit_score
        self.previous_killed_building_score = killed_building_score
        self.previous_spent_minerals = spent_minerals
        
        self.previous_state = current_state
        self.previous_action = rl_action
        
        

        if smart_action == ACTION_DO_NOTHING:
            return actions.FunctionCall(_NO_OP, [])
        
        elif smart_action == ACTION_BUILD_SUPPLY_DEPOT: # select scv, go to build depot flag
            if obs.observation['player'][1] >= 100:
                scvs = self.get_units_by_type(obs, units.Terran.SCV)
                if len(scvs) > 0:
                    self.flag_builddepot = True
                    scv = self.clip(random.choice(scvs))
                    return actions.FUNCTIONS.select_point('select', (scv.x, scv.y))
        
        elif smart_action == ACTION_BUILD_BARRACKS: #select scv, go to build barracks flag
            if obs.observation['player'][1] >= 150:
                scvs = self.get_units_by_type(obs, units.Terran.SCV)
                if len(scvs) > 0:
                    self.flag_buildbarracks = True
                    scv = self.clip(random.choice(scvs))           
                    return actions.FUNCTIONS.select_point('select', (scv.x, scv.y))
    
        elif smart_action == ACTION_BUILD_MARINE: #select barracks, go to train marine flag
            # unit_type = obs.observation['feature_screen'][_UNIT_TYPE]
            # unit_y, unit_x = (unit_type == _TERRAN_BARRACKS).nonzero()
            # if unit_y.any():
            #     self.flag_trainmarine = True
            #     target = [int(unit_x.mean()), int(unit_y.mean())]

            barracks = [unit for unit in obs.observation.feature_units if unit.unit_type == units.Terran.Barracks]
            if len(barracks) > 0:
                self.flag_trainmarine = True
                barrack = random.choice(barracks)
                target = [barrack.x, barrack.y]      
                return actions.FunctionCall(_SELECT_POINT, [_NOT_QUEUED, target])
        
        elif smart_action == ACTION_ATTACK_MARINE: #select army, go to attack flag
            if _SELECT_ARMY in obs.observation['available_actions']:
                self.flag_attack = True
                return actions.FunctionCall(_SELECT_ARMY, [_NOT_QUEUED])
            # marines = self.get_units_by_type(obs, units.Terran.Marine)
            # if len(marines) > 0:
            #     self.flag_attack = True
            #     marine = random.choice(marines)
            #     return actions.FUNCTIONS.select_point('select', (marine.x, marine.y))

        elif smart_action == ACTION_ATTACK_SCV: #select SCV, go to attack flag
            scvs = self.get_units_by_type(obs, units.Terran.SCV)
            if len(scvs) > 0:
                self.flag_attack = True
                scv = self.clip(random.choice(scvs))
                return actions.FUNCTIONS.select_point('select', (scv.x, scv.y))
        
        return actions.FunctionCall(_NO_OP, [])
