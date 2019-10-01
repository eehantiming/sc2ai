'''
hard coded zerg bot on abyssal reef
'''

from pysc2.agents import base_agent
from pysc2.env import sc2_env
from pysc2.lib import actions, features, units
from absl import app
import random
import time

class ZergAgent(base_agent.BaseAgent):
	def __init__(self):
		super(ZergAgent, self).__init__()
		#self.xmean = 0
		self.left = False
		self.attack_coordinates = None
		self.attack_coordinates_2 = None
		self.rallied = False
		self.flag_attack = False
	
	def unit_type_is_selected(self, obs, unit_type): # helper method checks if specified unit type is currently selected
		if (len(obs.observation.single_select) > 0 and obs.observation.single_select[0].unit_type == unit_type):
			return True

		if (len(obs.observation.multi_select) > 0 and obs.observation.multi_select[0].unit_type == unit_type):
			return True
		return False

	def get_units_by_type(self, obs, unit_type): # helper method to get stuff on screen by unit type
		return [unit for unit in obs.observation.feature_units if unit.unit_type == unit_type]

	def transformLocation(self, x, x_dist, y, y_dist):
		if self.left:
		#if self.xmean <= 31:
			return (x + x_dist, y + y_dist)
		return (x - x_dist, y - y_dist)

	def do_action(self, obs, action_name, *args):
		if (getattr(actions.FUNCTIONS, action_name).id in obs.observation.available_actions):
			return getattr(actions.FUNCTIONS, action_name)(*args)


	def step(self,obs):
		super(ZergAgent, self).step(obs)
		#time.sleep(0.2)

		if obs.first():
			player_y, player_x = ((obs.observation.feature_minimap.player_relative)==(features.PlayerRelative.SELF)).nonzero()
			xmean = player_x.mean()
			#self.xmean = xmean
			if xmean <= 31:
				self.left = True
				self.attack_coordinates = (40, 49)
				self.attack_coordinates_2 = (49,49)
			else: 
				self.attack_coordinates = (22, 16)
				self.attack_coordinates_2 = (12,16)
			hatch = self.get_units_by_type(obs, units.Zerg.Hatchery)[0]
			return actions.FUNCTIONS.select_point('select', (hatch.x, hatch.y))

		if not self.rallied:
			self.rallied = True
			hatch = self.get_units_by_type(obs, units.Zerg.Hatchery)[0]
			target = self.transformLocation(hatch.x, 25, hatch.y, 0)
			if actions.FUNCTIONS.Rally_Units_screen.id in obs.observation.available_actions:
				return actions.FUNCTIONS.Rally_Units_screen('now', target)

		if self.flag_attack:
			self.flag_attack = False
			return actions.FUNCTIONS.Attack_minimap('queued', self.attack_coordinates_2)

		if obs.observation.player.food_workers < 16:
			if actions.FUNCTIONS.Train_Drone_quick.id in obs.observation.available_actions:
				return actions.FUNCTIONS.Train_Drone_quick("now")	

		if len(self.get_units_by_type(obs, units.Zerg.SpawningPool)) == 0: # if no pool on screen
			if not self.unit_type_is_selected(obs, units.Zerg.Drone):
				drones = self.get_units_by_type(obs,units.Zerg.Drone)
				if len(drones) > 0: # check if theres drones on screen 
					#drone = random.choice(drones)
					drones_x = [temp.x for temp in drones]
					if self.left:
						drone = drones[drones_x.index(max(drones_x))]
					else: drone = drones[drones_x.index(min(drones_x))]
					#return actions.FUNCTIONS.select_point('select',(drone.x,drone.y)) #CTRL click
					return actions.FunctionCall(actions.FUNCTIONS.select_point.id, ([0], (drone.x,drone.y)))
			hatch = self.get_units_by_type(obs, units.Zerg.Hatchery)[0]
			target = self.transformLocation(hatch.x, 10, hatch.y, -15)
		
			temp = self.do_action(obs, "Build_SpawningPool_screen", "now", target)	
			if (temp != None): return temp			
			# if (actions.FUNCTIONS.Build_SpawningPool_screen.id in obs.observation.available_actions):
			# 	return actions.FUNCTIONS.Build_SpawningPool_screen("now", target)

		
		zerglings = self.get_units_by_type(obs, units.Zerg.Zergling)
		if len(zerglings) >= 20:
			if self.unit_type_is_selected(obs, units.Zerg.Zergling):
				if actions.FUNCTIONS.Attack_minimap.id in obs.observation.available_actions:
					self.flag_attack = True
					return actions.FUNCTIONS.Attack_minimap("now", self.attack_coordinates)
			
			ling = random.choice(zerglings)
			if actions.FUNCTIONS.select_army.id in obs.observation.available_actions:
				return actions.FUNCTIONS.select_army("select")
			#self.do_action(obs, "select_army", "select")

		if self.unit_type_is_selected(obs, units.Zerg.Larva):
			free_supply = obs.observation.player.food_cap - obs.observation.player.food_used
			if free_supply <= 1:
				if(actions.FUNCTIONS.Train_Overlord_quick.id in obs.observation.available_actions):
					return actions.FUNCTIONS.Train_Overlord_quick("now")

			if (actions.FUNCTIONS.Train_Zergling_quick.id in obs.observation.available_actions):
				return actions.FUNCTIONS.Train_Zergling_quick("now")

		larvae = self.get_units_by_type(obs, units.Zerg.Larva)
		if len(larvae) > 0:
			larva = random.choice(larvae)
			return actions.FUNCTIONS.select_point('select', (larva.x, larva.y))


		return actions.FUNCTIONS.no_op()


def main(unused_argv):
  agent = ZergAgent()
  try:
    while True:
      with sc2_env.SC2Env(
          map_name="AbyssalReef",
          players=[sc2_env.Agent(sc2_env.Race.zerg),
                   sc2_env.Bot(sc2_env.Race.random,
                               sc2_env.Difficulty.very_easy)],
          agent_interface_format=features.AgentInterfaceFormat(
              feature_dimensions=features.Dimensions(screen=84, minimap=64), use_feature_units=True),
          step_mul=16,
          #realtime=True,
          game_steps_per_episode=0,
          visualize=False) as env:
          
        agent.setup(env.observation_spec(), env.action_spec())
        
        timesteps = env.reset()
        agent.reset()
        
        while True:
          step_actions = [agent.step(timesteps[0])]
          if timesteps[0].last():
            break
          timesteps = env.step(step_actions)
      
  except KeyboardInterrupt:
    pass
  

if __name__ == "__main__":
  app.run(main)