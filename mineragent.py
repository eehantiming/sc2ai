'''
hard coded agent for collectmineralshards map
'''


from pysc2.agents import base_agent
from pysc2.env import sc2_env
from pysc2.lib import actions, features, units
import random
import math
import numpy as np
import pandas as pd
from absl import app


class MinerAgent(base_agent.BaseAgent):
	def __init__(self):
		super(MinerAgent, self).__init__()
		self.moving = False

	def unit_type_is_selected(self, obs, unit_type): # helper method checks if specified unit type is currently selected
		if (len(obs.observation.single_select) > 0 and obs.observation.single_select[0].unit_type == unit_type):
			return True

		if (len(obs.observation.multi_select) > 0 and obs.observation.multi_select[0].unit_type == unit_type):
			return True
		return False

	def get_units_by_type(self, obs, unit_type):
		return [unit for unit in obs.observation.feature_units if unit.unit_type == unit_type]

	def nearest(self, x, y, minerals):
		minerals = np.array(minerals)
		marine = np.array([x,y])
		dist = np.linalg.norm(minerals-marine, axis=1)
		index = np.argmin(dist)
		return(minerals[index])

	def step(self, obs):
		super(MinerAgent, self).step(obs)

		if not self.unit_type_is_selected(obs, units.Terran.Marine):
			marines = self.get_units_by_type(obs, units.Terran.Marine)
			marine = random.choice(marines)
			#return actions.FUNCTIONS.select_point('select', (marine.x,marine.y))
			return actions.FunctionCall(actions.FUNCTIONS.select_point.id, [[0], [marine.x,marine.y]])
		if self.moving:
			self.moving = False
			marines = self.get_units_by_type(obs, units.Terran.Marine)
			marines = [unit for unit in marines if unit.is_selected == False]
			marine = marines[0]
			return actions.FUNCTIONS.select_point("select", (marine.x,marine.y))

		if self.unit_type_is_selected(obs, units.Terran.Marine):
			marines = self.get_units_by_type(obs, units.Terran.Marine)
			marines = [unit for unit in marines if unit.is_selected == True]
			if len(marines) > 0:
				marine = marines[0]
			#marine = [unit for unit in marines if unit.is_selected == True][0]
				minerals_y, minerals_x = ((obs.observation.feature_screen.player_relative)==(features.PlayerRelative.NEUTRAL)).nonzero()
				minerals = list(zip(minerals_x, minerals_y))
				mineral_x,mineral_y = self.nearest(marine.x, marine.y, minerals)
			if actions.FUNCTIONS.Move_screen.id in obs.observation.available_actions:
				self.moving = True
				#return actions.FUNCTIONS.Move_screen("now", (mineral_x, mineral_y))
				return actions.FunctionCall(actions.FUNCTIONS.Move_screen.id, [[1], [mineral_x, mineral_y]])

		return actions.FUNCTIONS.no_op()


def main(unused_argv):
#py -m pysc2.bin.agent --map CollectMineralShards --agent mineragent.MinerAgent --use_feature_units True --step_mul 4
  agent = MinerAgent()
  try:
    while True:
      with sc2_env.SC2Env(
          map_name="CollectMineralShards",
          #players=[sc2_env.Agent(sc2_env.Race.Terran)],
          agent_interface_format=features.AgentInterfaceFormat(
              feature_dimensions=features.Dimensions(screen=84, minimap=64), use_feature_units=True),
          step_mul=4,
          realtime=True,
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