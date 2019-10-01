''' basic hard coded terran agent from steven brown 1.0 tutorial
'''


from pysc2.agents import base_agent
from pysc2.env import sc2_env
from pysc2.lib import actions, features, units
from absl import app
import random
import time


class TerranAgent(base_agent.BaseAgent):
    def __init__(self):
        super(TerranAgent, self).__init__()
        self.left = False
        self.attack_location = None
        self.flag_builddepot = False
        self.flag_buildbarracks = False
        self.flag_trainmarine = False

    def get_units_by_type(self, obs, unit_type):
        return [unit for unit in obs.observation.feature_units if unit.unit_type == unit_type]
    
    def transformLocation(self, x, x_dist, y, y_dist):
        if self.left:
            return (x + x_dist, y + y_dist)
        return (x - x_dist, y - y_dist)

    def step(self, obs):
        super(TerranAgent, self).step(obs)

        if obs.first():
            player_y, player_x = ((obs.observation.feature_minimap.player_relative)==(features.PlayerRelative.SELF)).nonzero()
            xmean = player_x.mean()
            if xmean <= 31:
                self.left = True
                self.attack_coordinates = (49,49)
            else: self.attack_coordinates = (12,16)

        if self.flag_builddepot:
            self.flag_builddepot = False
            cc = self.get_units_by_type(obs, units.Terran.CommandCenter)[0]
            target = self.transformLocation(cc.x, 15, cc.y, 15)
            return actions.FUNCTIONS.Build_SupplyDepot_screen('now', target)

        if self.flag_buildbarracks:
            self.flag_buildbarracks = False
            cc = self.get_units_by_type(obs, units.Terran.CommandCenter)[0]
            target = self.transformLocation(cc.x, 15, cc.y, -15)
            return actions.FUNCTIONS.Build_Barracks_screen('now', target)

        if len(self.get_units_by_type(obs, units.Terran.SupplyDepot))==0:
            if obs.observation['player'][1] >= 100:
                scvs = self.get_units_by_type(obs, units.Terran.SCV)
                scv = random.choice(scvs)
                self.flag_builddepot = True
                return actions.FUNCTIONS.select_point('select', (scv.x, scv.y))

        if len(self.get_units_by_type(obs, units.Terran.Barracks))==0:
            if (actions.FUNCTIONS.Build_Barracks_screen.id in obs.observation.available_actions) and obs.observation['player'][1] >= 150:
                scvs = self.get_units_by_type(obs, units.Terran.SCV)
                scv = random.choice(scvs)
                self.flag_buildbarracks = True
                return actions.FUNCTIONS.select_point('select', (scv.x, scv.y))

        return actions.FUNCTIONS.no_op()


def main(unused_argv):
  agent = TerranAgent()
  try:
    while True:
      with sc2_env.SC2Env(
          map_name="AbyssalReef",
          players=[sc2_env.Agent(sc2_env.Race.terran),
                   sc2_env.Bot(sc2_env.Race.random,
                               sc2_env.Difficulty.very_easy)],
          agent_interface_format=features.AgentInterfaceFormat(
              feature_dimensions=features.Dimensions(screen=84, minimap=64), use_feature_units=True),
          step_mul=16,
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