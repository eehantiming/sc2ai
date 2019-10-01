# pysc2 python scripts for Starcraft 2
## 1. Train agent
~~~ 
py -m pysc2.bin.agent --map Simple64 --agent smartagent.SmartAgent --agent_race terran --use_feature_units True --step_mul 8 --norender --parallel 3 --nosave_replay 
~~~
Agent will be saved in the form of a qtable in savedqtable.pickle
## 2. Watch agent
~~~
py -m pysc2.bin.agent --map Simple64 --agent watchsmartagent.SmartAgent --agent_race terran --use_feature_units True --step_mul 8 --norender --realtime True --nosave_replay
~~~
Agent will be loaded from savedqtable.pickle. No training during this
## 3. Read q table
In terminal, run readqtable.py and input '1' to investigate best action at each state
