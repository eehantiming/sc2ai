import pickle

# get user input for file to read
userinput = input('which table to read?: ')
while not userinput.isdigit():
	userinput = input('please input a valid integer: ')
if userinput == '1':
	filename = 'savedqtable.pickle'
else:
	filename = 'savedqtable_'+userinput+'.pickle'

#reading file
best_action = []
with open(filename, 'rb') as f:
	qtable = pickle.load(f)
for i in range(len(qtable.q_table.index)):
	best_action.append(qtable.q_table.iloc[i].idxmax())
qtable.q_table['Best'] = best_action #add column for best actions at each state
print('\n',qtable.q_table.head(50))
print('\n number of rows: ', len(qtable.q_table.index))
