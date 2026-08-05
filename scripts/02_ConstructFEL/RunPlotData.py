import os, sys

### for code test purposes select {remote=True,test=True} 

states = [[True,True]]

for i in range(len(states)):

    cmd = f'python PlotData.py {states[i][0]} {states[i][1]}'
    print(cmd)
    os.system(cmd)