
import numpy as np

filepath = '../data/trajdata_debug.txt'
output_filepath = '../data/trajdata_debug_reduced.txt'


# read it in
timestep = ''
ndefs = -1
defs = dict()
states = []
frames = []

with open(filepath, 'r') as infile:
    infile.readline() # skip empty first line
    timestep = infile.readline()
    
    ndefs = int(infile.readline())
    for i in range(ndefs):
        vehid = int(infile.readline())
        vehdef = infile.readline()
        defs[vehid] = vehdef
    
    nstates = int(infile.readline())
    for i in range(nstates):
        vehid = int(infile.readline())
        state = infile.readline()
        states.append((state, vehid))
    
    nframes = int(infile.readline())
    for i in range(nframes):
        frame = infile.readline()
        frames.append(frame)    


# reduce the size
max_nframes = 500
last_frame_index = int(frames[max_nframes].strip().split(' ')[-1])
frames = frames[:max_nframes]
states = states[:last_frame_index]
vehids = set()
for (state, vehid) in states:
    vehids.add(vehid)
keys = list(defs.keys())
for k in keys:
    if k not in vehids:
        defs.pop(k)

# set the counts
ndefs = len(defs.keys())
nstates = len(states)
nframes = len(frames)


# write it out
with open(output_filepath, 'w') as outfile:
    outfile.write('\n') # skip empty first line
    outfile.write(timestep)
    
    outfile.write('{}\n'.format(ndefs))
    for (vehid, vehdef) in defs.items():
        outfile.write('{}\n'.format(vehid))
        outfile.write(vehdef)
    
    outfile.write('{}\n'.format(nstates))
    for (state, vehid) in states:
        outfile.write('{}\n'.format(vehid))
        outfile.write(state)
    
    outfile.write('{}\n'.format(nframes))
    for i, frame in enumerate(frames):
        if i >= max_nframes:
            break
        outfile.write(frame)
