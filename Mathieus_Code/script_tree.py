from os import execlp, fork, wait, listdir
from sys import argv

PYTHON = 'python3.9'
#PYTHON = 'python3'


treedir = '/data/mdehouck/model_irrelevance/families'
#treedir = '/data/tbidewell/Tree'

# first prepare the parameters of the code you want to run
trains = {}
devs = {}
for file in listdir(treedir):

    '''
    for f in listdir(treedir + "/" + file):
        if 'train' in f:
            fam, par, cur = file.split('_')
            trains[fam, int(par), int(cur)] = treedir + "/" + file + "/" + f
        elif 'dev' in f:
            fam, par, cur = file.split('_')
            devs[fam, int(par), int(cur)] = treedir + "/" + file + "/" + f
    '''

    if 'train.numed' in file:
        fam, par, cur = file.split('-')[0].split('_')
        trains[fam, int(par), int(cur)] = treedir + file 

    elif 'dev.numed' in file:
        fam, par, cur = file.split('-')[0].split('_')
        devs[fam, int(par), int(cur)] = treedir + file 


todo = []
children = {}
for k, v in sorted(trains.items(), reverse = True):

    if k not in devs:
        continue

    if k[1] == 0:
        for mod in ['lstm', 'char']:#, 'transf']:
            todo.append((k, v, devs[k], mod))
            children[k[0], k[2], mod] = []

    else:
        for mod in ['lstm', 'char']:#, 'transf']:
            children[k[0], k[1], mod].append((k, v, devs[k], mod))
            children[k[0], k[2], mod] = []

print(todo)
print(children)

for k, v in sorted(children.items()):
    if len(v) == 0:
        del(children[k])


'''
# then run it
running = {'0':-1, '1':-1}# these are the two gpus on atropos
#running = {'1':-1}
procs = {}
done = set()

while todo != [] or len(children) != 0:

    if len(todo) != 0 and -1 in running.values(): # there at least one free GPU (for me)
            k, train, dev, mod = todo[0]

            #print(k, mod)#, done)
        
            todo = todo[1:]

            gpu = [x for x, y in running.items() if y == -1][0]

            pid = fork() # fork
        
            if pid == 0:
                # run you new code in the child process

                bits = train.split('/')[-1].split('-')[0].split('_')
                this = bits[0] + '_' + bits[-1]
                parent = '_'.join(bits[:2])

                print(k, train, dev, mod, this, parent, gpu)
                if mod == 'char':
                    out = 'results/' + this + '_charlstm'
                    if bits[1] == '0':
                        parent = 'NIL'
                    else:
                        parent = 'models/' + parent + '_charlstm_last'
                    #execlp(PYTHON, PYTHON, 'train_mi_lstm_tree.py', train, '-f', out, '-g', gpu, '-c', '-p', parent)
                else:
                    out = 'results/' + this + '_lstm'
                    if bits[1] == '0':
                        parent = 'NIL'
                    else:
                        parent = 'models/' + parent + '_lstm_last'
                    #execlp(PYTHON, PYTHON, 'train_mi_lstm_tree.py', train, '-f', out, '-g', gpu, '-p', parent)
                #print('running', k, mod)

                s = [i for i in range(1000000)]
                exit()

            
            else:
                # the parent remembers the pid,    could also store the parameters todo[0]
                procs[pid] = k[0], k[1], k[2], mod
                running[gpu] = pid

    else: # if no available GPU, or no next task, wait
        pid, status = wait() # here we could catch the failing parameters and put them back in todo...


        if status != 0:
            with open('failed', 'a') as out:
                print(pid, status, procs[pid], file=out)

        else:
            task = procs[pid]
            done.add(task)
            #print(pid, task)

            key = tuple(task[x] for x in [0, 2, 3])
            if key in children:
                #print(children[tuple(task[x] for x in [0, 2, 3])])
                todo += children[tuple(task[x] for x in [0, 2, 3])]
                del(children[tuple(task[x] for x in [0, 2, 3])])
            
        if running['0'] == pid:
            running['0'] = -1
        elif running['1'] == pid:
            running['1'] = -1
'''
            
        
