import multiprocessing
import os
from functools import partial
import platform
import binascii

baseSeed = int(str(int(binascii.hexlify(platform.node().encode('utf-8')), 16))[-3:])

def run(configuration,i):

    vi = int(i % 2**32)

    print("python singleRoomScenario.py %s %s" % (configuration,str(int(vi))))
    os.system("python singleRoomScenario.py %s %s" % (configuration,str(int(vi))))


workers = multiprocessing.cpu_count()
print(workers)

confList = ["runningConf.json"]

with multiprocessing.Pool(workers) as pool:
    for j,conf in enumerate(confList):
        rfunc = partial(run,conf)
        pool.map(rfunc,[1e5*baseSeed+1e4*j+x for x in range(1500)])
    