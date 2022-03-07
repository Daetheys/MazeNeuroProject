from ray.tune.logger import pretty_print
from agent import agent


import logging
logging.basicConfig(level=logging.INFO)

import time

l = ['episode_len_mean','episode_reward_max','episode_reward_mean','episode_reward_min']

for i in range(150):
    result = agent.train()
    print(i)
    for k in l:
        print(k,result[k])
    #print(pretty_print(result))

agent.save('checkpoint-'+str(time.time()).split('.')[0])