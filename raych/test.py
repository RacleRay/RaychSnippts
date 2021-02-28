import os
print(__file__)
print(os.path.join(os.path.dirname(__file__), '..'))
print(os.path.dirname(os.path.realpath(__file__)))
print(os.path.abspath(os.path.dirname(__file__)))


from util import logger
from util.info import get_machine_info

logger.info("只是一个测试")
logger.info(get_machine_info())



# import math

# import matplotlib.pyplot as plt

# lr = [1e-8 + (10.0 - 1e-8) * (1 + math.cos(1 * math.pi * batch_num / 300)) / 2 for batch_num in range(1, 301)]
# plt.plot(lr)
# plt.savefig("test.png")