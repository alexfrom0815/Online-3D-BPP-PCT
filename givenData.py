import numpy as np

###################################################
################# Gym Setting #####################
###################################################
container_size = [10,10,10]
# container_size = [1,1,1]

lower = 1
higher = 5
increase = 1
item_size_set = []
for i in range(lower, higher + 1):
    for j in range(lower, higher + 1):
        for k in range(lower, higher + 1):
                item_size_set.append((i * increase, j * increase , k *  increase))

