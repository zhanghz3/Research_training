import random
def parent_selection(pop, m):
    parents = random.choices(pop, k=m)
    return parents