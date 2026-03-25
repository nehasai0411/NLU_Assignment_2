def novelty(gen,train):

    new = [g for g in gen if g not in train]

    return len(new)/len(gen)

def diversity(gen):

    return len(set(gen))/len(gen)