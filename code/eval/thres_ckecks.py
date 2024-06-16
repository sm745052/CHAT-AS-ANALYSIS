def thres_checkQB(x, y, pred, cost, subword_len, threshold):
    if(subword_len=='-'):
        return False
    cost = float(cost)
    subword_len = float(subword_len)
    if(pred == ""):
        return False
    if(len(pred) == 0):
        return False
    return cost / (subword_len) < threshold

from string import punctuation
def thres_checkGPT2(x, y, pred, cost, subword_len, threshold):
    if(subword_len=='-'):
        return False
    cost = float(cost)
    subword_len = float(subword_len)
    # print(y, " --------- ", pred)
    # print(y==pred)
    # print(cost/subword_len)
    if(len(pred) == 0):
        return False
    return cost/subword_len  < threshold


def thres_checkMPC(x, y, pred, cost, subword_len, threshold):
    if(subword_len=='-'):
        return False
    if(pred==""):
        return False
    if(len(pred) == 0):
        return False
    return True

def thres_checkNGAME(x, y, pred, cost, subword_len, threshold):
    if(subword_len=='-'):
        return False
    cost = float(cost)
    if(pred==""):
        return False
    if(len(pred) == 0):
        return False
    return cost > threshold

def thres_checkGPT4(x, y, pred, cost, subword_len, threshold):
    if(subword_len=='-'):
        return False
    if(pred==""):
        return False
    if(len(pred) == 0):
        return False
    return True