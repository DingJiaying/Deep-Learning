import math

def ComputeDistance(x1,y1,x2,y2):
    d = math.sqrt(math.pow(x1 - x2, 2) + math.pow(y1 - y2, 2) )
    return d

d_ag = ComputeDistance(3,104,18,90)

print("The diStance is :" + str(d_ag))
