def get_closest_point2d(p1, p2, p3):

    '''
    Where p1 is a single point, and p2 and p3 are points of a line segment

    Adapted from: 
    https://math.stackexchange.com/questions/2483734/get-point-from-line-segment-that-is-closest-to-another-point, 
    https://www.desmos.com/calculator/qbtyssnnmf and 
    https://stackoverflow.com/questions/849211/shortest-distance-between-a-point-and-a-line-segment
    '''

    p1x, p1y = p1
    p2x, p2y = p2
    p3x, p3y = p3

    A = p1x - p2x
    B = p1y - p2y
    C = p3x - p2x
    D = p3y - p2y

    dot = A * C + B * D
    len_sq = C * C + D * D
    param = -1

    if (len_sq != 0): # in case of 0 length line
        param = dot / len_sq

    ax = p3x - p1x
    ay = p3y - p1y

    bx = p2x - p3x
    by = p2y - p3y

    t = - ( (ax * bx) + (ay * by) ) / ( bx ** 2 + by ** 2)

    if 0 < param <= 1:
        p4x = p3x + t * (p2x - p3x)
        p4y = p3y + t * (p2y - p3y)
        return (p4x, p4y)
    elif param < 0:
        return (p2x, p2y)
    else:
        return (p3x, p3y)
    

if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt


    p1 = (0,4)
    p2 = (2,5)
    p3 = (3,4)

    print(get_closest_point2d(p1,p2,p3))
