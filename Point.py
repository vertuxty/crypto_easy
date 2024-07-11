import numpy as np
import galois

class Point:
    isO = True
    x, y = None, None
    def __init__(self, x, y, x0, y0):
        self.x = x
        self.y = y
        self.x0 = x0
        self.y0 = y0
        self.isO = x == None and y == None

    def __str__(self):
        if self.isO:
            return "O"
        return "Point(%s,%s)"%(self.x, self.y)

    def __eq__(self, value: object) -> bool:
        return self.x == value.x and self.y == value.y
    
    def __repr__(self):
        return str(self)
    
    def __add__(self, other):
        return Point(None, None) 


class EcllipticCurve:

    def __init__(self, A, B, GF, q, m):
        self.A = A
        self.B = B
        self.GF = GF
        self.qm = q ** m
        self.q = q
        self.m = m
        self.__pt = []

    def __str__(self):
        return "y^2=x^3 + %sx + %s"%(self.A, self.B)

    def __eq__(self, value: object) -> bool:
        return self.A == value.A and self.B == value.B
    
    def has_point(self, point: Point):
        return (point.y * point.y) == ((point.x * point.x + self.A) * point.x + self.B)
    
    def get_points(self, x):
        left = x ** 3 + self.A * x + self.B
        for y in range(0, self.qm):
            if self.GF(y) ** 2 == left:
                return self.GF(y), -self.GF(y) 
        return None, None
        # if not left.is_square():
        #     return None, None
        # res = np.sqrt(self.GF([x**3 + self.A * x + self.B]))
        # if (res[0] ** 2 != left):
        #     if ((-res[0]) ** 2 != left):
        #         return None, None
        #     return None, -res[0]
        # return res[0], -res[0]

    def card_eclliptic(self):
        if len(self.__pt) != 0:
            return self.__pt, len(self.__pt)
        self.__pt = [Point(None, None)]
        for x in range(self.qm):
            y1, y2 = self.get_points(self.GF(x))
            if y1 == None and y2 == None:
                continue
            elif y1 == self.GF(0):
                self.__pt.append(Point(self.GF(x), y1, x, y1))
            else:
                self.__pt.append(Point(self.GF(x), y1, x, y1))
                self.__pt.append(Point(self.GF(x), y2, x, y1))
        return self.__pt, len(self.__pt)

    def is_supersingular(self):
        return self.__is_supersingular