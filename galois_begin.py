import galois
import random
import sagemath
import numpy as np
import utilsdiff.lecture1 as utils
from Point import Point
from Point import EcllipticCurve
from time import time

def gcd_my(a, b):
    if a == 0:
        return b, 0, 1
    d, x, y = gcd_my(b % a, a)
    return d, y - x * (b // a), x

def inverse_mod(a, p):
    g, x, y = gcd_my(a, p)
    return (x % p + p) % p

def kU_plus(A: int, B: int, q: int, m: int, P: Point, Q: Point, GL):
    # print("adds: ", P, P.isO)
    # print("adds: ", Q, Q.isO)
    if P.isO:
        return Q
    if Q.isO:
        return P
    if P.x == Q.x and P.y == -Q.y:
        return Point(None, None, None, None)
    s = None
    if P == Q:
        if P.y == 0:
            return Point(None, None, None, None)
        s = (GL(3) * P.x ** 2 + GL(A)) / (GL(2) * P.y)
    else:
        s = (P.y - Q.y) / (P.x - Q.x)

    xR = s ** 2 - P.x - Q.x
    yR = - P.y + s * (P.x - xR)

    return Point(xR, yR, xR, yR)


def frob_end(point: Point, q):
    x = point.x ** q
    y = point.y ** q
    return Point(x, y, point.x0 ** q % q, point.y0 ** q % q)


def is_supersingular(curve: EcllipticCurve):
    poly = galois.Poly([1, -3, curve.A, curve.B], field=GF) ** ((q - 1) // 2)
    return poly.coeffs[q - 1] == 0, poly.coeffs


def multn(n, P, curve):
    Q = P
    R = Point(None, None, None, None)
    while n > 0:
        if n % 2 == 1:
            R = kU_plus(curve.A, curve.B, curve.q, curve.m, R, Q, curve.GF)
        Q = kU_plus(curve.A, curve.B, curve.q, curve.m, Q, Q, curve.GF)
        n = n // 2
    return R


# def DiffieHellman(point: Point, curve: EcllipticCurve):
#     nA = random.randint(1, )

def solve(p, m):
    return 0

if __name__ == '__main__':
    q = 3623
    m = 1
    t = q ** m
    print(t)
    GF = galois.GF(t)
    print(GF.properties)
    A = GF(14)
    B = GF(19)
    curve = EcllipticCurve(A, B, GF, q, m)
    # with open('out.txt', 'w') as f:
    #     print(GF.arithmetic_table("*"), file=f)
    # print(GF.arithmetic_table("+"))
    P = Point(GF(6), GF(730), 6, 730)
    Q = Point(GF(85), GF(112), 85, 112)
    R = Point(None, None, None, None)
    print(multn(947, P, curve))
    for i in range(1, 947 + 1):
        R = kU_plus(A, B, q, m, P, R, GF)
    print(R)