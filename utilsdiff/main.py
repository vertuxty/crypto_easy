import math
import random
from typing import List
import lecture1 as lp
alpha, betta = 0, 0

def gcd_my(a, b):
    if a == 0:
        return b, 0, 1
    d, x, y = gcd_my(b % a, a)
    return d, y - x * (b // a), x


def F1(e: int, a: int, p: int): # p -is prime
    # find d: ed == 1 mod p - 1
    e %= (p - 1)
    gcd = gcd_my(e, p - 1)
    if gcd[0] == 1:
        return lp.modpow(a, gcd[1] + (p - 1) if gcd[1] < 0 else gcd[1], p)
    return -1


def F2(e: int, a: int, p: int, q: int): # N = p * q, (e, phi(N)) = 1
    gcd = gcd_my(e, (p - 1) * (q - 1))
    if gcd[0] == 1:
        return lp.modpow(a, gcd[1] + (p - 1) * (q - 1) if gcd[1] < 0 else gcd[1], p * q) 
    return -1


def RSA():
    p = lp.next_prime(random.randint(10**10, 10**11))
    q = lp.next_prime(random.randint(10**10, 10**11))
    N = p * q
    phi = (p - 1) * (q - 1)
    msg = 24332
    e = 2
    while True:
        e = random.randint(2, phi)
        if gcd_my(e, phi)[0] == 1:
            break

    def RSA_endcoding(msg: int, e: int, N: int):
        return lp.modpow(msg, e, N)
    
    def RSA_decoding(msg: int, e: int):
        return F2(e, msg, p=p, q=q)

    encoded = RSA_endcoding(msg=msg, e=e, N=N)
    decoded = RSA_decoding(e=e, msg=encoded)
    print(encoded, decoded)

def fact(N: int):
    if (N == 0 or N == 1):
        return 1
    else:
        return fact(N - 1) * N 

def pollard_factorization(N: int):
    a = 2
    n = 10
    fac = math.factorial(n)
    pow = lp.modpow(a, fac, N) - 1 
    while (True):
        gcd = math.gcd(pow, N) % N
        if gcd == 1:
            pow = lp.modpow(pow + 1, n + 1, N)
            n += 1 
        elif (1 < gcd < N):
            return gcd
        else:
            a = a + 1
            n = 10
            fac = math.factorial(n)
            pow = lp.modpow(a, fac, N) - 1 


def mat(array: List[int], N: int):
    prime = []
    for i in range(2, 50):
        if lp.probably(i):
            prime.append(i)
    n = len(array)
    matrix = [[0 for i in range(n)] for i in range(prime.__len__)]
    c_array = [lp.modpow(a, 2, N) for a in array]
    for i in range(c_array.__len__):
        c = c_array[i]
        for j in range(prime.__len__):
            while (c % prime[j] == 0):
                c //= prime[j]
                

    print(matrix)
    return 0

# Выделение полного квадрата. N = p * q. Пусть N = a^2 - b^2 = (a - b)(a + b)
# Пусть kN = a^2 - b^2 = k (a - b) (a + b). Найти разные a, b такие что a^2 == b^2 mod N
# Как находить комбинации с_i = (p1)^k11...(pm)^(k1m)
def solve():
    # print(pollard_factorization(168441398857))
    # print(scipy.array([[1, 2], [2, 4]]))
    mat([2, 3, 10, 23, 43, 43], N=33)
    return 0

if __name__ == '__main__': 
    solve()