import Point

def gcd_my(a, b):
    if a == 0:
        return b, 0, 1
    d, x, y = gcd_my(b % a, a)
    return d, y - x * (b // a), x

def inverse_mod(a, p):
    g, x, y = gcd_my(a, p)
    return (x % p + p) % p

def mod(v, p):
    return (v % p + p) % p

def modpow(a, pow, p):
    if pow == 0:
        return 1
    x = modpow(a, pow // 2, p)
    if pow % 2 == 0:
        return x**2 % p
    else:
        return x**2 * a % p

def legendre_symbol(n: int, p: int):
    return modpow(n, (p - 1) // 2, p)

def eratosphene(n: int):
    primes = [True for x in range(n + 1)]
    for p in range(2, n + 1):
        if primes[p]:
            for g in range(2 * p, n + 1, p):
                primes[g] = False
    ans = [x for x in range(2, n + 1) if primes[x] == True]
    return ans

def tonelli_shanks(a, p):
    assert legendre_symbol(a, p) == 1, "Not a square!"
        # print("Not a square!")
        # return None, None
    Q = p - 1
    S = 0
    while Q % 2 == 0:
        Q = Q // 2
        S += 1
    if S == 1:
        R = modpow(a, (p + 1)//4, p)
        return R, p - R
    y = 0
    for i in range(2, p):
        if legendre_symbol(i, p) == p - 1:
            y = i
            break    
    R = modpow(a, (Q + 1) // 2, p)
    c = modpow(y, Q, p)
    t = modpow(a, Q, p)
    E = S
    while t % p != 1:
        prep = modpow(t, 2, p)
        i_min = 0
        for i in range(1, E):
            if prep % p == 1:
                i_min = i
                break
            prep = modpow(prep, 2, p)
        b = modpow(c, 2 ** (E - i_min - 1), p)
        R = (R * b) % p
        c = (b * b) % p
        t = (c * t) % p
        E = i_min
    return R, p - R