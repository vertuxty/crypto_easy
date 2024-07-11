import lecture1 as lec
import main as mn
import math
import random
import numpy as np
import time
from itertools import chain

# ЗАПУСК В САМОМ НИЗУ.
# -------- BASE ---------
def gcd_my(a, b):
    if a == 0:
        return b, 0, 1
    d, x, y = gcd_my(b % a, a)
    return d, y - x * (b // a), x

def inverse_mod(a, p):
    g, x, y = gcd_my(a, p)
    return (x % p + p) % p

def modpow(a, pow, p):
    if pow == 0:
        return 1
    x = modpow(a, pow // 2, p)
    if pow % 2 == 0:
        return x**2 % p
    else:
        return x**2 * a % p

# ---------------------------------------- TASK 2.1 ----------------------------------------
def IsPrimitive(p: int, g: int) -> bool:
    g = g % p
    if p == 2 and g == 1:
        return True
    elif p == 2:
        return False  
    isPrime = lec.probably(p)
    if not isPrime:
        return False
    gcd = math.gcd(p, g)
    if gcd != 1:
        return False
    phi_p = p - 1
    diff = phi_p
    dict = {}
    pi = 2
    while pi * pi <= diff:
        if diff % pi == 0:
            if dict.get(pi) == None:
                dict[pi] = 1
            else:
                dict[pi] = dict[pi] + 1
            while diff % pi == 0:
                diff = diff // pi
        pi += 1
    if diff > 1:
        if dict.get(diff) == None:
            dict[diff] = 1
        else:
            dict[pi] = dict[pi] + 1
    for pi in list(dict):
        isOne = modpow(g, phi_p // pi, p) != 1
        if not isOne:
            return False
    return True

def gen_primitive(p: int): # for fun (not needed)
    dict = []
    phi_p = p - 1
    diff = phi_p
    pi = 2
    while (pi * pi <= diff):
        if diff % pi == 0:
            dict.append(pi)
        while (diff % pi == 0):
            diff /= pi
    if diff > 1:
        dict.append(diff)
    for pi in range(2, p + 1):
        fl = True
        for d in dict:
            fl = fl and modpow(pi, phi_p // d, p) != 1
        if fl:
            return pi
    return -1

# ---------------------------------------- TASK 2.3 ----------------------------------------

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

def factor_base(B: int, n: int):
    return [p for p in eratosphene(B) if legendre_symbol(n, p) == 1]

def find_factorization(fact_base, n):
    ans = []
    dicts = {}
    for p in fact_base:
        if dicts.get(p) == None:
            dicts[p] = 0
        while n % p == 0:
            n /= p
            dicts[p] = (dicts[p] + 1) % 2
    for p in fact_base:
        ans.append(dicts[p])
    return ans

def sieve_nums_p(sieve_nums, pos, p):
    for i in range(pos, len(sieve_nums), p):
        while sieve_nums[i] % p == 0:
            sieve_nums[i] = sieve_nums[i] // p
    return sieve_nums

def sieve_powers_of_two(sieve_nums): 
    pos = 0
    while sieve_nums[pos] % 2 != 0:
        pos += 1
    return sieve_nums_p(sieve_nums, pos, 2)

# tonelli_shanks algorithm 
# https://exploringnumbertheory.wordpress.com/2015/12/09/solving-quadratic-congruences-with-odd-prime-moduli/
def tonelli_shanks(a, p): 
    if (legendre_symbol(a, p) != 1):
        return None, None
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


def sieve_B_smooth(B: int, n: int, ranges: int):
    x_begin = int(math.ceil(math.sqrt(n)))
    nums = [x ** 2 - n for x in range(x_begin, x_begin + ranges)]
    sieve_nums = sieve_powers_of_two(nums.copy())
    fact_base = factor_base(B, n)
    sz = len(sieve_nums)
    for p in fact_base:
        R1, R2 = tonelli_shanks(n, p)
        sieve_nums = sieve_nums_p(sieve_nums, (R1 - x_begin) % p, p)
        sieve_nums = sieve_nums_p(sieve_nums, (R2 - x_begin) % p, p)
    ans_smooth = [nums[i] for i in range(sz) if sieve_nums[i] == 1]
    poses_plus_x_begin = [i + x_begin for i in range(sz) if sieve_nums[i] == 1]
    return ans_smooth, poses_plus_x_begin


def transpose_matrix(M):
    n = len(M[0])
    L = list(chain(*M))
    return [L[i::n] for i in range(n)]


def form_matrix(smooth_vector, fact_base):
    M = []
    for p in smooth_vector:
        factorization = find_factorization(fact_base, p)
        M.append(factorization)
    return M


def add_rows(row1, row2):
    ans = row1.copy()
    for i in range(len(row2)):
        ans[i] = (ans[i] + row2[i]) % 2
    return ans

# https://hyperelliptic.org/tanja/SHARCS/talks06/smith_revised.pdf
# (Binary Gaussian Elimination with Pivoting)
def gauss_elimination(M):
    print("\n\nM:", len(M), len(M[0]), "\n\n")
    M = transpose_matrix(M) #  удобнее вычитать строки а не столбцы.
    n = len(M)
    privot_marks = np.zeros(len(M[0]))
    for row in M:
        for j in range(len(row)):
            if row[j] == 1:
                privot_marks[j] = 1
                ind = M.index(row)
                for k in range(n):
                    if k != ind and M[k][j] == 1:
                        M[k] = add_rows(M[k], row)
                break
    gauss_M = transpose_matrix(M)
    rows = [gauss_M[i] for i in range(len(privot_marks)) if privot_marks[i] == 0]
    return gauss_M, privot_marks, rows, [i for i in range(len(privot_marks)) if privot_marks[i] == 0]

def finds_elements(gauss, privot_marks, fr, pos):
    ans = [pos]
    for k in range(len(gauss)):
        for i in range(len(fr)):
            if fr[i] == 1 and gauss[k][i] == 1 and privot_marks[k]:
                ans.append(k)
                break
    return ans
    
def sqrt_fast(n):
    x = n
    y = (x + 1) // 2
    while y < x:
        x = y
        y = (x + n // x) // 2
    return x

def square_construct(pos, sieve, sizev_pos, n):
    rhs = 1
    lhs = 1
    sqrtN = int(math.ceil(math.sqrt(n)))
    for p in pos:
        rhs *= sizev_pos[p]
        lhs *= sieve[p]
    lhs = sqrt_fast(lhs)
    gcd = mn.gcd_my(rhs - lhs, n)
    return gcd[0], n // gcd[0]

def Quadratic_Sieve(n, B, ranges):
    sievs = sieve_B_smooth(B, n, ranges)
    M = form_matrix(sievs[0], factor_base(B, n))
    gauss = gauss_elimination(M)
    p, q = 1, n
    for j in range(len(gauss[0])):
        p, q = square_construct(finds_elements(gauss[0], gauss[1], gauss[2][j], gauss[3][j]), sievs[0], sievs[1], n)
        if p == 1 or q == 1:
            continue
        return p, q
    return 1, n

# ---------------------------------------- TASK 2.4 ----------------------------------------
# https://ru.wikipedia.org/wiki/Алгоритм_Полига_—_Хеллмана
def Pohlig_Hellman(p, g, a):
    _p = p - 1
    pow_p = 0
    while _p % 2 == 0:
        pow_p += 1
        _p //= 2
    x = 0
    _a = a
    i = 0
    Q = []
    while i < pow_p:
        m = (p - 1) // (2 ** (i + 1))
        z = modpow(_a, m, p)
        if z == p - 1:
            Q.append(i)
            gn = inverse_mod(g, p)
            pow_of_2 = 2 ** i
            _a = (_a * modpow(gn, pow_of_2, p)) % p
        i += 1
    for i in Q:
        x += 2 ** i
    return x

# ---------------------------------------- TASK 2.5 AND part of 2.1----------------------------------------
def BruteForce(p, g, a):
    if not lec.probably(p):
        return None
    if not IsPrimitive(p, g):
        dict = {1: 1}
        prev = 1
        for pi in range(1, p + 1):
            if pi % 100000 == 0:
                print(pi)
            diff_a = prev * g % p
            prev = diff_a
            if diff_a == a:
                return pi
            if diff_a not in dict:
                dict[diff_a] = 1
            else:
                return None           
        return None
    a = a % p
    m = math.sqrt(p) + 1
    for pi in range(p + 1):
        if modpow(g, pi, p) == a:
            return pi


# ---------------------------------------- TASK 2.6 ----------------------------------------

def Babystep_giantstep(p, g, a):
    n = math.floor(math.sqrt(p)) + 1
    dict1 = {a : 0}
    dict2 = {1 : 0}
    prev_1 = a
    prev_2 = 1
    gn = inverse_mod(modpow(g, n, p), p)
    for pow in range(1, n + 1):
        prev_1 = prev_1 * gn % p
        prev_2 = prev_2 * g % p
        dict1[prev_1] = pow
        dict2[prev_2] = pow
    for e in dict1:
        if e in dict2:
            j = dict2[e]
            i = dict1[e]
            return (i * n + j)
    return None; 

# ---------------------------------------- TASK 2.7 ----------------------------------------

# МЕДЛЕННЫЙ на больших числах :(
def ProbabilityCollision(p, g, a):
    n = 3 * math.floor(math.sqrt(p)) + 1
    pw1 = [random.randrange(1, p + 1, 1) for i in range(n + 1)]
    random.seed(132342)
    pw2 = [random.randrange(1, p + 1, 1) for i in range(n + 1)]
    Z = {a : 1}
    Y = {1: 1}
    for i in range(1, n + 1):
        pow1 = pw1[i]
        pow2 = pw2[i]
        power1 = modpow(g, pow1, p) 
        power2 = (a * modpow(g, pow2, p)) % p
        Y[power1] = pow1
        Z[power2] = pow2
    for e in Z:
        if e in Y:
            j = Z[e]
            i = Y[e]
            if i < j:
                return i - j + p - 1
            return i - j
    return None; 

# ---------------------------------------- TASK 2.8 ----------------------------------------
    
def map(x: int, g: int, a: int, alpha: int, beta: int, p: int):
    x = x % p
    if 0 <= x and x < p / 3:
        return g * x % p, (alpha + 1) % (p - 1), beta 
    elif p / 3 <= x and x < 2 * p / 3:
        return x * x % p, 2 * alpha % (p - 1), 2 * beta % (p - 1)
    return a * x % p, alpha, (beta + 1) % (p - 1)

def PollardRho(p: int, g: int, a: int):
    x, alpha, beta = 1, 0, 0
    y, gamm, delt = 1, 0, 0
    i = 1
    while y != x or i == 1:
        i += 1
        x, alpha, beta = map(x, g, a, alpha, beta, p)
        y, gamm, delt = map(y, g, a, gamm, delt, p)
        y, gamm, delt = map(y, g, a, gamm, delt, p)
    if alpha - gamm < 0:
        pow_g = alpha - gamm + p - 1
    else:
        pow_g = alpha - gamm
    if delt - beta < 0:
        pow_a = delt - beta + p - 1
    else:
        pow_a = delt - beta
    gcd = gcd_my(pow_a, p - 1)
    hyt = gcd[1] 
    if hyt < 0:
        hyt += p - 1
    pow_g = pow_g * hyt % (p - 1)
    pow_a = gcd[0]
    div_p_d = (p - 1) // gcd[0]
    s = pow_g // pow_a
    for k in range(gcd[0] + 1):
        t = (s + k * div_p_d) % p
        if modpow(g, t, p) == a:
            return t
    return None

# ---------------------------------------- TESTS ----------------------------------------
# ---------------------------------------- TASK 1 ----------------------------------------
def IsPRIMITIVE(p, g):
    print("------------------")
    print("Testing IsPrimitive...")
    print("Input p, g: ", p, g)
    start = time.time()
    ans = IsPrimitive(p, g)
    end = time.time()
    print("Result: ", ans)
    print("Time: ", end - start)
    print("------------------")

def test_is_prim():
    data = [
        (233232332323327, 54), # 0
        (233232332323327, 119), # 1
        (423242452342342289, 423242452342342219),
        (423242452342342289, 423242452342342218),
    ]
    for i in data:
        p, g = i
        IsPRIMITIVE(p, g)

# ---------------------------------------- TASK 3 ----------------------------------------

def QS(n, B, range):
    print("------------------")
    print("Testing QS...")
    print("Input n, B, range: ", n, B, range)
    start = time.time()
    p, q = Quadratic_Sieve(n, B, range)
    end = time.time()
    print("Result: ", p, q)
    print("Is correct: ", p * q == n)
    print("Time: ", end - start)
    print("------------------")


def test_QS():
    data = [
        (15347, 29, 100), # 0
        (221, 11, 100), # 1
        (210593097461, 10000, 10000),
        (2518487 * 51295499, 10000, 10000) # 2
    ]
    for i in data:
        n, B, range = i
        QS(n, B, range)

# ---------------------------------------- TASK 4 ----------------------------------------

def pohlig_hellman_ts(p, g, a):
    print("------------------")
    print("Testing pohlig_hellman...")
    print("Input (p, g, a), g^x = a mod p: ", p, g, a)
    start = time.time()
    ans = Pohlig_Hellman(p, g, a)
    end = time.time()
    print("Result: ", ans)
    print("Is correct: ", modpow(g, ans, p) == a)
    print("Time: ", end - start)
    print("------------------")

def test_pohlig_hellman():
    data = [
        (17, 7, 11),
        (257, 103, 199),
        (65537, 57, 13242)
    ]
    for i in data:
        p, g, a = i
        pohlig_hellman_ts(p, g, a)

# ---------------------------------------- TASK 5-8 ----------------------------------------

def brut_force(p, g, a):
    print("------------------")
    print("Testing brutforce...")
    print("Input (p, g, a), g^x = a mod p: ", p, g, a)
    start = time.time()
    ans = BruteForce(p, g, a)
    end = time.time()
    print("Result: ", ans)
    print("Is correct: ", modpow(g, ans, p) == a)
    print("Time: ", end - start)
    print("------------------")
    return ans


def baby_step_gaint_step(p, g, a):
    print("------------------")
    print("Testing babystepgaintstep...")
    print("Input (p, g, a), g^x = a mod p: ", p, g, a)
    start = time.time()
    ans = Babystep_giantstep(p, g, a)
    end = time.time()
    print("Result: ", ans)
    print("Is correct: ", modpow(g, ans, p) == a)
    print("Time: ", end - start)
    print("------------------")
    return ans


def probability_step(p, g, a):
    print("------------------")
    print("Testing probability collitiion...")
    print("Input (p, g, a), g^x = a mod p: ", p, g, a)
    start = time.time()
    ans = ProbabilityCollision(p, g, a)
    end = time.time()
    print("Result: ", ans)
    print("Is correct: ", modpow(g, ans, p) == a)
    print("Time: ", end - start)
    print("------------------")
    return ans


def rho_pollard_test(p, g, a):
    print("------------------")
    print("Testing rhopollard...")
    print("Input (p, g, a), g^x = a mod p: ", p, g, a)
    start = time.time()
    ans = PollardRho(p, g, a)
    end = time.time()
    print("Result: ", ans)
    print("Is correct: ", modpow(g, ans, p) == a)
    print("Time: ", end - start)
    print("------------------")
    return ans


def test_last_tasks():
    data = [
        (17, 7, 11),
        (659, 2, 390),
        (48611, 19, 24717),
        (17389, 122, 13896),
        (17389, 9704, 13896), 
        (7597537516031, 234212331, 2312312324),
        (233232332323327, 141, 2312312324)
    ]
    for i in range(len(data)):
        p, g, a = data[i]
        baby_step_gaint_step(p, g, a)
        if IsPrimitive(p, g):
            print(p, g, a)
            rho_pollard_test(p, g, a)
        if i < 5: # т.к бурт форс долгий, а вероятностный не знаю как сделать быстрее, надо генерить числа + возводить в степень
            probability_step(p, g, a) # быстрое возведение не помогает
            brut_force(p, g, a)

def print_header(test_name):
    print()
    print("==================")
    print(test_name)
    print("==================")
    print()

if __name__ == '__main__':
    # print("START OF TESTS")
    # print_header("TASK 1 TESTs")
    # test_is_prim()
    # print_header("TASK 3 TESTs")
    test_QS()
    # print_header("TASK 4 TESTs")
    # test_pohlig_hellman()
    # print_header("TASK 5-8 TESTs")
    # test_last_tasks()
    # print("END OF TESTS")
