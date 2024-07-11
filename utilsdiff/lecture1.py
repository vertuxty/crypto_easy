import random
import math

def ferma_test(n: int):
    ans = []
    for a in range(1, n):
        if (pow(a, n - 1) % n != 1):
           ans.append(a)
    if len(ans) == 0 and n != 561:
        return n
    return -1


def check_ferma():
    for n in range(2, 100):
        if ferma_test(n) != -1:
            print(n)

def modpow(a, pow, p):
    if pow == 0:
        return 1
    x = modpow(a, pow // 2, p)
    if pow % 2 == 0:
        return x**2 % p
    else:
        return x**2 * a % p

def fast_pow(a, pow):
    if pow == 0:
        return 1
    x = fast_pow(a, pow // 2)
    if pow % 2 == 0:
        return x ** 2
    else:
        return x ** 2 * a

def my_pow(n: int):
    k = 0
    div = n - 1
    while ((div % 2) == 0):
        div = div // 2
        k += 1
    q = (n - 1) // pow(2, k)
    return [q, k]


def miller_ryabin(n: int, a: int, k:int, q: int):
    witnesses = []
    flag = 0
    if modpow(a, q, n) % n == 1:
        return False
    if (modpow(a, q, n) % n == (n - 1)):
        return False
    for m in range(1, k):
        if (pow(a, pow(2, m) * q, n) % n == n - 1):
            return False
    return True


def check_miller_ryabin(n: int):
    witnesses = []
    [q, k] = my_pow(n)
    count = 0
    for a in range(2, int(math.sqrt(n))):
        if miller_ryabin(n, a, k, q):
            return False
            # count += 1
            # witnesses.append(a)
    if len(witnesses) == 0 and n != 561:
        return True
    else:
        return False
        # print("n: ", n)
        # print("witnesses:", count / (n - 2))


def next_prime(n: int):
    next = n + 2
    while (True):
        for i in range(20):
            [q, k] = my_pow(next)
            if miller_ryabin(next, random.randint(2, next - 1), k, q):
                break
            else:
                return next
        next += 2


def q_simple_prime():
    for i in range(100):
        # print("i")
        num = next_prime(random.randint(10**5, 10**6))

        if probably(num * 2 + 1):
            print("ok", num, 2 * num + 1)
            return num, 2 * num + 1

def get_random_element(p: int):
    return random.randint(0, p - 1)

def get_key_g():
    q, p = q_simple_prime()
    for i in range(100):
        g = modpow(get_random_element(p), (p - 1) // q, p)
        if g != 1:
            return g
    return -1


def get_key(p: int, g: int, key: int):
    return modpow(g, key, p)

def probably(n: int):
    [q, k] = my_pow(n)
    for i in range(40):
        if miller_ryabin(n, random.randint(2, n - 1), k, q):
            return False
    return True


def diffie_hellman():
    p = 941
    g = 627
    g = get_key_g()
    print(g)
    a = random.randint(1, 10**4)
    b = random.randint(1, 10**4)
    ga = get_key(p, g, a)
    gb = get_key(p, g, b)
    privet_key_A = get_key(p, ga, b)
    privet_key_B = get_key(p, gb, a) 
    print(ga, gb)
    print(privet_key_A, privet_key_B)


def solve():
    t = 101000454895823409383242369
    g = diffie_hellman()
    # print(g)
    for i in range(20):
            [q, k] = my_pow(t)
            print(q, k)
            if miller_ryabin(t, random.randint(2, t - 1), k, q):
                print(False)
                break
            else:
                print(True)
                return next
    return 0


if __name__ == '__main__':
    solve()
