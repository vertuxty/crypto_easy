# ---------------------------------------- TASK 2.4 ----------------------------------------
def gcd_my(a, b):
    if a == 0:
        return b, 0, 1
    d, x, y = gcd_my(b % a, a)
    return d, y - x * (b // a), x

def inverse_mod(a, p):
    g, x, y = gcd_my(a, p)
    return (x % p + p) % p

def pohlig_hellman(p, g, a):
    if not IsPrimitive(p, g):
        print(g, "is not primitive")
        return None
    x = 0
    pow = 0
    pcopy = p - 1
    while pcopy % 2 == 0:
        pow +=1
        pcopy //= 2
    powcopy = pow
    while powcopy % 2 == 0:
        powcopy //=2
    if pcopy != 1 or powcopy != 1:
        print(p, "is not 2^(2^k) + 1")
        return None
    i = 1
    bits = []
    A = a
    while i < pow + 1:
        pw = (p - 1) // (2 ** i)
        b = lec.modpow(A, pw, p)
        G = lec.modpow(g, (p - 1) // 2, p)
        k = 0
        while True:
            if lec.modpow(G, k, p) == b:
                bits.append(k)
                break
            k += 1
        gn = inverse_mod(g, p)
        A = (A * lec.modpow(gn, k * 2 ** (i - 1), p)) % p
        i = i + 1
    pw = 0
    for i in range(len(bits)):
        x += bits[i] * (2 ** i)
    return x

def pohlig_hellman_ts(p, g, a):
    print("------------------")
    print("Testing polhlig_hellman...")
    print("Input (p, g, a), g^x = a mod p: ", p, g, a)
    ans = pohlig_hellman(p, g, a)
    print("Result: ", ans)
    print("Is correct: ", lec.modpow(g, ans, p) == a)
    print("------------------")

def test_pohlig_hellman():
    data = [
        (17, 7, 2),
        (65537, 57, 13242)
    ]
    p, g, a = data[0]
    pohlig_hellman_ts(p, g, a)
    p, g, a = data[1]
    pohlig_hellman_ts(p, g, a)