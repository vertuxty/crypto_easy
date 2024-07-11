import collections
import itertools
import lecture1 as lec

def prime_factors(n):
    i = 2
    while i * i <= n:
        if n % i == 0:
            n /= i
            yield i
        else:
            i += 1
    if n > 1:
        yield n


def prod(iterable):
    result = 1
    for i in iterable:
        result *= i
    return result


def get_divisors(n):
    pf = prime_factors(n)
    pf_with_multiplicity = collections.Counter(pf)
    powers = [
        [factor ** i for i in range(count + 1)]
        for factor, count in pf_with_multiplicity.items()
    ]
    for prime_power_combo in itertools.product(*powers):
        yield prod(prime_power_combo)

def divisors_list(n):
    list = []
    gen = get_divisors(n)
    for a in gen:
        list.append(int(a))
    return sorted(list)


def find_order(p, g):
    list = divisors_list(p - 1)
    for pow in list:
        if lec.modpow(g, pow, p) == 1:
            return pow
    return p - 1