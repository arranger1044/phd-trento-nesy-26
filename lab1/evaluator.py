import torch

EPSILON = 10e-12


# def log1mexp(x):
#     """
#     Numerically accurate evaluation of log(1 - exp(x)) for x < 0.
#     See [Maechler2012accurate]_ for details.
#     https://github.com/pytorch/pytorch/issues/39242
#     """
#     mask = CUTOFF < x  # x < 0
#     return torch.where(
#         mask,
#         (-x.expm1() + EPSILON).log(),
#         (-x.exp() + EPSILON).log1p(),
#     )


class ProbSemiring:

    @staticmethod
    def one():
        return 1.0

    @staticmethod
    def zero():
        return 0.0

    @staticmethod
    def plus(a, b):
        return a + b

    @staticmethod
    def times(a, b):
        return a * b

    @staticmethod
    def value(a):
        return a

    @staticmethod
    def negate(a):
        return 1.0 - a


class LogProbSemiring:

    @staticmethod
    def one():
        raise NotImplementedError()

    @staticmethod
    def zero():
        raise NotImplementedError()

    @staticmethod
    def plus(a, b):
        raise NotImplementedError()

    @staticmethod
    def times(a, b):
        raise NotImplementedError()

    @staticmethod
    def value(a):
        raise NotImplementedError()

    @staticmethod
    def negate(a):
        raise NotImplementedError()


def evaluate_formula(formula, probs, semiring):
    assert formula, manager.var_count() == probs.shape[-1]

    def init_cache(pos_literals, pos_values, semiring):
        cache = {}
        for l, v in zip(pos_literals, pos_values):
            neg_l = l.negate()

            cache[l] = semiring.value(v)
            cache[neg_l] = semiring.negate(v)
        return cache

    pos_literals = formula.manager.vars
    cache = init_cache(pos_literals, probs, semiring)

    result, _ = evaluate_node(formula, cache, semiring)

    return result


def evaluate_node(node, cache, semiring):
    if node in cache:
        return cache[node], cache
    if node.is_false():
        return semiring.zero(), cache
    if node.is_true():
        return semiring.one(), cache
    if node.is_decision():
        result = semiring.zero()
        for prime, sub in node.elements():
            rprime, cache = evaluate_node(prime, cache, semiring)
            rsub, cache = evaluate_node(sub, cache, semiring)
            r_primesub = semiring.times(rprime, rsub)
            result = semiring.plus(result, r_primesub)
        cache[node] = result
        return result, cache
