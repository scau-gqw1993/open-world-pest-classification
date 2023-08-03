import numpy as np


def sample(data, n_way, k_shot, q_query):
    res = []

    data = np.array(data, dtype=object)
    ns = np.random.choice(data, n_way, replace=False)
    for var in ns:
        res.append(np.random.choice(var, k_shot + q_query, replace=False).tolist())

    return res


if __name__ == '__main__':
    a = [
        [1, 2, 4, 5],
        ['a', 'b', 'c', 'd', 'e'],
        ['A', 'B', 'C', 'D'],
        [10, 20, 11, 32, 31]
    ]

    for _ in range(10):
        rs = sample(a, 3, 2, 1)
        print(rs)
