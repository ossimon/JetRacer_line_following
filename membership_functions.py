linguistic_variables = {
    'offset': ['farLeft', 'left', 'center', 'right', 'farRight'],
    'direction': ['hardLeft', 'left', 'straight', 'right', 'hardRight'],
    'speed': ['verySlow', 'slow', 'medium', 'fast', 'veryFast'],
    'steer': ['hardLeft', 'left', 'straight', 'right', 'hardRight']
}

default_membership_functions = {
    'offset': [-1, -0.5, 0, 0.5, 1],
    'direction': [-1, -0.5, 0, 0.5, 1],
    'speed': [0, 0.25, 0.5, 0.75, 1],
    'steer': [-1, -0.5, 0, 0.5, 1]
}

mf_bounds = [(0, 1) for _ in range(10)]  # 10 membership function bounds

def mf_genotype_to_membership_functions(membership_function_genotype):
    """
    Genotype translation to membership functions:
    genotype is like [a, b, c, d, e, f, g, h, i, j]

    'offset': [-1, -a / (a+b), 0, a / (a+b), 1],
    'direction': [-1, -c / (c+d), 0, c / (c+d), 1],
    'speed': [0, e / (e+f+g+h), (e+f) / (e+f+g+h), (e+f+g) / (e+f+g+h), 1],
    'steer': [-1, -i / (i+j), 0, i / (i+j), 1]
    """

    if len(membership_function_genotype) != 10:
        raise ValueError("Membership function genotype must have 10 elements.")

    membership_function_genotype = [max(1e-4, x) for x in membership_function_genotype]
    a, b, c, d, e, f, g, h, i, j = membership_function_genotype

    membership_functions = {
        'offset': [-1, -a / (a + b), 0, a / (a + b), 1],
        'direction': [-1, -c / (c + d), 0, c / (c + d), 1],
        'speed': [0, e / (e + f + g + h), (e + f) / (e + f + g + h), (e + f + g) / (e + f + g + h), 1],
        'steer': [-1, -i / (i + j), 0, i / (i + j), 1]
    }

    return membership_functions