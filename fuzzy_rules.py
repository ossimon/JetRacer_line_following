linguistic_variables = {
    'offset': ['farLeft', 'left', 'center', 'right', 'farRight'],
    'direction': ['hardLeft', 'left', 'straight', 'right', 'hardRight'],
    'speed': ['verySlow', 'slow', 'medium', 'fast', 'veryFast'],
    'steer': ['hardLeft', 'left', 'straight', 'right', 'hardRight']
}

speed_rule_bounds = [
    # (lower bound, upper bound) # offset    direction
    (0, 4),  # far       hardSame
    (1, 4),  # far       softSame
    (2, 4),  # far       straight
    (2, 4),  # far       softOpposite
    (2, 4),  # far       hardOpposite
    (0, 4),  # med       hardSame
    (1, 4),  # med       softSame
    (2, 4),  # med       straight
    (2, 4),  # med       softOpposite
    (2, 4),  # med       hardOpposite
    (2, 4),  # center     hard
    (3, 4),  # center     soft
    (4, 4),  # center     straight
]

steer_rule_bounds = [
    # (lower bound, upper bound) # offset    direction
    (0, 0),  # farLeft    hardLeft
    (0, 1),  # farLeft    left
    (0, 1),  # farLeft    straight
    (1, 3),  # farLeft    right
    (1, 4),  # farLeft    hardRight
    (0, 1),  # left       hardLeft
    (0, 1),  # left       left
    (0, 2),  # left       straight
    (1, 3),  # left       right
    (1, 4),  # left       hardRight
    (0, 1),  # center     hardLeft
    (1, 2),  # center     left
    (2, 2),  # center     straight
]

def loosen_rule_bounds(bounds):
    for i, (lb, ub) in enumerate(bounds):
        if lb != ub:
            continue
        lb = max(0, lb - 1)
        ub = min(4, ub + 1)
        bounds[i] = (lb, ub)
    return bounds

def speed_rule_genotype_to_speed_rules(speed_rule_genotype):
    if len(speed_rule_genotype) != 13:
        raise ValueError("Speed rule genotype must have 13 elements.")
    speed_rules = [
        linguistic_variables['speed'][speed_index] for speed_index in speed_rule_genotype
    ] + [
        linguistic_variables['speed'][speed_index] for speed_index in speed_rule_genotype[:-1][::-1]
    ]
    return speed_rules

def steer_rule_genotype_to_steer_rules(steer_rule_genotype):
    if len(steer_rule_genotype) != 13:
        raise ValueError("Steer rule genotype must have 13 elements.")
    steer_rules = [
        linguistic_variables['steer'][steer_index] for steer_index in steer_rule_genotype
    ] + [
        linguistic_variables['steer'][4 - steer_index] for steer_index in steer_rule_genotype[:-1][::-1]
    ]
    return steer_rules

def get_full_rules(speed_rules, steer_rules):
    speed_rules = speed_rule_genotype_to_speed_rules(speed_rules)
    steer_rules = steer_rule_genotype_to_steer_rules(steer_rules)
    rules = []
    offset_labels = linguistic_variables['offset']
    direction_labels = linguistic_variables['direction']
    for i in range(5):
        for j in range(5):
            idx = i * 5 + j
            if idx < len(speed_rules):
                rules.append({
                    'antecedents': [('offset', offset_labels[i]), ('direction', direction_labels[j])],
                    'consequent': ('speed', speed_rules[idx])
                })
            if idx < len(steer_rules):
                rules.append({
                    'antecedents': [('offset', offset_labels[i]), ('direction', direction_labels[j])],
                    'consequent': ('steer', steer_rules[idx])
                })
    return rules