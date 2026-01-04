#!/usr/bin/env python3
"""Create Phase 7 pulse mechanism configuration variants"""

import yaml

# Load base config
with open('configs/benchmark/phase7/p7_pulse_base.yaml', 'r') as f:
    base_config = yaml.safe_load(f)

# Define Phase 7 experiments
experiments = [
    {
        'id': '7.1',
        'name': 'pulse_off',
        'desc': 'Pulse mechanism disabled (baseline)',
        'enabled': False,
        'mode': 'adaptive',
        'boost': 0.3
    },
    {
        'id': '7.2',
        'name': 'pulse_static',
        'desc': 'Static threshold trigger, boost_ε=0.3',
        'enabled': True,
        'mode': 'static',
        'boost': 0.3
    },
    {
        'id': '7.3',
        'name': 'pulse_adaptive',
        'desc': 'Adaptive threshold (current default)',
        'enabled': True,
        'mode': 'adaptive',
        'boost': 0.3
    },
    {
        'id': '7.4',
        'name': 'pulse_strong',
        'desc': 'Adaptive with stronger boost_ε=0.5',
        'enabled': True,
        'mode': 'adaptive',
        'boost': 0.5
    },
    {
        'id': '7.5',
        'name': 'pulse_noboost',
        'desc': 'Adaptive trigger, no epsilon boost',
        'enabled': True,
        'mode': 'adaptive',
        'boost': None
    }
]

# Create config for each experiment
for idx, exp in enumerate(experiments, 1):
    config = base_config.copy()

    # Update pulse mechanism settings
    config['training']['learning_rate_scheduler']['pulse_mechanism']['enabled'] = exp['enabled']
    config['training']['learning_rate_scheduler']['pulse_mechanism']['trigger_mode'] = exp['mode']
    config['training']['learning_rate_scheduler']['pulse_mechanism']['boost_epsilon'] = exp['boost']

    # Disable adaptive threshold for static mode
    if exp['mode'] == 'static':
        config['training']['learning_rate_scheduler']['pulse_mechanism']['adaptive_threshold']['enabled'] = False

    # Save config
    output_file = f'configs/benchmark/phase7/p7_{exp["name"]}.yaml'
    with open(output_file, 'w') as f:
        # Write header comment
        f.write(f"# Phase {exp['id']}: Pulse Mechanism - {exp['desc']}\n")
        f.write("# Benchmark Campaign - Music Generation RL System\n")
        f.write("# Base: Phase 1-6 validated configuration\n\n")
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    print(f"Created: {output_file}")

print("\nPhase 7 configs created successfully!")
