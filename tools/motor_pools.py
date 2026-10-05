"""
The motor pools, in one place.

`POOLS` is the table step 2 measured the resistance reflex through and step 3
drove the animal with: each entry is a named *muscle set*, and a pool is the set
of motor neurons that BANC says innervate those muscles on one leg. It lives in
this file rather than in one of those tools because step 6 needs the same table
on the phone, and a second copy of a table like this is how the two ends of a
closed loop stop describing the same animal.

`pool_groups()` turns it into predicate/name pairs over the BANC meta table, in
the same form `build_group_specs()` in `tools/build_banc.py` already uses, so
the app's connectome carries one named group per (leg, side, muscle set) and
`SimulationEngine.groupRate()` can read the spikes of the pool that drives one
joint in one direction.
"""

# A named muscle set, in the animal's own vocabulary: the values are the BANC
# `peripheral_target_type` labels for the muscles the motor neuron innervates.
POOLS = {
    "tibia flexor":   {"tibia_flexor_muscle", "accessory_tibia_flexor_muscle"},
    "tibia extensor": {"tibia_extensor_muscle"},
    "trochanter flexor": {"trochanter_flexor_muscle",
                          "accessory_trochanter_flexor_muscle"},
    "trochanter extensor": {"trochanter_extensor_muscle",
                            "tergotrochanter_extensor_muscle",
                            "sternotrochanter_extensor_muscle"},
    "coxa rotator ant":  {"sternal_anterior_rotator_muscle"},
    "coxa rotator post": {"sternal_posterior_rotator_muscle"},
    "femur reductor":    {"femur_reductor_muscle"},
    "long tendon":       {"long_tendon_muscle"},
    "tarsus depressor":  {"tarsus_depressor_muscle"},
    "tarsus levator":    {"tarsus_levator_muscle"},
}

# Which joint each pool moves, and which way, in the model's vocabulary. The
# sign of qpos that "flexion" corresponds to is *not* here: step 3 measured it
# on the assembled animal, because the MJCF's joint axes are whatever the CT
# scan produced, and a sign read off a picture of an axis is a coin flip.
POOL_JOINT = {
    "tibia flexor":        ("tibia", "flexion"),
    "tibia extensor":      ("tibia", "extension"),
    "trochanter flexor":   ("femur", "flexion"),
    "trochanter extensor": ("femur", "extension"),
    "femur reductor":      ("femur", "flexion"),
    "coxa rotator ant":    ("coxa_abduct", "protraction"),
    "coxa rotator post":   ("coxa_abduct", "retraction"),
}

LEGS = {"front_leg": "T1", "middle_leg": "T2", "hind_leg": "T3"}
SIDES = ("left", "right")


def slug(pool, leg, side):
    """The name the pool carries in the app's connectome."""
    return "pool:%s_%s:%s" % (leg, side, pool.replace(" ", "_"))


def pool_groups():
    """
    (name, predicate) pairs, in `build_group_specs()`'s form.

    The predicate takes the BANC meta table and returns a boolean Series in its
    own row order — which, after the sort in `build_banc.py`, *is* the app's
    neuron index order.
    """
    specs = []
    for leg, neuro in LEGS.items():
        for side in SIDES:
            for pool, muscles in POOLS.items():
                specs.append((slug(pool, neuro, side),
                              _predicate(leg, side, muscles)))
    return specs


def _predicate(leg, side, muscles):
    muscles = list(muscles)

    def pred(d):
        return ((d["super_class"] == "motor")
                & (d["side"] == side)
                & (d["body_part_effector"] == leg)
                & d["peripheral_target_type"].isin(muscles))

    return pred
