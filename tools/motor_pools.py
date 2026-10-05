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


def pool_slug(neuro, side, pool):
    """
    The name the pool carries in the app's connectome.

    `neuro` is the segment the pool's motor neurons sit in (T1/T2/T3), which is
    the name BANC's own tables use for a leg's neuromere. `tools/build_body.py`
    writes the same string into the body asset, so the app asks the connectome
    for a pool by the name the connectome was built with.
    """
    return "pool:%s_%s:%s" % (neuro, side, pool.replace(" ", "_"))


# Older callers passed the body part; keep one spelling.
slug = pool_slug


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
                specs.append((pool_slug(neuro, side, pool),
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


# ---------------------------------------------------------------------------
# The sense organs, the same way.
# ---------------------------------------------------------------------------
#
# A closed loop needs both ends named in the connectome the app reads: the
# muscles that move a joint (the pools above) and the organ that reports the
# joint back (here). The predicates are *step 2's*, moved rather than copied:
# `tools/step2_reflex.py` imports ORGANS from this file, so the organ a Python
# experiment drives and the organ the phone drives are selected by one
# expression.
#
# The values are BANC's `cell_function_detailed` labels for the organ's cells —
# what the cell is, not what class it is in.
ORGANS = {
    "chordotonal": {"joint_angle", "vibro_position", "stretch"},
    "campaniform": {"mechanical_strain", "vibro_tactile"},
    "hairplate":   {"position", "direction"},
}


def organ_slug(neuro, side, organ):
    """The name the organ carries in the app's connectome."""
    return "organ:%s_%s:%s" % (neuro, side, organ)


def organ_groups():
    """
    (name, predicate) pairs for every leg's sense organs.

    The predicate is the one step 2 selected the leg's organs with: a
    proprioceptor, of the right detailed function, on the right side, either
    tagged to this leg or sitting in its neuromere. A cell that cannot be placed
    on a leg is not in any organ group, which is reported by
    `tools/verify_loop.py` rather than quietly dropped.
    """
    specs = []
    for leg, neuro in LEGS.items():
        for side in SIDES:
            for organ, funcs in ORGANS.items():
                specs.append((organ_slug(neuro, side, organ),
                              _organ_predicate(leg, neuro, side, funcs)))
    return specs


def _organ_predicate(leg, neuro, side, funcs):
    funcs = list(funcs)

    def pred(d):
        part = d["body_part_sensory"].fillna("").astype(str)
        return ((d["cell_function"] == "proprioception")
                & d["cell_function_detailed"].isin(funcs)
                & (d["side"] == side)
                & (part.str.contains(leg, regex=False) | (d["neuromere"] == neuro)))

    return pred
