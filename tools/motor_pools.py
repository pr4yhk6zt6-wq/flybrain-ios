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
# An organ's cells are selected by two of BANC's own columns: `cell_function` —
# what the cell is for (proprioception, tactile) — and, for the proprioceptors,
# the `cell_function_detailed` label, which is the *kind* of proprioceptor it is.
#
# `leg_tag_required` is item 12's addition. The three proprioceptors accept a
# cell that is tagged to this leg **or** sitting in this leg's neuromere, which
# is right for them: a femoral chordotonal organ's cells are not all annotated
# with the leg they are on. A thoracic neuromere also holds the wing's and the
# haltere's sensilla, so the same fallback would put another appendage's bristles
# into a leg's touch organ. (Measured: on BANC v888 the fallback would add
# nothing — every leg bristle carries its leg tag — but the predicate is not
# allowed to depend on that.)
ORGANS = {
    "chordotonal": {
        "cell_function": "proprioception",
        "functions": {"joint_angle", "vibro_position", "stretch"},
    },
    "campaniform": {
        "cell_function": "proprioception",
        "functions": {"mechanical_strain", "vibro_tactile"},
    },
    # In the table and in the shipped connectome (BANC annotates 37-68 hair-plate
    # cells per leg), but `driven: False`: the cord has no law for it yet. The
    # femoral chordotonal organ already reports the knee's angle, and a second
    # position channel with no measurement behind it would be a placeholder
    # wearing a sensor's name.
    "hairplate": {
        "cell_function": "proprioception",
        "functions": {"position", "direction"},
        "driven": False,
    },
    # Touch, item 12: the leg's tactile hairs — trichoid sensilla, "the primary
    # exteroceptive organs" (Tuthill & Wilson 2016, Curr Biol 26:R1022). BANC
    # gives a leg's tactile cells no detailed label of their own — 3,172 of the
    # 3,184 are blank — so `functions` is None and the label is not filtered;
    # that is BANC's own function call, which is the authority the other three
    # organs also use. What the class is, for the record: 3,011 bristle neurons,
    # 114 the mechanosensory neuron of a tarsal taste hair (a bimodal sensillum —
    # its own function is still tactile), 38 orphan neurons, 20 unclassed, 1
    # hair-plate neuron. All 3,184 are `super_class == "sensory"`, all sit in the
    # ventral nerve cord, and 43 carry no side, so they are in no organ group and
    # `tools/verify_loop.py` reports them rather than dropping them quietly.
    "tactile": {
        "cell_function": "tactile",
        "functions": None,
        "leg_tag_required": True,
    },
}

# The three proprioceptors, as one view of the table above. This is what
# `tools/step2_reflex.py` measures its reflex baseline on, and it is not a
# tidiness choice: that tool drives every cell of every organ it is given and
# measures what the pools do, so admitting 3,184 more afferents would move a
# baseline instead of describing touch. `organ_groups()` — what writes the
# shipped connectome — reads the whole table.
PROPRIOCEPTORS = {k: v for k, v in ORGANS.items()
                  if v["cell_function"] == "proprioception"}

# The organs the cord has a drive law for: exactly what `tools/build_body.py`
# writes into each leg of the body asset — where `FlyCord.swift` finds its
# `kind` — and what `tools/verify_loop.py` requires both ends to agree on.
DRIVEN_ORGANS = {k for k, v in ORGANS.items() if v.get("driven", True)}


def organ_slug(neuro, side, organ):
    """The name the organ carries in the app's connectome."""
    return "organ:%s_%s:%s" % (neuro, side, organ)


def organ_groups():
    """
    (name, predicate) pairs for every leg's sense organs.

    The predicate is the one step 2 selected the leg's organs with, widened by
    the organ's own entry: a cell of the right function, of the right detailed
    kind where the organ has kinds, on the right side, either tagged to this leg
    or — for an organ that accepts it — sitting in its neuromere. A cell that
    cannot be placed on a leg is not in any organ group, which is reported by
    `tools/verify_loop.py` rather than quietly dropped.
    """
    specs = []
    for leg, neuro in LEGS.items():
        for side in SIDES:
            for organ, spec in ORGANS.items():
                specs.append((organ_slug(neuro, side, organ),
                              _organ_predicate(leg, neuro, side, spec)))
    return specs


def _organ_predicate(leg, neuro, side, spec):
    funcs = None if spec["functions"] is None else list(spec["functions"])

    def pred(d):
        part = d["body_part_sensory"].fillna("").astype(str)
        m = ((d["cell_function"] == spec["cell_function"])
             & (d["side"] == side))
        if funcs is not None:
            m = m & d["cell_function_detailed"].isin(funcs)
        if spec.get("leg_tag_required"):
            return m & part.str.contains(leg, regex=False)
        return m & (part.str.contains(leg, regex=False) | (d["neuromere"] == neuro))

    return pred
