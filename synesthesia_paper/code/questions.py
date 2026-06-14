"""Question sets for the synesthetic-probing experiments.

Each synesthetic question targets one of the 5 senses. We manipulate METAPHORICITY:
the degree to which the sensory predicate carries a conventional, lexicalized
*non-sensory* (abstract/figurative) meaning in everyday English.

- SYN_LOW  : low-metaphoricity sensory predicates (a "purple"/"grainy"/"nasal" text has
             almost no conventional figurative meaning). These are the core stimuli.
- SYN_HIGH : high-metaphoricity sensory predicates (a "dark"/"bitter"/"warm"/"fishy" text
             has a strong lexicalized abstract meaning). Control for "is it just metaphor?".
- GIB      : gibberish / random-word probes (control for "does any probe work?").

metaphoricity in [0,1]: 0 = purely sensory, no figurative meaning; 1 = strongly lexicalized
abstract meaning. a_priori scores below are author judgments; we also collect LLM self-ratings.

Feature value for a (text, question) pair = logit("yes") - logit("no") at the answer slot.
"""

# sense, id, predicate, question, a_priori_metaphoricity
SYN_LOW = [
    # SIGHT
    ("sight", "low_orange",   "orange",   "Is this text orange?",                       0.10),
    ("sight", "low_purple",   "purple",   "Is this text purple?",                       0.10),
    ("sight", "low_angular",  "angular",  "Is this text angular (rather than rounded)?",0.20),
    ("sight", "low_opaque",   "opaque",   "Is this text opaque (rather than see-through)?",0.25),
    # TASTE
    ("taste", "low_salty",    "salty",    "Does this text taste salty?",                0.30),
    ("taste", "low_umami",    "umami",    "Does this text taste savory (umami)?",       0.05),
    ("taste", "low_starchy",  "starchy",  "Does this text taste starchy?",              0.05),
    ("taste", "low_minty",    "minty",    "Does this text taste minty?",                0.10),
    # TOUCH
    ("touch", "low_wet",      "wet",      "Does this text feel wet?",                   0.20),
    ("touch", "low_grainy",   "grainy",   "Does this text feel grainy?",                0.15),
    ("touch", "low_fuzzy",    "fuzzy",    "Does this text feel fuzzy?",                 0.30),
    ("touch", "low_spongy",   "spongy",   "Does this text feel spongy?",                0.15),
    # HEARING
    ("hearing","low_muffled", "muffled",  "Does this text sound muffled?",              0.20),
    ("hearing","low_echoey",  "echoey",   "Does this text sound echoey?",               0.10),
    ("hearing","low_nasal",   "nasal",    "Does this text sound nasal?",                0.10),
    ("hearing","low_squeaky", "squeaky",  "Does this text sound squeaky?",              0.20),
    # SMELL
    ("smell", "low_metallic", "metallic", "Does this text smell metallic?",             0.10),
    ("smell", "low_smoky",    "smoky",    "Does this text smell smoky?",                0.20),
    ("smell", "low_floral",   "floral",   "Does this text smell floral?",               0.15),
    ("smell", "low_musty",    "musty",    "Does this text smell musty?",                0.30),
]

SYN_HIGH = [
    # SIGHT  (dark=gloomy, bright=clever/cheerful, colorful=vivid, black=evil/illicit)
    ("sight", "high_dark",     "dark",     "Is this text dark?",                        0.85),
    ("sight", "high_bright",   "bright",   "Is this text bright?",                      0.80),
    ("sight", "high_colorful", "colorful", "Is this text colorful?",                    0.80),
    ("sight", "high_black",    "black",    "Is this text black?",                       0.75),
    # TASTE  (sweet=kind, bitter=resentful, sour=grumpy, bland=dull)
    ("taste", "high_sweet",    "sweet",    "Is this text sweet?",                       0.85),
    ("taste", "high_bitter",   "bitter",   "Is this text bitter?",                      0.90),
    ("taste", "high_sour",     "sour",     "Is this text sour?",                        0.80),
    ("taste", "high_bland",    "bland",    "Is this text bland?",                       0.90),
    # TOUCH  (warm=friendly, cold=unfeeling, rough=harsh, smooth=suave)
    ("touch", "high_warm",     "warm",     "Is this text warm?",                        0.90),
    ("touch", "high_cold",     "cold",     "Is this text cold?",                        0.90),
    ("touch", "high_rough",    "rough",    "Is this text rough?",                       0.80),
    ("touch", "high_smooth",   "smooth",   "Is this text smooth?",                      0.80),
    # HEARING (loud=ostentatious, quiet=understated, shrill=annoying, harmonious=agreeable)
    ("hearing","high_loud",    "loud",     "Is this text loud?",                        0.80),
    ("hearing","high_quiet",   "quiet",    "Is this text quiet?",                       0.75),
    ("hearing","high_shrill",  "shrill",   "Is this text shrill?",                      0.80),
    ("hearing","high_harmonious","harmonious","Is this text harmonious?",              0.85),
    # SMELL  (stinking=immoral, fragrant=pleasant, fishy=suspicious, fresh=new/original)
    ("smell", "high_stinking", "stinking", "Does this text stink?",                     0.85),
    ("smell", "high_fragrant", "fragrant", "Is this text fragrant?",                    0.80),
    ("smell", "high_fishy",    "fishy",    "Is this text fishy?",                       0.95),
    ("smell", "high_fresh",    "fresh",    "Is this text fresh?",                       0.85),
]

# Gibberish control: arbitrary random-word probes (no sensory content).
GIB = [
    ("gib", "gib_01", "gib", "Does this text evoke the words: lantern, gravel, cabbage, ledger, piston, marsh, velvet, anvil?", None),
    ("gib", "gib_02", "gib", "Does this text evoke the words: thimble, cactus, quorum, drizzle, brisket, pylon, sequin, otter?", None),
    ("gib", "gib_03", "gib", "Does this text evoke the words: meadow, transistor, mustard, gondola, pebble, kettle, fern, abacus?", None),
    ("gib", "gib_04", "gib", "Does this text evoke the words: walnut, satchel, glacier, turbine, parsley, lozenge, harbor, mallet?", None),
    ("gib", "gib_05", "gib", "Does this text evoke the words: trellis, mango, snorkel, granite, ferret, candle, axle, plankton?", None),
    ("gib", "gib_06", "gib", "Does this text evoke the words: bramble, socket, lichen, dossier, tangerine, harpoon, quartz, mitten?", None),
    ("gib", "gib_07", "gib", "Does this text evoke the words: cobweb, ledger, saffron, dynamo, walrus, trowel, pumice, lattice?", None),
    ("gib", "gib_08", "gib", "Does this text evoke the words: pelican, vellum, cinder, ratchet, basil, igloo, monsoon, scallop?", None),
    ("gib", "gib_09", "gib", "Does this text evoke the words: thistle, cog, lagoon, almanac, gherkin, beacon, marble, sled?", None),
    ("gib", "gib_10", "gib", "Does this text evoke the words: nutmeg, piston, fjord, lantern, badger, trellis, quilt, ember?", None),
]

ALL_QUESTIONS = SYN_LOW + SYN_HIGH + GIB

SENSES = ["sight", "taste", "touch", "hearing", "smell"]


def as_records():
    recs = []
    for sense, qid, pred, qtext, meta in ALL_QUESTIONS:
        recs.append({
            "qid": qid,
            "sense": sense,
            "predicate": pred,
            "question": qtext,
            "set": "SYN_LOW" if qid.startswith("low_") else ("SYN_HIGH" if qid.startswith("high_") else "GIB"),
            "a_priori_metaphoricity": meta,
        })
    return recs


if __name__ == "__main__":
    import json
    recs = as_records()
    print(f"{len(recs)} questions")
    print(json.dumps(recs, indent=2))
