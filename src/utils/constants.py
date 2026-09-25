REFERENCE_NAMES = ["reference1", "reference2"]
METRICS_BY_FAMILY = {
    "dialogue_gaps": ["cider", "meteor"],
    "aligned": ["soda-m", "soda-t"],
    "qa_based": ["qeval", "qeval-t"],
}
METRIC_FAMILIES = list(METRICS_BY_FAMILY)
METRIC_NAMES = [name for names in METRICS_BY_FAMILY.values() for name in names]
METRIC_LABELS = {
    "cider": "CIDEr",
    "meteor": "METEOR",
    "soda-m": "SODA-M",
    "soda-t": "SODA-T",
    "qeval": "QEval",
    "qeval-t": "QEval-T",
}


def metric_label(name):
    return METRIC_LABELS[name]


FAMILY_BY_METRIC = {
    name: family for family, names in METRICS_BY_FAMILY.items() for name in names
}
SPLITS = {
    "challenge": "challenge",
    "screenplay": "screenplay",
    "train": "videos",
    "validation": "videos",
    "test": "videos",
}
