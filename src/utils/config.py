import argparse
import pathlib

import omegaconf

import utils.constants

BASE_CONFIG = "configs/base.yaml"
REALISTIC_SUFFIX = "-realistic"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--analysis", action="store_true")
    parser.add_argument("--significance_testing", action="store_true")
    add_metric_overrides(parser)

    return parser.parse_args()


def parse_scorer_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    add_metric_overrides(parser)

    return parser.parse_args()


def add_metric_overrides(parser):
    parser.add_argument(
        "--metrics",
        nargs="+",
        choices=utils.constants.METRIC_FAMILIES,
    )
    parser.add_argument(
        "--metric_subset",
        nargs="+",
        choices=utils.constants.METRIC_NAMES,
    )


def selected_metrics(config, args):
    configured = {}

    for entry in config.metrics:
        family = entry.name

        if family not in utils.constants.METRICS_BY_FAMILY:
            raise ValueError(f"Unknown metric family: {family}")

        if family in configured:
            raise ValueError(f"Metric family listed more than once: {family}")

        if "metric_subset" in entry:
            names = list(entry.metric_subset)
        else:
            names = list(utils.constants.METRICS_BY_FAMILY[family])

        validate_metric_names(family, names)
        configured[family] = names

    if args.metrics is None:
        families = list(configured)
    else:
        families = list(args.metrics)

    if len(families) == 0:
        raise ValueError("At least one metric family is required")

    if len(families) != len(set(families)):
        raise ValueError("Metric families must not be repeated")

    selected = {}

    for family in families:
        if family in configured:
            selected[family] = list(configured[family])
        else:
            selected[family] = list(utils.constants.METRICS_BY_FAMILY[family])

    if args.metric_subset is None:
        return selected

    if len(args.metric_subset) != len(set(args.metric_subset)):
        raise ValueError("Metric subset entries must not be repeated")

    selected = {family: [] for family in families}

    for name in args.metric_subset:
        family = utils.constants.FAMILY_BY_METRIC[name]

        if family not in selected:
            raise ValueError(f"{name} requires {family} in --metrics")

        selected[family].append(name)

    for family, names in selected.items():
        validate_metric_names(family, names)

    return selected


def validate_metric_names(family, names):
    if len(names) == 0:
        raise ValueError(f"{family} metric_subset must not be empty")

    if len(names) != len(set(names)):
        raise ValueError(f"{family} metric_subset entries must not be repeated")

    available = utils.constants.METRICS_BY_FAMILY[family]

    for name in names:
        if name not in available:
            raise ValueError(f"{family} has no metric {name}, choose from {available}")


def realistic_enabled(entry):
    if "realistic" not in entry:
        return True

    if not isinstance(entry.realistic, bool):
        raise ValueError(f"{entry.name}: realistic must be true or false")

    return entry.realistic


def splitting_settings(entry):
    settings = []

    for name in ["split_sentences", "split_elements"]:
        if name in entry:
            value = entry[name]

            if not isinstance(value, bool):
                raise ValueError(f"{entry.name}: {name} must be true or false")
        else:
            value = True

        settings.append(value)

    return tuple(settings)


def system_variants(entry):
    variants = [{"name": entry.name, "label": entry.label, "is_realistic": False}]

    if realistic_enabled(entry):
        variants.append(
            {
                "name": f"{entry.name}{REALISTIC_SUFFIX}",
                "label": f"{entry.label}{REALISTIC_SUFFIX}",
                "is_realistic": True,
            }
        )

    return variants


def validate_system_variants(config):
    names = []

    for entry in config.systems:
        splitting_settings(entry)
        names += [variant["name"] for variant in system_variants(entry)]

    if len(names) != len(set(names)):
        raise ValueError(
            f"System names and realistic variant names must be unique: {names}"
        )


def load(args):
    config_path = pathlib.Path(args.config)

    if not config_path.is_file():
        raise ValueError(f"Config file not found: {config_path}")

    base_path = pathlib.Path(BASE_CONFIG)

    if not base_path.is_file():
        raise ValueError(f"{BASE_CONFIG} not found, run from the repository root")

    config = omegaconf.OmegaConf.merge(
        omegaconf.OmegaConf.load(base_path),
        omegaconf.OmegaConf.load(config_path),
    )

    if config.analysis is None:
        config.analysis = []

    if config.gaps.min_gap_seconds.cider != config.gaps.min_gap_seconds.meteor:
        raise ValueError(
            "cider and meteor min_gap_seconds must match because they share "
            "dialogue gap assignment"
        )

    validate_system_variants(config)

    split = config.split

    if split not in utils.constants.SPLITS:
        raise ValueError(f"Unknown split: {split}")

    return config, split
