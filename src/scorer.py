import importlib

import utils.config
import utils.data

SCORER_MODULES = {
    "dialogue_gaps": "scorers.dialogue_gaps",
    "aligned": "scorers.aligned",
    "qa_based": "scorers.qa_based",
}


def main():
    args = utils.config.parse_scorer_args()
    config, split = utils.config.load(args)
    data = utils.data.load(config, split)
    selected_metrics = utils.config.selected_metrics(config, args)
    items_by_system = {}

    for system in data["systems"]:
        items_by_system[system["name"]] = utils.data.system_items(data, system)

    for family, names in selected_metrics.items():
        scorer = importlib.import_module(SCORER_MODULES[family])
        scorer.run(config, data, items_by_system, names)


if __name__ == "__main__":
    main()
