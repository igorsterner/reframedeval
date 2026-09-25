import pycocoevalcap.cider.cider
import pycocoevalcap.meteor.meteor

import utils.constants
import utils.data
import utils.io
import utils.scoring
import utils.spans


def run(config, data, items_by_system, names):
    existing = existing_directories(items_by_system)
    utils.io.delete_after_confirmation(existing, "dialogue_gaps/ folders")
    scorers = {}

    if "cider" in names:
        scorers["cider"] = utils.scoring.TextScorer(pycocoevalcap.cider.cider.Cider())

    if "meteor" in names:
        scorers["meteor"] = utils.scoring.TextScorer(
            pycocoevalcap.meteor.meteor.Meteor()
        )

    for system in data["systems"]:
        items = items_by_system[system["name"]]
        item_records, scoreable = make_gap_records(
            items,
            names,
            config.gaps.collar_seconds,
        )

        if "cider" in names:
            add_cider_scores(scorers["cider"], scoreable)

        if "meteor" in names:
            add_meteor_scores(scorers["meteor"], scoreable)

        outputs = output_files(item_records)

        for path, output in outputs.items():
            utils.io.write_json(path, output)

        print(f"{system['name']}: wrote {len(outputs)} dialogue gap files")


def existing_directories(items_by_system):
    directories = []

    for items in items_by_system.values():
        for item in items:
            if not item["is_covered_in_movie_subset"]:
                continue

            directory = utils.data.comparison_path(item, "dialogue_gaps").parent

            if directory not in directories and directory.exists():
                directories.append(directory)

    return directories


def make_gap_records(items, names, collar_seconds):
    item_records = []
    scoreable = []

    for item in items:
        generations = utils.spans.assign_to_gaps(
            item["generations"],
            item["gaps"],
            collar_seconds,
        )
        records = []

        for index, (start, end) in enumerate(item["gaps"]):
            references = item["gap_references"][index]
            record = {
                "start": start,
                "end": end,
                "generation": generations[index],
                "reference1": references[0],
                "reference2": references[1],
            }

            if "cider" in names:
                record["cider"] = None

            if "meteor" in names:
                record["meteor"] = {
                    "reference1": None,
                    "reference2": None,
                }

            records.append(record)

            if any(reference != "" for reference in references):
                scoreable.append(record)

        item_records.append((item, records))

    return item_records, scoreable


def add_cider_scores(scorer, records):
    references = [[record["reference1"], record["reference2"]] for record in records]
    generations = [record["generation"] for record in records]
    scores = scorer.score(references, generations)

    for record, score in zip(records, scores):
        record["cider"] = score * 100.0


def add_meteor_scores(scorer, records):
    for reference_name in utils.constants.REFERENCE_NAMES:
        selected = [record for record in records if record[reference_name] != ""]
        references = [[record[reference_name]] for record in selected]
        generations = [record["generation"] for record in selected]
        scores = scorer.score(references, generations)

        for record, score in zip(selected, scores):
            record["meteor"][reference_name] = score * 100.0


def output_files(item_records):
    outputs = {}

    for item, records in item_records:
        if not item["is_covered_in_movie_subset"]:
            continue

        path = utils.data.comparison_path(item, "dialogue_gaps")

        if path not in outputs:
            outputs[path] = {}

        outputs[path][item["item"]] = records

    return outputs
