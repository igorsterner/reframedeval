import utils.constants
import utils.data
import utils.io
import utils.scoring


class DialogueGaps:
    METRIC_NAMES = ["cider", "meteor"]

    def __init__(self, config, names):
        self.names = names
        self.cider_max_gap_seconds = config.gaps.max_gap_seconds.cider
        self.meteor_max_gap_seconds = config.gaps.max_gap_seconds.meteor

    def score(self, items):
        gaps = load_gaps(items, self.names)
        scores = {}
        analysis_data = {}

        if "cider" in self.names:
            cider_gaps = [
                gap for gap in gaps if gap["seconds"] < self.cider_max_gap_seconds
            ]
            scores["cider"] = item_means(cider_gaps, "cider")
            analysis_data["cider"] = gap_analysis_data(gaps, "cider")

        if "meteor" in self.names:
            meteor_gaps = [
                gap for gap in gaps if gap["seconds"] < self.meteor_max_gap_seconds
            ]
            scores["meteor"] = item_means(meteor_gaps, "meteor")
            meteor_analysis_data = gap_analysis_data(gaps, "meteor")
            add_reference_scores(meteor_analysis_data, gaps)
            analysis_data["meteor"] = meteor_analysis_data

        return scores, analysis_data


def load_gaps(items, names):
    files = {}
    gaps = []

    for item in items:
        if not item["is_covered_in_movie_subset"]:
            continue

        path = utils.data.comparison_path(item, "dialogue_gaps")

        if path not in files:
            files[path] = utils.io.load_json(path)

        records = files[path][item["item"]]
        assert len(records) == len(item["gaps"]), (path, item["item"])

        for index, record in enumerate(records):
            start, end = item["gaps"][index]
            references = item["gap_references"][index]
            assert record["start"] == start, (path, item["item"], index)
            assert record["end"] == end, (path, item["item"], index)
            assert record["reference1"] == references[0], (
                path,
                item["item"],
                index,
            )
            assert record["reference2"] == references[1], (
                path,
                item["item"],
                index,
            )

            if record["reference1"] == "" and record["reference2"] == "":
                if "cider" in names:
                    assert record["cider"] is None, (path, item["item"], index)

                if "meteor" in names:
                    assert record["meteor"]["reference1"] is None, (
                        path,
                        item["item"],
                        index,
                    )
                    assert record["meteor"]["reference2"] is None, (
                        path,
                        item["item"],
                        index,
                    )

                continue

            gap = {
                "movie": item["movie"],
                "item": item["item"],
                "seconds": end - start,
            }

            if "cider" in names:
                assert record["cider"] is not None, (path, item["item"], index)
                gap["cider"] = record["cider"]

            if "meteor" in names:
                meteor_scores = {}

                for reference_name in utils.constants.REFERENCE_NAMES:
                    score = record["meteor"][reference_name]

                    if record[reference_name] == "":
                        assert score is None, (
                            path,
                            item["item"],
                            index,
                            reference_name,
                        )
                    else:
                        assert score is not None, (
                            path,
                            item["item"],
                            index,
                            reference_name,
                        )
                        meteor_scores[reference_name] = score

                gap["meteor"] = max(meteor_scores.values())
                gap["reference_scores"] = meteor_scores

            gaps.append(gap)

    return gaps


def item_means(gaps, metric_name):
    collected = {}

    for gap in gaps:
        key = (gap["movie"], gap["item"])

        if key not in collected:
            collected[key] = []

        collected[key].append(gap[metric_name])

    by_movie = {}

    for (movie, item), scores in collected.items():
        if movie not in by_movie:
            by_movie[movie] = {}

        by_movie[movie][item] = utils.scoring.mean(scores)

    return by_movie


def gap_analysis_data(gaps, metric_name):
    by_movie = {}

    for gap in gaps:
        movie = gap["movie"]

        if movie not in by_movie:
            by_movie[movie] = {}

        if gap["item"] not in by_movie[movie]:
            by_movie[movie][gap["item"]] = {"gaps": []}

        by_movie[movie][gap["item"]]["gaps"].append(
            {"seconds": gap["seconds"], "score": gap[metric_name]}
        )

    return by_movie


def add_reference_scores(analysis_data, gaps):
    collected = {}

    for gap in gaps:
        key = (gap["movie"], gap["item"])

        if key not in collected:
            collected[key] = {
                reference_name: [] for reference_name in utils.constants.REFERENCE_NAMES
            }

        for reference_name, score in gap["reference_scores"].items():
            collected[key][reference_name].append(score)

    for (movie, item), by_reference in collected.items():
        reference_scores = {}

        for reference_name, scores in by_reference.items():
            if len(scores) == 0:
                reference_scores[reference_name] = None
            else:
                reference_scores[reference_name] = utils.scoring.mean(scores)

        analysis_data[movie][item]["reference_scores"] = reference_scores
