import utils.data
import utils.io
import utils.spans


class QEval:
    METRIC_NAMES = ["qeval", "qeval-t"]

    def __init__(self, config, names):
        self.names = names
        self.temporal_tolerance_seconds = config.temporal.tolerance_seconds
        self.cache = {}

    def score(self, items):
        include_qeval_t = "qeval-t" in self.names
        qeval = {}
        qeval_t = {}
        analysis_data_plain = {}
        analysis_data_t = {}

        for item in items:
            if not item["is_covered_in_movie_subset"] or not utils.data.has_qas(item):
                continue

            path = item["qa_path"]

            if path not in self.cache:
                self.cache[path] = utils.io.load_json(path)

            records = reference_records(
                self.cache[path],
                path,
                self.temporal_tolerance_seconds,
                include_qeval_t,
            )
            movie = item["movie"]

            if movie not in qeval:
                qeval[movie] = {}
                qeval_t[movie] = {}
                analysis_data_plain[movie] = {}
                analysis_data_t[movie] = {}

            qeval[movie][item["item"]] = pooled_accuracy(records, "correct")
            qeval_t[movie][item["item"]] = pooled_accuracy(records, "hits")
            analysis_data_plain[movie][item["item"]] = {
                "reference_scores": reference_accuracies(records, "correct")
            }
            analysis_data_t[movie][item["item"]] = {
                "reference_scores": reference_accuracies(records, "hits"),
                "attribution": reference_attribution(records),
            }

        scores = {"qeval": qeval, "qeval-t": qeval_t}
        analysis_data = {
            "qeval": analysis_data_plain,
            "qeval-t": analysis_data_t,
        }

        return (
            {name: scores[name] for name in self.names},
            {name: analysis_data[name] for name in self.names},
        )


def reference_records(
    data,
    path,
    temporal_tolerance_seconds,
    include_qeval_t,
):
    records = {}

    for reference in utils.constants.REFERENCE_NAMES:
        correct = 0.0
        hits = 0.0
        total = 0
        distances = []

        for questions in data[reference].values():
            for result in questions.values():
                total += 1

                if include_qeval_t and "attribution" not in result:
                    raise ValueError(
                        f"{path}: qeval-t requires attribution; rerun the "
                        f"scorer with --metric_subset qeval-t"
                    )

                if result["predicted_answer"] == result["correct_answer"]:
                    correct += 1.0
                    attribution = None

                    if include_qeval_t:
                        attribution = result["attribution"]

                    if attribution is not None:
                        distance = abs(
                            utils.spans.midpoint(attribution)
                            - float(result["grounding"])
                        )
                        distances.append(distance)

                        if distance < temporal_tolerance_seconds:
                            hits += 1.0

        records[reference] = {
            "correct": correct,
            "hits": hits,
            "total": total,
            "distances": distances,
        }

    return records


def pooled_accuracy(records, key):
    numerator = sum(record[key] for record in records.values())
    denominator = sum(record["total"] for record in records.values())

    assert denominator > 0
    return numerator / float(denominator) * 100.0


def reference_accuracies(records, key):
    scores = {}

    for reference, record in records.items():
        if record["total"] == 0:
            scores[reference] = None
        else:
            scores[reference] = record[key] / float(record["total"]) * 100.0

    return scores


def reference_attribution(records):
    return {
        reference: {"distances": record["distances"], "total": record["total"]}
        for reference, record in records.items()
    }
