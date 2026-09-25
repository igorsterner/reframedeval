import utils.constants
import utils.data
import utils.io
import utils.spans


class SODA:
    METRIC_NAMES = ["soda-m", "soda-t"]

    def __init__(self, config, names):
        self.names = names
        self.temporal_tolerance_seconds = config.temporal.tolerance_seconds

    def score(self, items):
        soda_m = {}
        soda_t = {}
        analysis_data_m = {}
        analysis_data_t = {}
        files = {}

        for item in items:
            if not item["is_covered_in_movie_subset"]:
                continue

            if not any(len(reference) > 0 for reference in item["references"]):
                continue

            path = utils.data.comparison_path(item, "aligned")

            if path not in files:
                files[path] = utils.io.load_json(path)

            alignment = files[path][item["item"]]
            m_scores = {
                reference_name: None
                for reference_name in utils.constants.REFERENCE_NAMES
            }
            t_scores = {
                reference_name: None
                for reference_name in utils.constants.REFERENCE_NAMES
            }
            attribution = {}

            for reference_name, reference in zip(
                utils.constants.REFERENCE_NAMES, item["references"]
            ):
                saved = alignment[reference_name]

                if len(reference) == 0:
                    continue

                m, t, distances = reference_scores(
                    len(item["generations"]),
                    len(reference),
                    saved,
                    self.temporal_tolerance_seconds,
                )
                m_scores[reference_name] = m
                t_scores[reference_name] = t
                attribution[reference_name] = {
                    "distances": distances,
                    "total": len(reference),
                }

            movie = item["movie"]

            if movie not in soda_m:
                soda_m[movie] = {}
                soda_t[movie] = {}
                analysis_data_m[movie] = {}
                analysis_data_t[movie] = {}

            available_m = [score for score in m_scores.values() if score is not None]
            available_t = [score for score in t_scores.values() if score is not None]
            soda_m[movie][item["item"]] = max(available_m)
            soda_t[movie][item["item"]] = max(available_t)
            analysis_data_m[movie][item["item"]] = {"reference_scores": m_scores}
            analysis_data_t[movie][item["item"]] = {
                "reference_scores": t_scores,
                "attribution": attribution,
            }

        scores = {"soda-m": soda_m, "soda-t": soda_t}
        analysis_data = {
            "soda-m": analysis_data_m,
            "soda-t": analysis_data_t,
        }

        return (
            {name: scores[name] for name in self.names},
            {name: analysis_data[name] for name in self.names},
        )


def reference_scores(generation_count, reference_count, saved, tolerance_seconds):
    pairs = saved["optimal_alignment"]
    assert len(pairs) <= generation_count
    assert len(pairs) <= reference_count
    matched_sum = sum(float(pair["meteor"]) for pair in pairs)

    if generation_count == 0:
        precision = 0.0
    else:
        precision = matched_sum / float(generation_count)

    recall = matched_sum / float(reference_count)
    soda_m = 0.0

    if precision + recall > 0.0:
        soda_m = (2.0 * precision * recall) / (precision + recall)

    distances = []

    for pair in pairs:
        distances.append(
            abs(
                utils.spans.midpoint(pair["reference"])
                - utils.spans.midpoint(pair["generation"])
            )
        )

    midpoint_hits = sum(1.0 for distance in distances if distance < tolerance_seconds)
    soda_t = midpoint_hits / float(reference_count) * 100.0

    return soda_m, soda_t, distances
