import utils.constants
import utils.latex
import utils.scoring


class AttributionSweep:
    SUPPORTED_METRICS = ["soda-t", "qeval-t"]

    def __init__(self, config, entry, metric_names):
        self.metric_names = metric_names
        self.distances_seconds = entry.distances_seconds

    def report(self, data, items_by_system, analysis_data_by_system):
        labels = {system["name"]: system["label"] for system in data["systems"]}

        for metric_name in self.metric_names:
            combine = COMBINE[metric_name]
            headers = ["System"] + [
                f"{distance:g}" for distance in self.distances_seconds
            ]

            rows = []

            for system_name, analysis_data in analysis_data_by_system.items():
                row = [labels[system_name]]

                for distance in self.distances_seconds:
                    by_movie = threshold_scores(
                        analysis_data[metric_name], combine, distance
                    )
                    row.append(f"{utils.scoring.aggregate(by_movie):.1f}")

                rows.append(row)

            print(f"attribution_sweep: {utils.constants.metric_label(metric_name)}")
            print(utils.latex.tabular(headers, rows))


def threshold_scores(analysis_data_by_movie, combine, distance):
    by_movie = {}

    for movie, movie_analysis_data in analysis_data_by_movie.items():
        by_movie[movie] = {}

        for item, item_analysis_data in movie_analysis_data.items():
            by_movie[movie][item] = combine(item_analysis_data["attribution"], distance)

    return by_movie


def hit_count(distances, threshold):
    return sum(1.0 for distance in distances if distance < threshold)


def max_of_reference_ratios(attribution, threshold):
    ratios = []

    for reference_attribution in attribution.values():
        hits = hit_count(reference_attribution["distances"], threshold)
        ratios.append(hits / float(reference_attribution["total"]))

    return max(ratios) * 100.0


def pooled_ratio(attribution, threshold):
    hits = 0.0
    total = 0

    for reference_attribution in attribution.values():
        hits += hit_count(reference_attribution["distances"], threshold)
        total += reference_attribution["total"]

    return hits / float(total) * 100.0


COMBINE = {"soda-t": max_of_reference_ratios, "qeval-t": pooled_ratio}
