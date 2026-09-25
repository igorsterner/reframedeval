import utils.constants
import utils.latex
import utils.scoring


class PerReference:
    SUPPORTED_METRICS = ["meteor", "soda-m", "soda-t", "qeval", "qeval-t"]

    def __init__(self, config, entry, metric_names):
        self.metric_names = metric_names

    def report(self, data, items_by_system, analysis_data_by_system):
        labels = {system["name"]: system["label"] for system in data["systems"]}
        headers = ["System"]

        for metric_name in self.metric_names:
            for reference_name in utils.constants.REFERENCE_NAMES:
                headers.append(
                    f"{utils.constants.metric_label(metric_name)} {reference_name}"
                )

        rows = []

        for system_name, analysis_data in analysis_data_by_system.items():
            row = [labels[system_name]]

            for metric_name in self.metric_names:
                for reference_name in utils.constants.REFERENCE_NAMES:
                    score = reference_score(analysis_data[metric_name], reference_name)
                    row.append(f"{score:.1f}")

            rows.append(row)

        print("per_reference")
        print(utils.latex.tabular(headers, rows))


def reference_score(analysis_data_by_movie, reference_name):
    by_movie = {}

    for movie, movie_analysis_data in analysis_data_by_movie.items():
        for item, item_analysis_data in movie_analysis_data.items():
            score = item_analysis_data["reference_scores"][reference_name]

            if score is None:
                continue

            if movie not in by_movie:
                by_movie[movie] = {}

            by_movie[movie][item] = score

    return utils.scoring.aggregate(by_movie)
