import utils.constants
import utils.latex
import utils.scoring


class GapLength:
    SUPPORTED_METRICS = ["cider", "meteor"]

    def __init__(self, config, entry, metric_names):
        self.metric_names = metric_names
        self.bins = entry.bins
        self.max_gap_seconds = config.gaps.max_gap_seconds.cider

    def report(self, data, items_by_system, analysis_data_by_system):
        labels = {system["name"]: system["label"] for system in data["systems"]}

        for metric_name in self.metric_names:
            headers = ["System"] + [bin_label(low, high) for low, high in self.bins]
            rows = []

            for system_name, analysis_data in analysis_data_by_system.items():
                gaps = system_gaps(analysis_data[metric_name])
                row = [labels[system_name]]

                for low, high in self.bins:
                    scores = [
                        gap["score"] for gap in gaps if low <= gap["seconds"] < high
                    ]

                    if len(scores) == 0:
                        row.append("-")
                    else:
                        mean = utils.scoring.mean(scores)
                        row.append(f"{mean:.1f} ({len(scores)})")

                rows.append(row)

            print(f"gap_length: {utils.constants.metric_label(metric_name)}")
            print(utils.latex.tabular(headers, rows))

            if metric_name == "cider":
                print(
                    f"note: the main cider result only includes gaps shorter "
                    f"than {self.max_gap_seconds:g} seconds"
                )


def system_gaps(analysis_data_by_movie):
    gaps = []

    for movie_analysis_data in analysis_data_by_movie.values():
        for item_analysis_data in movie_analysis_data.values():
            gaps += item_analysis_data["gaps"]

    return gaps


def bin_label(low, high):
    if high == float("inf"):
        return f"{low:g}+"

    return f"{low:g}-{high:g}"
