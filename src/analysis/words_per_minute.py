import utils.constants
import utils.latex
import utils.scoring
import utils.spans


class WordsPerMinute:
    SUPPORTED_METRICS = []

    def __init__(self, config, entry, metric_names):
        self.metric_names = metric_names
        self.collar_seconds = config.gaps.collar_seconds

    def report(self, data, items_by_system, analysis_data_by_system):
        labels = {system["name"]: system["label"] for system in data["systems"]}
        rows = []

        for system_name, items in items_by_system.items():
            segment_lists = []

            for item in items:
                if item["is_covered_in_movie_subset"]:
                    segment_lists.append(
                        (
                            item["movie"],
                            item["item"],
                            item["generations"],
                            item["gaps"],
                            item["scoreable_gaps"],
                        )
                    )

            rows.append(
                words_per_minute_row(
                    labels[system_name], segment_lists, self.collar_seconds
                )
            )

        reference_items = all_items(data)

        for index, reference_name in enumerate(utils.constants.REFERENCE_NAMES):
            segment_lists = []

            for item in reference_items:
                segment_lists.append(
                    (
                        item["movie"],
                        item["item"],
                        item["references"][index],
                        item["gaps"],
                        item["scoreable_gaps"],
                    )
                )

            rows.append(
                words_per_minute_row(reference_name, segment_lists, self.collar_seconds)
            )

        print("words_per_minute")
        print(
            utils.latex.tabular(
                [
                    "System",
                    "WPM",
                    "\\# Words / Movie",
                    "\\# Dialogue-Gap Words / Movie",
                ],
                rows,
            )
        )


def words_per_minute_row(name, segment_lists, collar_seconds):
    by_movie = {}
    words_by_movie = {}
    dialogue_gap_words_by_movie = {}

    for movie, item, segments, gaps, scoreable_gaps in segment_lists:
        if len(segments) == 0:
            continue

        tokens = 0
        minutes = 0.0

        for segment in segments:
            tokens += len(segment["tokens"])
            minutes += (segment["end"] - segment["start"]) / 60.0

        assert minutes > 0.0, (name, movie, item)

        if movie not in by_movie:
            by_movie[movie] = {}
            words_by_movie[movie] = 0
            dialogue_gap_words_by_movie[movie] = 0

        by_movie[movie][item] = tokens / minutes
        words_by_movie[movie] += tokens
        assigned = utils.spans.assign_to_gaps(segments, gaps, collar_seconds)
        dialogue_gap_words_by_movie[movie] += sum(
            len(assigned[index].split()) for index in scoreable_gaps
        )

    if len(by_movie) == 0:
        return [name, "-", "-", "-"]

    words_per_minute = utils.scoring.mean_over_movies(by_movie)
    words_per_movie = utils.scoring.mean(list(words_by_movie.values()))
    dialogue_gap_words_per_movie = utils.scoring.mean(
        list(dialogue_gap_words_by_movie.values())
    )
    return [
        name,
        f"{words_per_minute:.1f}",
        f"{words_per_movie:.1f}",
        f"{dialogue_gap_words_per_movie:.1f}",
    ]


def all_items(data):
    items = []

    for movie_items in data["items"].values():
        items += movie_items

    return items
