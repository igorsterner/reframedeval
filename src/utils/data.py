import pathlib

import utils.config
import utils.constants
import utils.io
import utils.sequences
import utils.spans


def load(config, split):
    split_dir = utils.constants.SPLITS[split]
    is_challenge = split == "challenge"
    preprocessed = pathlib.Path(config.paths.preprocessed_data)
    reference_dir = preprocessed / "references" / split_dir
    generation_dir = preprocessed / "generations" / split_dir
    meta = load_meta(config, split)
    min_gap_seconds = config.gaps.min_gap_seconds.cider
    collar_seconds = config.gaps.collar_seconds
    items = {}

    for movie, movie_meta in meta.items():
        if is_challenge:
            items[movie] = utils.sequences.movie_items(
                reference_dir,
                movie,
                movie_meta,
                min_gap_seconds,
            )
        else:
            items[movie] = [
                video_item(reference_dir, movie, video, min_gap_seconds)
                for video in movie_meta
            ]

    for movie_items in items.values():
        for item in movie_items:
            prepare_item(item, collar_seconds)

    return {
        "is_challenge": is_challenge,
        "items": items,
        "systems": systems(config, generation_dir, len(meta)),
    }


def load_meta(config, split):
    return utils.io.load_json(
        pathlib.Path(config.paths.meta) / f"reframed-{split}.json"
    )


def systems(config, generation_dir, movie_count):
    out = []

    for entry in config.systems:
        expected = movie_count

        if "movie_subset" in entry:
            expected = entry.movie_subset

        for variant in utils.config.system_variants(entry):
            out.append(
                {
                    "name": variant["name"],
                    "label": variant["label"],
                    "path": generation_dir / variant["name"],
                    "expected_movies": expected,
                }
            )

    return out


def video_item(reference_dir, movie, video, min_gap_seconds):
    video_id = video["id"]
    subs = utils.io.load_segments(reference_dir / "subs" / movie / f"{video_id}.json")
    references = []

    for reference in utils.constants.REFERENCE_NAMES:
        path = reference_dir / reference / movie / f"{video_id}.json"
        references.append(utils.io.load_segments(path))

    end = float(video["duration"])

    return {
        "movie": movie,
        "item": video_id,
        "start": 0.0,
        "end": end,
        "gaps": utils.spans.build_dialogue_gaps(
            subs,
            0.0,
            end,
            min_gap_seconds,
        ),
        "references": references,
    }


def prepare_item(item, collar_seconds):
    item["gap_references"] = gap_references(
        item["references"],
        item["gaps"],
        collar_seconds,
    )
    item["scoreable_gaps"] = scoreable_gaps(item["gap_references"])
    item["qa"] = reference_qas(item["references"])


def gap_references(references, gaps, collar_seconds):
    texts = [
        utils.spans.assign_to_gaps(reference, gaps, collar_seconds)
        for reference in references
    ]

    return [list(gap_texts) for gap_texts in zip(*texts)]


def scoreable_gaps(gap_references):
    return [
        index
        for index, references in enumerate(gap_references)
        if any(reference != "" for reference in references)
    ]


def reference_qas(references):
    out = {}

    for name, segments in zip(utils.constants.REFERENCE_NAMES, references):
        out[name] = segment_qas(segments)

    return out


def segment_qas(segments):
    out = {}

    for segment in segments:
        if "qa" in segment and len(segment["qa"]) > 0:
            out[segment["id"]] = {
                "midpoint": utils.spans.midpoint(segment),
                "questions": segment["qa"],
            }

    return out


def system_items(data, system):
    covered_movies = 0
    out = []

    for movie, movie_items in data["items"].items():
        movie_items = [dict(item) for item in movie_items]

        if data["is_challenge"]:
            covered = utils.sequences.apply_generations(system, movie, movie_items)
        else:
            apply_video_generations(system, movie_items)
            covered = True

        for item in movie_items:
            item["is_covered_in_movie_subset"] = covered

        if covered:
            covered_movies += 1
        else:
            for item in movie_items:
                item["generations"] = []
                item["qa_path"] = None

        out += movie_items

    if covered_movies != system["expected_movies"]:
        raise ValueError(
            f"{system['name']}: expected generations for "
            f"{system['expected_movies']} movies, found {covered_movies}"
        )

    return out


def apply_video_generations(system, movie_items):
    for item in movie_items:
        movie_dir = system["path"] / item["movie"]
        path = movie_dir / f"{item['item']}.json"
        item["generations"] = utils.spans.deoverlapped(utils.io.load_segments(path))
        item["qa_path"] = movie_dir / "qa" / f"{item['item']}.json"


def has_qas(item):
    return any(len(qas) > 0 for qas in item["qa"].values())


def comparison_path(item, comparison):
    qa_path = item["qa_path"]
    assert qa_path is not None, (item["movie"], item["item"])

    return qa_path.parent.parent / comparison / qa_path.name
