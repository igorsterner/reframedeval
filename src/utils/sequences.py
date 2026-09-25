import utils.constants
import utils.io
import utils.spans


def movie_items(reference_dir, movie, sequence_ids, min_gap_seconds):
    subs = utils.io.load_segments(reference_dir / "subs" / f"{movie}.json")
    scenes = utils.io.load_json(reference_dir / "scenes" / f"{movie}.json")
    sequences = utils.io.load_json(reference_dir / "sequences" / f"{movie}.json")
    spans, scene_to_sequence, end_credits_id = sequence_layout(
        movie,
        scenes,
        sequences,
    )
    references = movie_references(reference_dir, movie, spans, scene_to_sequence)
    items = []

    for sequence_id in sequence_ids:
        if sequence_id == end_credits_id:
            continue

        items.append(
            sequence_item(
                movie,
                sequence_id,
                spans[sequence_id],
                subs,
                references,
                min_gap_seconds,
            )
        )

    return items


def sequence_item(movie, sequence_id, span, subs, references, min_gap_seconds):
    clipped = clipped_subs(subs, span["start"], span["end"])

    return {
        "movie": movie,
        "item": sequence_id,
        "start": span["start"],
        "end": span["end"],
        "gaps": utils.spans.build_dialogue_gaps(
            clipped,
            span["start"],
            span["end"],
            min_gap_seconds,
        ),
        "references": [reference[sequence_id] for reference in references],
    }


def movie_references(reference_dir, movie, spans, scene_to_sequence):
    references = []

    for reference in utils.constants.REFERENCE_NAMES:
        segments = utils.io.load_segments(reference_dir / reference / f"{movie}.json")
        references.append(by_sequence(segments, spans, scene_to_sequence))

    return references


def sequence_layout(movie, scenes, sequences):
    spans = {}
    scene_to_sequence = {}
    end_credits_id = ""

    for sequence_id, scene_names in sequences.items():
        starts = []
        ends = []

        for scene in scene_names:
            assert scene not in scene_to_sequence, (movie, scene)
            scene_to_sequence[scene] = sequence_id

            if scene == "end_credits":
                end_credits_id = sequence_id

            starts.append(float(scenes[scene]["start"]))
            ends.append(float(scenes[scene]["end"]))

        spans[sequence_id] = {"start": min(starts), "end": max(ends)}

    assert end_credits_id != "", movie
    return spans, scene_to_sequence, end_credits_id


def by_sequence(segments, spans, scene_to_sequence):
    out = {sequence_id: [] for sequence_id in spans}

    for segment in segments:
        out[scene_to_sequence[segment["scene"]]].append(segment)

    return out


def clipped_subs(subs, start, end):
    clipped = []

    for sub in subs:
        left = max(sub["start"], start)
        right = min(sub["end"], end)

        if right > left:
            clipped.append({"start": left, "end": right})

    return clipped


def apply_generations(system, movie, movie_items):
    path = system["path"] / f"{movie}.json"

    if not path.exists():
        return False

    segments = utils.spans.deoverlapped(utils.io.load_segments(path))
    generations = assign_to_sequences(segments, movie_items)

    for item in movie_items:
        item["generations"] = generations[item["item"]]
        item["qa_path"] = system["path"] / "qa" / f"{movie}.json"

    return True


def assign_to_sequences(segments, movie_items):
    assigned = {item["item"]: [] for item in movie_items}

    for segment in segments:
        best = None
        best_overlap = 0.0

        for item in movie_items:
            overlap = overlap_duration(segment, item)

            if overlap > best_overlap:
                best = item["item"]
                best_overlap = overlap

        if best is not None:
            assigned[best].append(segment)

    return assigned


def overlap_duration(segment, item):
    return min(segment["end"], item["end"]) - max(segment["start"], item["start"])
