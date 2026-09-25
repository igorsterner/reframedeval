def midpoint(segment):
    return (segment["start"] + segment["end"]) / 2.0


def assert_non_overlapping(segments, message):
    for left, right in zip(segments, segments[1:]):
        assert right["start"] >= left["end"], (message, left, right)


def deoverlapped(segments):
    out = []

    for segment in segments:
        segment = dict(segment)
        keep = True

        while len(out) > 0 and segment["start"] < out[-1]["end"]:
            previous = out[-1]

            if segment["start"] == previous["start"]:
                out.pop()
            elif segment["end"] <= previous["end"]:
                keep = False
                break
            else:
                overlap_midpoint = (previous["end"] + segment["start"]) / 2.0
                previous["end"] = overlap_midpoint
                segment["start"] = overlap_midpoint

        if keep:
            out.append(segment)

    assert_non_overlapping(out, "segments still overlap after deoverlapping")
    return out


def build_dialogue_gaps(dialogues, start, end, min_gap_seconds):
    assert_non_overlapping(dialogues, "subtitles must not overlap")

    if len(dialogues) == 0:
        return [(start, end)]

    assert dialogues[-1]["end"] <= end, (
        "subtitles must not run past the item end",
        dialogues[-1],
        end,
    )

    gaps = []

    if dialogues[0]["start"] - start >= min_gap_seconds:
        gaps.append((start, dialogues[0]["start"]))

    for left, right in zip(dialogues, dialogues[1:]):
        if right["start"] - left["end"] >= min_gap_seconds:
            gaps.append((left["end"], right["start"]))

    if end - dialogues[-1]["end"] >= min_gap_seconds:
        gaps.append((dialogues[-1]["end"], end))

    return gaps


def assign_to_gaps(segments, gaps, collar_seconds):
    token_lists = [[] for _ in gaps]

    for segment in sorted(segments, key=lambda item: (item["start"], item["end"])):
        for index, (gap_start, gap_end) in enumerate(gaps):
            extended_start = gap_start - collar_seconds
            extended_end = gap_end + collar_seconds

            if segment["start"] >= extended_start and segment["end"] <= extended_end:
                token_lists[index] += segment["tokens"]

    return [" ".join(tokens) for tokens in token_lists]


def is_assigned_to_any_gap(segment, gaps, collar_seconds):
    return any(
        is_assigned_to_gap(segment, start, end, collar_seconds) for start, end in gaps
    )


def is_assigned_to_gap(segment, gap_start, gap_end, collar_seconds):
    extended_start = gap_start - collar_seconds
    extended_end = gap_end + collar_seconds

    return segment["start"] >= extended_start and segment["end"] <= extended_end
