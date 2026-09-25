import numpy as np
import pycocoevalcap.meteor.meteor

import utils.constants
import utils.data
import utils.io
import utils.scoring


def run(config, data, items_by_system, names):
    existing = existing_directories(items_by_system)
    utils.io.delete_after_confirmation(existing, "aligned/ folders")
    meteor = utils.scoring.TextScorer(pycocoevalcap.meteor.meteor.Meteor())

    for system in data["systems"]:
        items = items_by_system[system["name"]]
        outputs = alignment_files(meteor, items)

        for path, output in outputs.items():
            utils.io.write_json(path, output)

        print(f"{system['name']}: wrote {len(outputs)} aligned files")


def existing_directories(items_by_system):
    directories = []

    for items in items_by_system.values():
        for item in items:
            if not item["is_covered_in_movie_subset"]:
                continue

            directory = utils.data.comparison_path(item, "aligned").parent

            if directory not in directories and directory.exists():
                directories.append(directory)

    return directories


def alignment_files(meteor, items):
    outputs = {}

    for item in items:
        if not item["is_covered_in_movie_subset"]:
            continue

        path = utils.data.comparison_path(item, "aligned")

        if path not in outputs:
            outputs[path] = {}

        alignment = {}

        for reference_name, reference in zip(
            utils.constants.REFERENCE_NAMES, item["references"]
        ):
            similarity = similarity_matrix(meteor, item["generations"], reference)
            alignment[reference_name] = {
                "optimal_alignment": optimal_alignment(
                    similarity,
                    item["generations"],
                    reference,
                )
            }

        outputs[path][item["item"]] = alignment

    return outputs


def similarity_matrix(meteor, generations, reference):
    similarity = np.zeros((len(generations), len(reference)), dtype=float)

    if similarity.size == 0:
        return similarity

    pair_references = []
    pair_generations = []

    for generation in generations:
        for segment in reference:
            pair_references.append([" ".join(segment["tokens"])])
            pair_generations.append(" ".join(generation["tokens"]))

    scores = meteor.score(pair_references, pair_generations)
    index = 0

    for i in range(len(generations)):
        for j in range(len(reference)):
            similarity[i, j] = scores[index]
            index += 1

    return similarity


def alignment_path(similarity):
    n, m = similarity.shape
    scores = np.zeros((n + 1, m + 1), dtype=float)
    move = np.zeros((n + 1, m + 1), dtype=np.int8)

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            up = scores[i - 1, j]
            diagonal = scores[i - 1, j - 1] + similarity[i - 1, j - 1]
            left = scores[i, j - 1]
            scores[i, j] = max(up, diagonal, left)

            if diagonal >= up and diagonal >= left:
                move[i, j] = 1
            elif up >= left:
                move[i, j] = 0
            else:
                move[i, j] = 2

    i, j = n, m
    path = [(i, j)]

    while i > 0 or j > 0:
        if i == 0:
            j -= 1
        elif j == 0:
            i -= 1
        elif move[i, j] == 1:
            i -= 1
            j -= 1
        elif move[i, j] == 0:
            i -= 1
        else:
            j -= 1

        path.append((i, j))

    path.reverse()

    assert path[0] == (0, 0)
    assert path[-1] == (n, m)

    return path


def optimal_alignment(similarity, generations, reference):
    path = alignment_path(similarity)
    pairs = []

    for (i0, j0), (i1, j1) in zip(path[:-1], path[1:]):
        if i1 == i0 + 1 and j1 == j0 + 1:
            pairs.append(
                {
                    "generation": saved_segment(generations[i0]),
                    "reference": saved_segment(reference[j0]),
                    "meteor": float(similarity[i0, j0] * 100.0),
                }
            )

    return pairs


def saved_segment(segment):
    return {
        "text": segment["text"],
        "start": segment["start"],
        "end": segment["end"],
    }
