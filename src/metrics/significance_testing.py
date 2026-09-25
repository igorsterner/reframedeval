import numpy as np
import tqdm

import utils.scoring


def run_tests(config, metric_names, scores, results):
    rounds = config.significance_testing_permutation_test.rounds
    alpha = config.significance_testing_permutation_test.alpha
    seed = config.significance_testing_permutation_test.seed
    system_names = list(scores)
    p_values = {name: {} for name in metric_names}
    unpaired = {name: [] for name in metric_names}
    test_count = len(metric_names) * len(system_names) * (len(system_names) - 1) // 2

    with tqdm.tqdm(
        total=test_count, desc="Significance tests", unit="test"
    ) as progress:
        for index, system_x in enumerate(system_names):
            for system_y in system_names[index + 1 :]:
                for name in metric_names:
                    x = scores[system_x][name]
                    y = scores[system_y][name]

                    if list(x) == list(y):
                        p = paired_p_value(x, y, rounds, seed)
                    else:
                        p = unpaired_p_value(x, y, rounds, seed)
                        unpaired[name].append((system_x, system_y))

                    p_values[name][(system_x, system_y)] = p
                    progress.update()

    return (
        p_values,
        win_loss_records(metric_names, system_names, p_values, results, alpha),
        unpaired,
    )


def win_loss_records(metric_names, system_names, p_values, results, alpha):
    records = {}

    for name in metric_names:
        records[name] = {system_name: [0, 0] for system_name in system_names}

        for (system_x, system_y), p in p_values[name].items():
            if p >= alpha or results[system_x][name] == results[system_y][name]:
                continue

            if results[system_x][name] > results[system_y][name]:
                winner, loser = system_x, system_y
            else:
                winner, loser = system_y, system_x

            records[name][winner][0] += 1
            records[name][loser][1] += 1

    return records


def paired_p_value(x, y, rounds, seed):
    rng = np.random.RandomState(seed)
    pairs = aligned_pairs(x, y)

    movie_count = len(x)
    x_values = np.array([x[movie][item] for movie, item in pairs])
    y_values = np.array([y[movie][item] for movie, item in pairs])
    weights = np.array([1.0 / (movie_count * len(x[movie])) for movie, item in pairs])

    reference_stat = abs((x_values - y_values) @ weights)
    expected_reference_stat = abs(
        utils.scoring.aggregate(x) - utils.scoring.aggregate(y)
    )
    assert np.isclose(reference_stat, expected_reference_stat), (
        reference_stat,
        expected_reference_stat,
    )

    flips = rng.randn(rounds, len(pairs)) > 0.0
    sample_x = np.where(flips, y_values, x_values)
    sample_y = np.where(flips, x_values, y_values)
    diffs = np.abs((sample_x - sample_y) @ weights)
    at_least_as_extreme = (diffs > reference_stat) | np.isclose(diffs, reference_stat)

    return np.mean(at_least_as_extreme)

    # Looped implementation:
    # reference_stat = abs(utils.scoring.aggregate(x) - utils.scoring.aggregate(y))
    # sample_x = {movie: dict(items) for movie, items in x.items()}
    # sample_y = {movie: dict(items) for movie, items in y.items()}
    # at_least_as_extreme = 0.0
    #
    # for _ in range(rounds):
    #     flips = rng.randn(len(pairs)) > 0.0
    #
    #     for (movie, item), flip in zip(pairs, flips):
    #         if flip:
    #             sample_x[movie][item] = y[movie][item]
    #             sample_y[movie][item] = x[movie][item]
    #         else:
    #             sample_x[movie][item] = x[movie][item]
    #             sample_y[movie][item] = y[movie][item]
    #
    #     diff = abs(
    #         utils.scoring.aggregate(sample_x)
    #         - utils.scoring.aggregate(sample_y)
    #     )
    #
    #     if diff > reference_stat or np.isclose(diff, reference_stat):
    #         at_least_as_extreme += 1.0
    #
    # return at_least_as_extreme / rounds


def aligned_pairs(x, y):
    assert list(x) == list(y), (list(x), list(y))

    pairs = []

    for movie in x:
        assert list(x[movie]) == list(y[movie]), (
            movie,
            list(x[movie]),
            list(y[movie]),
        )

        for item in x[movie]:
            pairs.append((movie, item))

    return pairs


def unpaired_p_value(x, y, rounds, seed):
    rng = np.random.RandomState(seed)

    x_values = np.array(flattened(x))
    y_values = np.array(flattened(y))
    pooled = np.concatenate((x_values, y_values))

    x_movie_count = len(x)
    x_weights = np.array(
        [1.0 / (x_movie_count * len(items)) for items in x.values() for _ in items]
    )
    y_movie_count = len(y)
    y_weights = np.array(
        [1.0 / (y_movie_count * len(items)) for items in y.values() for _ in items]
    )
    weights = np.concatenate((x_weights, -y_weights))

    reference_stat = abs(pooled @ weights)
    expected_reference_stat = abs(
        utils.scoring.aggregate(x) - utils.scoring.aggregate(y)
    )
    assert np.isclose(reference_stat, expected_reference_stat), (
        reference_stat,
        expected_reference_stat,
    )

    shuffled = np.stack([rng.permutation(pooled) for _ in range(rounds)])
    diffs = np.abs(shuffled @ weights)
    at_least_as_extreme = (diffs > reference_stat) | np.isclose(diffs, reference_stat)

    return np.mean(at_least_as_extreme)

    # Looped implementation:
    # pooled = np.array(flattened(x) + flattened(y))
    # reference_stat = abs(utils.scoring.aggregate(x) - utils.scoring.aggregate(y))
    # sample_x = {movie: dict(items) for movie, items in x.items()}
    # sample_y = {movie: dict(items) for movie, items in y.items()}
    # at_least_as_extreme = 0.0
    #
    # for _ in range(rounds):
    #     shuffled = rng.permutation(pooled)
    #     offset = refill(sample_x, shuffled, 0)
    #     offset = refill(sample_y, shuffled, offset)
    #
    #     assert offset == len(shuffled)
    #
    #     diff = abs(
    #         utils.scoring.aggregate(sample_x)
    #         - utils.scoring.aggregate(sample_y)
    #     )
    #
    #     if diff > reference_stat or np.isclose(diff, reference_stat):
    #         at_least_as_extreme += 1.0
    #
    # return at_least_as_extreme / rounds


def flattened(scores_by_movie):
    values = []

    for items in scores_by_movie.values():
        values += list(items.values())

    return values


def refill(sample, values, offset):
    for movie in sample:
        for item in sample[movie]:
            sample[movie][item] = values[offset]
            offset += 1

    return offset
