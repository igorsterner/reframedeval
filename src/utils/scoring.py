def mean(values):
    assert len(values) > 0
    return sum(values) / float(len(values))


def mean_over_movies(scores_by_movie):
    movie_means = [mean(list(items.values())) for items in scores_by_movie.values()]

    return mean(movie_means)


def aggregate(scores_by_movie):
    return mean_over_movies(scores_by_movie)


class TextScorer:
    def __init__(self, scorer):
        self.scorer = scorer

    def score(self, references, generations):
        assert len(references) == len(generations)

        reference_map = dict(enumerate(references))
        generation_map = {
            index: [generation] for index, generation in enumerate(generations)
        }
        _, scores = self.scorer.compute_score(reference_map, generation_map)

        return [float(score) for score in scores]
