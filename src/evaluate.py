import analysis.attribution_sweep
import analysis.gap_length
import analysis.per_reference
import analysis.words_per_minute
import metrics.aligned
import metrics.dialogue_gaps
import metrics.qa_based
import metrics.significance_testing
import utils.config
import utils.constants
import utils.data
import utils.latex
import utils.scoring

METRIC_CLASSES = {
    "dialogue_gaps": metrics.dialogue_gaps.DialogueGaps,
    "aligned": metrics.aligned.SODA,
    "qa_based": metrics.qa_based.QEval,
}

ANALYSIS_CLASSES = {
    "gap_length": analysis.gap_length.GapLength,
    "per_reference": analysis.per_reference.PerReference,
    "attribution_sweep": analysis.attribution_sweep.AttributionSweep,
    "words_per_minute": analysis.words_per_minute.WordsPerMinute,
}


def load_metrics(config, selected_metrics):
    out = []

    for family, names in selected_metrics.items():
        metric_class = METRIC_CLASSES[family]
        out.append(metric_class(config, names))

    return out


def load_analyses(config, metric_names):
    out = []

    for entry in config.analysis:
        analysis_class = ANALYSIS_CLASSES[entry.name]
        names = analysis_metric_names(entry, analysis_class, metric_names)

        if len(analysis_class.SUPPORTED_METRICS) > 0 and len(names) == 0:
            continue

        out.append(analysis_class(config, entry, names))

    return out


def analysis_metric_names(entry, analysis_class, metric_names):
    supported = analysis_class.SUPPORTED_METRICS

    if "metric_subset" not in entry:
        return [name for name in supported if name in metric_names]

    names = list(entry.metric_subset)

    if len(names) == 0:
        raise ValueError(f"{entry.name} metric_subset must not be empty")

    if len(names) != len(set(names)):
        raise ValueError(f"{entry.name} metric_subset entries must not be repeated")

    for name in names:
        if name not in supported:
            raise ValueError(
                f"{entry.name} does not support {name}, choose from {supported}"
            )

        if name not in metric_names:
            raise ValueError(f"{name} must be selected in metrics before {entry.name}")

    return names


def evaluate_items(items, configured_metrics):
    scores = {}
    analysis_data = {}

    for metric in configured_metrics:
        metric_scores, metric_analysis_data = metric.score(items)

        for metric_name, scores_by_movie in metric_scores.items():
            scores[metric_name] = scores_by_movie

        for metric_name, data_by_movie in metric_analysis_data.items():
            analysis_data[metric_name] = data_by_movie

    return scores, analysis_data


def results_table(metric_names, results, labels):
    rows = []

    for system_name, scores in results.items():
        row = [labels[system_name]]

        for metric_name in metric_names:
            row.append(f"{scores[metric_name]:.1f}")

        rows.append(row)

    headers = ["System"] + [utils.constants.metric_label(name) for name in metric_names]
    return utils.latex.tabular(headers, rows)


def significance_ranking(metric_names, system_names, win_loss_records):
    totals = {}

    for system_name in system_names:
        totals[system_name] = sum(
            win_loss_records[name][system_name][0]
            - win_loss_records[name][system_name][1]
            for name in metric_names
        )

    return sorted(system_names, key=lambda system_name: -totals[system_name])


def win_loss_table(metric_names, ranked, results, win_loss_records, labels):
    rows = []

    for system_name in ranked:
        row = [labels[system_name]]

        for name in metric_names:
            wins, losses = win_loss_records[name][system_name]
            row.append(f"{results[system_name][name]:.1f} ({wins}--{losses})")

        rows.append(row)

    headers = ["System"] + [utils.constants.metric_label(name) for name in metric_names]
    return utils.latex.tabular(headers, rows)


def p_value_table(metric_name, results, pair_p_values, labels):
    ranked = sorted(results, key=lambda system_name: -results[system_name][metric_name])
    rows = []

    for system_name in ranked:
        row = [labels[system_name]]

        for other in ranked:
            if other == system_name:
                row.append("-")
            else:
                row.append(f"{pair_p_value(pair_p_values, system_name, other):.3f}")

        rows.append(row)

    headers = ["System"] + [labels[system_name] for system_name in ranked]

    return utils.latex.tabular(headers, rows)


def pair_p_value(pair_p_values, system_x, system_y):
    if (system_x, system_y) in pair_p_values:
        return pair_p_values[(system_x, system_y)]

    return pair_p_values[(system_y, system_x)]


def main():
    args = utils.config.parse_args()
    config, split = utils.config.load(args)
    data = utils.data.load(config, split)
    selected_metrics = utils.config.selected_metrics(config, args)
    configured_metrics = load_metrics(config, selected_metrics)
    metric_names = [name for names in selected_metrics.values() for name in names]

    configured_analyses = []

    if args.analysis:
        configured_analyses = load_analyses(config, metric_names)
    scores = {}
    results = {}
    items_by_system = {}
    analysis_data_by_system = {}
    labels = {system["name"]: system["label"] for system in data["systems"]}

    for system in data["systems"]:
        items = utils.data.system_items(data, system)
        system_scores, system_analysis_data = evaluate_items(items, configured_metrics)
        items_by_system[system["name"]] = items
        analysis_data_by_system[system["name"]] = system_analysis_data
        scores[system["name"]] = system_scores
        results[system["name"]] = {
            name: utils.scoring.aggregate(system_scores[name]) for name in metric_names
        }

    if not args.significance_testing:
        print(results_table(metric_names, results, labels))
    else:
        p_values, win_loss_records, unpaired = metrics.significance_testing.run_tests(
            config,
            metric_names,
            scores,
            results,
        )
        ranked = significance_ranking(metric_names, results, win_loss_records)
        print(win_loss_table(metric_names, ranked, results, win_loss_records, labels))

        for name in metric_names:
            print()
            print(utils.constants.metric_label(name))
            print(p_value_table(name, results, p_values[name], labels))

            if len(unpaired[name]) > 0:
                pairs = ", ".join(
                    f"{labels[x]} vs {labels[y]}" for x, y in unpaired[name]
                )
                print(f"unpaired: {pairs}")

        items_by_system = {name: items_by_system[name] for name in ranked}
        analysis_data_by_system = {
            name: analysis_data_by_system[name] for name in ranked
        }

    for configured_analysis in configured_analyses:
        print()
        configured_analysis.report(data, items_by_system, analysis_data_by_system)


if __name__ == "__main__":
    main()
