import pathlib

import tqdm
import pycocoevalcap.tokenizer.ptbtokenizer

import utils.config
import utils.constants
import utils.data
import utils.io
import utils.sequences
import utils.spans

TOKENIZER = pycocoevalcap.tokenizer.ptbtokenizer.PTBTokenizer()


def relative_stems(meta, is_challenge):
    stems = []

    for movie, movie_meta in meta.items():
        if is_challenge:
            stems.append(movie)
        else:
            for video in movie_meta:
                stems.append(f"{movie}/{video['id']}")

    return stems


def discover_files(stems, generations_dir, out_dir):
    files = []

    for stem in stems:
        json_path = generations_dir / f"{stem}.json"
        srt_path = generations_dir / f"{stem}.srt"
        output = out_dir / f"{stem}.json"

        if json_path.exists() and srt_path.exists():
            raise ValueError(f"Both JSON and SRT exist for {generations_dir / stem}")

        if json_path.exists():
            files.append({"input": json_path, "output": output})

        if srt_path.exists():
            files.append({"input": srt_path, "output": output})

    if len(files) == 0:
        raise ValueError(f"No generation files found under {generations_dir}")

    suffixes = {file["input"].suffix for file in files}

    if len(suffixes) != 1:
        raise ValueError(f"Mixed JSON and SRT generations in {generations_dir}")

    return files


def read_segments(path):
    if path.suffix == ".json":
        return utils.io.load_segments(path)

    return srt_segments(path)


def srt_segments(path):
    import pysrt

    segments = []

    for item in pysrt.open(str(path)):
        segments.append(
            {
                "start": item.start.ordinal / 1000.0,
                "end": item.end.ordinal / 1000.0,
                "text": utils.io.normalize_text(item.text_without_tags),
            }
        )

    segments.sort(key=lambda segment: (segment["start"], segment["end"]))
    return segments


def split_elements_batched(sentences_by_segment, sat_element):
    sentences = []

    for segment_sentences in sentences_by_segment:
        sentences += segment_sentences

    if len(sentences) == 0:
        return sentences_by_segment

    elements_by_sentence = list(
        sat_element.split(sentences, split_on_input_newlines=False)
    )
    assert len(elements_by_sentence) == len(sentences)

    elements_by_segment = []
    sentence_index = 0

    for segment_sentences in sentences_by_segment:
        segment_elements = []

        for _ in segment_sentences:
            segment_elements += elements_by_sentence[sentence_index]
            sentence_index += 1

        elements_by_segment.append(segment_elements)

    assert sentence_index == len(elements_by_sentence)
    return elements_by_segment


def split_texts(texts, sat_sentence, sat_element):
    sentences_by_segment = [[text] for text in texts]

    if sat_sentence is not None and len(texts) > 0:
        sentences_by_segment = list(
            sat_sentence.split(texts, split_on_input_newlines=False)
        )

    assert len(sentences_by_segment) == len(texts)

    if sat_element is not None:
        pieces_by_segment = split_elements_batched(sentences_by_segment, sat_element)
    else:
        pieces_by_segment = sentences_by_segment

    assert len(pieces_by_segment) == len(texts)
    return pieces_by_segment


def tokenize_segments(segments):
    if len(segments) == 0:
        return

    raw = {}

    for index, segment in enumerate(segments):
        raw[index] = [{"caption": piece} for piece in segment["pieces"]]

    if not any(raw.values()):
        for segment in segments:
            segment["piece_tokens"] = []
        return

    tokenized = TOKENIZER.tokenize(raw)

    for index, segment in enumerate(segments):
        segment["piece_tokens"] = [
            tokenized[index][position].split()
            for position in range(len(segment["pieces"]))
        ]


def tokenized_texts(texts):
    if len(texts) == 0:
        return []

    raw = {index: [{"caption": text}] for index, text in enumerate(texts)}
    tokenized = TOKENIZER.tokenize(raw)

    return [tokenized[index][0].split() for index in range(len(texts))]


def realistic_segments(segments, gaps, collar_seconds):
    if len(segments) == 0:
        return []

    tokens = tokenized_texts([segment["text"] for segment in segments])
    candidates = []

    for segment, segment_tokens in zip(segments, tokens):
        duration = segment["end"] - segment["start"]
        assert duration > 0.0, segment
        words_per_minute = len(segment_tokens) / duration * 60.0

        if words_per_minute >= 300.0:
            continue

        if not utils.spans.is_assigned_to_any_gap(
            segment,
            gaps,
            collar_seconds,
        ):
            continue

        candidates.append(dict(segment))

    out = []

    for segment in candidates:
        if len(out) == 0 or segment["start"] >= out[-1]["end"]:
            out.append(segment)

    return out


def interpolate_time(start, end, total_tokens, token_offset):
    return start + (end - start) * (token_offset / total_tokens)


def segment_outputs(segment):
    kept = []

    for piece, tokens in zip(segment["pieces"], segment["piece_tokens"]):
        if len(tokens) > 0:
            kept.append({"text": piece.strip(), "tokens": tokens})

    total = 0

    for piece in kept:
        total += len(piece["tokens"])

    cursor = 0
    out = []

    for piece in kept:
        piece_start = interpolate_time(segment["start"], segment["end"], total, cursor)
        cursor += len(piece["tokens"])
        piece_end = interpolate_time(segment["start"], segment["end"], total, cursor)
        out.append(
            {
                "start": round(piece_start, 3),
                "end": round(piece_end, 3),
                "text": piece["text"],
                "tokens": piece["tokens"],
            }
        )

    return out


def write_file(path, segments):
    data = {}

    for index, segment in enumerate(segments):
        data[str(index + 1)] = segment

    utils.io.write_json(path, data)


def process_original_system(
    stems,
    generations_dir,
    out_dir,
    sat_sentence,
    sat_element,
):
    files = discover_files(stems, generations_dir, out_dir)
    segments = []

    for file in files:
        file["segments"] = read_segments(file["input"])
        segments += file["segments"]

    texts = [segment["text"] for segment in segments]
    piece_lists = split_texts(texts, sat_sentence, sat_element)

    for segment, pieces in zip(segments, piece_lists):
        segment["pieces"] = pieces

    tokenize_segments(segments)
    written = 0

    for file in files:
        outputs = []

        for segment in file["segments"]:
            outputs += segment_outputs(segment)

        write_file(file["output"], outputs)
        written += len(outputs)

    return written


def process_system(
    stems,
    generations_dir,
    variants,
    gaps_by_stem,
    collar_seconds,
    sat_sentence,
    sat_element,
):
    original = variants[0]
    assert not original["is_realistic"], original
    files = discover_files(stems, generations_dir, original["path"])

    for file in files:
        file["segments"] = read_segments(file["input"])

    variant_files = []

    for variant in variants:
        for file in files:
            relative = file["output"].relative_to(original["path"])
            stem = str(relative.with_suffix(""))

            if variant["is_realistic"]:
                segments = realistic_segments(
                    file["segments"],
                    gaps_by_stem[stem],
                    collar_seconds,
                )
            else:
                segments = [dict(segment) for segment in file["segments"]]

            variant_files.append(
                {
                    "name": variant["name"],
                    "output": variant["path"] / relative,
                    "segments": segments,
                }
            )

    segments = [segment for file in variant_files for segment in file["segments"]]

    texts = [segment["text"] for segment in segments]
    piece_lists = split_texts(texts, sat_sentence, sat_element)

    for segment, pieces in zip(segments, piece_lists):
        segment["pieces"] = pieces

    tokenize_segments(segments)
    written = {variant["name"]: 0 for variant in variants}

    for file in variant_files:
        outputs = []

        for segment in file["segments"]:
            outputs += segment_outputs(segment)

        write_file(file["output"], outputs)
        written[file["name"]] += len(outputs)

    return written


def dialogue_gaps_by_stem(meta, is_challenge, reference_dir, min_gap_seconds):
    out = {}

    for movie, movie_meta in meta.items():
        if is_challenge:
            items = utils.sequences.movie_items(
                reference_dir,
                movie,
                movie_meta,
                min_gap_seconds,
            )
            out[movie] = [gap for item in items for gap in item["gaps"]]
        else:
            for video in movie_meta:
                item = utils.data.video_item(
                    reference_dir,
                    movie,
                    video,
                    min_gap_seconds,
                )
                out[f"{movie}/{video['id']}"] = item["gaps"]

    return out


def reference_jobs(meta, is_challenge, reference_in, reference_out):
    copies = []
    references = []

    for movie, movie_meta in meta.items():
        if is_challenge:
            for folder in ["subs", "scenes", "sequences"]:
                copies.append(
                    (
                        reference_in / folder / f"{movie}.json",
                        reference_out / folder / f"{movie}.json",
                    )
                )

            for reference in utils.constants.REFERENCE_NAMES:
                references.append(
                    (
                        reference_in / reference / f"{movie}.json",
                        reference_out / reference / f"{movie}.json",
                    )
                )
        else:
            for video in movie_meta:
                clip = f"{video['id']}.json"
                copies.append(
                    (
                        reference_in / movie / "subs" / clip,
                        reference_out / "subs" / movie / clip,
                    )
                )

                for reference in utils.constants.REFERENCE_NAMES:
                    references.append(
                        (
                            reference_in / movie / reference / clip,
                            reference_out / reference / movie / clip,
                        )
                    )

    return copies, references


def preprocess_references(copies, references):
    for source, output in copies:
        utils.io.write_json(output, utils.io.load_json(source))

    files = [utils.io.load_json(source) for source, _ in references]
    raw = {}

    for index, data in enumerate(files):
        raw[index] = [
            {"caption": utils.io.normalize_text(segment["text"])}
            for segment in data.values()
        ]

    tokenized = TOKENIZER.tokenize(raw) if any(raw.values()) else {}

    for index, (data, pair) in enumerate(zip(files, references)):
        source, output = pair

        for position, segment in enumerate(data.values()):
            tokens = tokenized[index][position].split()

            assert len(tokens) > 0, (source, segment["text"])
            segment["tokens"] = tokens

        utils.io.write_json(output, data)


def main():
    args = utils.config.parse_args()
    config, split = utils.config.load(args)
    split_dir = utils.constants.SPLITS[split]
    is_challenge = split == "challenge"
    raw_generations = pathlib.Path(config.paths.generations) / split_dir
    preprocessed = pathlib.Path(config.paths.preprocessed_data)
    generation_out = preprocessed / "generations" / split_dir
    reference_in = pathlib.Path(config.paths.reference_data) / split_dir
    reference_out = preprocessed / "references" / split_dir
    meta = utils.data.load_meta(config, split)
    stems = relative_stems(meta, is_challenge)
    jobs = []

    for entry in config.systems:
        name = entry.name
        generations_dir = raw_generations / name
        variants = []

        for variant in utils.config.system_variants(entry):
            variant["path"] = generation_out / variant["name"]
            variants.append(variant)

        if not generations_dir.is_dir():
            if utils.config.realistic_enabled(entry):
                raise ValueError(
                    f"No raw generations for {name}; set realistic: false to keep "
                    f"an existing preprocessed-only system"
                )

            if not (generation_out / name).is_dir():
                raise ValueError(f"No generations or preprocessed data for {name}")

            print(f"{name}: no raw generations, keeping preprocessed files")
            continue

        jobs.append(
            {
                "name": name,
                "generations": generations_dir,
                "variants": variants,
                "splitting": utils.config.splitting_settings(entry),
            }
        )

    existing = [
        variant["path"]
        for job in jobs
        for variant in job["variants"]
        if variant["path"].exists()
    ]
    utils.io.delete_after_confirmation(
        existing, "preprocessing output folders (including QA folders)"
    )

    copies, references = reference_jobs(meta, is_challenge, reference_in, reference_out)
    preprocess_references(copies, references)
    print(f"references: wrote {len(copies) + len(references)} files")

    # Load each model at most once, according to per-system splitting settings.
    need_sentences = any(job["splitting"][0] for job in jobs)
    need_elements = any(job["splitting"][1] for job in jobs)

    sat_sentence = None
    sat_element = None

    if need_sentences or need_elements:
        import torch
        import wtpsplit

        use_cuda = torch.cuda.is_available()

        if need_sentences:
            sat_sentence = wtpsplit.SaT(config.preprocessing.sentence_model)

            if use_cuda:
                sat_sentence.half().to("cuda")

        if need_elements:
            sat_element = wtpsplit.SaT(
                config.preprocessing.element_model,
                hub_prefix=None,
            )

            if use_cuda:
                sat_element.half().to("cuda")

    has_realistic = any(
        variant["is_realistic"] for job in jobs for variant in job["variants"]
    )
    gaps_by_stem = {}

    if has_realistic:
        gaps_by_stem = dialogue_gaps_by_stem(
            meta,
            is_challenge,
            reference_out,
            config.gaps.min_gap_seconds.cider,
        )

    for job in tqdm.tqdm(jobs):
        split_sentences, split_elements = job["splitting"]
        job_sentence = sat_sentence if split_sentences else None
        job_element = sat_element if split_elements else None

        if len(job["variants"]) == 1:
            variant = job["variants"][0]
            assert not variant["is_realistic"], variant

            written = process_original_system(
                stems,
                job["generations"],
                variant["path"],
                job_sentence,
                job_element,
            )
            tqdm.tqdm.write(f"{variant['name']}: wrote {written} segments")
            continue

        written = process_system(
            stems,
            job["generations"],
            job["variants"],
            gaps_by_stem,
            config.gaps.collar_seconds,
            job_sentence,
            job_element,
        )

        for name, count in written.items():
            tqdm.tqdm.write(f"{name}: wrote {count} segments")


if __name__ == "__main__":
    main()
