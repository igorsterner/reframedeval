import pathlib

import datasets
import pysrt

import utils.config
import utils.constants
import utils.io


def parse_srt(text):
    segments = []

    for item in pysrt.SubRipFile.from_string(text or ""):
        segments.append(
            {
                "start": item.start.ordinal / 1000.0,
                "end": item.end.ordinal / 1000.0,
                "text": item.text_without_tags,
            }
        )

    return segments


def write_segments(path, segments):
    data = {str(index + 1): segment for index, segment in enumerate(segments)}
    utils.io.write_json(path, data)


def main():
    args = utils.config.parse_scorer_args()
    config, split = utils.config.load(args)
    reference_dir = pathlib.Path(config.paths.reference_data) / utils.constants.SPLITS[split]
    rows = datasets.load_dataset("igorsterner/reframed", split=split)
    meta = {}

    for row in rows:
        movie = row["imdb_id"]
        video_id = row["video_id"]
        movie_dir = reference_dir / movie

        write_segments(movie_dir / "subs" / f"{video_id}.json", parse_srt(row["subtitles"]))
        write_segments(movie_dir / "reference1" / f"{video_id}.json", parse_srt(row["audio_description_us"]))
        write_segments(movie_dir / "reference2" / f"{video_id}.json", parse_srt(row["audio_description_uk"]))

        if movie not in meta:
            meta[movie] = []

        meta[movie].append({"id": video_id, "duration": float(row["video_duration"])})

    utils.io.write_json(pathlib.Path(config.paths.meta) / f"reframed-{split}.json", meta)
    print(f"wrote reference_data and meta for {len(rows)} clips")


if __name__ == "__main__":
    main()
