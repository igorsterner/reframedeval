import json
import shutil


def normalize_text(text):
    return " ".join(text.split())


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path, data):
    text = json.dumps(data, indent=4, ensure_ascii=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def load_segments(path):
    segments = []

    for key, segment in load_json(path).items():
        segment["id"] = key
        segment["start"] = float(segment["start"])
        segment["end"] = float(segment["end"])
        segment["text"] = normalize_text(segment["text"])
        segments.append(segment)

    segments.sort(key=lambda item: (item["start"], item["end"], item["id"]))
    return segments


def delete_after_confirmation(paths, description):
    if len(paths) == 0:
        return

    print(f"Existing {description} found:")

    for path in paths:
        print(path)

    answer = input("Permanently delete these and start again? [yes/no] ")

    if answer != "yes":
        raise SystemExit("Nothing deleted, stopping.")

    for path in paths:
        shutil.rmtree(path)
