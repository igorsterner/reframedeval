import torch
import vllm
import vllm.sampling_params

import utils.constants
import utils.data
import utils.io

ANSWER_LABELS = ["A", "B", "C", "D", "E"]
ANSWER_TOKENS = [" " + label for label in ANSWER_LABELS]


def documents(segments, attribution):
    if len(segments) == 0:
        return [""]

    texts = [segment["text"] for segment in segments]
    out = [" ".join(texts)]

    if attribution:
        for index in range(len(texts)):
            out.append(" ".join(texts[:index] + texts[index + 1 :]))

    return out


def make_prompt(document, question, answers):
    assert len(answers) == len(ANSWER_LABELS)

    lines = [document, f"Question: {question}"]

    for label, answer in zip(ANSWER_LABELS, answers):
        lines.append(f"{label}. {answer['answer']}")

    lines.append("Answer:")

    return "\n".join(lines)


def correct_index(answers):
    correct = [index for index, answer in enumerate(answers) if answer["is_correct"]]

    assert len(correct) == 1
    return correct[0]


def compile_jobs(items, attribution):
    jobs = []

    for item in items:
        jobs += item_jobs(item, attribution)

    assert len(jobs) > 0
    return jobs


def item_jobs(item, attribution):

    item_documents = documents(item["generations"], attribution)
    generation_segments = [saved_segment(segment) for segment in item["generations"]]
    jobs = []

    for reference in item["qa"]:
        for segment_id, segment_qa in item["qa"][reference].items():
            for question_id, question in segment_qa["questions"].items():
                jobs.append(
                    make_job(
                        item,
                        reference,
                        segment_id,
                        segment_qa["midpoint"],
                        question_id,
                        question,
                        item_documents,
                        generation_segments,
                        attribution,
                    )
                )

    return jobs


def make_job(
    item,
    reference,
    segment_id,
    grounding,
    question_id,
    question,
    item_documents,
    generation_segments,
    attribution,
):
    prompts = []
    answer_index = correct_index(question["answers"])

    for document in item_documents:
        prompts.append(make_prompt(document, question["question"], question["answers"]))

    return {
        "qa_path": item["qa_path"],
        "reference": reference,
        "segment": segment_id,
        "question_id": question_id,
        "question_text": question["question"],
        "correct_answer": question["answers"][answer_index]["answer"],
        "answer_texts": [answer["answer"] for answer in question["answers"]],
        "grounding": grounding,
        "correct_index": answer_index,
        "generation_segments": generation_segments,
        "attribution": attribution,
        "prompts": prompts,
    }


def existing_qa_dirs(jobs):
    dirs = []

    for job in jobs:
        qa_dir = job["qa_path"].parent

        if qa_dir not in dirs and qa_dir.exists():
            dirs.append(qa_dir)

    return dirs


def answer_token_ids(llm):
    tokenizer = llm.get_tokenizer()
    out = []

    for token in ANSWER_TOKENS:
        encoded = tokenizer.encode(token, add_special_tokens=False)
        assert len(encoded) == 1
        out.append(encoded[0])

    return out


def output_scores(output, answer_ids):
    logprobs = output.outputs[0].logprobs[0]
    scores = []

    for token_id in answer_ids:
        assert token_id in logprobs
        scores.append(logprobs[token_id].logprob)

    return torch.tensor(scores, dtype=torch.float32)


def make_llm(config):
    llm = vllm.LLM(
        model=config.qa_based.model,
        tensor_parallel_size=config.qa_based.tensor_parallel_size,
        max_model_len=config.qa_based.max_model_len,
        max_logprobs=config.qa_based.max_logprobs,
        # disable_custom_all_reduce=True,  # optional
    )
    answer_ids = answer_token_ids(llm)
    sampling_params = vllm.SamplingParams(
        max_tokens=1,
        temperature=0.0,
        logprobs=config.qa_based.max_logprobs,
        structured_outputs=vllm.sampling_params.StructuredOutputsParams(
            choice=ANSWER_TOKENS,
        ),
        allowed_token_ids=answer_ids,
    )

    return llm, answer_ids, sampling_params


def generate_scores(llm, answer_ids, sampling_params, prompts):
    outputs = llm.generate(prompts, sampling_params=sampling_params)
    assert len(outputs) == len(prompts)

    return [output_scores(output, answer_ids) for output in outputs]


def run_llm(llm, answer_ids, sampling_params, jobs, attribution_all_answers):
    full_prompts = [job["prompts"][0] for job in jobs]
    full_scores = generate_scores(llm, answer_ids, sampling_params, full_prompts)
    scores = [[score] for score in full_scores]
    attribution_prompts = []
    attribution_jobs = []

    for index, job in enumerate(jobs):
        predicted = full_scores[index].argmax().item()
        correct = predicted == job["correct_index"]

        if (
            job["attribution"]
            and len(job["generation_segments"]) > 0
            and (correct or attribution_all_answers)
        ):
            attribution_prompts += job["prompts"][1:]
            attribution_jobs += [index] * len(job["generation_segments"])

    if len(attribution_prompts) > 0:
        attribution_scores = generate_scores(
            llm,
            answer_ids,
            sampling_params,
            attribution_prompts,
        )

        for index, score in zip(attribution_jobs, attribution_scores):
            scores[index].append(score)

    return scores


def job_results(job, scores, attribution_all_answers):
    full = scores[0]
    predicted = full.argmax().item()
    qeval = predicted == job["correct_index"]
    expected = 1

    if job["attribution"] and (qeval or attribution_all_answers):
        expected += len(job["generation_segments"])

    assert len(scores) == expected

    predicted_answer = job["answer_texts"][predicted]
    assert qeval == (predicted_answer == job["correct_answer"]), (
        predicted,
        job["correct_index"],
        predicted_answer,
        job["correct_answer"],
    )
    attribution = None

    if (
        job["attribution"]
        and len(job["generation_segments"]) > 0
        and (qeval or attribution_all_answers)
    ):
        full_prob = torch.softmax(full, dim=0)[predicted].item()
        impacts = []

        for ablated in scores[1:]:
            ablated_prob = torch.softmax(ablated, dim=0)[predicted].item()
            impacts.append(full_prob - ablated_prob)

        attribution = job["generation_segments"][impacts.index(max(impacts))]

    return predicted_answer, attribution


def fill_outputs(jobs, scores, attribution_all_answers):
    outputs = {}

    for job, job_scores in zip(jobs, scores):
        predicted_answer, attribution = job_results(
            job,
            job_scores,
            attribution_all_answers,
        )
        path = job["qa_path"]

        if path not in outputs:
            outputs[path] = {
                reference: {} for reference in utils.constants.REFERENCE_NAMES
            }

        by_reference = outputs[path][job["reference"]]

        if job["segment"] not in by_reference:
            by_reference[job["segment"]] = {}

        result = {
            "question": job["question_text"],
            "correct_answer": job["correct_answer"],
            "predicted_answer": predicted_answer,
            "grounding": job["grounding"],
        }

        if job["attribution"]:
            result["attribution"] = attribution

        by_reference[job["segment"]][job["question_id"]] = result

    return outputs


def write_outputs(outputs):
    for path, data in outputs.items():
        utils.io.write_json(path, data)


def saved_segment(segment):
    return {
        "text": segment["text"],
        "start": segment["start"],
        "end": segment["end"],
    }


def run(config, data, items_by_system, names):
    attribution = "qeval-t" in names
    system_jobs = []

    for system in data["systems"]:
        items = [
            item
            for item in items_by_system[system["name"]]
            if item["is_covered_in_movie_subset"]
        ]
        system_jobs.append(
            {"name": system["name"], "jobs": compile_jobs(items, attribution)}
        )

    all_jobs = []

    for entry in system_jobs:
        all_jobs += entry["jobs"]

    attribution_prompt_count = 0

    for job in all_jobs:
        attribution_prompt_count += len(job["prompts"]) - 1

    print(
        f"Compiled {len(all_jobs):,} QEval prompts and "
        f"{attribution_prompt_count:,} potential QEval-T attribution prompts"
    )
    utils.io.delete_after_confirmation(existing_qa_dirs(all_jobs), "QA folders")

    llm, answer_ids, sampling_params = make_llm(config)
    completed_attribution_prompt_count = 0

    for entry in system_jobs:
        scores = run_llm(
            llm,
            answer_ids,
            sampling_params,
            entry["jobs"],
            config.qa_based.attribution_all_answers,
        )
        completed_attribution_prompt_count += sum(
            len(job_scores) - 1 for job_scores in scores
        )
        outputs = fill_outputs(
            entry["jobs"],
            scores,
            config.qa_based.attribution_all_answers,
        )
        write_outputs(outputs)

        print(f"{entry['name']}: wrote {len(outputs)} qa files")

    print(
        f"Ran {len(all_jobs):,} QEval prompts and "
        f"{completed_attribution_prompt_count:,} QEval-T attribution prompts"
    )
    print(
        "Skipped "
        f"{attribution_prompt_count - completed_attribution_prompt_count:,} "
        f"QEval-T attribution prompts for incorrect answers"
    )
