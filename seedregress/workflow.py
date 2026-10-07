"""Patch a ComfyUI API-format workflow with one case."""

from __future__ import annotations

import copy

from seedregress.errors import SuiteError
from seedregress.suite import Bindings, Case


def apply_case(workflow: dict, case: Case, bindings: Bindings) -> dict:
    graph = copy.deepcopy(workflow)
    _require_node(graph, bindings.positive, "positive")["inputs"]["text"] = case.positive
    _require_node(graph, bindings.negative, "negative")["inputs"]["text"] = case.negative

    sampler = _require_node(graph, bindings.sampler, "sampler")["inputs"]
    sampler["seed"] = int(case.seed)
    sampler["steps"] = int(case.steps)
    sampler["sampler_name"] = case.sampler
    if case.scheduler:
        sampler["scheduler"] = case.scheduler
    if case.cfg is not None:
        sampler["cfg"] = float(case.cfg)
    if case.denoise is not None:
        sampler["denoise"] = float(case.denoise)

    latent = _require_node(graph, bindings.latent, "latent")["inputs"]
    latent["width"] = int(case.width)
    latent["height"] = int(case.height)

    if case.ckpt_name:
        if not bindings.checkpoint:
            raise SuiteError(f"Case {case.id} sets ckpt_name but the suite has no checkpoint binding")
        _require_node(graph, bindings.checkpoint, "checkpoint")["inputs"]["ckpt_name"] = case.ckpt_name

    if bindings.lora:
        _apply_loras(graph, bindings.lora, case)
    elif case.loras:
        raise SuiteError(f"Case {case.id} lists LoRAs but the suite has no lora binding")
    return graph


def _require_node(graph: dict, node_id: str, label: str) -> dict:
    node = graph.get(node_id)
    if not isinstance(node, dict) or "inputs" not in node:
        raise SuiteError(f"Workflow node {node_id} ({label}) has no inputs")
    return node


def _apply_loras(graph: dict, lora_id: str, case: Case) -> None:
    node = _require_node(graph, lora_id, "lora")
    if node.get("class_type") != "LoraLoader":
        raise SuiteError(f"Node {lora_id} is not a LoraLoader")
    model_in = list(node["inputs"]["model"])
    clip_in = list(node["inputs"]["clip"])
    if not case.loras:
        _rewire(graph, lora_id, 0, model_in)
        _rewire(graph, lora_id, 1, clip_in)
        del graph[lora_id]
        return

    ids = [lora_id]
    for index in range(1, len(case.loras)):
        clone_id = f"{lora_id}_sr{index}"
        graph[clone_id] = copy.deepcopy(node)
        ids.append(clone_id)
    if len(ids) > 1:
        _rewire(graph, lora_id, 0, [ids[-1], 0], skip=set(ids))
        _rewire(graph, lora_id, 1, [ids[-1], 1], skip=set(ids))

    prev_model, prev_clip = model_in, clip_in
    for node_id, spec in zip(ids, case.loras, strict=True):
        inputs = graph[node_id]["inputs"]
        inputs["lora_name"] = spec.name
        inputs["strength_model"] = float(spec.strength_model)
        inputs["strength_clip"] = float(spec.strength_clip)
        inputs["model"] = list(prev_model)
        inputs["clip"] = list(prev_clip)
        prev_model = [node_id, 0]
        prev_clip = [node_id, 1]


def _rewire(graph: dict, node_id: str, slot: int, new_link: list, skip: set[str] | None = None) -> None:
    skip = skip or set()
    for nid, node in graph.items():
        if nid in skip or not isinstance(node, dict):
            continue
        inputs = node.get("inputs")
        if not isinstance(inputs, dict):
            continue
        for key, value in inputs.items():
            if (
                isinstance(value, list)
                and len(value) == 2
                and str(value[0]) == str(node_id)
                and value[1] == slot
            ):
                inputs[key] = list(new_link)
