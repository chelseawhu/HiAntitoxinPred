"""MAIA window/RNG/aggregation protocol from the frozen scientific implementation."""

import hashlib

import json

import math

AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"

OPERATORS = ("U19", "SHUFFLE_COMPOSITION")

RNG_SERIALIZATION = "sha256-compact-json-array-v1"

RNG_GENERATOR = "sha256-counter-big-endian64-rejection256-v1"


def _positive_integer(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(name + " must be a positive integer")
    return value


def _finite(value, name):
    try:
        value = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(name + " must be finite") from exc
    if not math.isfinite(value):
        raise ValueError(name + " must be finite")
    return value


def full_windows(length):
    """All complete W20/stride1 windows; one actual-length window if L<20."""
    _positive_integer(length, "length")
    if length < 20:
        return [(1, length)]
    return [(start, start + 19) for start in range(1, length - 18)]


class _CounterRNG:
    def __init__(self, seed_hex):
        self.seed_bytes = bytes.fromhex(seed_hex)
        self.counter = 0

    def randbelow(self, bound):
        _positive_integer(bound, "random bound")
        space = 1 << 256
        limit = space - space % bound
        while True:
            payload = self.seed_bytes + self.counter.to_bytes(8, "big")
            self.counter += 1
            value = int.from_bytes(hashlib.sha256(payload).digest(), "big")
            if value < limit:
                return value % bound


def mutate_window(
    sequence, start_1based, end_1based, operator_id, group_seed, repeat_id
):
    """Return one registered mutation shared by all model seeds.

    A shuffle that equals the original is retained, with an empty changes list.
    """
    if (
        not isinstance(sequence, str)
        or not sequence
        or set(sequence) - set(AMINO_ACIDS)
    ):
        raise ValueError(
            "sequence must contain only the 20 uppercase standard amino acids"
        )
    if (start_1based, end_1based) not in full_windows(len(sequence)):
        raise ValueError("coordinates must identify one complete registered W20 window")
    if operator_id not in OPERATORS:
        raise ValueError("unknown operator_id")
    _positive_integer(group_seed, "group_seed")
    _positive_integer(repeat_id, "repeat_id")
    input_sha256 = hashlib.sha256(sequence.encode("ascii")).hexdigest()
    rng_tuple = [operator_id, input_sha256, start_1based, group_seed, repeat_id]
    rng_payload = json.dumps(rng_tuple, ensure_ascii=True, separators=(",", ":"))
    rng_sha256 = hashlib.sha256(rng_payload.encode("utf-8")).hexdigest()
    rng = _CounterRNG(rng_sha256)
    start, stop = start_1based - 1, end_1based
    original_window = sequence[start:stop]
    result = list(original_window)
    selected = list(range(len(result)))
    if operator_id == "SHUFFLE_COMPOSITION":
        for index in range(len(result) - 1, 0, -1):
            other = rng.randbelow(index + 1)
            result[index], result[other] = result[other], result[index]
    else:
        for index in selected:
            alternatives = AMINO_ACIDS.replace(original_window[index], "")
            result[index] = alternatives[rng.randbelow(19)]
    mutant_sequence = sequence[:start] + "".join(result) + sequence[stop:]
    changes = [
        dict(position_1based=start_1based + index, original=original, mutant=mutant)
        for index, (original, mutant) in enumerate(zip(original_window, result))
        if original != mutant
    ]
    return dict(
        operator_id=operator_id,
        group_seed=group_seed,
        repeat_id=repeat_id,
        start_1based=start_1based,
        end_1based=end_1based,
        actual_window_length=len(result),
        input_sequence_sha256=input_sha256,
        rng_tuple=rng_tuple,
        rng_payload_utf8=rng_payload,
        rng_sha256=rng_sha256,
        rng_serialization=RNG_SERIALIZATION,
        rng_generator=RNG_GENERATOR,
        selected_positions_1based=[start_1based + index for index in selected],
        changes=changes,
        mutant_sequence=mutant_sequence,
        mutant_sequence_sha256=hashlib.sha256(
            mutant_sequence.encode("ascii")
        ).hexdigest(),
    )


def aggregate_signed_sensitivity(length, original_logit, records):
    """d(w)=mean(zorig-zmut), S=mean_covering(d/W), A=mean_covering(abs(d)/W).

    Complete windows are required, exactly once each. Repeat-group quotas/model
    identity are checked by the caller; this math function
    also accepts small analytic fixtures with fewer repeats.
    """
    windows = full_windows(length)
    original_logit = _finite(original_logit, "original logit")
    by_window = {}
    for record in records:
        key = (record["start_1based"], record["end_1based"])
        if key in by_window or key not in windows:
            raise ValueError("duplicate or nonregistered window")
        mutants = [_finite(value, "mutant logit") for value in record["mutant_logits"]]
        if not mutants:
            raise ValueError("window has no repeat scores")
        by_window[key] = math.fsum(original_logit - value for value in mutants) / len(
            mutants
        )
    if set(by_window) != set(windows):
        raise ValueError("incomplete window coverage")
    signed_parts = [[] for _ in range(length)]
    absolute_parts = [[] for _ in range(length)]
    means = []
    for start, end in windows:
        delta = by_window[(start, end)]
        width = end - start + 1
        means.append(
            dict(
                start_1based=start,
                end_1based=end,
                actual_window_length=width,
                mean_delta=delta,
            )
        )
        for index in range(start - 1, end):
            signed_parts[index].append(delta / width)
            absolute_parts[index].append(abs(delta) / width)
    coverage = [len(parts) for parts in signed_parts]
    return dict(
        window_means=means,
        coverage=coverage,
        signed_support=[math.fsum(parts) / len(parts) for parts in signed_parts],
        absolute_sensitivity=[
            math.fsum(parts) / len(parts) for parts in absolute_parts
        ],
    )


def explain(sequence, score_sequences, *, operator="U19", batch_size=2, progress=None):
    """Full MAIA with fixed registered RNG groups and no annotation input.

    score_sequences must return one raw logit per supplied full protein. It must
    use the same encoding/padding settings for original and perturbed inputs.
    """
    if operator not in ("U19", "SHUFFLE_COMPOSITION"):
        raise ValueError("Unsupported public MAIA operator")
    _positive_integer(batch_size, "batch_size")
    ledger = [
        mutate_window(sequence, s, e, operator, g, r)
        for s, e in full_windows(len(sequence))
        for g in (1001, 2001)
        for r in range(1, 6)
    ]
    original = _finite(score_sequences([sequence])[0], "original logit")
    cache = {sequence: original}
    raw = []
    for offset in range(0, len(ledger), batch_size):
        rows = ledger[offset : offset + batch_size]
        pending = list(
            dict.fromkeys(
                r["mutant_sequence"] for r in rows if r["mutant_sequence"] not in cache
            )
        )
        if pending:
            scores = list(score_sequences(pending))
            if len(scores) != len(pending):
                raise ValueError("Scorer output length mismatch")
            cache.update(zip(pending, [_finite(z, "mutant logit") for z in scores]))
        raw.extend(cache[r["mutant_sequence"]] for r in rows)
        if progress is not None:
            progress(min(offset + batch_size, len(ledger)), len(ledger))
    records = []
    for s, e in full_windows(len(sequence)):
        records.append(
            dict(
                start_1based=s,
                end_1based=e,
                mutant_logits=[
                    z
                    for r, z in zip(ledger, raw)
                    if (r["start_1based"], r["end_1based"]) == (s, e)
                ],
            )
        )
    return dict(
        sequence=sequence,
        original_logit=original,
        ledger=ledger,
        raw_logits=raw,
        aggregate=aggregate_signed_sensitivity(len(sequence), original, records),
        protocol=dict(
            window=20,
            stride=1,
            operator=operator,
            group_seeds=[1001, 2001],
            repeats_per_group=5,
            rng_serialization=RNG_SERIALIZATION,
            rng_generator=RNG_GENERATOR,
        ),
    )
