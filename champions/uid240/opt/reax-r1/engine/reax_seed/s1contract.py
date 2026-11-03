"""The REAX typed-decision contract for the reference models s1-fast and s1-pro.

Follows the reference engine's request parsing and answer normalisation so that a
miner's answer object is exactly what the reference models return for ``s1-fast``
and ``s1-pro``. Limits are kept:
one question per decision, no images.

The validator additionally checks that the derived fields of an answer
(``choice``, ``noul``, ``score``, ``confidence``, ``legend``) follow from its
probabilities, so reference-looking fields over wrong probabilities fail.
"""

from __future__ import annotations

import base64
import binascii
import json
import math
import re
import struct
import zlib
from dataclasses import dataclass
from typing import Any

MAX_STATE_BYTES = 16 * 1024
# The reference models of the s1 mechanism (miner protocol ``reax-s1/1``): only these may appear in a miner request.
REFERENCE_MAX_OPTIONS = {"s1-fast": 16, "s1-pro": 64, "s1-lite": 16, "s1-cpu": 16}
# Pools of engine competition v2 (engcomposite.py): the decisionmodels.io models that have a contract of their own.
#   quyet-1-large  Quyet-1.0-Large (Gemma-4-31B fine-tune): at most 10 options (node-letters configs/quyet.json max_options).
#   jeffan-31b     Jeffan-31b-1710 (Gemma-4-31B): one question, 2..26 options = the public decisionmodels.io limit (gateway
#                  PR #103 model_versions.MAX_OPTIONS[JEFFAN] = 26, board #22 #26863; the engine itself takes 255), no images.
#                  Prompt: the shared 8192-byte instructions / 16 KiB state limits below; 8192 input tokens per question.
#   s1-vision      Blink v0.3 with one image per request: 26 options (the Blink engine bound of s1-vision-2).
ENGINE_MAX_OPTIONS = {"quyet-1-large": 10, "jeffan-31b": 26, "s1-vision": 26}
MAX_OPTIONS = {**REFERENCE_MAX_OPTIONS, **ENGINE_MAX_OPTIONS}
MODELS = tuple(REFERENCE_MAX_OPTIONS)  # NOT the engine pools: the miner protocol never takes a vision or Quyet request
# Only this pool takes the optional ``images`` field (production: s1-vision, exactly one image per request).
IMAGE_POOLS = frozenset({"s1-vision"})
# The gateway's published image limits (services/gateway images.py): 4 MiB decoded, 2,000,000 pixels. The validator
# only sends PNG/JPEG (the gateway also accepts WebP).
MAX_IMAGE_BYTES = 4 * 1024 * 1024
MAX_IMAGE_PIXELS = 2_000_000
MIN_IMAGE_SIDE = 8
# The reference engine checks sum(p) within 1e-3; the validator is stricter on what it accepts back.
PROB_SUM_TOLERANCE = 1e-3
DERIVED_TOLERANCE = 1e-6


class ContractError(ValueError):
    """The decision or answer violates the REAX decision contract."""


def _text(value: Any, name: str, limit: int = 8192) -> str:
    if not isinstance(value, str) or not value or len(value.encode("utf-8")) > limit:
        raise ContractError(f"{name} must be nonempty text within {limit} bytes")
    return value


def _json_data(value: Any, name: str, limit: int, *, null_ok: bool = False) -> Any:
    if not isinstance(value, (str, dict, list)) and not (null_ok and value is None):
        raise ContractError(f"{name} must be text, object, or array")
    try:
        size = len(json.dumps(value, ensure_ascii=False, allow_nan=False).encode("utf-8"))
    except (TypeError, ValueError) as exc:
        raise ContractError(f"{name} must contain finite JSON values") from exc
    if size > limit:
        raise ContractError(f"{name} exceeds {limit} bytes")
    return value


@dataclass(frozen=True)
class Question:
    ident: str
    kind: str
    instructions: Any
    criteria: Any
    keys: tuple[str, ...]
    descriptions: tuple[Any, ...]

    def author_dict(self) -> dict[str, Any]:
        """What the reference engine sends to the model runtime (the reference engine's Question.author_dict)."""
        result = {"type": self.kind, "instructions": self.instructions}
        if self.kind != "noul" or self.criteria:
            result["criteria"] = self.criteria
        return result


@dataclass(frozen=True)
class Decision:
    ident: str
    state: Any
    question: Question
    images: tuple[str, ...] = ()  # s1-vision only: exactly one ``data:image/png;base64,...`` URL (production field ``images``)

    def body(self) -> dict[str, Any]:
        """The decision as sent on the wire (``questions`` map with one entry)."""
        q: dict[str, Any] = {"type": self.question.kind, "instructions": self.question.instructions}
        if self.question.criteria is not None and not (self.question.kind == "noul" and self.question.criteria == {}):
            q["criteria"] = self.question.criteria
        out: dict[str, Any] = {"id": self.ident, "state": self.state, "questions": {self.question.ident: q}}
        if self.images:
            out["images"] = list(self.images)
        return out


def parse_question(ident: str, value: Any, model: str) -> Question:
    _text(ident, "question id", 128)
    if not isinstance(value, dict):
        raise ContractError("question must be an object")
    if set(value) - {"type", "instructions", "criteria"}:
        raise ContractError("question has unknown fields")
    kind = value.get("type")
    if not isinstance(kind, str) or kind not in {"noul", "choice", "score"}:
        raise ContractError("question has invalid type")
    instruction = _json_data(value.get("instructions", ""), "instructions", 8192, null_ok=True)
    criteria = value.get("criteria")
    if kind == "noul":
        if criteria is None:
            criteria = {}
        if not isinstance(criteria, dict) or set(criteria) - {"true", "false"}:
            raise ContractError("question has invalid noul criteria")
        keys: tuple[str, ...] = ("true", "false")
        descriptions: tuple[Any, ...] = (criteria.get("true"), criteria.get("false"))
    elif kind == "choice":
        if not isinstance(criteria, dict):
            raise ContractError("question requires choice criteria")
        keys = tuple(criteria)
        descriptions = tuple(criteria.values())
    else:
        if not isinstance(criteria, list):
            raise ContractError("question requires score criteria")
        keys = tuple(str(i) for i in range(len(criteria)))
        descriptions = tuple(criteria)
    if not 2 <= len(keys) <= MAX_OPTIONS[model]:
        raise ContractError("question has invalid option count")
    if any(not isinstance(k, str) or not k or len(k.encode()) > 256 for k in keys):
        raise ContractError("question has invalid option keys")
    for desc in descriptions:
        if model != "s1-fast" and desc is not None and not isinstance(desc, (str, dict, list)):
            raise ContractError("this model requires text, object, or array option descriptions")
        try:
            encoded = json.dumps(desc, ensure_ascii=False, allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise ContractError("option description must be finite JSON") from exc
        if len(encoded.encode("utf-8")) > 4096:
            raise ContractError("option description exceeds 4096 bytes")
        if kind == "score" and desc is not None and not isinstance(desc, (str, dict, list)):
            raise ContractError("score level must be text, object, or array")
    return Question(ident, kind, instruction, criteria, keys, descriptions)


_DATA_URL = re.compile(r"data:image/(png|jpeg);base64,([A-Za-z0-9+/]+={0,2})")
_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
_MAX_DATA_URL_CHARS = len("data:image/jpeg;base64,") + 4 * ((MAX_IMAGE_BYTES + 2) // 3)


@dataclass(frozen=True)
class ImageInfo:
    mime: str  # image/png | image/jpeg
    width: int
    height: int
    size: int  # decoded bytes


def _png_size(data: bytes) -> tuple[int, int]:
    """Width and height of a PNG after walking every chunk (length, CRC, IHDR first, IDAT present, IEND last)."""
    if not data.startswith(_PNG_SIGNATURE):
        raise ContractError("image is not a PNG")
    pos, size, seen_idat = len(_PNG_SIGNATURE), None, False
    while True:
        if pos + 12 > len(data):
            raise ContractError("PNG is truncated")
        length, kind = struct.unpack(">I4s", data[pos:pos + 8])
        end = pos + 12 + length
        if end > len(data):
            raise ContractError("PNG is truncated")
        if zlib.crc32(data[pos + 4:pos + 8 + length]) != struct.unpack(">I", data[end - 4:end])[0]:
            raise ContractError("PNG chunk checksum is wrong")
        if size is None:
            if kind != b"IHDR" or length != 13:
                raise ContractError("PNG does not start with IHDR")
            width, height, depth, colour, _comp, _filt, interlace = struct.unpack(">IIBBBBB", data[pos + 8:pos + 21])
            if colour not in (0, 2, 3, 4, 6) or depth not in (1, 2, 4, 8, 16) or interlace not in (0, 1):
                raise ContractError("PNG header is invalid")
            size = (width, height)
        elif kind == b"IDAT":
            seen_idat = True
        elif kind == b"IEND":
            if length or end != len(data) or not seen_idat:
                raise ContractError("PNG has no image data or trailing bytes")
            return size
        pos = end


def _jpeg_size(data: bytes) -> tuple[int, int]:
    """Width and height from the start-of-frame marker of a JPEG (ends with EOI)."""
    if data[:3] != b"\xff\xd8\xff" or data[-2:] != b"\xff\xd9":
        raise ContractError("image is not a JPEG")
    pos = 2
    while pos + 4 <= len(data):
        if data[pos] != 0xFF:
            raise ContractError("JPEG marker is invalid")
        marker = data[pos + 1]
        if marker == 0xFF:  # fill byte
            pos += 1
            continue
        if marker in (0x01, *range(0xD0, 0xD8)):  # markers without a length
            pos += 2
            continue
        length = struct.unpack(">H", data[pos + 2:pos + 4])[0]
        if length < 2 or pos + 2 + length > len(data):
            raise ContractError("JPEG is truncated")
        if 0xC0 <= marker <= 0xCF and marker not in (0xC4, 0xC8, 0xCC):
            if length < 8:
                raise ContractError("JPEG frame header is invalid")
            height, width = struct.unpack(">HH", data[pos + 5:pos + 9])
            return width, height
        if marker == 0xDA:  # scan data before any frame header
            break
        pos += 2 + length
    raise ContractError("JPEG has no frame header")


def parse_image(url: Any) -> ImageInfo:
    """Validate one image data URL as the s1-vision gateway does: a ``data:image/png|jpeg;base64,`` URL, at most 4 MiB
    decoded, valid base64, real PNG/JPEG structure whose type matches the URL, a decoded size of at most 2,000,000 pixels
    and at least 8 x 8. Raises ContractError."""
    if not isinstance(url, str) or len(url) > _MAX_DATA_URL_CHARS:
        raise ContractError("image must be a data URL within the size limit")
    m = _DATA_URL.fullmatch(url)
    if m is None:
        raise ContractError("image must be a data:image/png or data:image/jpeg base64 URL")
    try:
        data = base64.b64decode(m.group(2), validate=True)
    except (binascii.Error, ValueError):
        raise ContractError("image is not valid base64") from None
    if not data or len(data) > MAX_IMAGE_BYTES:
        raise ContractError(f"image must be 1..{MAX_IMAGE_BYTES} bytes decoded")
    kind = m.group(1)
    width, height = _png_size(data) if kind == "png" else _jpeg_size(data)
    if width < MIN_IMAGE_SIDE or height < MIN_IMAGE_SIDE or width * height > MAX_IMAGE_PIXELS:
        raise ContractError(f"image must be at least {MIN_IMAGE_SIDE} x {MIN_IMAGE_SIDE} and at most "
                            f"{MAX_IMAGE_PIXELS} pixels")
    return ImageInfo(f"image/{kind}", width, height, len(data))


def parse_decision(obj: Any, model: str) -> Decision:
    """One decision: ``{"id", "state", "questions": {<one question>}}``; on the s1-vision pool also an optional
    ``"images"``: a list of exactly one image data URL (every other pool is text only)."""
    if model not in MAX_OPTIONS:
        raise ContractError("unknown model")
    if not isinstance(obj, dict) or not {"id", "state", "questions"} <= set(obj) <= {"id", "state", "questions", "images"}:
        raise ContractError("decision must have exactly the fields id, state, questions")
    images: tuple[str, ...] = ()
    if "images" in obj:
        if model not in IMAGE_POOLS:
            raise ContractError("this model takes text only")
        raw_images = obj["images"]
        if not isinstance(raw_images, list) or len(raw_images) != 1:
            raise ContractError("exactly one image is accepted")
        parse_image(raw_images[0])
        images = (raw_images[0],)
    state = _json_data(obj["state"], "state", MAX_STATE_BYTES)
    raw = obj["questions"]
    if not isinstance(raw, dict) or len(raw) != 1:
        raise ContractError("a decision carries exactly one question (reference contract limit)")
    (ident, value), = raw.items()
    return Decision(str(obj["id"]), state, parse_question(ident, value, model), images)


def normalize_answer(question: Question, answer: dict[str, Any]) -> dict[str, Any]:
    """The reference answer normalisation (the reference engine's normalize_answer), unchanged."""
    if not isinstance(answer, dict):
        raise RuntimeError("backend returned an invalid answer")
    if question.kind == "noul":
        p = float(answer["noul"])
        probs = {"true": p, "false": 1.0 - p}
    else:
        given = answer["probabilities"]
        probs = {key: float(given[key]) for key in question.keys}
    if any(not math.isfinite(p) or p < 0 or p > 1 for p in probs.values()) or abs(sum(probs.values()) - 1) > 1e-3:
        raise RuntimeError("backend returned invalid probabilities")
    best = max(range(len(question.keys)), key=lambda i: probs[question.keys[i]])
    out: dict[str, Any] = {"type": question.kind, "probabilities": probs}
    if question.kind == "noul":
        out["noul"] = probs["true"]
    elif question.kind == "choice":
        out["choice"] = question.keys[best]
        out["confidence"] = probs[question.keys[best]]
    else:
        expected = sum(i * probs[str(i)] for i in range(len(probs)))
        score = float(answer.get("score", expected))
        if not math.isfinite(score) or abs(score - expected) > 1e-3:
            raise RuntimeError("backend returned an invalid score")
        confidence = float(answer.get("confidence", max(probs.values())))
        if not math.isfinite(confidence) or not 0 <= confidence <= 1:
            raise RuntimeError("backend returned an invalid confidence")
        out.update(score=score, confidence=confidence, legend=dict(zip(question.keys, question.descriptions)))
    return out


def _num(value: Any, what: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ContractError(f"{what} must be a number")
    try:
        f = float(value)
    except OverflowError:  # a JSON integer too large for a float
        raise ContractError(f"{what} must be finite") from None
    if not math.isfinite(f):
        raise ContractError(f"{what} must be finite")
    return f


def check_answer(question: Question, answer: Any) -> tuple[float, ...]:
    """Validate a miner's answer object; return its probabilities in ``question.keys`` order.

    The fields must be exactly those ``normalize_answer`` produces and every derived
    field must follow from the probabilities.
    """
    expected_fields = {
        "noul": {"type", "probabilities", "noul"},
        "choice": {"type", "probabilities", "choice", "confidence"},
        "score": {"type", "probabilities", "score", "confidence", "legend"},
    }[question.kind]
    if not isinstance(answer, dict) or set(answer) != expected_fields:
        raise ContractError(f"answer must have exactly the fields {sorted(expected_fields)}")
    if answer["type"] != question.kind:
        raise ContractError("answer type does not match the question")
    probs = answer["probabilities"]
    if not isinstance(probs, dict) or set(probs) != set(question.keys):
        raise ContractError("probabilities must cover exactly the option keys")
    values = tuple(_num(probs[k], "probability") for k in question.keys)
    if any(v < 0.0 or v > 1.0 for v in values):
        raise ContractError("probabilities must be within [0, 1]")
    if abs(math.fsum(values) - 1.0) > PROB_SUM_TOLERANCE:
        raise ContractError("probabilities must sum to 1")
    by_key = dict(zip(question.keys, values))
    if question.kind == "noul":
        if abs(_num(answer["noul"], "noul") - by_key["true"]) > DERIVED_TOLERANCE:
            raise ContractError("noul must equal p(true)")
        if abs(by_key["true"] + by_key["false"] - 1.0) > DERIVED_TOLERANCE:
            raise ContractError("noul probabilities must be complementary")
    elif question.kind == "choice":
        best = max(range(len(question.keys)), key=lambda i: values[i])
        if answer["choice"] != question.keys[best]:
            raise ContractError("choice must be the most probable option")
        if abs(_num(answer["confidence"], "confidence") - values[best]) > DERIVED_TOLERANCE:
            raise ContractError("confidence must be the top probability")
    else:
        expected = math.fsum(i * values[i] for i in range(len(values)))
        if abs(_num(answer["score"], "score") - expected) > 1e-3:
            raise ContractError("score must be the expected level")
        # Both reference backends report the top probability as the score confidence.
        if abs(_num(answer["confidence"], "confidence") - max(values)) > DERIVED_TOLERANCE:
            raise ContractError("confidence must be the top probability")
        if answer["legend"] != dict(zip(question.keys, question.descriptions)):
            raise ContractError("legend must repeat the score levels")
    return values
