"""Fresh typed-decision items for ``reax-s1/1``.

Each item is a typed decision (the s1 request body) without the model field: a
*state* (text, object or array) and exactly one question of type ``choice``,
``noul`` or ``score``. As in rc3 there is no answer key: the validator scores a
miner against its own run of the pinned model, so nothing can leak.

The states are short, realistic records (client messages, reviews,
profiles, schedules, incident notes) assembled from word pools, and the
questions ask for judgement on them. Such items give spread-out, model-specific
distributions; a seeded generator with 128-bit seeds makes every round new.
"""

from __future__ import annotations

import random
import secrets
from dataclasses import dataclass
from typing import Any, Callable

MAX_CHOICE_OPTIONS = 8


@dataclass(frozen=True)
class S1Item:
    item_id: str
    state: Any
    kind: str  # choice | noul | score
    instructions: Any
    criteria: Any  # dict (choice / noul or None) or list (score)
    images: tuple[str, ...] = ()  # s1-vision items: exactly one image data URL (s1vision.py); text items: none

    def question(self) -> dict[str, Any]:
        q: dict[str, Any] = {"type": self.kind, "instructions": self.instructions}
        if self.criteria is not None:
            q["criteria"] = self.criteria
        return q

    def decision(self) -> dict[str, Any]:
        out = {"id": self.item_id, "state": self.state, "questions": {"q": self.question()}}
        if self.images:
            out["images"] = list(self.images)
        return out

    def keys(self) -> tuple[str, ...]:
        if self.kind == "noul":
            return ("true", "false")
        if self.kind == "choice":
            return tuple(self.criteria)
        return tuple(str(i) for i in range(len(self.criteria)))


_FIRST = (
    "Ada", "Bruno", "Chiara", "Dmitri", "Elif", "Farid", "Greta", "Hiro", "Ines", "Jonas", "Kofi", "Lena", "Mateo",
    "Nadia", "Oskar", "Priya", "Quentin", "Rosa", "Sven", "Tariq", "Uma", "Viktor", "Wen", "Ximena", "Yusuf", "Zofia",
)
_PRODUCTS = (
    "wireless headphones", "standing desk", "espresso machine", "running shoes", "rain jacket", "e-reader",
    "air purifier", "electric toothbrush", "camping tent", "blender", "office chair", "smart watch", "backpack",
    "robot vacuum", "winter boots", "desk lamp", "yoga mat", "kettle", "phone case", "bike helmet",
)
_ISSUES = (
    "was charged twice this month", "has not received the parcel after two weeks", "cannot log in since the update",
    "got the wrong size", "wants to change the delivery address", "found a scratch on the item",
    "asks whether the warranty covers water damage", "wants to cancel the subscription",
    "says the app crashes when uploading photos", "needs an invoice with the company name",
    "asks if the item ships to Norway", "reports that the discount code did not apply",
    "wants to know when the item is back in stock", "says the battery drains overnight",
)
_TONES = (
    "", "This is the third time I am writing. ", "Thanks in advance! ", "I am really disappointed. ",
    "No rush, just curious. ", "Please fix this today. ", "Honestly, not a big deal. ", "I love your shop, but ",
    "My lawyer will hear about this. ", "Quick question: ",
)
_TEAMS = {
    "billing": "Payments, refunds, invoices and charges",
    "shipping": "Deliveries, tracking and addresses",
    "technical": "Apps, logins, crashes and devices",
    "returns": "Returns, exchanges and damaged items",
    "sales": "Product questions, stock and availability",
    "account": "Subscriptions, account settings and cancellations",
    "legal": "Complaints with legal or regulatory threats",
    "general": "Anything else",
}
_ASPECTS = ("price", "durability", "comfort", "design", "noise", "battery life", "ease of use", "customer service",
            "size", "weight", "delivery speed", "packaging")
_PRAISE = ("works great", "feels solid", "looks beautiful", "was cheap", "arrived quickly", "is very quiet",
           "is easy to set up", "is comfortable all day")
_COMPLAINTS = ("broke after a month", "is louder than expected", "is a bit heavy", "smells of plastic",
               "came without a manual", "is overpriced", "has a confusing app", "runs small")
_HOBBIES = ("hiking", "baking", "chess", "photography", "gardening", "jazz", "climbing", "board games", "painting",
            "cycling", "birdwatching", "knitting", "running", "pottery", "astronomy", "sailing")
_CITIES = ("Lisbon", "Oslo", "Kyoto", "Nairobi", "Lima", "Reykjavik", "Hanoi", "Vienna", "Montreal", "Tbilisi",
           "Cape Town", "Krakow", "Seville", "Auckland", "Edinburgh", "Passau")
_JOBS = ("nurse", "teacher", "pilot", "architect", "student", "retired engineer", "chef", "lawyer", "farmer",
         "software developer", "translator", "firefighter")
_GIFTS = ("a cookbook", "a board game", "concert tickets", "a plant", "a scarf", "a camera strap", "a puzzle",
          "a spa voucher", "a travel guide", "headphones", "a sketchbook", "a bottle of wine", "climbing chalk",
          "a telescope map", "a yoga class", "a fountain pen")
_EVENTS = ("team offsite", "product launch", "budget review", "client workshop", "hiring panel", "board meeting",
           "training day", "office move", "security audit", "customer visit")
_DAYS = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday")
_SYSTEMS = ("checkout", "login service", "search", "email delivery", "payment gateway", "mobile app", "reporting",
            "image upload", "notifications", "admin panel")
_SYMPTOMS = ("error rate rose to 3 %", "latency doubled", "a few users see blank pages", "one region is down",
             "requests time out after 30 s", "memory grows slowly", "a nightly job failed twice",
             "customers report duplicate emails")
_GENRES = ("mystery", "science fiction", "romance", "travel memoir", "history", "children's", "horror", "poetry")
_MOODS = {
    "calm": "Quiet, peaceful, relaxed",
    "tense": "Suspense or worry",
    "joyful": "Happy, celebratory",
    "sad": "Loss, grief, melancholy",
    "angry": "Conflict, irritation",
    "nostalgic": "Looking back fondly",
    "hopeful": "Looking forward with optimism",
}
_SCENES = (
    "The kettle clicked off and nobody moved to pour it.",
    "She folded the letter twice and put it back in the drawer.",
    "Rain hammered the tin roof while the radio played an old song.",
    "The train was late again, and the platform smelled of coffee.",
    "He laughed so hard the waiter dropped a spoon.",
    "The last box was taped shut; the flat suddenly felt enormous.",
    "Lanterns swayed above the square as the band tuned up.",
    "The phone buzzed at 3 a.m. with an unknown number.",
    "Snow covered the garden their grandmother had planted.",
    "The interview ended with a handshake that lasted a beat too long.",
)
_PLACES_SCENE = ("station", "harbour", "old house", "hospital", "bakery", "border crossing", "wedding hall", "library",
                 "mountain hut", "airport", "school gate", "market")
_URGENCY = ["Not urgent: can wait a week", "Low: handle within days", "Medium: handle today",
            "High: handle within the hour", "Critical: drop everything"]
_SENTIMENT = ["Very negative", "Negative", "Mixed or neutral", "Positive", "Very positive"]
_FIT = ["Poor fit", "Acceptable fit", "Good fit", "Excellent fit"]
_RISK = ["Low risk", "Moderate risk", "High risk"]


def _ticket(rng: random.Random) -> tuple[Any, str]:
    name = rng.choice(_FIRST)
    product = rng.choice(_PRODUCTS)
    text = f"{rng.choice(_TONES)}Note from support: {name} ({product} order) {rng.choice(_ISSUES)}."
    if rng.random() < 0.5:
        return {"channel": rng.choice(["email", "chat", "phone note"]), "customer": name, "message": text,
                "orders_last_year": rng.randint(0, 12)}, "ticket"
    return text, "ticket"


def _review(rng: random.Random) -> tuple[Any, str]:
    product = rng.choice(_PRODUCTS)
    parts = [f"The {product} {rng.choice(_PRAISE)}"]
    if rng.random() < 0.7:
        parts.append(f"but it {rng.choice(_COMPLAINTS)}")
    if rng.random() < 0.4:
        parts.append(f"and it {rng.choice(_PRAISE + _COMPLAINTS)}")
    text = " ".join(parts) + "."
    if rng.random() < 0.5:
        return {"product": product, "stars": rng.randint(1, 5), "review": text}, "review"
    return text, "review"


def _profile(rng: random.Random) -> tuple[Any, str]:
    return {
        "name": rng.choice(_FIRST),
        "age": rng.randint(9, 82),
        "job": rng.choice(_JOBS),
        "city": rng.choice(_CITIES),
        "hobbies": rng.sample(_HOBBIES, rng.randint(1, 3)),
        "budget_eur": rng.choice([10, 20, 35, 50, 80, 150]),
    }, "profile"


def _schedule(rng: random.Random) -> tuple[Any, str]:
    days = rng.sample(_DAYS, rng.randint(2, 4))
    return [
        {"day": d, "event": rng.choice(_EVENTS), "attendees": rng.randint(2, 40), "hours": rng.choice([1, 2, 3, 4, 8])}
        for d in days
    ], "schedule"


def _incident(rng: random.Random) -> tuple[Any, str]:
    return {
        "ticket": f"INC-{rng.randint(1000, 99999)}",
        "system": rng.choice(_SYSTEMS),
        "symptom": rng.choice(_SYMPTOMS),
        "started_minutes_ago": rng.randint(2, 900),
        "deploy_in_last_hour": rng.random() < 0.5,
        "customers_affected": rng.choice(["unknown", "a few", "about 1 %", "about 20 %", "all"]),
        "on_call": rng.choice(_FIRST),
    }, "incident"


def _scene(rng: random.Random) -> tuple[Any, str]:
    opening = f"{rng.choice(_FIRST)} reached the {rng.choice(_PLACES_SCENE)} at {rng.randint(0, 23)}:{rng.randint(0, 59):02d}."
    return " ".join([opening] + rng.sample(_SCENES, rng.randint(1, 2))), "scene"


_STATES: tuple[Callable[[random.Random], tuple[Any, str]], ...] = (_ticket, _review, _profile, _schedule, _incident, _scene)


def _subset(rng: random.Random, pool: dict[str, str], low: int = 2, high: int = 6) -> dict[str, Any]:
    keys = rng.sample(list(pool), rng.randint(low, min(high, len(pool), MAX_CHOICE_OPTIONS)))
    # Descriptions are sometimes omitted (the reference contract accepts null descriptions).
    return {k: (pool[k] if rng.random() < 0.8 else None) for k in keys}


def _levels(rng: random.Random, levels: list[str]) -> list[str]:
    n = rng.randint(3, len(levels))
    start = rng.randint(0, len(levels) - n)
    return levels[start : start + n]


def _question(rng: random.Random, kind_of_state: str) -> tuple[str, Any, Any]:
    kind = rng.choice(("choice", "choice", "noul", "score"))
    if kind_of_state == "ticket":
        if kind == "choice":
            return kind, "Which team should handle this message?", _subset(rng, _TEAMS)
        if kind == "noul":
            q = rng.choice(["Does the customer ask for money back?", "Is the customer likely to leave?",
                            "Does this need a human reply?", "Is the customer polite?"])
            return kind, q, (None if rng.random() < 0.5 else {"true": "Yes", "false": "No"})
        return kind, "How urgent is this message?", _levels(rng, _URGENCY)
    if kind_of_state == "review":
        if kind == "choice":
            return kind, "Which aspect matters most to the reviewer?", {
                a.replace(" ", "_"): a for a in rng.sample(_ASPECTS, rng.randint(2, 6))}
        if kind == "noul":
            return kind, rng.choice(["Would the reviewer buy this again?", "Is the reviewer recommending it?",
                                     "Does the review mention a defect?"]), None
        return kind, "What is the overall sentiment?", _levels(rng, _SENTIMENT)
    if kind_of_state == "profile":
        if kind == "choice":
            gifts = rng.sample(_GIFTS, rng.randint(2, 7))
            return kind, "Which gift should we suggest?", {f"g{i}": g for i, g in enumerate(gifts)}
        if kind == "noul":
            return kind, f"Would this person enjoy a {rng.choice(_HOBBIES)} weekend in {rng.choice(_CITIES)}?", None
        return kind, f"How well does a job as a {rng.choice(_JOBS)} suit this person?", _levels(rng, _FIT)
    if kind_of_state == "schedule":
        if kind == "choice":
            days = rng.sample(_DAYS, rng.randint(2, 5))
            return kind, "On which day should the extra review meeting go?", {d.lower(): d for d in days}
        if kind == "noul":
            return kind, "Is this week overloaded?", {"true": "Too much for one team", "false": "Manageable"}
        return kind, "How risky is this plan for the team's wellbeing?", _levels(rng, _RISK)
    if kind_of_state == "incident":
        if kind == "choice":
            acts = {"rollback": "Roll back the last deploy", "page": "Page the on-call engineer",
                    "watch": "Keep watching, no action yet", "status": "Post on the status page",
                    "scale": "Add capacity", "ticket": "Open a normal ticket"}
            return kind, "What is the best first action?", _subset(rng, acts, 2, 6)
        if kind == "noul":
            return kind, "Should customers be informed now?", None
        return kind, "How severe is this incident?", _levels(rng, _URGENCY)
    # scene
    if kind == "choice":
        return kind, "What is the mood of this passage?", _subset(rng, _MOODS, 2, 7)
    if kind == "noul":
        return kind, f"Could this be the opening of a {rng.choice(_GENRES)} book?", None
    return kind, "How much tension does the passage carry?", _levels(rng, ["None", "A little", "Some", "A lot", "Extreme"])


def generate_s1_items(n: int, seed: int | None = None) -> list[S1Item]:
    """Return ``n`` fresh typed items. ``seed`` is for tests and calibration only."""
    rng = random.Random(secrets.randbits(128) if seed is None else seed)
    items = []
    for _ in range(n):
        state, kind_of_state = rng.choice(_STATES)(rng)
        kind, instructions, criteria = _question(rng, kind_of_state)
        if kind == "choice":
            keys = list(criteria)
            rng.shuffle(keys)
            criteria = {k: criteria[k] for k in keys}
        items.append(S1Item("%032x" % rng.getrandbits(128), state, kind, instructions, criteria))
    return items


def items_for_pool(pool: str, n: int, seed: int | None = None) -> list[S1Item]:
    """Fresh items for a contract pool: image items for ``s1-vision`` (s1vision.py), the text items for every other pool."""
    from .s1contract import IMAGE_POOLS

    if pool in IMAGE_POOLS:
        from .s1vision import generate_vision_items

        return generate_vision_items(n, seed)
    return generate_s1_items(n, seed)
