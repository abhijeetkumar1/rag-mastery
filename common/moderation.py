"""Input moderation (Phase 7): is the user's text harmful (harassment, hate, self-harm, violence, sexual, illicit)?

This is a different question from Phase 3's router (in scope? investment advice?) and Phase 5's injection detector
(does a passage instruct the AI?). Options, cheapest first:
  OpenAI moderation API    free, ~100-300 ms, 13 categories (omni-moderation-latest). Needs the API to be up.
  Llama Guard 3 (1B / 8B)  open-weights classifier for prompts AND responses with a configurable taxonomy; runs
                           locally on a GPU (the 1B on CPU, slowly). No data leaves your infrastructure.
  NeMo Guardrails          a framework (Colang flows) that calls such classifiers and LLM self-checks as "rails".
Policy when the moderation call FAILS is a product decision, made in advance:
  fail open   serve the request, log moderation_unavailable. Right for a low-harm domain like this one: the router,
              the corpus restriction and the output checks still apply.
  fail closed refuse. Right where one harmful answer costs more than an outage (health, minors, self-harm).
"""
from common.resilience import BREAKERS

MODEL = "omni-moderation-latest"


def _call(text: str) -> dict:
    from common.llm import client
    r = client().moderations.create(model=MODEL, input=text).results[0]
    return {"flagged": r.flagged, "categories": sorted(k for k, v in r.categories.model_dump().items() if v),
            "source": MODEL}


def moderate(text: str, fail_open: bool = True) -> dict:
    """{flagged, categories, source}. source = "unavailable" when the API failed or the breaker is open."""
    def fallback(_):
        return {"flagged": not fail_open, "categories": [], "source": "unavailable"}
    out, _ = BREAKERS["moderation"].call(_call, fallback, text)
    return out
