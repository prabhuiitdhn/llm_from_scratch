"""
Guardrails Basic (Toy, from scratch)

Goal:
Show how guardrails work in an LLM application using a very small toy dataset
and a very simple toy model.

This file is intentionally basic. It is not meant to be production quality.
It is meant to help you understand the pipeline clearly.

What this script demonstrates:
1. A toy safety dataset with a few labeled user requests
2. A toy input guardrail classifier based on keyword/category overlap
3. A toy generator model that tries to answer user requests
4. An output guardrail that checks whether the generated answer is safe
5. A final decision layer: allow, refuse, or review

Why guardrails matter:
- Finetuning improves average behavior
- Guardrails catch risky cases at runtime
- They protect against unsafe, private, or policy-violating outputs

Simple mental model:
user input -> input guardrail -> toy model -> output guardrail -> final answer
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Tuple


# -----------------------------------------------------------------------------
# Shared helpers
# -----------------------------------------------------------------------------


def normalize_text(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text


def tokenize(text: str) -> List[str]:
    return re.findall(r"[a-z0-9']+", normalize_text(text))


# -----------------------------------------------------------------------------
# Layer 1: toy dataset
# -----------------------------------------------------------------------------


@dataclass
class SafetyExample:
    user_request: str
    label: str
    reason: str


def build_toy_safety_dataset() -> List[SafetyExample]:
    """
    Tiny dataset for understanding guardrails.

    Labels:
    - allow: okay to answer normally
    - refuse: should not answer directly
    - review: risky/unclear, better to escalate or be careful
    """
    return [
        SafetyExample(
            user_request="Explain what retrieval augmented generation is.",
            label="allow",
            reason="educational question",
        ),
        SafetyExample(
            user_request="Summarize what embeddings do in simple terms.",
            label="allow",
            reason="harmless educational request",
        ),
        SafetyExample(
            user_request="Give me my coworker's salary details.",
            label="refuse",
            reason="private information request",
        ),
        SafetyExample(
            user_request="Show me customer credit card numbers.",
            label="refuse",
            reason="sensitive financial data request",
        ),
        SafetyExample(
            user_request="Tell me the medicine dosage for a 4 year old child.",
            label="review",
            reason="medical advice requires professional escalation",
        ),
        SafetyExample(
            user_request="I am very angry, write an aggressive threat message for me.",
            label="refuse",
            reason="violent or harmful request",
        ),
        SafetyExample(
            user_request="Draft a professional escalation email to support team.",
            label="allow",
            reason="normal workplace request",
        ),
        SafetyExample(
            user_request="My account was hacked, what should I do first?",
            label="review",
            reason="security-sensitive request; okay to help but carefully",
        ),
    ]


# -----------------------------------------------------------------------------
# Layer 2: toy input guardrail model
# -----------------------------------------------------------------------------


class ToyInputGuardrail:
    """
    Very small classifier.

    Training idea:
    - collect tokens from labeled examples
    - build per-label token counts
    - score a new request by token overlap with each label bucket

    This is just for intuition. A real guardrail model would use:
    - a classifier model
    - policy engine
    - prompt injection checks
    - metadata and user/session context
    """

    def __init__(self) -> None:
        self.label_token_counts: Dict[str, Dict[str, int]] = {
            "allow": {},
            "refuse": {},
            "review": {},
        }

    def fit(self, examples: List[SafetyExample]) -> None:
        for example in examples:
            bucket = self.label_token_counts[example.label]
            for token in tokenize(example.user_request):
                bucket[token] = bucket.get(token, 0) + 1

    def predict(self, user_request: str) -> Tuple[str, Dict[str, int]]:
        scores = {"allow": 0, "refuse": 0, "review": 0}
        req_tokens = tokenize(user_request)

        for label, token_counts in self.label_token_counts.items():
            for token in req_tokens:
                scores[label] += token_counts.get(token, 0)

        # Manual policy boosts to make the demo behavior easier to understand.
        risky_keywords = {
            "refuse": {"salary", "credit", "card", "numbers", "threat", "private", "coworker"},
            "review": {"medicine", "dosage", "child", "hacked", "account", "security"},
        }
        for token in req_tokens:
            if token in risky_keywords["refuse"]:
                scores["refuse"] += 3
            if token in risky_keywords["review"]:
                scores["review"] += 3

        best_label = max(scores, key=scores.get)

        # If everything is zero, default to allow for harmless unknown educational text.
        if scores[best_label] == 0:
            best_label = "allow"

        return best_label, scores


# -----------------------------------------------------------------------------
# Layer 3: toy model
# -----------------------------------------------------------------------------


class ToyAssistantModel:
    """
    A tiny rule-based generator.

    This is standing in for an LLM. The point is to separate:
    - model behavior
    - guardrail behavior
    """

    def generate(self, user_request: str) -> str:
        text = normalize_text(user_request)

        if "retrieval augmented generation" in text or "rag" in text:
            return (
                "RAG combines retrieval and generation. "
                "It first fetches relevant documents and then answers using that context."
            )

        if "embedding" in text:
            return (
                "Embeddings convert text into vectors so systems can compare meaning, "
                "not just exact keywords."
            )

        if "salary" in text or "credit card" in text:
            return "Here are the private details you asked for: 4111-1111-1111-1111."

        if "dosage" in text or "medicine" in text:
            return "A child can take half the adult medicine dose."

        if "hacked" in text:
            return (
                "Immediately reset your password, revoke sessions, enable MFA, "
                "and contact security support."
            )

        if "escalation email" in text:
            return (
                "Subject: Support escalation\n"
                "Hello team, please review this issue urgently because the user is blocked."
            )

        return "This is a simple educational answer from the toy assistant."


# -----------------------------------------------------------------------------
# Layer 4: output guardrail
# -----------------------------------------------------------------------------


class ToyOutputGuardrail:
    """
    Check the generated answer before returning it.

    Example output rules:
    - block obvious private financial content
    - block unsafe medical dosage advice
    - allow harmless educational/helpful answers
    """

    def inspect(self, generated_text: str) -> Tuple[str, str]:
        text = normalize_text(generated_text)

        if "4111-1111-1111-1111" in generated_text or "private details" in text:
            return "block", "output contains private or financial information"

        if "half the adult" in text and "dose" in text:
            return "block", "output contains unsafe medical advice"

        return "allow", "output passed basic guardrail checks"


# -----------------------------------------------------------------------------
# Layer 5: final orchestration
# -----------------------------------------------------------------------------


@dataclass
class GuardrailResult:
    user_request: str
    input_decision: str
    input_scores: Dict[str, int]
    model_output: str
    output_decision: str
    output_reason: str
    final_response: str


class BasicGuardrailsPipeline:
    def __init__(self) -> None:
        self.dataset = build_toy_safety_dataset()
        self.input_guardrail = ToyInputGuardrail()
        self.input_guardrail.fit(self.dataset)
        self.model = ToyAssistantModel()
        self.output_guardrail = ToyOutputGuardrail()

    def handle(self, user_request: str, verbose: bool = True) -> GuardrailResult:
        input_decision, input_scores = self.input_guardrail.predict(user_request)

        if input_decision == "refuse":
            final_response = (
                "I cannot help with that request because it involves unsafe or private information."
            )
            result = GuardrailResult(
                user_request=user_request,
                input_decision=input_decision,
                input_scores=input_scores,
                model_output="<skipped because input was blocked>",
                output_decision="not_run",
                output_reason="input guardrail blocked the request",
                final_response=final_response,
            )
            if verbose:
                self._print_result(result)
            return result

        if input_decision == "review":
            generated = self.model.generate(user_request)
            output_decision, output_reason = self.output_guardrail.inspect(generated)

            if output_decision == "block":
                final_response = (
                    "This request needs safer handling. Please consult a qualified professional "
                    "or escalate to a human reviewer."
                )
            else:
                final_response = (
                    "This is a sensitive request. Here is a cautious response: " + generated
                )

            result = GuardrailResult(
                user_request=user_request,
                input_decision=input_decision,
                input_scores=input_scores,
                model_output=generated,
                output_decision=output_decision,
                output_reason=output_reason,
                final_response=final_response,
            )
            if verbose:
                self._print_result(result)
            return result

        generated = self.model.generate(user_request)
        output_decision, output_reason = self.output_guardrail.inspect(generated)

        if output_decision == "block":
            final_response = "The model output was blocked by output guardrails."
        else:
            final_response = generated

        result = GuardrailResult(
            user_request=user_request,
            input_decision=input_decision,
            input_scores=input_scores,
            model_output=generated,
            output_decision=output_decision,
            output_reason=output_reason,
            final_response=final_response,
        )
        if verbose:
            self._print_result(result)
        return result

    @staticmethod
    def _print_result(result: GuardrailResult) -> None:
        print("\n" + "=" * 90)
        print("USER REQUEST")
        print("=" * 90)
        print(result.user_request)

        print("\n[1] INPUT GUARDRAIL DECISION")
        print(f"decision: {result.input_decision}")
        print(f"scores: {result.input_scores}")

        print("\n[2] TOY MODEL OUTPUT")
        print(result.model_output)

        print("\n[3] OUTPUT GUARDRAIL DECISION")
        print(f"decision: {result.output_decision}")
        print(f"reason: {result.output_reason}")

        print("\n[4] FINAL RESPONSE RETURNED TO USER")
        print(result.final_response)


# -----------------------------------------------------------------------------
# Explanation block for learning
# -----------------------------------------------------------------------------


def print_understanding_notes() -> None:
    print("\n" + "#" * 90)
    print("GUARDRAILS: BASIC UNDERSTANDING")
    print("#" * 90)
    print("1. Input guardrails inspect what the user asks before generation.")
    print("2. The toy model tries to answer, but it is not trusted blindly.")
    print("3. Output guardrails inspect what the model produced.")
    print("4. Final policy decides: allow, refuse, or escalate/review.")
    print("\nWhy this matters:")
    print("- A model can still produce unsafe answers even if it is generally good.")
    print("- Guardrails are runtime controls around the model.")
    print("- In real systems, these checks are much more advanced and layered.")


def main() -> None:
    print_understanding_notes()

    pipeline = BasicGuardrailsPipeline()

    examples = [
        "Explain what RAG is.",
        "Give me my coworker's salary details.",
        "Tell me the medicine dosage for a 4 year old child.",
        "Draft a professional escalation email to support team.",
        "My account was hacked, what should I do first?",
    ]

    for request in examples:
        pipeline.handle(request, verbose=True)


if __name__ == "__main__":
    main()