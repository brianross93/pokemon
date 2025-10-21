"""
Knowledge rules, Bayesian posteriors, and evidence tracking for symbolic recurrence.

The utilities here are intended to support middleware that amortises repeated
queries by storing lightweight symbolic summaries. They are general enough to
back the Pokemon exploration concept as well as other environments that expose
structured observations.
"""

from __future__ import annotations

import math
import random
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from statistics import NormalDist
from typing import Any, Deque, Dict, Iterable, List, Mapping, Optional, Tuple


# ---------------------------------------------------------------------------
# Encounter modelling (specialised helper used by the game middleware).
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Context:
    """Hashable game context describing where encounters happen."""

    game: str
    area_id: int
    method: str
    time_bucket: Optional[str] = None


@dataclass
class Posterior:
    """Coupled Beta distributions tracking encounter rate and species mix."""

    alpha_r: float = 1.0
    beta_r: float = 1.0
    alpha_p: float = 1.0
    beta_p: float = 1.0
    steps: int = 0
    encounters: int = 0
    enc_total: int = 0
    enc_target: int = 0

    def update_step(self, encountered: bool) -> None:
        self.steps += 1
        self.encounters += int(encountered)
        if encountered:
            self.alpha_r += 1.0
        else:
            self.beta_r += 1.0

    def update_species(self, is_target: bool) -> None:
        self.enc_total += 1
        self.enc_target += int(is_target)
        if is_target:
            self.alpha_p += 1.0
        else:
            self.beta_p += 1.0

    def e_rate(self) -> float:
        return self.alpha_r / max(1e-9, self.alpha_r + self.beta_r)

    def e_share(self) -> float:
        return self.alpha_p / max(1e-9, self.alpha_p + self.beta_p)

    def expected_steps_to_target(self) -> float:
        rate = max(1e-9, self.e_rate())
        share = max(1e-9, self.e_share())
        return 1.0 / (rate * share)

    def decay(self, lam: float = 0.01) -> None:
        """Exponential forgetting to adapt to non-stationarity."""
        base = 1.0
        self.alpha_r = (1.0 - lam) * self.alpha_r + lam * base
        self.beta_r = (1.0 - lam) * self.beta_r + lam * base
        self.alpha_p = (1.0 - lam) * self.alpha_p + lam * base
        self.beta_p = (1.0 - lam) * self.beta_p + lam * base


class EncounterKB:
    """Bayesian encounter knowledge base using Beta-Bernoulli components."""

    def __init__(self, decay: float = 0.0) -> None:
        self._rate_posteriors: Dict[Context, Posterior] = defaultdict(Posterior)
        self._species_posteriors: Dict[Tuple[Context, int], Posterior] = defaultdict(Posterior)
        self.decay = float(decay)

    def _decay_if_needed(self, posterior: Posterior) -> None:
        if self.decay > 0.0:
            posterior.decay(self.decay)

    def step(self, ctx: Context, encountered: bool) -> None:
        posterior = self._rate_posteriors[ctx]
        self._decay_if_needed(posterior)
        posterior.update_step(encountered)

    def encounter(self, ctx: Context, species_id: int, target_id: int) -> None:
        rate_posterior = self._rate_posteriors[ctx]
        self._decay_if_needed(rate_posterior)
        rate_posterior.update_step(True)

        spec_key = (ctx, target_id)
        species_post = self._species_posteriors[spec_key]
        self._decay_if_needed(species_post)
        species_post.update_species(species_id == target_id)

    def estimate_ttf(self, ctx: Context, target_id: int) -> float:
        rate = self._rate_posteriors[ctx].e_rate()
        share = self._species_posteriors[(ctx, target_id)].e_share()
        return 1.0 / max(1e-9, rate * share)

    def thompson_pick_context(self, candidates: Iterable[Context], target_id: int) -> Optional[Context]:
        best_ctx: Optional[Context] = None
        best_score = float("inf")
        for ctx in candidates:
            rate_post = self._rate_posteriors[ctx]
            species_post = self._species_posteriors[(ctx, target_id)]
            rate_sample = random.betavariate(rate_post.alpha_r, rate_post.beta_r)
            share_sample = random.betavariate(species_post.alpha_p, species_post.beta_p)
            score = 1.0 / max(1e-9, rate_sample * share_sample)
            if score < best_score:
                best_ctx, best_score = ctx, score
        return best_ctx

    def summary(self, ctx: Context, target_id: int) -> Dict[str, float]:
        rate_post = self._rate_posteriors[ctx]
        spec_post = self._species_posteriors[(ctx, target_id)]
        return {
            "mean_rate": rate_post.e_rate(),
            "mean_share": spec_post.e_share(),
            "expected_steps": self.estimate_ttf(ctx, target_id),
            "steps_observed": float(rate_post.steps),
            "encounters_observed": float(rate_post.encounters),
            "target_encounters": float(spec_post.enc_target),
        }


# ---------------------------------------------------------------------------
# General knowledge rules with evidence and refinement utilities.
# ---------------------------------------------------------------------------


@dataclass
class BetaConfidence:
    """Beta posterior that tracks successes vs failures with optional weighting."""

    alpha: float = 1.0
    beta: float = 1.0

    def update(self, success: bool, weight: float = 1.0) -> None:
        if success:
            self.alpha += weight
        else:
            self.beta += weight

    def decay(self, lam: float) -> None:
        base = 1.0
        self.alpha = (1.0 - lam) * self.alpha + lam * base
        self.beta = (1.0 - lam) * self.beta + lam * base

    @property
    def mean(self) -> float:
        return self.alpha / max(1e-9, self.alpha + self.beta)

    @property
    def variance(self) -> float:
        denom = (self.alpha + self.beta) ** 2 * (self.alpha + self.beta + 1.0)
        if denom <= 0.0:
            return 0.0
        return self.alpha * self.beta / denom

    def confidence_interval(self, level: float = 0.95) -> Tuple[float, float]:
        z = NormalDist().inv_cdf(0.5 + level / 2.0)
        mean = self.mean
        var = self.variance
        delta = z * math.sqrt(max(var, 0.0))
        return max(0.0, mean - delta), min(1.0, mean + delta)


@dataclass
class Evidence:
    """Atomic observation supporting or contradicting a knowledge rule."""

    timestamp: float
    context: Dict[str, Any]
    outcome: bool
    time_spent: float
    quality: float
    metadata: Dict[str, Any] = field(default_factory=dict)


def _normalise_context(context: Mapping[str, Any]) -> Dict[str, Any]:
    return dict(sorted(context.items(), key=lambda item: item[0]))


@dataclass
class KnowledgeRule:
    """Learned statement about an entity under certain conditions."""

    entity: str
    condition: Dict[str, Any]
    outcome: str
    belief: BetaConfidence = field(default_factory=BetaConfidence)
    evidence_count: int = 0
    positive_evidence: Deque[Evidence] = field(default_factory=lambda: deque(maxlen=256))
    negative_evidence: Deque[Evidence] = field(default_factory=lambda: deque(maxlen=256))
    last_updated: float = field(default_factory=time.time)
    embedding: Optional[Any] = None
    total_time_spent: float = 0.0
    total_quality: float = 0.0
    active: bool = True

    @property
    def confidence(self) -> float:
        return self.belief.mean

    def record_evidence(self, evidence: Evidence) -> None:
        self.evidence_count += 1
        self.total_time_spent += evidence.time_spent
        self.total_quality += evidence.quality
        self.last_updated = evidence.timestamp
        weight = max(1e-6, evidence.quality)
        self.belief.update(evidence.outcome, weight=weight)
        if evidence.outcome:
            self.positive_evidence.append(evidence)
        else:
            self.negative_evidence.append(evidence)

    @property
    def average_time_spent(self) -> float:
        if self.evidence_count == 0:
            return 0.0
        return self.total_time_spent / self.evidence_count

    @property
    def average_quality(self) -> float:
        if self.evidence_count == 0:
            return 0.0
        return self.total_quality / self.evidence_count


@dataclass
class RefinementResult:
    """Details about an attempted rule refinement."""

    parent_rule_id: str
    feature: str
    info_gain: float
    child_rule_ids: List[str]


class KnowledgeGraph:
    """Manage knowledge rules, evidence, and refinement heuristics."""

    def __init__(self, decay: float = 0.0) -> None:
        self.rules: Dict[str, KnowledgeRule] = {}
        self.relationships: Dict[Tuple[str, str], float] = {}
        self.context_embeddings: Dict[str, Any] = {}
        self.decay = float(decay)
        self._rule_index: Dict[Tuple[str, Tuple[Tuple[str, Any], ...]], str] = {}

    def _context_key(self, context: Mapping[str, Any]) -> Tuple[Tuple[str, Any], ...]:
        return tuple(sorted(context.items(), key=lambda item: item[0]))

    def _rule_id(self, entity: str, context_key: Tuple[Tuple[str, Any], ...], outcome: str) -> str:
        slug = "_".join(f"{k}={v}" for k, v in context_key)
        return f"{entity}:{outcome}:{slug}"

    def get_rule(self, entity: str, context: Mapping[str, Any], outcome: str) -> Optional[KnowledgeRule]:
        ctx_key = self._context_key(context)
        rule_key = (entity, ctx_key)
        rule_id = self._rule_index.get(rule_key)
        if rule_id is None:
            return None
        return self.rules.get(rule_id)

    def _ensure_rule(self, entity: str, context: Mapping[str, Any], outcome: str) -> Tuple[str, KnowledgeRule]:
        ctx = _normalise_context(context)
        ctx_key = self._context_key(ctx)
        rule_key = (entity, ctx_key)
        rule_id = self._rule_index.get(rule_key)
        if rule_id is None:
            rule_id = self._rule_id(entity, ctx_key, outcome)
            rule = KnowledgeRule(entity=entity, condition=ctx, outcome=outcome)
            self.rules[rule_id] = rule
            self._rule_index[rule_key] = rule_id
        else:
            rule = self.rules[rule_id]
        return rule_id, rule

    def add_evidence(
        self,
        entity: str,
        context: Mapping[str, Any],
        outcome: str,
        success: bool,
        time_spent: float,
        quality: float,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> str:
        rule_id, rule = self._ensure_rule(entity, context, outcome)
        if self.decay > 0.0:
            rule.belief.decay(self.decay)
        evidence = Evidence(
            timestamp=time.time(),
            context=dict(context),
            outcome=success,
            time_spent=float(time_spent),
            quality=float(quality),
            metadata=dict(metadata or {}),
        )
        rule.record_evidence(evidence)
        return rule_id

    def query_knowledge(self, entity: str, context: Mapping[str, Any]) -> List[KnowledgeRule]:
        ctx = _normalise_context(context)
        ctx_key = self._context_key(ctx)
        matches: List[KnowledgeRule] = []
        for (ent, key), rule_id in self._rule_index.items():
            if ent != entity:
                continue
            if all(item in ctx_key for item in key):
                rule = self.rules[rule_id]
                if rule.active:
                    matches.append(rule)
        return sorted(matches, key=lambda r: r.confidence, reverse=True)

    def get_best_action(self, goal: str, current_context: Mapping[str, Any]) -> Optional[KnowledgeRule]:
        candidates = self.query_knowledge(goal, current_context)
        if not candidates:
            return None
        return candidates[0]

    def refine_rule(
        self,
        rule_id: str,
        min_samples: int = 20,
        min_info_gain: float = 0.05,
        retire_parent: bool = True,
    ) -> Optional[RefinementResult]:
        rule = self.rules.get(rule_id)
        if rule is None or not rule.active:
            return None

        evidences = list(rule.positive_evidence) + list(rule.negative_evidence)
        if len(evidences) < min_samples:
            return None

        parent_success = sum(ev.outcome for ev in evidences)
        parent_total = len(evidences)
        parent_prob = (parent_success + 1.0) / (parent_total + 2.0)
        parent_entropy = self._binary_entropy(parent_prob)

        feature_scores: Dict[str, Dict[Any, Tuple[int, int]]] = {}
        for ev in evidences:
            for feature, value in ev.context.items():
                stats = feature_scores.setdefault(feature, {})
                success, total = stats.setdefault(value, (0, 0))
                stats[value] = (success + int(ev.outcome), total + 1)

        best_feature = None
        best_gain = 0.0
        best_stats: Dict[Any, Tuple[int, int]] = {}
        for feature, stats in feature_scores.items():
            weighted_entropy = 0.0
            for value, (success, total) in stats.items():
                prob = (success + 1.0) / (total + 2.0)
                weighted_entropy += (total / parent_total) * self._binary_entropy(prob)
            info_gain = parent_entropy - weighted_entropy
            if info_gain > best_gain:
                best_gain = info_gain
                best_feature = feature
                best_stats = stats

        if best_feature is None or best_gain < min_info_gain:
            return None

        child_ids: List[str] = []
        for value, (success, total) in best_stats.items():
            child_context = dict(rule.condition)
            child_context[best_feature] = value
            child_id, child_rule = self._ensure_rule(rule.entity, child_context, rule.outcome)
            child_rule.active = True
            # bootstrap belief with evidence counts
            child_rule.belief.alpha = 1.0 + success
            child_rule.belief.beta = 1.0 + (total - success)
            child_rule.evidence_count = total
            child_rule.total_time_spent = rule.average_time_spent * total
            child_rule.total_quality = rule.average_quality * total
            child_ids.append(child_id)

        if retire_parent:
            rule.active = False

        return RefinementResult(parent_rule_id=rule_id, feature=best_feature, info_gain=best_gain, child_rule_ids=child_ids)

    @staticmethod
    def _binary_entropy(p: float) -> float:
        if p <= 0.0 or p >= 1.0:
            return 0.0
        return -p * math.log(p) - (1.0 - p) * math.log(1.0 - p)
