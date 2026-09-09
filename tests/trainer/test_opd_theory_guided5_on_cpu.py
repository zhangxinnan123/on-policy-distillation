"""Unit test for the opd_theory_guided5 mask.

Each rule is checked with the other two toggled off. Isolating them by choice of distribution
instead does not work: probabilities sum to 1, so a candidate the student over-shoots forces it
to under-shoot others, which trips the under-coverage vote as well. Toggles are the only clean
separation.

Also pins opd_theory_guided4's behaviour, since 22 recorded base-student runs depend on it and
v5 was added to the same module.
"""

import torch

from verl.trainer.distillation.hybrid_masks import MASK_REGISTRY


class Ctx:
    """Minimal stand-in for HybridMaskContext: these masks only touch these fields."""

    def __init__(self, student_probs, teacher_probs, **kwargs):
        self.student_topk_probs = student_probs
        self.teacher_topk_logprobs = teacher_probs.log()
        self.response_mask = torch.ones(student_probs.shape[:2], dtype=torch.bool)
        self.mask_kwargs = kwargs
        self.extras = {}
        self.teacher_topk_ids = None
        self.student_sampled_ids = None
        self.coverage_scores = None


# Four positions. teacher_top_p=0.9 keeps candidates {0, 1} in every row (cum = [0, .6, .9]).
#   pos 0  student == teacher
#   pos 1  student starves candidate 0     -> pi_T/pi_S = 30 there, vote 0.667
#   pos 2  student piles onto candidate 1  -> pi_S/pi_T = 2.167 there
#   pos 3  student holds 2% of the nucleus -> coverage 0.02
CASES = [
    ([0.60, 0.30, 0.10], [0.60, 0.30, 0.10]),
    ([0.02, 0.30, 0.68], [0.60, 0.30, 0.10]),
    ([0.30, 0.65, 0.05], [0.60, 0.30, 0.10]),
    ([0.01, 0.01, 0.98], [0.60, 0.30, 0.10]),
]
COMMON = dict(
    teacher_top_p=0.9,
    coverage_threshold=0.2,
    fkl_vote_threshold=0.3,
    prob_floor=1e-6,
    overshoot_floor=0.0,  # off by default in these cases; exercised in its own test
)
ONLY = dict(use_coverage_rule=False, use_low_coverage_rule=False, use_overconfident_rule=False)


def v5(**over):
    s = torch.tensor([[c[0] for c in CASES]], dtype=torch.float32)
    t = torch.tensor([[c[1] for c in CASES]], dtype=torch.float32)
    kw = dict(COMMON, ratio_low=1.5, ratio_high=3.0)
    kw.update(over)
    ctx = Ctx(s, t, **kw)
    return MASK_REGISTRY["opd_theory_guided5"](ctx)[0].tolist(), ctx.extras


def test_coverage_rule_alone():
    """Only pos 3 holds under 20% of the teacher nucleus."""
    pg, extras = v5(**dict(ONLY, use_coverage_rule=True))
    assert pg == [True, True, True, False], pg
    assert extras["opd_student_mass_on_teacher_topp"][0].tolist()[3] < 0.2


def test_under_coverage_vote_alone():
    """pos 1/2/3 all have a starved candidate carrying >30% of the nucleus's teacher mass."""
    pg, extras = v5(**dict(ONLY, use_low_coverage_rule=True))
    assert pg == [True, False, False, False], pg
    assert torch.allclose(
        extras["opd_fkl_vote"][0], torch.tensor([0.0, 2 / 3, 2 / 3, 1.0]), atol=1e-6
    )


def test_overconfidence_rule_alone():
    """Only pos 2 over-shoots a nucleus candidate, at pi_S/pi_T = 0.65/0.30 = 2.167."""
    pg, _ = v5(**dict(ONLY, use_overconfident_rule=True, ratio_high=2.0))
    assert pg == [True, True, False, True], pg


def test_ratio_high_brackets_the_firing_point():
    """2.167 is the only over-shoot present, so it fires below that ratio and not above."""
    kw = dict(ONLY, use_overconfident_rule=True)
    assert v5(**dict(kw, ratio_high=2.0))[0][2] is False, "should route at ratio_high=2.0"
    assert v5(**dict(kw, ratio_high=2.5))[0][2] is True, "should not route at ratio_high=2.5"


def test_ratio_low_is_a_raw_ratio_not_one_plus_eps():
    """pos 1 has pi_T/pi_S = 30 on candidate 0.

    v5 takes the ratio directly, so ratio_low=10 still fires and ratio_low=100 does not. Under
    v4's `> 1 + eps_low` convention the same thresholds would be eps_low=9 and 99 -- the naming
    difference v5 exists to remove.
    """
    kw = dict(ONLY, use_low_coverage_rule=True)
    assert v5(**dict(kw, ratio_low=10.0))[0][1] is False
    assert v5(**dict(kw, ratio_low=100.0))[0][1] is True


def test_all_rules_together_are_the_union():
    pg, extras = v5(ratio_high=2.0)
    assert pg == [True, False, False, False], pg
    assert extras["opd_coverage_low_mask"][0].tolist() == [False, False, False, True]
    assert extras["opd_low_coverage_mask"][0].tolist() == [False, True, True, True]
    assert extras["opd_high_coverage_mask"][0].tolist() == [False, False, True, False]


def test_v4_has_no_overconfidence_rule():
    """v4 must be unaffected by v5 living in the same module."""
    s = torch.tensor([[c[0] for c in CASES]], dtype=torch.float32)
    t = torch.tensor([[c[1] for c in CASES]], dtype=torch.float32)
    ctx = Ctx(s, t, eps_low=0.5, **COMMON)
    pg = MASK_REGISTRY["opd_theory_guided4"](ctx)[0].tolist()
    assert pg == [True, False, False, False], pg
    assert "opd_high_coverage_mask" not in ctx.extras


def test_overshoot_floor_gates_r3_on_absolute_mass():
    """pos 2 has pi_S = 0.65 on the over-shot candidate.

    Ratio alone is not enough: R3 also requires pi_S(c) > overshoot_floor, so a floor under
    0.65 lets it fire and a floor above it does not. Without this gate a ratio of 3 fires on
    pi_S=0.003 vs pi_T=0.001, which is noise, and under ANY-over-K semantics that would route
    nearly every position.
    """
    kw = dict(ONLY, use_overconfident_rule=True, ratio_high=2.0)
    assert v5(**dict(kw, overshoot_floor=0.5))[0][2] is False, "0.65 > 0.5, should route"
    assert v5(**dict(kw, overshoot_floor=0.9))[0][2] is True, "0.65 < 0.9, should not route"


def test_floor_above_half_admits_at_most_one_candidate():
    """A floor >= 0.5 makes ANY-semantics unambiguous: probabilities sum to 1, so only one
    candidate can clear it. Verified on a row where the student is nearly deterministic."""
    s_ = torch.tensor([[[0.97, 0.02, 0.01]]], dtype=torch.float32)
    t_ = torch.tensor([[[0.30, 0.60, 0.10]]], dtype=torch.float32)
    ctx = Ctx(s_, t_, ratio_low=1.5, ratio_high=3.0, overshoot_floor=0.9,
              **{k: v for k, v in COMMON.items() if k != "overshoot_floor"},
              **dict(use_coverage_rule=False, use_low_coverage_rule=False,
                     use_overconfident_rule=True))
    pg = MASK_REGISTRY["opd_theory_guided5"](ctx)
    # pi_S/pi_T on candidate 0 is 0.97/0.30 = 3.23 > 3.0 and 0.97 > 0.9 -> routed
    assert pg[0].tolist() == [False]
    assert ctx.extras["opd_high_coverage_mask"][0].tolist() == [True]
