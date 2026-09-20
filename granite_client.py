"""AI placeholder and transparent rule-based study plan."""

from __future__ import annotations

USE_AI = False


def build_prompt(student_profile: dict, risk_factors: list[str]) -> str:
    """Build the future AI prompt without names or identifying information."""

    return f"""You are a supportive study coach for a student.
Use simple English and do not diagnose, label, punish, or mention protected traits.
Student profile: {student_profile}
Main factors to talk about: {", ".join(risk_factors)}

Return:
1. A kind two-line summary.
2. A realistic four-week study plan with small steps.
3. Three motivational tips.
4. Advice for a teacher or parent.
"""


def rule_based_plan(student_profile: dict, risk_factors: list[str]) -> str:
    """Return a realistic local demo plan while the AI model is disconnected."""

    study_band = int(student_profile.get("studytime", 2))
    study_minutes = {1: 15, 2: 25, 3: 35, 4: 45}.get(study_band, 25)
    factor_text = ", ".join(risk_factors[:3]) or "recent learning patterns"
    return f"""### Two-line summary
This student may benefit from a calm check-in around {factor_text.lower()}.
Start with one small win before adding more practice.

### Four-week plan
**Week 1 · Reconnect**
- Spend {study_minutes} minutes twice this week on one familiar concept.
- Ask the student to explain one answer aloud without correcting too quickly.

**Week 2 · Build one bridge**
- Complete two short retrieval sets from the current topic.
- Mark one question as “not yet” and bring it to the next check-in.

**Week 3 · Practice with support**
- Work through one example together, then let the student try a similar one.
- Celebrate the strategy used, not only the final answer.

**Week 4 · Notice progress**
- Revisit the first task and compare what feels easier now.
- Agree on one next habit that can continue next month.

### Three motivational tips
1. Keep the first five minutes low-stakes.
2. Ask “What part feels clear?” before asking what went wrong.
3. Make progress visible with one small check mark each session.

### Teacher or parent note
Use this as a conversation starter. Ask what support feels useful, adapt the pace,
and keep the student involved in choosing the next step.
"""


def generate_study_plan(student_profile: dict, risk_factors: list, prompt: str) -> str:
    """PLACEHOLDER: replace the body with the AI model call later."""
    # TODO: AI integration goes here
    return rule_based_plan(student_profile, risk_factors)