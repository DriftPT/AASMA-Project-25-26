from dataclasses import dataclass
from typing import List, Dict, Any


@dataclass
class EpisodeResult:
    success: bool
    survivors: int
    steps: int
    attack_events: int
    heal_events: int
    scan_events: int
    # Raw step counts per role (not ratios) so we can aggregate correctly
    adaptive_scout_steps: int = 0
    adaptive_defender_steps: int = 0
    adaptive_support_steps: int = 0


def collect_episode_result(model) -> EpisodeResult:
    """
    Collects the final metrics from a finished simulation episode.
    """
    from src.agents.adaptive_agent import AdaptiveAgent

    adaptive_scout_steps = 0
    adaptive_defender_steps = 0
    adaptive_support_steps = 0

    for agent in model.agents:
        if isinstance(agent, AdaptiveAgent):
            adaptive_scout_steps = agent.role_steps_scout
            adaptive_defender_steps = agent.role_steps_defender
            adaptive_support_steps = agent.role_steps_support
            break

    return EpisodeResult(
        success=model.is_successful(),
        survivors=len(model.get_all_survivors()),
        steps=model.current_step,
        attack_events=model.attack_events,
        heal_events=model.heal_events,
        scan_events=model.scan_events,
        adaptive_scout_steps=adaptive_scout_steps,
        adaptive_defender_steps=adaptive_defender_steps,
        adaptive_support_steps=adaptive_support_steps,
    )


def summarize_results(results: List[EpisodeResult]) -> Dict[str, Any]:
    """
    Aggregates multiple episode results into average metrics.
    """
    if not results:
        return {
            "success_rate": 0.0,
            "avg_survivors": 0.0,
            "avg_steps": 0.0,
            "avg_attack_events": 0.0,
            "avg_heal_events": 0.0,
            "avg_scan_events": 0.0,
        }

    episodes = len(results)
    total_steps = sum(result.steps for result in results)
    avg_survivors = sum(result.survivors for result in results) / episodes
    avg_steps = total_steps / episodes

    summary = {
        "success_rate": sum(result.success for result in results) / episodes,
        "avg_survivors": avg_survivors,
        "avg_steps": avg_steps,
        "avg_attack_events": sum(result.attack_events for result in results) / episodes,
        "avg_heal_events": sum(result.heal_events for result in results) / episodes,
        "avg_scan_events": sum(result.scan_events for result in results) / episodes,
    }

    total_scout = sum(result.adaptive_scout_steps for result in results)
    total_defender = sum(result.adaptive_defender_steps for result in results)
    total_support = sum(result.adaptive_support_steps for result in results)
    total_active = total_scout + total_defender + total_support

    if total_active > 0:
        summary["adaptive_scout_ratio"] = total_scout / total_active
        summary["adaptive_defender_ratio"] = total_defender / total_active
        summary["adaptive_support_ratio"] = total_support / total_active

    return summary