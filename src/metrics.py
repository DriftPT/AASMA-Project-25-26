from dataclasses import dataclass
from typing import List, Dict, Any

from src.utils import manhattan_distance


@dataclass
class EpisodeResult:
    success: bool
    survivors: int
    steps: int
    attack_events: int
    heal_events: int
    scan_events: int
    avg_distance: float


def collect_episode_result(model) -> EpisodeResult:
    """
    Collects the final metrics from a finished simulation episode.
    """
    return EpisodeResult(
        success=model.is_successful(),
        survivors=len(model.get_alive_survivors()),
        steps=model.current_step,
        attack_events=model.attack_events,
        heal_events=model.heal_events,
        scan_events=model.scan_events,
        avg_distance=model.sum_avg_distance / model.count_avg if model.count_avg > 0 else 0.0,
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
            "avg_cooperation_rate": 0.0,
            "avg_distance": 0.0,
        }

    episodes = len(results)
    total_steps = sum(result.steps for result in results)

    return {
        "success_rate": sum(result.success for result in results) / episodes,
        "avg_survivors": sum(result.survivors for result in results) / episodes,
        "avg_steps": sum(result.steps for result in results) / episodes,
        "avg_attack_events": sum(result.attack_events for result in results) / episodes,
        "avg_heal_events": sum(result.heal_events for result in results) / episodes,
        "avg_scan_events": sum(result.scan_events for result in results) / episodes,
        "avg_cooperation_rate": sum(result.attack_events + result.heal_events + result.scan_events for result in results) / total_steps,
        "avg_distance": sum(result.avg_distance for result in results) / episodes,
    }
