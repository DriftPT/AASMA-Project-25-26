from dataclasses import dataclass
from typing import List, Dict, Any

from src.utils import manhattan_distance


@dataclass
class EpisodeResult:
    success: bool
    survivors: int
    steps: int
    cooperation_events: int
    avg_distance: float

def collect_episode_result(model) -> EpisodeResult:
    """
    Collects the final metrics from a finished simulation episode.
    """
    return EpisodeResult(
        success=model.is_successful(),
        survivors=len(model.get_alive_survivors()),
        steps=model.current_step,
        cooperation_events=model.cooperation_events,
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
            "avg_cooperation_events": 0.0,
            "avg_distance": 0.0,
        }

    episodes = len(results)

    return {
        "success_rate": sum(result.success for result in results) / episodes,
        "avg_survivors": sum(result.survivors for result in results) / episodes,
        "avg_steps": sum(result.steps for result in results) / episodes,
        "avg_cooperation_events": sum(result.cooperation_events for result in results) / episodes,
        "avg_distance": sum(result.avg_distance for result in results) / episodes,
    }
