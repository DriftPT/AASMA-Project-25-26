from dataclasses import dataclass
from typing import List, Dict, Any

from src.utils.utils import manhattan_distance


@dataclass
class EpisodeResult:
    success: bool
    survivors: int
    steps: int
    cooperation_events: int
    avg_distance: float


def compute_avg_distance_between_survivors(model) -> float:
    """
    Computes the average Manhattan distance between alive survivor agents.
    This metric is used as a simple measure of team cohesion.
    """
    survivors = model.get_alive_survivors()

    if len(survivors) < 2:
        return 0.0

    distances = []

    for i in range(len(survivors)):
        for j in range(i + 1, len(survivors)):
            distances.append(
                manhattan_distance(survivors[i].pos, survivors[j].pos)
            )

    return sum(distances) / len(distances)


def collect_episode_result(model) -> EpisodeResult:
    """
    Collects the final metrics from a finished simulation episode.
    """
    return EpisodeResult(
        success=model.is_successful(),
        survivors=len(model.get_alive_survivors()),
        steps=model.current_step,
        cooperation_events=model.cooperation_events,
        avg_distance=compute_avg_distance_between_survivors(model),
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


def model_success(model) -> bool:
    return model.is_successful()


def model_alive_survivors(model) -> int:
    return len(model.get_alive_survivors())


def model_cooperation_events(model) -> int:
    return model.cooperation_events


def model_avg_distance(model) -> float:
    return compute_avg_distance_between_survivors(model)