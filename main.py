from src.experiments.experiments import run_all_experiments


def print_results(experiment_name: str, results: dict):
    print("=" * 70)
    print(experiment_name)
    print("=" * 70)

    for metric_name, value in results.items():
        print(f"{metric_name}: {value:.3f}")

    print()


def main():
    all_results = run_all_experiments()

    for experiment_name, results in all_results.items():
        print_results(experiment_name, results)


if __name__ == "__main__":
    main()