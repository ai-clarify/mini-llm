from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def read(path: str) -> str:
    return (REPO_ROOT / path).read_text(encoding="utf-8")


def assert_contains(path: str, needle: str) -> None:
    data = read(path)
    assert needle in data, f"{path} missing: {needle}"


def main() -> None:
    assert_contains("pipelines/run_mlx.sh", "# [tb] auto-start")
    assert_contains("pipelines/run_mlx.sh", "TB_AUTO")
    assert_contains("pipelines/run_mlx.sh", "tensorboard --logdir")
    assert_contains("pipelines/run_mlx_distill_ollama.sh", "# [tb] auto-start")
    assert_contains("pipelines/run_mlx_distill_ollama.sh", "--tensorboard_dir")
    assert_contains("pipelines/run_mlx_distill_ollama.sh", "TB_AUTO")


if __name__ == "__main__":
    main()
