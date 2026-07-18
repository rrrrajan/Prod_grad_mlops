import subprocess

from src.pipeline.stage_08_docker_builder import (
    DockerBuilderPipeline,
)


def test_docker_builder_pipeline():
    """
    Tests the complete Docker Builder pipeline by
    building the Docker image and verifying that it exists.
    """

    pipeline = DockerBuilderPipeline()

    pipeline.main()

    result = subprocess.run(
        [
            "docker",
            "images",
            "--format",
            "{{.Repository}}:{{.Tag}}",
        ],
        capture_output=True,
        text=True,
        check=True,
    )

    assert (
        "customer-churn-api:latest"
        in result.stdout
    )

if __name__ == "__main__":
    test_docker_builder_pipeline()