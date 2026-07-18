from src.components.docker_builder import DockerBuilder
from src.config.configuration import ConfigurationManager


def test_build_command():
    """
    Tests that the Docker build command is generated correctly.
    """

    config = ConfigurationManager().get_docker_builder_config()

    docker_builder = DockerBuilder(config)

    command = docker_builder._build_command()

    expected_image = (
        f"{config.image_name}:{config.image_tag}"
    )

    expected_command = [
        "docker",
        "build",
        "-f",
        str(config.dockerfile_path),
        "-t",
        expected_image,
        str(config.context_path),
    ]

    assert command == expected_command