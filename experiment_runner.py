import logging
import shutil
import subprocess
from pathlib import Path

runner_path = Path(__file__).resolve().parent


def main(
    input_folder: Path = runner_path / "executions_queue_dir",
    output_folder: Path = runner_path / "done_tasks_dir",
) -> None:
    """_summary_

    Args:
        input_folder (Path): _description_
        output_folder (Path): _description_

    Returns:
        None
    """

    # --- Logging Setup ---
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(runner_path / "experiment_runner_log.log"),
            logging.StreamHandler(),
        ],
    )

    for file in input_folder.glob("*.py"):
        src_file = input_folder / file.name
        try:
            result = subprocess.run(
                ["python", str(src_file)], capture_output=True, text=True, check=True
            )

            logging.info(f"Executed {file.name} successfully.")
            logging.info(f"Output:\n{result.stdout}")

            if not output_folder.exists():
                output_folder.mkdir()

            dst_file = output_folder / (src_file.stem + "_old" + src_file.suffix)
            try:
                shutil.move(src_file, dst_file)
                logging.info(f"Moved {file.name} to {output_folder}")
            except FileExistsError:
                logging.warning(
                    f"{dst_file} already exists. Skipping move. Check {output_folder}"
                )

        except subprocess.CalledProcessError as e:
            logging.error(f"Error running {file.name}")
            logging.error(f"Return Code: {e.returncode}")
            logging.error(f"Error Output:\n{e.stderr}")


if __name__ == "__main__":
    main()
