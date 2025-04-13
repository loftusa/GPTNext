# transfer_from_branches.py
import subprocess
import os
from typing import List, Optional


def run_command(command: List[str]) -> Optional[str]:
    """
    Runs a shell command and returns its stdout if successful.

    Args:
        command: The command to run as a list of strings.

    Returns:
        The stdout of the command as a string, or None if an error occurred.
    """
    try:
        result = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        return result.stdout
    except FileNotFoundError:
        print(
            f"Error: Command '{command[0]}' not found. Is Git installed and in your PATH?"
        )
        return None
    except subprocess.CalledProcessError:
        # Errors (like file not found on branch) are handled later
        return None


def copy_files_from_branches(
    branch_names: List[str],
    files_to_copy: List[str],
    target_dir: str = "experiments",
    start_experiment_number: int = 4,
) -> None:
    """
    Copies specified files from given git branches to the target directory,
    renaming them using the format _<#_branch_name_{model,train}.py.

    Args:
        branch_names: A list of git branch names to copy files from.
        files_to_copy: A list of relative file paths (from repo root) to copy.
        target_dir: The directory where copied files will be saved.
        start_experiment_number: The starting number for experiment naming.
    """
    os.makedirs(target_dir, exist_ok=True)
    print(f"Target directory: '{os.path.abspath(target_dir)}'")

    current_branch_cmd = ["git", "rev-parse", "--abbrev-ref", "HEAD"]
    current_branch = run_command(current_branch_cmd)
    if current_branch:
        current_branch = current_branch.strip()
        print(f"Current branch: '{current_branch}'")
    else:
        print("Warning: Could not determine the current branch.")

    experiment_counter = start_experiment_number

    for branch in branch_names:
        branch = branch.strip()
        if not branch:
            continue
        # Skip copying from the current branch to avoid overwriting local changes
        # or copying files that are already in the target directory structure.
        if branch == current_branch:
            print(f"Skipping current branch: '{branch}'")
            continue

        print(
            f"Processing branch: '{branch}' (Experiment #{experiment_counter:02d})..."
        )
        branch_processed = False  # Flag to check if any file was copied for this branch

        for file_path in files_to_copy:
            # Check if the file exists on the branch *before* trying to get content
            git_check_cmd = ["git", "cat-file", "-e", f"{branch}:{file_path}"]
            check_result = run_command(git_check_cmd)
            # run_command returns None on error (like file not found)
            if check_result is None:
                print(f"  Skipping: '{file_path}' not found on branch '{branch}'.")
                continue

            # If file exists, proceed to get content
            git_show_cmd = ["git", "show", f"{branch}:{file_path}"]
            content = run_command(git_show_cmd)

            if content is not None:
                base_name = os.path.basename(file_path)
                name, ext = os.path.splitext(base_name)
                # Sanitize branch name for use in filename
                sanitized_branch_name = branch.replace("/", "_")
                # New filename format: _<#>_<branch_name>_<original_name>.py
                new_filename = (
                    f"_{experiment_counter:02d}_{sanitized_branch_name}_{name}{ext}"
                )
                target_path = os.path.join(target_dir, new_filename)

                print(f"  Attempting to copy '{branch}:{file_path}' to '{target_path}'")
                try:
                    with open(target_path, "w", encoding="utf-8") as f:
                        f.write(content)
                    print(f"    Successfully copied to '{target_path}'")
                    branch_processed = True  # Mark that we copied at least one file
                except IOError as e:
                    print(f"    Error writing file '{target_path}': {e}")
            else:
                # This case might occur if `git show` fails for other reasons after `cat-file` succeeded
                print(
                    f"    Failed to retrieve content for '{file_path}' from branch '{branch}' (unexpected error)."
                )

        # Only increment the counter if we actually processed (found files on) the branch
        if branch_processed:
            experiment_counter += 1
        else:
            print(
                f"  No files specified in files_to_copy found on branch '{branch}'. Not incrementing experiment counter."
            )

    print("File copying process finished.")


if __name__ == "__main__":
    # --- Configuration ---
    # Using branch names from your context (excluding master)
    branches = [
        "baseline",
        "int8",
        "no_compile",
        "noflash",
        "rmsnorm",
        "soap",
        "swa",
        "swiglu",
    ]

    # Files to copy from the root of the repository in each branch
    files = [
        "train.py",
        "model.py",
    ]

    # Destination directory (relative to where the script is run)
    destination = "experiments"

    # Starting experiment number
    start_exp_num = 4
    # --- End Configuration ---

    copy_files_from_branches(branches, files, destination, start_exp_num)
