import subprocess
import os
import time
import glob
from collections import deque
from typing import List, Dict, Tuple, Set, Optional
import sys
import argparse

FAILED_LOG_FILE = "failed_experiments.log"

def get_train_scripts(directory: str = "experiments", retry_failed: bool = False) -> List[str]:
    """Finds training scripts, either all or only previously failed ones.

    Args:
        directory: The directory to search for training scripts (used if not retrying).
        retry_failed: If True, reads script paths from FAILED_LOG_FILE. Otherwise,
                      finds all *_train.py files in the directory and clears the log.

    Returns:
        A sorted list of paths to the training scripts to run.
    """
    if retry_failed:
        print(f"Attempting to retry failed scripts listed in '{FAILED_LOG_FILE}'.")
        scripts: List[str] = []
        if os.path.exists(FAILED_LOG_FILE):
            with open(FAILED_LOG_FILE, 'r') as f:
                scripts = [line.strip() for line in f if line.strip()]
            
            if scripts:
                 # Clear the log file *after* successfully reading it for the retry
                 print(f"Read {len(scripts)} scripts to retry. Clearing '{FAILED_LOG_FILE}' for the new run.")
                 open(FAILED_LOG_FILE, 'w').close() 
            else:
                 print(f"'{FAILED_LOG_FILE}' was found but is empty. No scripts to retry.")
        else:
            print(f"'{FAILED_LOG_FILE}' not found. No scripts to retry.")
        return sorted(scripts) # Return sorted list, even if empty
    else:
        # Standard run: find all scripts and clear any previous failure log
        if os.path.exists(FAILED_LOG_FILE):
             print(f"Clearing previous failure log '{FAILED_LOG_FILE}'.")
             os.remove(FAILED_LOG_FILE)
        pattern = os.path.join(directory, "*_train.py")
        scripts = sorted(glob.glob(pattern))
        to_skip = [
            "experiments/_01_spec_decoding_train.py",
            "experiments/_05_int8_train.py",
            "experiments/_06_no_compile_train.py",
            "experiments/_07_noflash_train.py",
        ]
        scripts = [script for script in scripts if script not in to_skip]
        print(f"Found {len(scripts)} scripts matching '{pattern}' for a standard run.")
        return scripts

def get_available_gpu_ids() -> List[int]:
    """Gets the list of available GPU IDs.

    Currently assumes 8 GPUs (0-7) based on user-provided context.
    A more dynamic approach could involve parsing nvidia-smi or using torch.

    Returns:
        A list of integer GPU IDs.
    """
    # Assuming 8 GPUs based on the provided nvidia-smi output.
    # Consider making this dynamic if GPU availability changes frequently.
    num_gpus = 8 
    print(f"Assuming {num_gpus} GPUs are available (IDs 0 to {num_gpus - 1}).")
    return list(range(num_gpus))

def run_experiments(retry_failed: bool = False) -> None:
    """Runs training scripts in parallel across available GPUs."""
    train_scripts: List[str] = get_train_scripts(retry_failed=retry_failed)
    if not train_scripts:
        # Message adjusted based on retry status
        if retry_failed:
            print("No failed scripts to retry. Exiting.")
        else:
            print("No training scripts found in experiments/ directory. Exiting.")
        return

    available_gpu_ids: List[int] = get_available_gpu_ids()
    if not available_gpu_ids:
        print("No GPUs seem to be available. Running sequentially on CPU might be an option, but this script requires GPUs. Exiting.")
        return
        
    num_gpus: int = len(available_gpu_ids)
    print(f"Managing {len(train_scripts)} scripts across {num_gpus} GPUs.")

    script_queue: deque[str] = deque(train_scripts)
    # {gpu_id: (subprocess.Popen, script_path)}
    running_processes: Dict[int, Tuple[subprocess.Popen, str]] = {}
    available_gpus: Set[int] = set(available_gpu_ids)
    
    output_dir = "experiment_logs"
    os.makedirs(output_dir, exist_ok=True)
    print(f"Redirecting script outputs to '{output_dir}/'.")

    try:
        while script_queue or running_processes:
            # --- Check for finished processes ---
            finished_gpus: List[int] = []
            for gpu_id, (process, script_path) in running_processes.items():
                return_code: Optional[int] = process.poll()
                if return_code is not None:  # Process finished
                    finished_gpus.append(gpu_id)
                    # Adjusted log file naming slightly for clarity
                    log_base = os.path.basename(script_path).replace('.py', '')
                    stdout_log = os.path.join(output_dir, f"{log_base}_gpu{gpu_id}_stdout.log")

                    if return_code == 0:
                        print(f"✅ Script {script_path} on GPU {gpu_id} finished successfully. Log: {stdout_log}")
                    else:
                        print(f"❌ Script {script_path} on GPU {gpu_id} failed with code {return_code}. Log: {stdout_log}")
                        # --- Log failed script ---
                        try:
                            with open(FAILED_LOG_FILE, 'a') as f_fail:
                                f_fail.write(f"{script_path}\n")
                            print(f"   Logged failed script to '{FAILED_LOG_FILE}'.")
                        except Exception as e:
                             print(f"   ⚠️ Could not log failed script {script_path} to '{FAILED_LOG_FILE}': {e}")
            
            # --- Free up GPUs ---
            for gpu_id in finished_gpus:
                if gpu_id in running_processes: # Ensure key exists before deleting
                     # Close log file handles before removing process reference
                    proc_tuple = running_processes[gpu_id]
                    if hasattr(proc_tuple[0], 'stdout') and proc_tuple[0].stdout:
                        proc_tuple[0].stdout.close()
                    if hasattr(proc_tuple[0], 'stderr') and proc_tuple[0].stderr:
                        proc_tuple[0].stderr.close()
                    del running_processes[gpu_id]
                available_gpus.add(gpu_id) # Make GPU available again

            # --- Launch new processes ---
            while script_queue and available_gpus:
                script_path: str = script_queue.popleft()
                gpu_id: int = available_gpus.pop()

                print(f"🚀 Launching {script_path} on GPU {gpu_id}...")
                env: Dict[str, str] = os.environ.copy()
                env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
                
                cmd: List[str] = ['uv', 'run', 'python', script_path]
                
                # Create log files for stdout and stderr
                base_name = os.path.basename(script_path)
                stdout_log = os.path.join(output_dir, f"{base_name}_gpu{gpu_id}_stdout.log")
                stderr_log = os.path.join(output_dir, f"{base_name}_gpu{gpu_id}_stderr.log")

                try:
                    # Open log files
                    stdout_f = open(stdout_log, 'w')
                    stderr_f = open(stderr_log, 'w')
                    
                    # Launch the process, redirecting output
                    process: subprocess.Popen = subprocess.Popen(
                        cmd, 
                        env=env, 
                        stdout=stdout_f, 
                        stderr=stderr_f
                    )
                    running_processes[gpu_id] = (process, script_path)
                except Exception as e:
                    print(f"⚠️ Failed to start {script_path} on GPU {gpu_id}: {e}")
                    # Clean up if process failed to start
                    if 'stdout_f' in locals() and not stdout_f.closed:
                        stdout_f.close()
                    if 'stderr_f' in locals() and not stderr_f.closed:
                        stderr_f.close()
                    # Make GPU available again and log failure
                    available_gpus.add(gpu_id)
                    print(f"↪️ Skipping {script_path} due to launch failure.")


            # --- Wait before next check ---
            if script_queue or running_processes:
                 # Only sleep if there's potential work left or processes running
                 time.sleep(5) # Check status every 5 seconds


    except KeyboardInterrupt:
        print("\n🛑 KeyboardInterrupt received. Terminating running processes...")
        for gpu_id, (process, script_path) in running_processes.items():
            print(f"   Terminating {script_path} on GPU {gpu_id} (PID: {process.pid})...")
            process.terminate() # Send SIGTERM
            try:
                process.wait(timeout=10) # Wait briefly for graceful termination
                print(f"   Process {process.pid} terminated.")
            except subprocess.TimeoutExpired:
                print(f"   Process {process.pid} did not terminate gracefully, sending SIGKILL...")
                process.kill() # Force kill
                process.wait()
                print(f"   Process {process.pid} killed.")
            # Ensure log files are closed even during interruption
            if hasattr(process, 'stdout') and process.stdout:
                process.stdout.close()
            if hasattr(process, 'stderr') and process.stderr:
                 process.stderr.close()
        print("Cleanup complete.")
        sys.exit(1) # Indicate abnormal termination


    print("\n🎉 All training scripts have been processed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run training experiments in parallel on available GPUs.")
    parser.add_argument(
        "--retry-failed",
        action="store_true",
        help=f"Run only the scripts listed in {FAILED_LOG_FILE} from a previous failed run."
    )
    args = parser.parse_args()
    
    run_experiments(retry_failed=args.retry_failed) 