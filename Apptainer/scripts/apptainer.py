import logging

import os
import subprocess
import tempfile
import sys

logger = logging.getLogger()




# def start_apptainer(notebooks_dir, container_image):
#     process = None
#     try:
#         with tempfile.TemporaryDirectory() as working_directory:
#             tmp_dir = os.path.join(working_directory, "tmp")
#             os.makedirs(tmp_dir, exist_ok=True)

#             # Step 2: Set environment variables
#             os.environ["SINGULARITY_TMPDIR"] = tmp_dir
#             os.environ["APPTAINER_TMPDIR"] = tmp_dir
#             os.environ["JUPYTER_RUNTIME_DIR"] = tmp_dir
#             os.environ["JUPYTER_DATA_DIR"] = tmp_dir
#             os.environ["JUPYTER_CONFIG_DIR"] = tmp_dir

#             run_command = [
#                 "apptainer",
#                 "run",
#                 "--bind",
#                 f"{notebooks_dir}:/notebooks",
#                 container_image,
#                 "jupyter",
#                 "notebook",
#                 "--no-browser",
#                 "--ip=0.0.0.0",
#             ]
#             process = subprocess.Popen(run_command)
#             process.wait()
#             logger.info(
#                 "Jupyter Notebook started successfully in the Apptainer container."
#             )
#     except KeyboardInterrupt:
#         if process:
#             process.terminate()
#             process.wait()
#         logger.info("Operation cancelled by user (Ctrl-C).")
       
#         sys.exit(0)



def rerun(container_image: str = "hpc-notebook.sif"):
 
    container_image = os.path.abspath(os.path.join(os.path.dirname(__file__), "../hpc-notebook.sif"))
    py_file = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../main.py"))

    if not os.path.exists(container_image):
        logger.error("Not found.")

    else:
        # start_apptainer(notebooks_dir, container_image)
        command = ["apptainer", "exec", container_image, "bash", "-c", f"python3 {py_file}"]
        subprocess.run(command, check=True)

      
#create Apptainer Image
def start(container_image: str = "hpc-notebook.sif"):
    """Open a notebook file in HPC."""

    # Get the absolute path of the definition file located three levels up from the current script's directory
    definition_file = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "Singularity.def")
    )
    
    current_directory = os.getcwd()
    notebooks_dir = os.path.join(current_directory, "notebooks")

    os.makedirs(notebooks_dir, exist_ok=True)

    try:
        # Ensure the definition file exists
        if not os.path.isfile(definition_file):
            logger.error(f"Definition file {definition_file} not found.")
  

        # Step 3: Build the Apptainer container
        build_command = ["apptainer", "build", container_image, definition_file]
        subprocess.run(build_command, check=True)
        logger.info("Apptainer container built successfully.")

        # Step 4: Run the Apptainer container and start Jupyter Notebook
        start_apptainer(notebooks_dir, container_image)
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to start Jupyter Notebook: {e}")

    except KeyboardInterrupt:
        logger.info("Operation cancelled by user (Ctrl-C).")
        
        sys.exit(0)
    except Exception as e:
        logger.error(f"An error occurred: {e}")
      


# start()

rerun()
