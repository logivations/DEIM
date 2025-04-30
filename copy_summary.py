import os
import shutil

# Source and target directories
source_dir = '/data/trainings'
destination_dir = '/mnt/nfs_training/models'

# Walk through all folders in the source directory
for root, dirs, files in os.walk(source_dir):
    # Check if there is a "summary" folder
    if 'summary' in dirs:
        # Path to the summary folder
        summary_dir = os.path.join(root, 'summary')
        
        # Determine the relative path to the model directory
        relative_path = os.path.relpath(root, source_dir)
        
        # Create the corresponding structure in the target directory
        target_dir = os.path.join(destination_dir, relative_path, 'summary')
        if os.path.exists(target_dir):
            logger.info("Target dir already exists")
            continue
        os.makedirs(target_dir, exist_ok=True)
        
        # Transfer all files from the summary folder
        for file_name in os.listdir(summary_dir):
            source_file = os.path.join(summary_dir, file_name)
            if os.path.isfile(source_file):
                shutil.copy2(source_file, target_dir)  # Copy preserving metadata
        print(f"Files from {summary_dir} have been transferred to {target_dir}")
