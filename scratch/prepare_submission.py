import os
import shutil
import zipfile

def create_submission_zip(output_filename, project_dir):
    exclude_dirs = {
        'dataset', 
        'data', 
        'venv', 
        '.git', 
        '__pycache__', 
        'runs', 
        'inference_results', 
        'inference_results_v2',
        '.gemini'
    }
    
    with zipfile.ZipFile(output_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(project_dir):
            # Modify dirs in-place to skip excluded directories
            dirs[:] = [d for d in dirs if d not in exclude_dirs]
            
            for file in files:
                if file == output_filename:
                    continue
                if file.endswith('.pyc'):
                    continue
                    
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, project_dir)
                
                # Further check to avoid adding files from excluded dirs (extra safety)
                parts = arcname.split(os.sep)
                if any(p in exclude_dirs for p in parts):
                    continue
                    
                print(f"Adding: {arcname}")
                zipf.write(file_path, arcname)

if __name__ == "__main__":
    create_submission_zip("offroad_segmentation_project.zip", ".")
    print("\nDone! offroad_segmentation_project.zip created.")
