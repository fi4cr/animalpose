
Set up virtualenv

python3 -m venv my_venv
source ~/my_venv/bin/activate
pip install -U -r requirements.txt

change dataset_path in files

To download / process:
python download.py
python animal_pose_dataset.py
python make_dataset_from_folders.py

Then copy dataset.py to <dataset_path>/<dataset_folder>/<dataset_folder>.py

