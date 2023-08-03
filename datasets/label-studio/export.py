import os
import time
from label_studio_sdk import Client
from label_studio_converter import Converter

LABEL_STUDIO_URL = os.getenv('LABEL_STUDIO_URL', default='http://localhost:8080')
API_KEY = "xxxxx"
PROJECT_ID = int("7")
VIEW_ID = False # int("18")

# connect to Label Studio
ls = Client(url=LABEL_STUDIO_URL, api_key=API_KEY)
ls.check_connection()

# get existing project
project = ls.get_project(PROJECT_ID)

# get the first tab
views = project.get_views()

for view in views:
    if VIEW_ID and VIEW_ID != view['id']:
        continue
    
    task_filter_options = {'view': view['id']} if views else {}
    view_name = view["data"]["title"]

    # create new export snapshot
    export_result = project.export_snapshot_create(
        title='Export SDK Snapshot', task_filter_options=task_filter_options
    )
    assert 'id' in export_result
    export_id = export_result['id']

    # wait until snapshot is ready
    while project.export_snapshot_status(export_id).is_in_progress():
        time.sleep(1.0)

    # download snapshot file
    status, file_name = project.export_snapshot_download(export_id)
    assert status == 200
    assert file_name is not None
    os.rename(file_name, view_name + ".json")
    print(f"Status of the export is {status}.\nFile name is {view_name}.json")

    # Run:
    # label-studio-converter export -i train.json --config config.xml -o "train" -f YOLO


    # c = Converter('config.xml', "/")
    # c.convert_to_yolo(view_name + ".json", view_name)
