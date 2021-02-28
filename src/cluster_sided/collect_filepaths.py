import os
import json
from subprocess import check_output

USER_PATH = os.environ["PROJECT"] + "/users/funk1/"
SSGET_COMMAND = USER_PATH + "src/ssget/ssget"
META_PATH = USER_PATH+'data/meta.json'
PATHS_PATH = USER_PATH+'data/paths.json'

meta_json = open(META_PATH)
meta_dict = json.load(meta_json)

matrices_to_download = [meta["id"] for meta in meta_dict if meta["rows"] == meta["cols"]]

file_paths = []
for id in matrices_to_download:
    file_path = check_output([SSGET_COMMAND, f"-ei{id}"]).decode().strip()
    file_paths.append({
        "id": id,
        "path": file_path
    })

with open(PATHS_PATH, 'w') as outfile:
    json.dump(file_paths, outfile, indent=4, sort_keys=True)
    print("Saved collected filepaths into paths.json")
