import json
from subprocess import check_output

SSGET_COMMAND = "/usr/src/ssget/ssget"

meta_json = open("../meta.json")
meta_dict = json.load(meta_json)

matrices_to_download = [meta["id"] for meta in meta_dict if meta["rows"] < 5000 and meta["rows"] == meta["cols"]]

file_paths = []
for id in matrices_to_download:
    file_path = check_output([SSGET_COMMAND, f"-ei{id}"]).decode().strip()
    file_paths.append({
        "id": id,
        "path": file_path
    })

with open('../paths.json', 'w') as outfile:
    json.dump(file_paths, outfile, indent=4, sort_keys=True)
    print("Saved collected filepaths into paths.json")
