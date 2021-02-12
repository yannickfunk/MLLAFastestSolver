from subprocess import check_output
import json

GINKGO_BENCHMARK_COMMAND = "/usr/src/ginkgo/build/benchmark/solver/solver"
SSGET_COMMAND = "/usr/src/ssget/ssget"

matrix_meta = []
num_matrices = int(check_output([SSGET_COMMAND, "-n"]).decode())
print(f"Found {num_matrices} matrices in archive")

print("Collecting matrix metadata")
for i in range(1,num_matrices):
    print(f"{i}/{num_matrices-1}")
    matrix_meta.append(json.loads(check_output([SSGET_COMMAND, f"-ji{i}"]).decode()))

with open('../meta.json', 'w') as outfile:
    json.dump(matrix_meta, outfile, indent=4, sort_keys=True)
    print("Saved collected metadata into meta.json")

