from subprocess import check_output
import json

GINKGO_BENCHMARK_COMMAND = "/usr/src/ginkgo/build/benchmark/solver/solver"

paths_json = open("../paths.json")
paths_dict = json.load(paths_json)

arguments = []
for path in paths_dict:
    arguments.append({
        "filename": path["path"],
        "optimal": {
            "spmv": "csr"
        }
    })


executor = "omp"
solvers = "cg,bicg,bicgstab,fcg,cgs,idr"
max_iters = "1000"
rel_res_goal = "0"
input_json = bytes(json.dumps(arguments), encoding="utf-8")
benchmark_results = json.loads(check_output(
    [
        GINKGO_BENCHMARK_COMMAND,
        "-executor", executor,
        "-solvers", solvers,
        "-max_iters", max_iters,
        "-rel_res_goal", rel_res_goal
    ],
    input=input_json).decode())
with open('../benchmark_results.json', 'w') as outfile:
    json.dump(benchmark_results, outfile, indent=4, sort_keys=True)
    print("Saved benchmark results into benchmark_results.json")
