import os
from subprocess import check_output
import json
from mpi4py import MPI
rank = MPI.COMM_WORLD.Get_rank()


USER_PATH = os.environ["PROJECT"] + "/users/funk1/"
PATHS_PATH = USER_PATH+'data/paths.json'
BENCHMARK_RESULTS_PATH = USER_PATH+'data/'
GINKGO_BENCHMARK_COMMAND = USER_PATH+"src/ginkgo/build/benchmark/solver/solver"

paths_json = open(PATHS_PATH)
paths_dict = json.load(paths_json)

arguments = []
for path in paths_dict:
    arguments.append({
        "filename": path["path"],
        "optimal": {
            "spmv": "csr"
        }
    })


executor = "cuda"
solvers = "cg,bicg,fcg,cgs,idr,gmres,bicgstab"
max_iters = "1000"
rel_res_goal = "0"
if rank == 0:
    input_json = bytes(json.dumps(arguments[:350]), encoding="utf-8")
    benchmark_results = json.loads(check_output(
	[
            GINKGO_BENCHMARK_COMMAND,
            "-executor", executor,
            "-solvers", solvers,
            "-max_iters", max_iters,
            "-rel_res_goal", rel_res_goal
        ],
        input=input_json).decode())
    with open(BENCHMARK_RESULTS_PATH+'benchmark_results0.json', 'w') as outfile:
        json.dump(benchmark_results, outfile, indent=4, sort_keys=True)
        print("Saved benchmark results into benchmark_results0.json")
elif rank == 1:
    input_json = bytes(json.dumps(arguments[350:700]), encoding="utf-8")
    benchmark_results = json.loads(check_output(
        [
            GINKGO_BENCHMARK_COMMAND,
            "-executor", executor,
            "-solvers", solvers,
            "-max_iters", max_iters,
            "-rel_res_goal", rel_res_goal
        ],
        input=input_json).decode())
    with open(BENCHMARK_RESULTS_PATH+'benchmark_results1.json', 'w') as outfile:
        json.dump(benchmark_results, outfile, indent=4, sort_keys=True)
        print("Saved benchmark results into benchmark_results1.json")
elif rank == 2:
    input_json = bytes(json.dumps(arguments[700:1050]), encoding="utf-8")
    benchmark_results = json.loads(check_output(
        [
            GINKGO_BENCHMARK_COMMAND,
            "-executor", executor,
            "-solvers", solvers,
            "-max_iters", max_iters,
            "-rel_res_goal", rel_res_goal
        ],
        input=input_json).decode())
    with open(BENCHMARK_RESULTS_PATH+'benchmark_results2.json', 'w') as outfile:
        json.dump(benchmark_results, outfile, indent=4, sort_keys=True)
        print("Saved benchmark results into benchmark_results2.json")
elif rank == 3:
    input_json = bytes(json.dumps(arguments[1050:1400]), encoding="utf-8")
    benchmark_results = json.loads(check_output(
        [
            GINKGO_BENCHMARK_COMMAND,
            "-executor", executor,
            "-solvers", solvers,
            "-max_iters", max_iters,
            "-rel_res_goal", rel_res_goal
        ],
        input=input_json).decode())
    with open(BENCHMARK_RESULTS_PATH+'benchmark_results3.json', 'w') as outfile:
        json.dump(benchmark_results, outfile, indent=4, sort_keys=True)
        print("Saved benchmark results into benchmark_results3.json")
elif rank == 4:
    input_json = bytes(json.dumps(arguments[1400:1750]), encoding="utf-8")
    benchmark_results = json.loads(check_output(
        [
            GINKGO_BENCHMARK_COMMAND,
            "-executor", executor,
            "-solvers", solvers,
            "-max_iters", max_iters,
            "-rel_res_goal", rel_res_goal
        ],
        input=input_json).decode())
    with open(BENCHMARK_RESULTS_PATH+'benchmark_results4.json', 'w') as outfile:
        json.dump(benchmark_results, outfile, indent=4, sort_keys=True)
        print("Saved benchmark results into benchmark_results4.json")
elif rank == 5:
    input_json = bytes(json.dumps(arguments[1750:2100]), encoding="utf-8")
    benchmark_results = json.loads(check_output(
        [
            GINKGO_BENCHMARK_COMMAND,
            "-executor", executor,
            "-solvers", solvers,
            "-max_iters", max_iters,
            "-rel_res_goal", rel_res_goal
        ],
        input=input_json).decode())
    with open(BENCHMARK_RESULTS_PATH+'benchmark_results5.json', 'w') as outfile:
        json.dump(benchmark_results, outfile, indent=4, sort_keys=True)
        print("Saved benchmark results into benchmark_results5.json")
elif rank == 6:
    input_json = bytes(json.dumps(arguments[2450:]), encoding="utf-8")
    benchmark_results = json.loads(check_output(
        [
            GINKGO_BENCHMARK_COMMAND,
            "-executor", executor,
            "-solvers", solvers,
            "-max_iters", max_iters,
            "-rel_res_goal", rel_res_goal
        ],
        input=input_json).decode())
    with open(BENCHMARK_RESULTS_PATH+'benchmark_results6.json', 'w') as outfile:
        json.dump(benchmark_results, outfile, indent=4, sort_keys=True)
        print("Saved benchmark results into benchmark_results6.json")


