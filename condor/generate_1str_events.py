import os
import stat
import htcondor
import htcondor.dags
import classad
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("--shape-model", required=True, dest="shape_model")
parser.add_argument("--counts-model", required=True, dest="counts_model")
parser.add_argument("-o", required=True, dest="outfolder")
parser.add_argument("--dag-dir", required=True, dest="dag_dir")
parser.add_argument("--singularity-image", required=True, dest="simage")
parser.add_argument("--for-slurm", action="store_true", dest="for_slurm")
parser.add_argument("--repo-path", required=True, dest="repo_path")
parser.add_argument("--config", required=True, dest="config")

args = parser.parse_args()

outfile_path = os.path.join(args.outfolder, "events_$(type)_$(seed).pickle")

runsh_cont = f"""ulimit -c 0
export PYTHONPATH=/opt/PROPOSAL/build/src/pyPROPOSAL:{args.repo_path}/olympus:{args.repo_path}/hyperion
"$@"
"""
runsh_file = os.path.join(args.repo_path, "run.sh")

with open(runsh_file, "w") as hdl:
    hdl.write(runsh_cont)

st = os.stat(runsh_file)
os.chmod(runsh_file, st.st_mode | stat.S_IEXEC)

if not args.for_slurm:
    exec = runsh_file
    exec_args = f"python {args.repo_path}/olympus/generate_events.py -n 100 -o {outfile_path} --seed $(seed) --shape-model {args.shape_model} --counts-model {args.counts_model} --config {args.config} --type $(type)"
else:
    exec = runsh_file
    exec_args = f"singularity exec --bind /mnt:/mnt --nv {args.simage} python {args.repo_path}/olympus/generate_events.py -n 100 -o {outfile_path} --seed $(seed) --shape-model {args.shape_model} --counts-model {args.counts_model} --config {args.config} --type $(type)"

description = htcondor.Submit(
    executable=exec,  # the program we want to run
    arguments=exec_args,
    log="logs/log",  # the HTCondor job event log
    output="logs/events.out.$(seed)_$(type)",  # stdout from the job goes here
    error="logs/events.err.$(seed)_$(type)",  # stderr from the job goes here
    request_gpus="1",
    Requirements="HasSingularity",
    should_transfer_files="YES",
    when_to_transfer_output="ON_EXIT",
    request_memory="2.5GB",
)
description["+SingularityImage"] = classad.quote(args.simage)


dagvars = []
for seed in range(100):
    for type in ["track", "starting_track", "cascade"]:
        dagvars.append({"seed": seed, "type": type})

dag = htcondor.dags.DAG()

layer = dag.layer(
    name="generate_events",
    submit_description=description,
    vars=dagvars,
)

dag_file = htcondor.dags.write_dag(
    dag, args.dag_dir, dag_file_name="generate_events.dag"
)
