import os
import stat
import htcondor
import htcondor.dags
import classad
import numpy as np
from itertools import product
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("--shape-model", required=True, dest="shape_model")
parser.add_argument("--counts-model", required=True, dest="counts_model")
parser.add_argument("-o", required=True, dest="outfolder")
parser.add_argument("--dag-dir", required=True, dest="dag_dir")
parser.add_argument("--singularity-image", required=True, dest="simage")
parser.add_argument("--for-slurm", action="store_true", dest="for_slurm")
parser.add_argument("--repo-path", required=True, dest="repo_path")
parser.add_argument(
    "--mode", required=True, choices=["tfirst", "full", "counts"], dest="mode"
)


args = parser.parse_args()

outfile_path = os.path.join(
    args.outfolder, "fisher_$(spacing)_$(energy)_$(pmts)_$(seed)_$(mode)_$(det).npz"
)


runsh_cont = f"""ulimit -c 0
export PYTHONPATH={args.repo_path}/olympus:{args.repo_path}/hyperion
"$@"
"""
runsh_file = os.path.join(args.repo_path, "run.sh")

with open(runsh_file, "w") as hdl:
    hdl.write(runsh_cont)

st = os.stat(runsh_file)
os.chmod(runsh_file, st.st_mode | stat.S_IEXEC)


arg_list = (
    f"-s $(spacing) --seed $(seed) --shape_model {args.shape_model} "
    f"--counts_model {args.counts_model} --pmts $(pmts) --mode {args.mode} --det $(det) "
    f"--nsamples 75 --nev 1 -o {outfile_path} double_casc -e $(energy)"
)

if not args.for_slurm:
    exec = runsh_file
    exec_args = f"python {args.repo_path}/olympus/run_fisher.py {arg_list}"
else:
    exec = runsh_file
    exec_args = f"singularity exec --bind /mnt:/mnt --nv {args.simage} python {args.repo_path}/olympus/run_fisher.py {arg_list}"

description = htcondor.Submit(
    executable=exec,  # the program we want to run
    arguments=exec_args,
    log="logs/log",  # the HTCondor job event log
    output="logs/fisher.out.$(spacing)_$(energy)_$(seed)_$(pmts)",  # stdout from the job goes here
    error="logs/fisher.err.$(spacing)_$(energy)_$(seed)_$(pmts)",  # stderr from the job goes here
    request_gpus="1",
    Requirements="HasSingularity",
    should_transfer_files="YES",
    when_to_transfer_output="ON_EXIT",
    request_memory="2.5GB",
)
description["+SingularityImage"] = classad.quote(args.simage)

spacings = np.linspace(50, 200, 6)
energies = np.logspace(3, 5.5, 6)
seeds = np.arange(200)
pmts = [16]  # , 20, 24]
dets = ["triangle", "cluster"]

dagvars = []
for spacing, energy, seed, pmt, det in product(spacings, energies, seeds, pmts, dets):
    dagvars.append(
        {"spacing": spacing, "energy": energy, "seed": seed, "pmts": pmt, "det": det}
    )

dag = htcondor.dags.DAG()

layer = dag.layer(
    name="fisher",
    submit_description=description,
    vars=dagvars,
)


dag_file = htcondor.dags.write_dag(dag, args.dag_dir, dag_file_name="fisher.dag")
