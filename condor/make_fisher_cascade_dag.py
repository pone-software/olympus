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

args = parser.parse_args()


shape_model_path = args.shape_model
counts_model_path = args.counts_model

outfile_path = "fisher_$(spacing)_$(energy)_$(pmts)_$(seed)_tfirst.npz"

description = htcondor.Submit(
    executable="/data/p-one/chaack/run.sh",  # the program we want to run
    arguments=f"python /data/p-one/chaack/olympus/run_fisher.py -o {outfile_path} -s $(spacing) -e $(energy) --seed $(seed) --shape_model {shape_model_path} --counts_model {counts_model_path} --pmts $(pmts) --mode tfirst",
    log="logs/log",  # the HTCondor job event log
    output="logs/fisher.out.$(spacing)_$(energy)_$(seed)_$(pmts)",  # stdout from the job goes here
    error="logs/fisher.err.$(spacing)_$(energy)_$(seed)_$(pmts)",  # stderr from the job goes here
    request_gpus="1",
    Requirements="HasSingularity",
    should_transfer_files="YES",
    when_to_transfer_output="ON_EXIT",
)
description["+SingularityImage"] = classad.quote(args.simage)

spacings = np.linspace(50, 200, 7)
energies = np.logspace(3, 5.5, 7)
seeds = np.arange(100)
pmts = [16, 20, 24]

dagvars = []
for spacing, energy, seed, pmt in product(spacings, energies, seeds, pmts):
    dagvars.append({"spacing": spacing, "energy": energy, "seed": seed, "pmts": pmt})

dag = htcondor.dags.DAG()

layer = dag.layer(
    name="fisher",
    submit_description=description,
    vars=dagvars,
)


dag_file = htcondor.dags.write_dag(dag, args.dag_dir, dag_file_name="fisher.dag")
