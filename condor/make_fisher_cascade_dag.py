import htcondor
import htcondor.dags
import classad
import numpy as np
from itertools import product

dag = htcondor.dags.DAG()

shape_model_path = (
    "/data/p-one/chaack/hyperion/data/photon_arrival_time_nflow_params.pickle"
)
counts_model_path = (
    "/data/p-one/chaack/hyperion/data/photon_arrival_time_counts_params.pickle"
)

outfile_path = "fisher_$(spacing)_$(energy)_$(pmts)_$(seed).npz"

description = htcondor.Submit(
    executable="/data/p-one/chaack/run.sh",  # the program we want to run
    arguments=f"python /data/p-one/chaack/olympus/run_fisher.py -o {outfile_path} -s $(spacing) -e $(energy) --seed $(seed) --shape_model {shape_model_path} --counts_model {counts_model_path} --pmts $(pmts)",
    log="logs/log",  # the HTCondor job event log
    output="logs/fisher.out.$(spacing)_$(energy)_$(seed)_$(pmts)",  # stdout from the job goes here
    error="logs/fisher.err.$(spacing)_$(energy)_$(seed)_$(pmts)",  # stderr from the job goes here
    request_gpus="1",  # resource requests; we don't need much per job for this problem
    Requirements="HasSingularity",
    should_transfer_files="YES",
    when_to_transfer_output="ON_EXIT",
)
description["+SingularityImage"] = classad.quote("/data/p-one/pytorch-geo.sif")

spacings = np.linspace(50, 200, 7)
energies = np.logspace(3, 5.5, 7)
seeds = np.arange(100)
pmts = [16, 20, 24]

dagvars = []
for spacing, energy, seed, pmt in product(spacings, energies, seeds, pmts):
    dagvars.append({"spacing": spacing, "energy": energy, "seed": seed, "pmts": pmt})

layer = dag.layer(
    name="fisher",
    submit_description=description,
    vars=dagvars,
)

dag_dir = "/scratch/chaack/condor"
dag_file = htcondor.dags.write_dag(dag, dag_dir, dag_file_name="fisher.dag")
