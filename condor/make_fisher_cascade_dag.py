import htcondor
import numpy as np
from itertools import product

dag = htcondor.dags.DAG()

shape_model_path = (
    "/data/p-one/chaack/hyperion/data/photon_arrival_time_nflow_params.pickle"
)
counts_model_path = (
    "/data/p-one/chaack/hyperion/data/photon_arrival_time_counts_params.pickle"
)

outfile_path = "/data2/p-one/chaack/fisher/fisher_$(spacing)_$(energy)_$(seed).npz"

description = htcondor.Submit(
    executable="/bin/env",  # the program we want to run
    arguments=f"python /data/p-one/chaack/olympus/run_fisher.py -o {outfile_path} -s $(spacing) -e $(energy) --seed $(seed) --shape_model {shape_model_path} --counts_model {counts_model_path}",
    log="logs/log",  # the HTCondor job event log
    output="logs/fisher.out.$(spacing)_$(energy)",  # stdout from the job goes here
    error="ogs/fisher.err.$(spacing)_$(energy)",  # stderr from the job goes here
    request_gpus="1",  # resource requests; we don't need much per job for this problem
    **{
        "+SingularityImage": htcondor.classad.quote(
            "/data/p-one/chaack/container/pytorch-geo.sif"
        ),
    }
    # request_memory="128MB",
    # request_disk="1GB",
)

spacings = np.linspace(50, 200, 10)
energies = np.logspace(3, 5.5, 10)
seeds = np.arange(100)

dagvars = []
for spacing, energy, seed in product(spacings, energies, seeds):
    dagvars.append({"spacing": spacing, "energy": energy, "seed": seed})

layer = dag.layer(
    name="fisher",
    submit_description=description,
    vars=dagvars,
)

dag_dir = "/scratch/chaack/condor"
dag_file = htcondor.dags.write_dag(dag, dag_dir)
