import logging

from ananke.configurations.collection import (
    HDF5StorageConfiguration,
    MergeConfiguration, MergeContentConfiguration,
)
from ananke.configurations.events import (
    Interval,
    RedistributionConfiguration,
    EventRedistributionMode,
)
from ananke.models.collection import Collection
from ananke.schemas.event import RecordType

# os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"  # add this
# os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "\"platform\""
#
# import jax
#
# # Global flag to set a specific platform, must be used at startup.
# jax.config.update('jax_platform_name', 'cpu')

logging.getLogger().setLevel(logging.INFO)

data_path = 'data/example/06_digital_twin.h5'
merged_data_path = 'data/example/06_digital_twin_merged.h5'

storage_configuration = HDF5StorageConfiguration(
    data_path=data_path,
    read_only=False
)

merged_storage_configuration = HDF5StorageConfiguration(
    data_path=merged_data_path,
    read_only=False
)

collection = Collection(storage_configuration)

with collection:
    collection.redistribute(
        RedistributionConfiguration(
            interval=Interval(),
            mode=EventRedistributionMode.CONTAINS_PERCENTAGE
        )
    )

merge_configuration = MergeConfiguration(
    in_collections=[storage_configuration],
    out_collection=merged_storage_configuration,
    content=[
        MergeContentConfiguration(
            primary_type=RecordType.CASCADE,
            secondary_types=[
                RecordType.ELECTRICAL,
                RecordType.BIOLUMINESCENCE
            ],
            interval=Interval(),
            number_of_records=3
        ),
        MergeContentConfiguration(
            primary_type=RecordType.BIOLUMINESCENCE,
            secondary_types=[
                RecordType.ELECTRICAL
            ],
            interval=Interval(),
            number_of_records=3
        )
    ]
)

merged_collection = Collection.from_merge(merge_configuration)
