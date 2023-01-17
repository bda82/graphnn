import logging

from gns.dataset.sfedu_dataset import sfedu_dataset_fabric

logger = logging.getLogger(__name__)

# Init dataset object

logger.info("Init dataset object...")

ds = sfedu_dataset_fabric(n_samples=100)

# Download dataset

logger.info("Download dataset...")

ds.download()

# Read content of dataset

logger.info("Read content of dataset...")

out = ds.read()

# Print result
print(out)
