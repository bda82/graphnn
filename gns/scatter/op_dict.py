from gns.scatter.scatter_max import scatter_max
from gns.scatter.scatter_mean import scatter_mean
from gns.scatter.scatter_min import scatter_min
from gns.scatter.scatter_prod import scatter_prod
from gns.scatter.scatter_sum import scatter_sum

OP_DICT = {
    "sum": scatter_sum,
    "mean": scatter_mean,
    "max": scatter_max,
    "min": scatter_min,
    "prod": scatter_prod,
}
