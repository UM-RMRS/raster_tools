import xarray as xr
import pathlib
from os import cpu_count
import operator
import time


def timing(f):
    def wrap(*args, **kwargs):
        time1 = time.time()
        ret = f(*args, **kwargs)
        time2 = time.time()
        print(
            "{:s} function took {:.3f} ms".format(
                f.__name__, (time2 - time1) * 1000.0
            )
        )
        return ret

    return wrap


@timing
def arithmetic(
    input1, input2, operation, chunk_description={"x": 512, "y": 512}
):
    """
    INPUTS:
        input1, input2 these are either floats or paths to files
        operation: a single text character that defines an operation to be
        performed
    OUTPUT:
        file1 operation file2
    """
    ops = {
        "+": operator.add,
        "-": operator.sub,
        "*": operator.mul,
        "/": operator.truediv,
    }

    if type(input1) == pathlib.PosixPath:
        input1 = xr.open_rasterio(input1).chunk(chunks=chunk_description)

    if type(input2) == pathlib.PosixPath:
        input2 = xr.open_rasterio(input2).chunk(chunks=chunk_description)

    return ops[operation](input1, input2)


if __name__ == "__main__":
    # Based on the batch file WorkingFiles/batchFiles/CRA/ccfSaw.bch

    # Note the following should have a batch file subtracted from them, but
    # I'm waiting for clarifation on what that means.
    # Also, the original files are missing, so I arbitrarily selected files
    # from the same directory...
    base_path = pathlib.Path("../datasets/WorkingFiles")
    c = base_path.joinpath(
        "NoMGT_2022/Total3Run_IC_10_NoMGT_2022_V20210420.tif"
    )
    d = base_path.joinpath(
        "NoMGT_2022/Total3Run_SP_15_NoMGT_2022_V20210420.tif"
    )
    f = base_path.joinpath(
        "NoMGT_2022/Total3Run_RF_15_NoMGT_2022_V20210420.tif"
    )
    p = base_path.joinpath(
        "NoMGT_2022/Total3Run_RCON4_Alt1_MC_GF_150_2022_V20210416.tif"
    )

    # These parameters should probably be cleaned up and placed in variables.
    cp = arithmetic(c, 0.001801, "*")
    dp = arithmetic(d, 0.001902, "*")
    fp = arithmetic(f, 0.001882, "*")
    pp = arithmetic(p, 0.001801, "*")

    saw1 = arithmetic(cp, dp, "+")

    saw2 = arithmetic(saw1, fp, "+")
    t1 = arithmetic(saw2, pp, "+")

    # I probably need to know more about how this works.
    # ccfSaw=REMAP(t1;-100000:0.0001:0)
    time1 = time.time()
    t1.compute()
    time2 = time.time()
    print(
        "Dask compute function took {:.3f} ms".format((time2 - time1) * 1000.0)
    )
