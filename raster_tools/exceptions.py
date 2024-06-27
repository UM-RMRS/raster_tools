class RasterToolsError(Exception):
    """The base exception for errors in raster_tools."""


class NoDataError(RasterToolsError):
    """Raised when an operation produces a raster with no data."""


# TODO: Deprecate and eventually remove
RasterNoDataError = NoDataError


class FileParseError(RasterToolsError):
    """Raised if an error occurs while parsing a non-raster file."""


# TODO: Deprecate and eventually remove
RemapFileParseError = FileParseError


class RasterIOError(RasterToolsError):
    """Raised if an error occurs when trying to read a raster file."""


class RasterDataError(RasterToolsError):
    """Raised if there is a problem with the data in a raster."""


class AffineEncodingError(RasterToolsError):
    """
    Raised if there is a problem getting the affine matrix from a dataset.
    """


class DimensionsError(RasterToolsError):
    """Raised if there is a problem with the dimensions in a dataset."""


class BatchScriptParseError(FileParseError):
    """Raised if an error occurs while parsing a batch script."""


class VectorError(RasterToolsError):
    """The base exception for errors dealing with vectors."""
