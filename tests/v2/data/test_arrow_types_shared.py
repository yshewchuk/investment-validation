"""One physical-type map: the generic writers re-export ``query``'s."""
import pyarrow as pa

from engine.v2.data import generic_incremental, objects, query


def test_generic_incremental_reexports_query_arrow_types():
    assert generic_incremental.ARROW_TYPES is query.ARROW_TYPES


def test_arrow_type_for_maps_date32():
    assert query.arrow_type_for("date32") == pa.date32()


def test_null_array_for_date32_is_typed():
    array = query.null_array_for("date32", 2)
    assert array.type == pa.date32()
    assert len(array) == 2
    assert array.null_count == 2


def test_date32_parquet_type_normalizes():
    assert objects.normalize_physical_type(str(pa.date32())) == "date32"