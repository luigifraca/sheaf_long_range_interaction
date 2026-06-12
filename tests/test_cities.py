from slri.datasets.cities import CITY_METADATA


def test_city_metadata_matches_pyg_release():
    assert CITY_METADATA["paris"] == {
        "nodes": 114_127,
        "edges": 182_511,
        "features": 37,
        "classes": 10,
        "split": "10%/10%/80%",
    }
    assert CITY_METADATA["shanghai"]["features"] == 37
    assert CITY_METADATA["shanghai"]["classes"] == 10

