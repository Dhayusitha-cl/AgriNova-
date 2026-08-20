import ee


PROJECT_ID = "project-3dc0f771-1142-477c-9b2"


def initialize_earth_engine():
    """Initialize Google Earth Engine."""
    try:
        ee.Initialize(project=PROJECT_ID)
        return True
    except Exception as e:
        print("Earth Engine initialization failed:", e)
        return False


def get_soil_moisture(latitude, longitude):
    """
    Get soil moisture information from ERA5-Land
    for the requested location.
    """

    try:
        initialize_earth_engine()

        point = ee.Geometry.Point([longitude, latitude])

        dataset = (
            ee.ImageCollection("ECMWF/ERA5_LAND/DAILY_AGGR")
            .filterBounds(point)
            .sort("system:time_start", False)
            .first()
        )

        moisture = dataset.select(
            "volumetric_soil_water_layer_1"
        ).reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=point,
            scale=10000
        ).getInfo()

        value = moisture.get(
            "volumetric_soil_water_layer_1"
        )

        if value is None:
            return None

        return float(value)

    except Exception as e:
        print("Earth Engine data error:", e)
        return None