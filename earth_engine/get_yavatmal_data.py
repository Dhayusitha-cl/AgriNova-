import ee
import json

PROJECT_ID = "project-3dc0f771-1142-477c-9b2"

ee.Initialize(project=PROJECT_ID)

# Yavatmal, Maharashtra
yavatmal = ee.Geometry.Point([78.12, 20.39])
region = yavatmal.buffer(10000)

# Sentinel-2 Surface Reflectance
collection = (
    ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
    .filterBounds(region)
    .filterDate("2026-06-01", "2026-08-20")
    .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 30))
)

image_count = collection.size().getInfo()

print("Satellite images found:", image_count)

if image_count > 0:

    image = collection.sort("system:time_start", False).first()

    date = (
        ee.Date(image.get("system:time_start"))
        .format("YYYY-MM-dd")
        .getInfo()
    )

    # NDVI = vegetation condition indicator
    ndvi = image.normalizedDifference(["B8", "B4"]).rename("NDVI")

    ndvi_value = ndvi.reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=region,
        scale=10,
        maxPixels=1e9
    ).get("NDVI").getInfo()

    result = {
        "district": "Yavatmal",
        "state": "Maharashtra",
        "satellite": "Sentinel-2",
        "image_date": date,
        "ndvi": ndvi_value,
        "source": "Google Earth Engine"
    }

    with open("earth_engine/yavatmal_data.json", "w") as f:
        json.dump(result, f, indent=4)

    print("\nEarth Engine data retrieved successfully!")
    print(json.dumps(result, indent=4))

else:
    print("No suitable satellite images found.")