import rasterio

mosaic_path = "../../data/processed/university_city_pilot_mosaic.tif"

with rasterio.open(mosaic_path) as src:
    print("CRS:", src.crs)
    print("Width:", src.width)
    print("Height:", src.height)
    print("Bounds:", src.bounds)
    print("Resolution:", src.res)
    print("Count:", src.count)
    print("Dtype:", src.dtypes)