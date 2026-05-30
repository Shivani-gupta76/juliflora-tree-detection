import rasterio
from rasterio.transform import Affine
from shapely.geometry import box, Point
from pyproj import Transformer, Geod

def load_raster(path):
    ds = rasterio.open(path)
    return ds

def pixel_to_coord(dataset, row, col):
    transform = dataset.transform
    x, y = rasterio.transform.xy(transform, row, col, offset="center")
    return x, y

def bbox_pixels_to_coords(dataset, bbox):
    x1, y1, x2, y2 = bbox
    x_tl, y_tl = pixel_to_coord(dataset, int(y1), int(x1))
    x_br, y_br = pixel_to_coord(dataset, int(y2), int(x2))
    minx = min(x_tl, x_br)
    maxx = max(x_tl, x_br)
    miny = min(y_tl, y_br)
    maxy = max(y_tl, y_br)
    return box(minx, miny, maxx, maxy)

def sample_raster_at_coords(dataset, x, y):
    for val in dataset.sample([(x, y)]):
        v = val[0]
        if v == dataset.nodata:
            return None
        return float(v)

def get_elevation_at_bbox_centroid(orth_ds, dsm_ds, dtm_ds, bbox):
    poly = bbox_pixels_to_coords(orth_ds, bbox)
    centroid = poly.centroid
    x, y = centroid.x, centroid.y
    dsm_val = sample_raster_at_coords(dsm_ds, x, y)
    dtm_val = sample_raster_at_coords(dtm_ds, x, y)
    if dsm_val is None or dtm_val is None:
        return {"dsm": dsm_val, "dtm": dtm_val, "height": None, "x": x, "y": y}
    height = dsm_val - dtm_val
    return {"dsm": dsm_val, "dtm": dtm_val, "height": float(height), "x": x, "y": y}

def bbox_real_world_dimensions(orth_ds, bbox):
    poly = bbox_pixels_to_coords(orth_ds, bbox)
    minx, miny, maxx, maxy = poly.bounds
    crs = orth_ds.crs
    if crs.is_geographic:
        geod = Geod(ellps="WGS84")
        lon1, lat1 = minx, (miny + maxy) / 2
        lon2, lat2 = maxx, (miny + maxy) / 2
        width_m = abs(geod.inv(lon1, lat1, lon2, lat2)[2])
        lon3, lat3 = (minx + maxx) / 2, miny
        lon4, lat4 = (minx + maxx) / 2, maxy
        height_m = abs(geod.inv(lon3, lat3, lon4, lat4)[2])
    else:
        width_m = maxx - minx
        height_m = maxy - miny
    area_m2 = width_m * height_m
    return {"width_m": float(width_m), "height_m": float(height_m), "area_m2": float(area_m2), "bounds": [minx, miny, maxx, maxy]}
