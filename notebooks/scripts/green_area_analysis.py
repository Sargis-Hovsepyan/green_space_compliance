import cv2
import rasterio

import numpy as np
import pandas as pd
import geopandas as gpd

import detectree as dtr
import positron as pt

from io import BytesIO

from rasterio.features import rasterize
from rasterio.io import MemoryFile
from rasterio.mask import mask

from shapely.geometry import Point

from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm.auto import tqdm

from skimage.exposure import match_histograms


def read_image_from_path(img_path):
    with rasterio.open(img_path) as src:
        img = src.read([1, 2, 3])  # Read RGB bands only
        transform = src.transform
        crs = src.crs
    return np.moveaxis(img, 0, -1), transform, crs  # (C, H, W) → (H, W, C)

def filter_grid_data(index, grid, buildings, roads_gdf):
    tile = grid.loc[grid['Index'] == index]
    blds = buildings.loc[buildings['img_id'] == index]
    roads = roads_gdf.sjoin(tile, predicate='intersects').drop(columns='index_right', errors='ignore')
    return tile, blds, roads

def paint_buildings_and_roads_white(img, buildings_gdf, roads_gdf, transform, road_buffer=0.00002):
    all_geoms = list(buildings_gdf.geometry)  # optionally: + list(roads_gdf.geometry.buffer(road_buffer))
    mask = rasterize(
        [(geom, 1) for geom in all_geoms if geom and not geom.is_empty],
        out_shape=img.shape[:2],
        transform=transform,
        fill=0,
        dtype='uint8'
    )
    result = img.copy()
    result[mask == 1] = 255
    return result

def detect_trees_with_canopy_estimation(img, transform):
    # ----- Subfunctions -----
    def compute_exg(img):
        # Vectorized Excess Green calculation
        r, g, b = img[..., 0], img[..., 1], img[..., 2]
        return 2 * g.astype(np.int16) - r - b

    # ----- Model Inference -----
    _, encoded = cv2.imencode('.jpg', cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    classifier = dtr.Classifier()
    y_pred = classifier.predict_img(BytesIO(encoded))  # Get float32 prob mask
    pred_mask = (y_pred > 0.85).astype(np.uint8)

    # ----- ExG Thresholding -----
    exg_mask = (compute_exg(img) > 20).astype(np.uint8)

    # ----- Combine and Clean Mask -----
    combined = cv2.bitwise_and(pred_mask, exg_mask)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    cleaned = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel, iterations=2)

    # ----- Connected Components & Area Filtering -----
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(cleaned, connectivity=8)
    canopy_data = [(stats[i, cv2.CC_STAT_AREA], centroids[i]) for i in range(1, num_labels) if stats[i, cv2.CC_STAT_AREA] >= 120]

    if canopy_data:
        areas, centers = zip(*canopy_data)
        avg_area = np.median(areas)
        est_trees = int(round(sum(areas) / avg_area))
    else:
        centers = []
        est_trees = 0

    # ----- Convert to Geo Points -----
    latlon_points = [
        Point(*rasterio.transform.xy(transform, int(cy), int(cx)))
        for cx, cy in centers
    ]

    # ----- Visualization -----
    marked = img.copy()
    for cx, cy in centers:
        cv2.circle(marked, (int(cx), int(cy)), 15, (0, 255, 0), -1)

    return est_trees, marked, latlon_points


def mask_image_by_geometry(image, transform, geometry):
    """
    Efficiently clips the image to a given geometry (e.g., a buffer zone).

    Parameters:
    - image: np.ndarray, shape (bands, height, width)
    - transform: Affine
    - geometry: shapely.geometry.Polygon or MultiPolygon

    Returns:
    - clipped_image: np.ndarray, masked image (bands, H', W')
    - clipped_transform: Affine transform of the clipped image
    """
    num_bands, height, width = image.shape
    geometry_geojson = [geometry.__geo_interface__]  # avoids overhead of mapping()

    with MemoryFile() as memfile:
        with memfile.open(
            driver='GTiff',
            height=height,
            width=width,
            count=num_bands,
            dtype=image.dtype,
            transform=transform,
            crs='EPSG:4326'  # update to match your actual CRS
        ) as dataset:
            dataset.write(image)  # write all bands at once

            clipped_image, clipped_transform = mask(
                dataset, 
                geometry_geojson, 
                crop=True,
                all_touched=False,  # faster and cleaner edge, can be set True for trees if needed
                filled=True
            )

    return clipped_image, clipped_transform

def add_nearest_parks(parks, buildings, radius=300):
    distances = []

    # Ensure all GeoDataFrames are in the same projection (WGS 84)
    parks = parks.to_crs(epsg=4326)
    buildings = buildings.to_crs(epsg=4326)

    for idx, row in buildings.iterrows():
        # Create buffer around centroid
        centroid = row.geometry.centroid
        buffer = centroid.buffer(radius)

        # Wrap buffer in a GeoDataFrame
        buffer_gdf = gpd.GeoDataFrame(geometry=[buffer], crs='EPSG:4326')

        # Spatial join to get parks within buffer
        nearby_parks = gpd.sjoin(parks, buffer_gdf, predicate='intersects')

        if not nearby_parks.empty:
            # Use your routing function to get distance from building to each nearby park
            travel_info = pt.single_source_travel_info_keyless(
                centroid, 
                nearby_parks, 
                base_url='https://routing.homenas.us', 
                mode='pedestrian'
            )
            travel_info['building_id'] = row['building_id']
            distances.append(travel_info)

    if distances:
        return pd.concat(distances, ignore_index=True)
    else:
        return pd.DataFrame()  # Return empty DataFrame if no distances

def annotate_buildings_with_nearby_parks(parks, buildings, radius=300):
    # Reproject both to meters for distance calculations
    parks = parks.to_crs(epsg=3857)
    buildings = buildings.to_crs(epsg=3857)

    # Add centroids and buffers
    buildings["centroid"] = buildings.geometry.centroid
    buildings["buffer"] = buildings.centroid.buffer(radius)
    buildings["has_park_within_radius"] = False

    records = []

    for idx, row in tqdm(buildings.iterrows(), total=len(buildings)):
        buffer_geom = gpd.GeoDataFrame([[row["building_id"], row["buffer"]]],
                                       columns=["building_id", "geometry"],
                                       crs=buildings.crs)

        nearby_parks = gpd.sjoin(parks, buffer_geom, predicate="intersects")

        if not nearby_parks.empty:
            buildings.at[idx, "has_park_within_radius"] = True

            # Find nearest park among those within radius
            nearby_parks["distance"] = nearby_parks.geometry.centroid.distance(row["centroid"])
            nearest_park = nearby_parks.loc[nearby_parks["distance"].idxmin()]

            records.append({
                "building_id": row["building_id"],
                "img_id": row.get("img_id", None),
                "nearest_park_name": nearest_park.get("name", None)
            })

    # Clean up
    buildings = buildings.drop(columns=["centroid", "buffer"])
    buildings = buildings.to_crs(epsg=4326)

    records_df = pd.DataFrame(records)
    return buildings, records_df

def histogram_match_image(geotiff_path, reference_path):
    """
    Matches the histogram of a georeferenced image to a reference image.
    
    Parameters:
        geotiff_path (str): Path to the georeferenced GeoTIFF image.
        reference_path (str): Path to the reference RGB image (e.g., JPEG or PNG).

    Returns:
        matched (np.ndarray): Histogram-matched image as a (H, W, 3) RGB array.
        transform (Affine): Affine transform of the original GeoTIFF image.
    """
    # Read georeferenced image
    with rasterio.open(geotiff_path) as src:
        img = src.read([1, 2, 3])  # shape: (3, H, W)
        transform = src.transform  # capture transform
        img = np.transpose(img, (1, 2, 0))  # to (H, W, 3)

        # Normalize if values are not in 0-255
        if img.max() > 255:
            img = (img / img.max()) * 255
        img = img.astype(np.uint8)

    # Read reference image
    reference = cv2.cvtColor(cv2.imread(reference_path), cv2.COLOR_BGR2RGB)

    # Match histograms
    matched = match_histograms(img, reference, channel_axis=-1)

    return matched, transform


################## THERADED TREE DETECTION ###################

def detect_all_trees(ids, grid, buildings, edges, detect_func, max_workers=8):
    """
    Runs tree detection around buildings in parallel using threads.
    
    Parameters:
        ids (array-like): List of tile indexes.
        grid, buildings, edges (GeoDataFrames): Data inputs.
        detect_func (function): A function like detect_trees_around_buildings.
        max_workers (int): Number of threads.
    
    Returns:
        merged_trees_gdf, merged_buildings_gdf (GeoDataFrames)
    """

    def process_tile(tile_index):
        try:
            trees_gdf, blds, _, _ = detect_func(tile_index, grid, buildings, edges)
            return trees_gdf, blds
        except Exception as e:
            print(f"Error processing tile {tile_index}: {e}")
            return None, None

    all_tree_detections = []
    all_buildings = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_tile, idx) for idx in ids]

        for future in tqdm(as_completed(futures), total=len(futures), desc='Images Processed', dynamic_ncols=True):
            trees_gdf, blds = future.result()
            if trees_gdf is not None and blds is not None:
                all_tree_detections.append(trees_gdf)
                all_buildings.append(blds)

    merged_trees_gdf = gpd.GeoDataFrame(pd.concat(all_tree_detections, ignore_index=True), crs=grid.crs)
    merged_buildings_gdf = gpd.GeoDataFrame(pd.concat(all_buildings, ignore_index=True), crs=grid.crs)

    return merged_trees_gdf, merged_buildings_gdf