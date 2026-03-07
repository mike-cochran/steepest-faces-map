import pystac_client
import planetary_computer
import rasterio
import numpy as np
import csv
from scipy.ndimage import map_coordinates
import matplotlib.pyplot as plt
import os
import requests
from rasterio.merge import merge

# --- CONFIGURATION PARAMETERS ---
SEARCH_REGIONS = [
    {'name': 'Karakoram', 'bbox': [73.5, 35.0, 77.5, 36.5]},
    {'name': 'Nepal Himalaya', 'bbox': [80.0, 27.0, 88.5, 29.5]},
]
MIN_PEAK_ELEVATION_M = 6000              # Minimum elevation of peaks to fetch
GEONAMES_USERNAME = 'mjcochran16'        # GeoNames API username (free account at geonames.org)
NUM_DIRECTIONS = 8                       # Number of radial directions to check per peak (e.g. 8 = 45 deg apart)
MAX_RADIUS_KM = 3                       # Maximum horizontal radial distance to search for a drop (km)
MIN_DROP_M = 1000                        # Minimum required vertical drop elevation (meters)
OUTPUT_CSV = 'himalaya_steepest_faces.csv'

def download_file(url, local_path):
    if not os.path.exists(local_path):
        print(f"Downloading {url} to {local_path}...")
        session = requests.Session()
        adapter = requests.adapters.HTTPAdapter(max_retries=3)
        session.mount('http://', adapter)
        session.mount('https://', adapter)
        
        with session.get(url, stream=True, timeout=30) as r:
            r.raise_for_status()
            with open(local_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
    else:
        try:
            with rasterio.open(local_path) as src:
                _ = src.profile
        except Exception:
            print(f"Corrupt file {local_path}, redownloading...")
            os.remove(local_path)
            download_file(url, local_path)
    return local_path

def analyze_radial_profiles(src, arr, peak_name, start_lon, start_lat, num_directions=16, max_radius_km=10, min_drop_m=1000):
    transform = src.transform
    
    try:
        start_row, start_col = src.index(start_lon, start_lat)
    except ValueError:
        return []
    
    if not (0 <= start_row < arr.shape[0] and 0 <= start_col < arr.shape[1]):
        return []

    start_elev = arr[start_row, start_col]
    safe_name = peak_name.encode('ascii', 'replace').decode('ascii')
    print(f"Analyzing {safe_name} (Elevation at pixel: {start_elev:.0f}m)")
    
    lat_rad = np.radians(start_lat)
    m_per_deg_lat = 111132.0
    m_per_deg_lon = 111319.0 * np.cos(lat_rad)
    
    step_m = 10.0
    max_steps = int(max_radius_km * 1000 / step_m)
    
    angles_deg = np.linspace(0, 360, num_directions, endpoint=False)
    results = []
    inv_transform = ~transform
    
    for angle_deg in angles_deg:
        angle_rad = np.radians(angle_deg)
        t = np.arange(max_steps)
        dist_m = t * step_m
        
        dy_m = dist_m * np.cos(angle_rad)
        dx_m = dist_m * np.sin(angle_rad)
        
        d_lat = dy_m / m_per_deg_lat
        d_lon = dx_m / m_per_deg_lon
        
        lat_coords = start_lat + d_lat
        lon_coords = start_lon + d_lon
        
        col_coords, row_coords = inv_transform * (lon_coords, lat_coords)
        
        profile_elevs = map_coordinates(np.nan_to_num(arr, nan=-9999), np.vstack((row_coords, col_coords)), order=1, mode='nearest')
        profile_elevs[profile_elevs <= -9990] = np.nan
        
        best_gradient = 0
        best_segment = None
        
        for j in range(len(profile_elevs)):
            if np.isnan(profile_elevs[j]): continue
            
            future_elevs = profile_elevs[j:]
            drops = profile_elevs[j] - future_elevs
            
            valid_drops = np.where(~np.isnan(drops) & (drops >= min_drop_m))[0]
            if len(valid_drops) > 0:
                k = valid_drops[0] + j
                dist = dist_m[k] - dist_m[j]
                if dist > 0:
                    gradient = drops[valid_drops[0]] / dist
                    if gradient > best_gradient:
                        best_gradient = gradient
                        best_segment = (dist_m[j], dist_m[k], profile_elevs[j], profile_elevs[k], drops[valid_drops[0]])
        
        direction_name = f"{angle_deg:.0f} deg"
        if angle_deg == 0: direction_name = "N"
        elif angle_deg == 45: direction_name = "NE"
        elif angle_deg == 90: direction_name = "E"
        elif angle_deg == 135: direction_name = "SE"
        elif angle_deg == 180: direction_name = "S"
        elif angle_deg == 225: direction_name = "SW"
        elif angle_deg == 270: direction_name = "W"
        elif angle_deg == 315: direction_name = "NW"
        
        if best_segment:
            start_d, end_d, start_e, end_e, total_drop = best_segment
            start_idx = int(start_d / step_m)
            min_elev_so_far = profile_elevs[start_idx]
            for k_idx in range(start_idx, len(profile_elevs)):
                if np.isnan(profile_elevs[k_idx]): continue
                if profile_elevs[k_idx] < min_elev_so_far:
                    min_elev_so_far = profile_elevs[k_idx]
                elif profile_elevs[k_idx] > min_elev_so_far + 150:
                    break
            total_face_height = start_elev - min_elev_so_far
            
            # calculate lat/lon for the face itself (midpoint)
            t_mid = (start_d + end_d) / 2.0
            lat_mid = start_lat + (t_mid * np.cos(angle_rad)) / m_per_deg_lat
            lon_mid = start_lon + (t_mid * np.sin(angle_rad)) / m_per_deg_lon

            results.append({
                'peak': peak_name,
                'direction': direction_name,
                'gradient': best_gradient,
                'drop': total_drop,
                'dist_m': end_d - start_d,
                'start_offset_m': start_d,
                'end_offset_m': end_d,
                'summit_elev': start_elev,
                'start_elev': start_e,
                'end_elev': end_e,
                'total_face_height': total_face_height,
                'latitude': lat_mid,
                'longitude': lon_mid
            })
    
    return results

def fetch_peaks_for_bbox(bbox):
    """Fetch peaks from GeoNames + OSM for a single bounding box."""
    raw_peaks = []
    
    # --- GeoNames ---
    geonames_base_url = "http://api.geonames.org/searchJSON"
    for fc in ['PK', 'MT']:
        start_row = 0
        while True:
            params = {
                'featureCode': fc, 'north': bbox[3], 'south': bbox[1],
                'east': bbox[2], 'west': bbox[0],
                'maxRows': 1000, 'startRow': start_row,
                'username': GEONAMES_USERNAME, 'style': 'FULL'
            }
            try:
                response = requests.get(geonames_base_url, params=params, timeout=30)
                response.raise_for_status()
                data = response.json()
            except Exception as e:
                print(f"    GeoNames warning ({fc}): {e}")
                break
            results = data.get('geonames', [])
            if not results:
                break
            for entry in results:
                try:
                    ele_val = float(entry.get('elevation', 0) or entry.get('srtm3', 0) or 0)
                    if ele_val >= MIN_PEAK_ELEVATION_M:
                        raw_peaks.append({'name': entry.get('name', 'Unnamed Peak'),
                                        'lon': float(entry['lng']), 'lat': float(entry['lat']),
                                        'ele': ele_val, 'source': 'geonames'})
                except (ValueError, TypeError):
                    pass
            total = data.get('totalResultsCount', 0)
            start_row += 1000
            if start_row >= total:
                break
    
    gn_count = len(raw_peaks)
    
    # --- OSM Overpass ---
    overpass_query = f"""
    [out:json][timeout:90];
    node["natural"="peak"]({bbox[1]},{bbox[0]},{bbox[3]},{bbox[2]});
    out body;
    """
    osm_count = 0
    try:
        response = requests.post("http://overpass-api.de/api/interpreter",
                                data={'data': overpass_query}, timeout=90)
        response.raise_for_status()
        for element in response.json().get('elements', []):
            tags = element.get('tags', {})
            ele_str = tags.get('ele', '')
            if ele_str:
                try:
                    ele_val = float(''.join(c for c in ele_str if c.isdigit() or c == '.'))
                    if ele_val >= MIN_PEAK_ELEVATION_M:
                        name = tags.get('name', tags.get('name:en', 'Unnamed Peak'))
                        raw_peaks.append({'name': name, 'lon': element['lon'], 'lat': element['lat'],
                                        'ele': ele_val, 'source': 'osm'})
                        osm_count += 1
                except ValueError:
                    pass
    except Exception as e:
        print(f"    OSM warning: {e}")
    
    print(f"    GeoNames: {gn_count}, OSM: {osm_count}, Total: {len(raw_peaks)}")
    return raw_peaks

def main():
    from rasterio.vrt import WarpedVRT
    
    catalog = pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=planetary_computer.sign_inplace,
    )
    
    os.makedirs("data", exist_ok=True)
    
    # Download DEM tiles for all regions
    all_dem_files = []
    for region in SEARCH_REGIONS:
        print(f"\nSearching for DEM tiles for {region['name']}...")
        search = catalog.search(
            collections=["cop-dem-glo-30"],
            bbox=region['bbox'],
        )
        items = list(search.items())
        print(f"  Found {len(items)} tiles. Downloading...")
        
        for item in items:
            url = item.assets["data"].href
            local_path = f"data/{item.id}.tif"
            download_file(url, local_path)
            if local_path not in all_dem_files:
                all_dem_files.append(local_path)
    
    print(f"\nTotal DEM tiles across all regions: {len(all_dem_files)}")
    
    # Create a VRT (virtual raster) instead of merging all tiles
    vrt_path = "data/all_regions.vrt"
    if not os.path.exists(vrt_path) or True:  # Always rebuild VRT
        print("Building virtual raster (VRT) from all DEM tiles...")
        from rasterio.merge import merge as rasterio_merge
        import subprocess
        # Use gdalbuildvrt if available, otherwise fall back to rasterio merge
        try:
            cmd = ['gdalbuildvrt', vrt_path] + all_dem_files
            subprocess.run(cmd, check=True, capture_output=True)
            print(f"  VRT created at {vrt_path}")
        except (FileNotFoundError, subprocess.CalledProcessError):
            # gdalbuildvrt not available, fall back to rasterio merge
            print("  gdalbuildvrt not found, falling back to rasterio merge...")
            print("  This may use significant memory for large regions.")
            vrt_path = None  # Signal to use direct merge
    
    # If VRT failed, merge tiles directly
    if vrt_path is None or not os.path.exists(vrt_path):
        merged_path = "data/merged_dem.tif"
        if not os.path.exists(merged_path):
            print("Merging DEM tiles (this may take a while)...")
            src_files = []
            for fp in all_dem_files:
                try:
                    src_files.append(rasterio.open(fp))
                except Exception as e:
                    print(f"  Failed to open {fp}: {e}")
            if src_files:
                mosaic, out_trans = merge(src_files)
                out_meta = src_files[0].meta.copy()
                out_meta.update({"driver": "GTiff", "height": mosaic.shape[1],
                                "width": mosaic.shape[2], "transform": out_trans})
                with rasterio.open(merged_path, "w", **out_meta) as dest:
                    dest.write(mosaic)
                for s in src_files:
                    s.close()
        vrt_path = merged_path
    # ========================================================
    # Fetch peaks from all regions using multiple sources
    # ========================================================
    raw_peaks = []
    for region in SEARCH_REGIONS:
        print(f"\nFetching peaks for {region['name']}...")
        region_peaks = fetch_peaks_for_bbox(region['bbox'])
        raw_peaks.extend(region_peaks)
    
    print(f"\nCombined raw total: {len(raw_peaks)} peaks (before dedup)")
                
    raw_peaks.sort(key=lambda x: x['ele'], reverse=True)
    
    # Deduplicate peaks that are too close (within ~2km)
    peaks = []
    for p in raw_peaks:
        too_close = False
        for up in peaks:
            dist = np.sqrt((p['lon']-up['lon'])**2 + (p['lat']-up['lat'])**2)
            if dist < 0.02: # ~2km
                too_close = True
                break
        if not too_close:
            peaks.append(p)
            
    print(f"Found {len(peaks)} independent peaks >{MIN_PEAK_ELEVATION_M}m to analyze.")
    
    all_faces = []
    
    if os.path.exists(vrt_path):
        print("Loading DEM into memory for analysis...")
        with rasterio.open(vrt_path) as src:
            arr = src.read(1)
            arr = arr.astype(np.float32)
            arr[arr == src.nodata] = np.nan
            
            total_peaks = len(peaks)
            for i, p in enumerate(peaks):
                safe_name = p['name'].encode('ascii', 'replace').decode('ascii')
                print(f"  [{i+1}/{total_peaks}] Analyzing {safe_name}...")
                faces = analyze_radial_profiles(
                    src,
                    arr,
                    p['name'], 
                    p['lon'], 
                    p['lat'], 
                    num_directions=NUM_DIRECTIONS, 
                    max_radius_km=MAX_RADIUS_KM, 
                    min_drop_m=MIN_DROP_M
                )
                all_faces.extend(faces)
    else:
        print("DEM path doesn't exist, cannot analyze right now.")
    
    if not all_faces:
        print(f"No faces >{MIN_DROP_M}m drop found from these peaks.")
        return
        
    all_faces.sort(key=lambda f: f['gradient'], reverse=True)
    
    csv_file = OUTPUT_CSV
    print(f"\nExporting {len(all_faces)} results to {csv_file}...")
    with open(csv_file, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'rank', 'peak', 'direction', 'gradient', 'drop', 'horizontal_dist', 
            'start_offset_m', 'end_offset_m', 'summit_elev', 'start_elev', 'end_elev', 'total_face_height',
            'latitude', 'longitude'
        ])
        writer.writeheader()
        for i, face in enumerate(all_faces):
            writer.writerow({
                'rank': i + 1,
                'peak': face['peak'],
                'direction': face['direction'],
                'gradient': round(face['gradient'], 3),
                'drop': round(face['drop'], 1),
                'horizontal_dist': round(face['dist_m'], 1),
                'start_offset_m': round(face['start_offset_m'], 1),
                'end_offset_m': round(face['end_offset_m'], 1),
                'summit_elev': round(face['summit_elev'], 1),
                'start_elev': round(face['start_elev'], 1),
                'end_elev': round(face['end_elev'], 1),
                'total_face_height': round(face['total_face_height'], 1),
                'latitude': round(face['latitude'], 6),
                'longitude': round(face['longitude'], 6)
            })
    print(f"Results successfully saved to {csv_file}")
    
    region_names = ' + '.join(r['name'] for r in SEARCH_REGIONS)
    print(f"\n--- TOP 10 STEEPEST FACES ({region_names}) ---")
    for i, f in enumerate(all_faces[:10]):
        print(f"Rank {i+1}: {f['peak']} facing {f['direction']}")
        print(f"  Gradient : {f['gradient']:.2f}")
        print(f"  Steepest 2km Drop : {f['drop']:.0f} m over {f['dist_m']:.0f} m horizontal distance")
        print(f"  Total Face Height : ~{f['total_face_height']:.0f} m (from top of face to valley floor)")
        print()

if __name__ == "__main__":
    main()
