import geopandas as gpd
import contextily as ctx
import matplotlib.pyplot as plt
import koreanize_matplotlib


def main():
    rp_gdf = gpd.read_file('./shape/실시간 도시 데이터 집계구역.shp')
    rp_gdf = rp_gdf.to_crs(epsg=3857)
    lp_gdf = gpd.read_file('./shape/집계구.shp')
    lp_gdf = lp_gdf.to_crs(epsg=3857)
    lp_gdf['area_total'] = lp_gdf.geometry.area
    intersection = gpd.overlay(rp_gdf, lp_gdf, how='intersection', keep_geom_type=True)
    intersection['area_section'] = intersection.geometry.area
    intersection['rate'] = intersection['area_section']/intersection['area_total']
    final_df = intersection[['AREA_NM', 'TOT_REG_CD', 'ADM_NM', 'rate']]
    final_df.to_csv('Rate_v2.csv', index=0)
    
if __name__ == "__main__":
    main()