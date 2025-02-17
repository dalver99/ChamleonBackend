import geopandas as gpd
import pandas as pd
import shapely
from shapely.geometry import Point
import folium

def import_geo_data():
    ta_gdf = gpd.read_file('./geospatial_util/shape_data/대상지역 경계.shp')
    ta_gdf['centroid'] = ta_gdf.geometry.centroid
    ta_gdf['buffer_1km'] = ta_gdf.geometry.centroid.buffer(1000)
    df = pd.read_csv('./geospatial_util/shape_data/소상공인시장진흥공단_상가(상권)정보_서울_202412.csv')
    geometry = [Point(xy) for xy in zip(df['경도'], df['위도'])]
    ag_gdf = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")
    #ag_gdf = gpd.read_file('./geospatial_util/shape_data/서울시 상권정보 v2.shp')
    umd_gdf = gpd.read_file('./geospatial_util/shape_data/행정구역 경계.shp')
    return ta_gdf, ag_gdf, umd_gdf
    
def visualization_all(ta_gdf, ag_gdf, option=False):
    if option == True:
        # 좌표계 변환
        ta_gdf = ta_gdf.to_crs(epsg=4326)
        ag_gdf = ag_gdf.to_crs(epsg=4326)

        # 중심점 계산
        center = [ta_gdf.centroid.y.mean(), ta_gdf.centroid.x.mean()]

        # 지도 생성
        f = folium.Figure(width=600, height=400)
        m = folium.Map(location=center, zoom_start=12).add_to(f)
        
        # 카테고리별 색상 정의
        color_dict = {
            '교육': '#FF0000',    # 빨강
            '음식': '#00FF00',    # 초록
            '부동산': '#0000FF',   # 파랑
            '소매': '#FF00FF',    # 마젠타
            '수리·개인': '#FFA500', # 주황
            '과학·기술': '#800080', # 보라
            '시설관리·임대': '#008080', # 청록
            '예술·스포츠': '#FFD700', # 골드
            '숙박': '#4B0082',    # 남색
            '보건의료': '#FF1493'  # 진한 분홍
        }

        # Legend 생성
        legend_html = '''
        <div style="position: fixed; 
                    bottom: 25px; right: 25px; 
                    border:1px solid grey; z-index:9999; 
                    background-color:white;
                    padding: 10px;
                    font-size:10px;
                    ">
        <p><strong>상권 유형</strong></p>
        '''
        
        for category in ag_gdf['Category'].unique():
            category_points = ag_gdf[ag_gdf['Category'] == category]
            color = color_dict.get(category, '#000000')
            
            # Legend에 항목 추가
            legend_html += f'''
            <p><i class="fa fa-circle fa-1x" style="color:{color}"></i> {category}</p>
            '''
            
            for idx, row in category_points.iterrows():
                # HTML로 popup 내용 구성
                popup_content = f"""
                <div style="font-size:12px">
                    <b>상권명:</b> {row['Name']}<br>
                    <b>상권유형:</b> {category}
                </div>
                """
                
                folium.CircleMarker(
                    location=[row.geometry.y, row.geometry.x],
                    radius=3,
                    color=color,
                    fill=True,
                    fill_color=color,
                    fill_opacity=0.7,
                    popup=folium.Popup(popup_content, max_width=200),
                    name=category
                ).add_to(m)

        legend_html += '</div>'
        m.get_root().html.add_child(folium.Element(legend_html))
        
        # ta_gdf 데이터 추가 (centroid에 팝업 추가)
        for idx, row in ta_gdf.iterrows():
            folium.Marker(location = [row.geometry.centroid.y, row.geometry.centroid.x],
                        popup=row['AREA_NM']).add_to(m)
            
        folium.GeoJson(
        ta_gdf['buffer_1km'],
        name='1km 버퍼',
        style_function=lambda x: {'color': 'blue', 'fillOpacity': 0.2}
        ).add_to(m)

        # 레이어 컨트롤 추가
        folium.LayerControl().add_to(m)
        return m
