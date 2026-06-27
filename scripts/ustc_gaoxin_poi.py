import json
import requests
from shapely.geometry import Point, Polygon
from tqdm import tqdm

AMAP_KEY = "731df7f33e6cff66bcf24378ce6a2b05"

# ✅ 中科大高新校区（手动精确围栏）
CAMPUS_POLYGON = Polygon([
    [117.1358, 31.8265],
    [117.1410, 31.8265],
    [117.1410, 31.8325],
    [117.1358, 31.8325],
    [117.1358, 31.8265],
])

# ✅ 校内常见 POI 类型（收敛到几十个）
POI_TYPES = [
    "科教文化服务",
    "餐饮服务",
    "购物服务",
    "体育休闲服务",
    "医疗保健",
    "公共设施",
]

CENTER_LNG = 117.1380
CENTER_LAT = 31.8295
RADIUS = 1500


def fetch_poi(poi_type):
    pois = []
    page = 1
    while True:
        r = requests.get("https://restapi.amap.com/v3/place/around", params={
            "key": AMAP_KEY,
            "location": f"{CENTER_LNG},{CENTER_LAT}",
            "radius": RADIUS,
            "types": poi_type,
            "offset": 25,
            "page": page,
            "extensions": "base"
        }).json()

        if r["status"] != "1" or not r["pois"]:
            break

        for p in r["pois"]:
            lng, lat = map(float, p["location"].split(","))
            if CAMPUS_POLYGON.contains(Point(lng, lat)):
                pois.append({
                    "name": p["name"],
                    "type": poi_type,
                    "address": p.get("address", ""),
                    "lat": lat,
                    "lng": lng,
                })

        page += 1
    return pois


if __name__ == "__main__":
    all_pois = []
    for t in tqdm(POI_TYPES):
        all_pois.extend(fetch_poi(t))

    # 去重
    seen = set()
    unique_pois = []
    for p in all_pois:
        key = (p["name"], p["lat"], p["lng"])
        if key not in seen:
            seen.add(key)
            unique_pois.append(p)

    with open("ustc_gaoxin_campus_poi.json", "w", encoding="utf-8") as f:
        json.dump(unique_pois, f, ensure_ascii=False, indent=2)

    print(f"校内 POI 数量: {len(unique_pois)}")