#!/usr/bin/env python3
import sys
import os
import json
import rospy
import requests
from openai import OpenAI
from sensor_msgs.msg import NavSatFix
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped
import math
# ================= 添加路径，确保能找到 coordTransform_utils =================
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)

# ================= 导入开源项目的坐标转换函数 =================
try:
    from coordTransform_utils import wgs84_to_gcj02
except ImportError as e:
    rospy.logerr(f"无法导入 coordTransform_utils: {e}")
    rospy.logerr("请确保 coordTransform_utils.py 文件在以下目录：")
    rospy.logerr(f"  {script_dir}")
    sys.exit(1)

# ================= 1. 配置 =================
POI_PATH = "/home/robot/nav_ws/src/pathplanning/scripts/ustc_gaoxin_poi.json"
AMAP_KEY = "731df7f33e6cff66bcf24378ce6a2b05"

# ================= 2. 加载 POI =================
with open(POI_PATH, "r", encoding="utf-8") as f:
    pois = json.load(f)

# ================= 3. LLM 客户端 =================
client = OpenAI(
    api_key="sk-5e6f579ceb024b11ad9212ea13344c8a",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

# ================= 4. 全局状态 =================
start_lng = None
start_lat = None
gps_received = False
navigation_started = False

path_pub = None


def ask_llm(user_instruction: str) -> str:
    prompt = f"""
你是一个智能导航助手。
下面是可选目的地列表（包含名称、类别、经纬度）：

{json.dumps(pois, ensure_ascii=False, indent=2)}

用户指令如下：
\"\"\"{user_instruction}\"\"\"

请严格按以下规则执行：
1. 结合“type”和“name"字段判断用户意图主要是name,同时参考type。
2. 选出 3 个最相关的候选目的地，按相关性排序。
3. 只输出编号，例如：
0, 3, 7
"""
    resp = client.chat.completions.create(
        model="qwen3.7-max",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )
    return resp.choices[0].message.content.strip()


def get_walking_route(origin, destination):
    url = "https://restapi.amap.com/v3/direction/walking"
    params = {
        "key": AMAP_KEY,
        "origin": origin,
        "destination": destination,
        "output": "json"
    }
    r = requests.get(url, params=params).json()
    if r["status"] != "1":
        raise RuntimeError(f"路径规划失败: {r.get('info')}")

    steps = r["route"]["paths"][0]["steps"]
    points = []
    for step in steps:
        for seg in step["polyline"].split(";"):
            lng, lat = map(float, seg.split(","))
            points.append((lng, lat))
    return points


def haversine_distance(lng1, lat1, lng2, lat2):
    """计算两点间距离（米），使用 Haversine 公式"""
    R = 6371000  # 地球平均半径（米）
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lng2 - lng1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c


def interpolate_path(points, max_interval=5.0):
    """对路径点进行加密：若相邻两点距离 > max_interval 米，则线性插值补点"""
    if len(points) < 2:
        return points

    new_points = [points[0]]
    for i in range(1, len(points)):
        lng1, lat1 = points[i - 1]
        lng2, lat2 = points[i]

        dist = haversine_distance(lng1, lat1, lng2, lat2)

        if dist <= max_interval:
            new_points.append((lng2, lat2))
        else:
            n = int(math.ceil(dist / max_interval))  # 需要插入的段数
            for j in range(1, n + 1):
                frac = j / n
                interp_lng = lng1 + (lng2 - lng1) * frac
                interp_lat = lat1 + (lat2 - lat1) * frac
                new_points.append((interp_lng, interp_lat))

    return new_points


# ================= 5. GPS 回调（只接收一次） =================
def gps_callback(msg: NavSatFix):
    global start_lng, start_lat, gps_received, navigation_started

    if gps_received:
        return

    if math.isnan(msg.latitude) or math.isnan(msg.longitude):
        rospy.logwarn("GPS 为空，等待有效定位...")
        return

    if msg.status.status < 0:
        rospy.logwarn("GPS 未锁定，忽略")
        return

    # ================= WGS84 → GCJ-02（使用开源项目函数） =================
    gcj_result = wgs84_to_gcj02(msg.longitude, msg.latitude)
    gcj_lon, gcj_lat = gcj_result[0], gcj_result[1]

    start_lng = gcj_lon
    start_lat = gcj_lat
    gps_received = True

    rospy.loginfo("✅ GPS 原始(WGS84)：%.6f, %.6f", msg.latitude, msg.longitude)
    rospy.loginfo("✅ 转换后(GCJ02)：%.6f, %.6f", start_lat, start_lng)

    if not navigation_started:
        navigation_started = True
        run_navigation()

# ================= 6. 导航流程（控制台交互） =================
def run_navigation():
    rospy.loginfo("请输入导航指令（例如：带我去学生食堂）：")

    # ✅ 控制台输入（ROS 中可用）
    user_input = input(">>> ").strip()

    rospy.loginfo("正在分析目的地...")
    result = ask_llm(user_input)
    rospy.loginfo("模型返回：%s", result)

    try:
        indices = [int(x.strip()) for x in result.replace("，", ",").split(",")]
    except Exception:
        rospy.logerr("模型输出解析失败")
        return

    rospy.loginfo("候选目的地：")
    for idx in indices[:3]:
        if 0 <= idx < len(pois):
            p = pois[idx]
            rospy.loginfo("[%d] %s（%s）", idx, p["name"], p.get("type", ""))

    # ================= 控制台选择 =================
    while True:
        try:
            choice = int(input("\n请选择目标编号：").strip())
            if choice in indices:
                break
            else:
                rospy.logwarn("请输入列表中的编号")
        except ValueError:
            rospy.logwarn("请输入数字")

    target = pois[choice]

    rospy.loginfo("\n✅ 已选择目标：")
    rospy.loginfo("名称：%s", target["name"])
    rospy.loginfo("纬度：%s", target["lat"])
    rospy.loginfo("经度：%s", target["lng"])

    # ================= 高德路径规划 =================
    origin = f"{start_lng},{start_lat}"
    destination = f"{target['lng']},{target['lat']}"

    rospy.loginfo("正在调用高德路径规划...")
    points = get_walking_route(origin, destination)

    # ================= 路径点加密：相邻点间距超过 5m 时线性插值 =================
    rospy.loginfo("加密前路径点数量：%d", len(points))
    points = interpolate_path(points, max_interval=5.0)
    rospy.loginfo("加密后路径点数量：%d（相邻点间距 ≤ 5m）", len(points))

    rospy.loginfo("路径规划完成，共 %d 个路径点：", len(points))

    # ================= 发布 ROS 路径 =================
    publish_path(points)

    # ✅ 控制台输出（你原来的格式）
    output_lines = ["["]
    for i, (lng, lat) in enumerate(points):
        comma = "," if i < len(points) - 1 else ""
        output_lines.append(f"    ({lng}, {lat}){comma}")
    output_lines.append("]")
    rospy.loginfo("\n".join(output_lines))


def publish_path(points):
    path = Path()
    path.header.frame_id = "map"
    path.header.stamp = rospy.Time.now()

    for lng, lat in points:
        pose = PoseStamped()
        pose.header.frame_id = "map"
        pose.pose.position.x = lng
        pose.pose.position.y = lat
        pose.pose.orientation.w = 1.0
        path.poses.append(pose)

    path_pub.publish(path)
    rospy.loginfo("✅ 已发布 /planned_path")


# ================= 7. 主函数 =================
if __name__ == "__main__":
    rospy.init_node("llm_gps_planner")

    path_pub = rospy.Publisher("/planned_path", Path, queue_size=1)
    rospy.Subscriber("/fix", NavSatFix, gps_callback, queue_size=1)

    rospy.loginfo("等待 GPS 信号...")
    rospy.spin()