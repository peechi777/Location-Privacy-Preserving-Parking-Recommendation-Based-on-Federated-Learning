import numpy as np
from shapely.geometry import Point

def create_circle(center, radius=1):
    return Point(center).buffer(radius)

#計算重疊率
def calculate_overlap(circle1, circle2):
    if circle1.intersects(circle2):
        return circle1.intersection(circle2).area / circle1.union(circle2).area
    return 0

# 參數
num_attacks = 10000
num_users_per_attack = 1000
area_center = (0, 0) #目前位置
area_radius = 5  # 半徑

average_overlaps = []


#每次攻擊，隨機生成使用者的真實位置和推斷位置
np.random.seed(42)
for i in range(num_attacks):
    real_positions = [(np.random.uniform(-area_radius, area_radius),
                       np.random.uniform(-area_radius, area_radius)) for _ in range(num_users_per_attack)]
    inferred_positions = [(np.random.uniform(-area_radius, area_radius),
                           np.random.uniform(-area_radius, area_radius)) for _ in range(num_users_per_attack)]

    overlaps = []
    for real_pos, inf_pos in zip(real_positions, inferred_positions):
        real_circle = create_circle(real_pos)
        inf_circle = create_circle(inf_pos)
        overlap = calculate_overlap(real_circle, inf_circle)
        overlaps.append(overlap)
    
    average_overlap = np.mean(overlaps)
    average_overlaps.append(average_overlap)
    print(f'Attack {i+1}: Average Overlap = {average_overlap:.4f}')


overall_average = np.mean(average_overlaps)
print(f'\nOverall Average Overlap: {overall_average:.4f}')
