import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Wedge

def highlight_and_label_circle(radii, num_segments, angle_per_segment, center_x, center_y, time_circle, start_angle, end_angle):
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlim(center_x - 10, center_x + 10)
    ax.set_ylim(center_y - 10, center_y + 10)
    ax.set_aspect('equal', 'box')
    ax.axhline(center_y, color='black', linewidth=0.5)
    ax.axvline(center_x, color='black', linewidth=0.5)
    
    
    for time, rad in radii.items():
        circle = plt.Circle((center_x, center_y), rad, fill=False, color='black', linestyle='--', linewidth=1)
        ax.add_artist(circle)
    
   
    radius = radii[time_circle]
    
    
    wedge = Wedge((center_x, center_y), radius, start_angle, end_angle, color='red', alpha=0.5)
    ax.add_patch(wedge)
    
    
    for rad in radii.values():
        for i in range(num_segments):
            theta = np.deg2rad(i * angle_per_segment)
            end_theta = np.deg2rad((i + 1) * angle_per_segment)
            x = np.cos(theta) * rad + center_x
            y = np.sin(theta) * rad + center_y
            
            
            ax.plot([center_x, x], [center_y, y], color='grey', linestyle=':', linewidth=0.5)
            
            
            if rad == radius:
                text_theta = np.deg2rad(i * angle_per_segment + angle_per_segment / 2)
                text_x = np.cos(text_theta) * (rad + 0.5) + center_x  
                text_y = np.sin(text_theta) * (rad + 0.5) + center_y
                ax.text(text_x, text_y, f"{angle_per_segment * i}° to {angle_per_segment * (i + 1)}°", 
                        horizontalalignment='center', verticalalignment='center', fontsize=8)

    
    plt.show()


radii = {'10min': 2, '20min': 5, '30min': 7}  # 依照交通情況設定圓半徑
num_segments = 16  # 切割區塊
angle_per_segment = 360 / num_segments  

# 使用者輸入
center_x = float(input("使用者位置 x : "))
center_y = float(input("使用者位置 y : "))
time_circle = input("圓位置(10min,20min,30min): ")
start_deg = float(input("開始角度: "))
end_deg = float(input("結束角度: "))


if time_circle in radii:
    highlight_and_label_circle(radii, num_segments, angle_per_segment, center_x, center_y, time_circle, start_deg, end_deg)
else:
    print("Invalid time circle. Please enter one of '10min', '20min', '30min'.")
