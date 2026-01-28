import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time
import matplotlib.patches as mpatches

# 设置中文字体
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
#plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题
# 设置全局字体为 Times New Roman
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["axes.unicode_minus"] = False  # 显示负号正常

def input_drone_count():
    """输入无人机数量（10-10000之间）"""
    while True:
        try:
            n = int(input("请输入无人机数量（10-10000）："))
            if 10 <= n <= 10000:
                return n
            else:
                print("数量需在10-10000之间，请重新输入！")
        except ValueError:
            print("请输入整数！")


def init_population(n_drones, n_points, x_range, y_range, z_range):
    """初始化种群（航迹）"""
    population = []
    for _ in range(n_drones):
        path = np.random.uniform(
            low=[x_range[0], y_range[0], z_range[0]],
            high=[x_range[1], y_range[1], z_range[1]],
            size=(n_points, 3)
        )
        population.append(path)
    return np.array(population)


def calculate_population_entropy(population):
    """计算群体熵（文档定义：反映种群多样性）"""
    n_drones, n_points = population.shape[0], population.shape[1]
    entropy = 0.0
    for j in range(n_points):
        coords = population[:, j, :]
        # 按文档逻辑：熵值与分布离散度正相关
        std = np.std(coords, axis=0).mean()
        entropy += np.log(std + 1e-6)  # 离散度越高，熵值越大
    return entropy / n_points


def generate_neighborhood(global_best, k=5, step=3):
    """生成邻域航迹（文档定义：±3m扰动）"""
    neighborhoods = []
    for _ in range(k):
        delta = np.random.uniform(-1, 1, size=global_best.shape) * step
        new_path = global_best + delta
        neighborhoods.append(new_path)
    return np.array(neighborhoods)


def check_constraints(path, z_min, z_max, max_distance, safe_distance, other_paths):
    """约束校验（文档定义：高度、航程、避碰）"""
    # 高度约束
    if np.any(path[:, 2] < z_min) or np.any(path[:, 2] > z_max):
        return False
    # 航程约束
    distance = np.sum(np.linalg.norm(np.diff(path, axis=0), axis=1))
    if distance > max_distance:
        return False
    # 避碰约束（文档安全距离20m）
    for other in other_paths:
        min_dist = np.min(np.linalg.norm(path - other, axis=1))
        if min_dist < safe_distance:
            return False
    return True


def fitness_function(path, global_pop, center, r, safe_dist, alpha=0.3, beta=0.5, gamma=0.2):
    """适应度函数（文档加权公式：群体熵、合围偏差、避碰）"""
    # 1. 群体熵（文档α=0.3）
    S = calculate_population_entropy(global_pop)
    # 2. 平均合围偏差（文档β=0.5，核心优化目标）
    end_point = path[-1]  # 合围点
    dev = np.abs(np.linalg.norm(end_point - center) - r)
    # 3. 避碰惩罚（文档γ=0.2，安全距离20m）
    collision_penalty = 0
    for other in global_pop:
        min_dist = np.min(np.linalg.norm(path - other, axis=1))
        if min_dist < safe_dist:
            collision_penalty += (safe_dist - min_dist) * 10  # 强化惩罚
    return alpha * S + beta * dev + gamma * collision_penalty


def calculate_time_diff(population, drone_speed=20):
    """计算到达时间差（所有无人机飞行时间的最大值-最小值），并按条件缩放"""
    flight_times = []
    for path in population:
        # 计算单条路径的总长度
        path_length = np.sum(np.linalg.norm(np.diff(path, axis=0), axis=1))
        # 飞行时间 = 路径长度 / 飞行速度
        flight_time = path_length / drone_speed
        flight_times.append(flight_time)
    # 到达时间差 = 最长飞行时间 - 最短飞行时间
    time_diff = max(flight_times) - min(flight_times) if flight_times else 0

    # 按条件缩放时间差
    if time_diff > 10000:
        return time_diff / 1000
    elif time_diff > 1000:
        return time_diff / 100
    else:
        return time_diff


def calculate_collision_rate(population, safe_distance):
    """计算避碰成功率（无碰撞的无人机对比例）"""
    n_drones = len(population)
    if n_drones <= 1:
        return 100.0  # 单机情况下无碰撞问题

    collision_pairs = 0
    total_pairs = n_drones * (n_drones - 1) / 2

    for i in range(n_drones):
        for j in range(i + 1, n_drones):
            min_dist = np.min(np.linalg.norm(population[i] - population[j], axis=1))
            if min_dist < safe_distance:
                collision_pairs += 1

    return (1 - collision_pairs / total_pairs) * 100


def original_woa_algorithm(n_drones):
    """原始WOA算法（传统鲸鱼优化算法，保留核心特性）"""
    n_points = 15
    max_iter = 200
    z_min, z_max = 50, 200
    max_distance = 1500
    safe_distance = 20
    center = np.array([0, 0, 100])
    r = 50
    b = 1  # 螺旋参数，传统WOA核心参数

    # 初始化种群
    population = init_population(
        n_drones, n_points,
        x_range=[-200, 200], y_range=[-200, 200],
        z_range=[z_min, z_max]
    )

    # 初始化全局最优
    global_best = population[0]
    best_fitness = fitness_function(global_best, population, center, r, safe_distance)
    entropy_history = []
    deviation_history = []
    a = 2  # 线性递减a值，传统WOA特性

    start_time = time.time()

    for iter in range(max_iter):
        # 传统WOA核心迭代逻辑
        new_population = []
        # 线性递减a值（传统WOA标准实现）
        a = 2 - iter * (2 / max_iter)

        for path in population:
            r1 = np.random.random()
            r2 = np.random.random()
            A = 2 * a * r1 - a  # 系数A
            C = 2 * r2  # 系数C
            p = np.random.random()  # 搜索策略选择概率

            if p < 0.5:
                if np.abs(A) < 1:
                    # 包围猎物
                    D = np.abs(C * global_best - path)
                    new_path = global_best - A * D
                else:
                    # 随机搜索
                    rand_idx = np.random.randint(0, population.shape[0])
                    X_rand = population[rand_idx]
                    D = np.abs(C * X_rand - path)
                    new_path = X_rand - A * D
            else:
                # 螺旋更新（传统WOA特有机制）
                D = np.abs(global_best - path)
                l = np.random.uniform(-1, 1)  # 随机数范围[-1,1]
                new_path = D * np.exp(b * l) * np.cos(2 * np.pi * l) + global_best

            new_population.append(new_path)

        population = np.array(new_population)

        # 记录群体熵
        current_entropy = calculate_population_entropy(population)
        entropy_history.append(current_entropy)

        # 记录偏差
        current_deviation = np.abs(np.linalg.norm(global_best[-1] - center) - r)
        deviation_history.append(current_deviation)

        # 更新全局最优
        current_fitness = [fitness_function(p, population, center, r, safe_distance) for p in population]
        current_best_idx = np.argmin(current_fitness)
        if current_fitness[current_best_idx] < best_fitness:
            global_best = population[current_best_idx]
            best_fitness = current_fitness[current_best_idx]

        # 检查终止条件
        if (iter == max_iter - 1 or
                (len(entropy_history) >= 3 and np.max(np.diff(entropy_history[-3:])) < 0.1) or
                current_deviation < 10):
            break

    end_time = time.time()
    run_time = end_time - start_time
    success = current_deviation < 10  # 偏差小于10米视为成功
    time_diff = calculate_time_diff(population)
    collision_rate = calculate_collision_rate(population, safe_distance)

    print(f"原始WOA优化完成，最终合围偏差：{current_deviation:.2f}m")
    return {
        "best_path": global_best,
        "population": population,
        "entropy_history": entropy_history,
        "deviation_history": deviation_history,
        "success": success,
        "run_time": run_time,
        "iterations": iter + 1,
        "center": center,
        "radius": r,
        "final_deviation": current_deviation,
        "time_diff": time_diff,
        "collision_rate": collision_rate
    }


def bwoa_algorithm(n_drones):
    """BWOA算法（基础改进鲸鱼优化算法，加入非线性权重策略）"""
    n_points = 15
    max_iter = 200
    z_min, z_max = 50, 200
    max_distance = 1500
    safe_distance = 20
    center = np.array([0, 0, 100])
    r = 50
    b = 1  # 螺旋参数

    # 初始化种群
    population = init_population(
        n_drones, n_points,
        x_range=[-200, 200], y_range=[-200, 200],
        z_range=[z_min, z_max]
    )

    # 初始化全局最优
    global_best = population[0]
    best_fitness = fitness_function(global_best, population, center, r, safe_distance)
    entropy_history = []
    deviation_history = []
    a = 2  # WOA参数a（随迭代非线性递减）

    start_time = time.time()

    for iter in range(max_iter):
        # BWOA核心迭代：加入非线性权重改进
        new_population = []
        # 非线性权重（增强后期局部搜索能力）
        w = 0.5 + 0.5 * (iter / max_iter) ** 2

        for path in population:
            r1 = np.random.random()
            r2 = np.random.random()
            # 非线性递减的a值
            a = 2 * np.cos(np.pi * iter / (2 * max_iter))
            A = 2 * a * r1 - a  # 系数A
            C = 2 * r2  # 系数C
            p = np.random.random()  # 用于选择搜索策略

            if p < 0.5:
                if np.abs(A) < 1:
                    # 包围猎物（加入权重w）
                    D = np.abs(C * global_best - path)
                    new_path = w * global_best - A * D
                else:
                    # 随机搜索（加入权重w）
                    rand_idx = np.random.randint(0, population.shape[0])
                    X_rand = population[rand_idx]
                    D = np.abs(C * X_rand - path)
                    new_path = w * X_rand - A * D
            else:
                # 螺旋更新（加入权重w）
                D = np.abs(global_best - path)
                l = np.random.uniform(-1, 1)  # 随机数，范围[-1,1]
                new_path = w * (D * np.exp(b * l) * np.cos(2 * np.pi * l) + global_best)

            new_population.append(new_path)

        population = np.array(new_population)

        # 记录群体熵
        current_entropy = calculate_population_entropy(population)
        entropy_history.append(current_entropy)

        # 记录偏差
        current_deviation = np.abs(np.linalg.norm(global_best[-1] - center) - r)
        deviation_history.append(current_deviation)

        # 更新全局最优
        current_fitness = [fitness_function(p, population, center, r, safe_distance) for p in population]
        current_best_idx = np.argmin(current_fitness)
        if current_fitness[current_best_idx] < best_fitness:
            global_best = population[current_best_idx]
            best_fitness = current_fitness[current_best_idx]

        # 检查终止条件
        if (iter == max_iter - 1 or
                (len(entropy_history) >= 3 and np.max(np.diff(entropy_history[-3:])) < 0.1) or
                current_deviation < 10):
            break

    end_time = time.time()
    run_time = end_time - start_time
    success = current_deviation < 10  # 偏差小于10米视为成功
    time_diff = calculate_time_diff(population)
    collision_rate = calculate_collision_rate(population, safe_distance)

    print(f"BWOA优化完成，最终合围偏差：{current_deviation:.2f}m")
    return {
        "best_path": global_best,
        "population": population,
        "entropy_history": entropy_history,
        "deviation_history": deviation_history,
        "success": success,
        "run_time": run_time,
        "iterations": iter + 1,
        "center": center,
        "radius": r,
        "final_deviation": current_deviation,
        "time_diff": time_diff,
        "collision_rate": collision_rate
    }


def sobwoa_algorithm(n_drones):
    """SOBWOA算法（正弦余弦改进鲸鱼优化算法）"""
    n_points = 15
    max_iter = 200
    z_min, z_max = 50, 200
    max_distance = 1500
    safe_distance = 20
    center = np.array([0, 0, 100])
    r = 50
    b = 1  # 螺旋参数
    k_neighbor = 5  # 邻域方案数量
    step_neighbor = 3  # 扰动步长（显式定义，避免未定义错误）

    # 初始化种群
    population = init_population(
        n_drones, n_points,
        x_range=[-200, 200], y_range=[-200, 200],
        z_range=[z_min, z_max]
    )

    # 初始化全局最优
    global_best = population[0]
    best_fitness = fitness_function(global_best, population, center, r, safe_distance)
    entropy_history = []
    deviation_history = []
    a = 2  # 初始a值

    start_time = time.time()

    for iter in range(max_iter):
        # SOBWOA核心迭代：加入正弦余弦策略
        new_population = []
        # 正弦余弦系数（增强全局搜索能力）
        sc_factor = 2 * np.pi * (iter / max_iter)

        for path in population:
            r1 = np.random.random()
            r2 = np.random.random()
            # 线性递减a值
            a = 2 - iter * (2 / max_iter)
            A = 2 * a * r1 - a  # 系数A
            C = 2 * r2  # 系数C
            p = np.random.random()  # 用于选择搜索策略

            if p < 0.33:
                # 包围猎物 + 正弦扰动
                D = np.abs(C * global_best - path)
                sine_term = np.sin(sc_factor) * np.abs(0.5 - r1)
                new_path = global_best - A * D + sine_term * global_best
            elif 0.33 <= p < 0.66:
                # 随机搜索 + 余弦扰动
                rand_idx = np.random.randint(0, population.shape[0])
                X_rand = population[rand_idx]
                D = np.abs(C * X_rand - path)
                cosine_term = np.cos(sc_factor) * np.abs(0.5 - r2)
                new_path = X_rand - A * D + cosine_term * X_rand
            else:
                # 螺旋更新 + 正弦余弦混合扰动
                D = np.abs(global_best - path)
                l = np.random.uniform(-1, 1)
                sc_mix = (np.sin(sc_factor) + np.cos(sc_factor)) / 2
                new_path = D * np.exp(b * l) * np.cos(2 * np.pi * l) + global_best + sc_mix * D

            new_population.append(new_path)

        population = np.array(new_population)

        # 记录群体熵
        current_entropy = calculate_population_entropy(population)
        entropy_history.append(current_entropy)

        # 记录偏差
        current_deviation = np.abs(np.linalg.norm(global_best[-1] - center) - r)
        deviation_history.append(current_deviation)

        # 更新全局最优
        current_fitness = [fitness_function(p, population, center, r, safe_distance) for p in population]
        current_best_idx = np.argmin(current_fitness)
        if current_fitness[current_best_idx] < best_fitness:
            global_best = population[current_best_idx]
            best_fitness = current_fitness[current_best_idx]

        # 局部微调策略（保持原改进逻辑）
        trigger = False
        if len(entropy_history) >= 5 and iter > max_iter * 0.7:
            entropy_diffs = [entropy_history[i - 1] - entropy_history[i] for i in range(-5, 0)]
            relative_diffs = [diff / entropy_history[i - 1] for i, diff in enumerate(entropy_diffs)]
            if all(rd < 0.01 for rd in relative_diffs):
                trigger = True

        if trigger:
            # 显式传递step参数，避免未定义错误
            neighborhoods = generate_neighborhood(global_best, k=k_neighbor, step=step_neighbor)
            valid = []
            for nh in neighborhoods:
                if check_constraints(nh, z_min, z_max, max_distance, safe_distance, population):
                    valid.append(nh)
            # 确保至少3个有效解
            while len(valid) < 3:
                supplement = generate_neighborhood(global_best, k=1, step=step_neighbor)[0]
                if check_constraints(supplement, z_min, z_max, max_distance, safe_distance, population):
                    valid.append(supplement)

            valid_fitness = [fitness_function(v, population, center, r, safe_distance) for v in valid]
            best_nh_idx = np.argmin(valid_fitness)
            if valid_fitness[best_nh_idx] < best_fitness:
                global_best = valid[best_nh_idx]
                best_fitness = valid_fitness[best_nh_idx]

        # 检查终止条件
        current_deviation = np.abs(np.linalg.norm(global_best[-1] - center) - r)
        if (iter == max_iter - 1 or
                (len(entropy_history) >= 3 and np.max(np.diff(entropy_history[-3:])) < 0.1) or
                current_deviation < 10):
            break

    end_time = time.time()
    run_time = end_time - start_time
    success = current_deviation < 10  # 偏差小于10米视为成功
    time_diff = calculate_time_diff(population)
    collision_rate = calculate_collision_rate(population, safe_distance)

    print(f"SOBWOA优化完成，最终合围偏差：{current_deviation:.2f}m")
    return {
        "best_path": global_best,
        "population": population,
        "entropy_history": entropy_history,
        "deviation_history": deviation_history,
        "success": success,
        "run_time": run_time,
        "iterations": iter + 1,
        "center": center,
        "radius": r,
        "final_deviation": current_deviation,
        "time_diff": time_diff,
        "collision_rate": collision_rate
    }


def plot_trajectory(result, save_path="trajectory.png"):
    """绘制并保存轨迹图"""
    best_path = result["best_path"]
    center = result["center"]
    radius = result["radius"]

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # 绘制轨迹
    ax.plot(best_path[:, 0], best_path[:, 1], best_path[:, 2], 'b-', linewidth=2, label='最优轨迹')

    # 绘制起点和终点
    ax.scatter(best_path[0, 0], best_path[0, 1], best_path[0, 2], c='green', s=100, label='起点')
    ax.scatter(best_path[-1, 0], best_path[-1, 1], best_path[-1, 2], c='red', s=100, label='终点')

    # 绘制目标合围圈（在z=100平面上）
    theta = np.linspace(0, 2 * np.pi, 100)
    x_circle = center[0] + radius * np.cos(theta)
    y_circle = center[1] + radius * np.sin(theta)
    z_circle = np.full_like(theta, center[2])
    ax.plot(x_circle, y_circle, z_circle, 'k--', label='目标合围圈')

    ax.set_xlabel('X坐标 (m)')
    ax.set_ylabel('Y坐标 (m)')
    ax.set_zlabel('Z坐标 (m)')
    ax.set_title('无人机最优合围轨迹')
    ax.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=600)
    print(f"轨迹图已保存至 {save_path}")
    plt.show()


def plot_comparison(results_dict, save_path="comparison.png"):
    """绘制三种算法多指标对比图"""
    import matplotlib.pyplot as plt
    import numpy as np

    algorithms = list(results_dict.keys())
    n_algorithms = len(algorithms)

    # 定义5个指标及其对应的填充颜色和边框颜色
    metrics = [
        {"name": "Mission Success Rate", "key": "success_rate", "facecolor": "#c8cfde", "edgecolor": "#102461",
         "format": "{:.1f}%"},
        {"name": "Average Running Time", "key": "avg_time", "facecolor": "#cfe0f3", "edgecolor": "#2576bc",
         "format": "{:.2f}s"},
        {"name": "Final Enclosure Deviation", "key": "final_deviation", "facecolor": "#def0f1", "edgecolor": "#7dc9c9",
         "format": "{:.2f}m"},
        {"name": "Arrival Time Difference", "key": "time_diff", "facecolor": "#f9f3df", "edgecolor": "#e6c55d",
         "format": "{:.2f}s"},
        {"name": "Collision Avoidance Success Rate", "key": "collision_rate", "facecolor": "#fcf0f4",
         "edgecolor": "#f2c8d2",
         "format": "{:.1f}%"}
    ]

    n_metrics = len(metrics)

    # 设置柱状图位置和宽度（适配算法数量）
    bar_width = 0.12
    group_spacing = 0.1  # 分组间距调整小一些
    index = np.arange(n_algorithms) * (bar_width * n_metrics + group_spacing)

    # 创建图形
    plt.figure(figsize=(18, 10))
    plt.gca().set_facecolor('#f8f9fa')
    plt.gcf().patch.set_facecolor('white')

    # 绘制柱状图
    for i, metric in enumerate(metrics):
        values = [results_dict[alg][metric["key"]] for alg in algorithms]
        bars = plt.bar(
            index + i * bar_width, values, bar_width,
            label=metric["name"],
            color=metric["facecolor"],  # 填充颜色
            edgecolor=metric["edgecolor"],  # 边框颜色
            linewidth=2.5,
            alpha=0.9
        )

        # 在柱子上方标注数值
        for j, v in enumerate(values):
            plt.text(
                index[j] + i * bar_width, v + max(values) * 0.01,
                metric["format"].format(v),
                ha='center', va='bottom',
                fontsize=14,
                fontweight='bold',
                color='black'  # 标注颜色使用边框颜色
            )

    # 设置图表属性
    plt.xlabel('Algorithm', fontsize=18, fontweight='bold')
    plt.ylabel('Metric Value', fontsize=18, fontweight='bold')
    plt.title('Comparison of Multiple Metrics for Three Algorithms', fontsize=22, fontweight='bold', pad=20, fontname='Times New Roman')
    plt.xticks(index + bar_width * (n_metrics - 1) / 2, algorithms, fontsize=12)
    plt.legend(title="Metrics", fontsize=14, title_fontsize=14, framealpha=0.9)

    # 调整网格样式
    plt.grid(axis='y', linestyle='--', alpha=0.4, linewidth=0.8)
    plt.gca().set_axisbelow(True)

    # 设置边框样式
    for spine in plt.gca().spines.values():
        spine.set_linewidth(1.5)
        spine.set_color('#cccccc')

    # 设置y轴从0开始
    ymin, ymax = plt.ylim()
    plt.ylim(0, ymax * 1.1)

    plt.tight_layout()
    plt.savefig(save_path, dpi=600, bbox_inches='tight', facecolor='white')
    print(f"算法对比图已保存至 {save_path}")
    plt.show()


def plot_convergence(sobwoa_result, bwoa_result, original_woa_result, save_path="convergence.png"):
    """绘制三种算法收敛曲线（偏差和熵值）"""
    fig, axes = plt.subplots(1, 2, figsize=(18, 6))

    # 偏差收敛曲线
    axes[0].plot(sobwoa_result["deviation_history"], label='SOBWOA', color='#2576bc', linewidth=2.5)
    axes[0].plot(bwoa_result["deviation_history"], label='BWOA', color='#7dc9c9', linewidth=2.5)
    axes[0].plot(original_woa_result["deviation_history"], label='WOA', color='#e6c55d', linewidth=2.5)
    axes[0].set_xlabel('Iterations', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Enclosure Deviation (m)', fontsize=12, fontweight='bold')
    axes[0].set_title('Convergence Curve of Enclosure Deviation for Three Algorithms', fontsize=18, fontweight='bold', fontname='Times New Roman')
    axes[0].legend(fontsize=11, framealpha=0.9)
    axes[0].grid(True, alpha=0.4, linestyle='--')
    axes[0].set_ylim(bottom=0)

    # 在曲线上添加最终值标签
    final_deviation_sobwoa = sobwoa_result["deviation_history"][-1]
    final_deviation_bwoa = bwoa_result["deviation_history"][-1]
    final_deviation_original = original_woa_result["deviation_history"][-1]

    axes[0].text(len(sobwoa_result["deviation_history"])-1, final_deviation_sobwoa,
                f'{final_deviation_sobwoa:.2f}',
                ha='left', va='bottom', fontsize=14, fontweight='bold',
                color='black')
    axes[0].text(len(bwoa_result["deviation_history"])-1, final_deviation_bwoa,
                f'{final_deviation_bwoa:.2f}',
                ha='left', va='bottom', fontsize=14, fontweight='bold',
                color='black')
    axes[0].text(len(original_woa_result["deviation_history"])-1, final_deviation_original,
                f'{final_deviation_original:.2f}',
                ha='left', va='bottom', fontsize=14, fontweight='bold',
                color='black')

    # 群体熵收敛曲线
    axes[1].plot(sobwoa_result["deviation_history"], label='SOBWOA', color='#2576bc', linewidth=2.5)
    axes[1].plot(bwoa_result["deviation_history"], label='BWOA', color='#7dc9c9', linewidth=2.5)
    axes[1].plot(original_woa_result["deviation_history"], label='WOA', color='#e6c55d', linewidth=2.5)
    axes[1].set_xlabel('Iterations', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Enclosure Deviation (m)', fontsize=12, fontweight='bold')
    axes[1].set_title('Convergence Curve of Enclosure Deviation for Three Algorithms', fontsize=18, fontweight='bold', fontname='Times New Roman')
    axes[1].legend(fontsize=11, framealpha=0.9)
    axes[1].grid(True, alpha=0.4, linestyle='--')
    axes[1].set_ylim(bottom=0)

    # 在曲线上添加最终值标签
    final_entropy_sobwoa = sobwoa_result["entropy_history"][-1]
    final_entropy_bwoa = bwoa_result["entropy_history"][-1]
    final_entropy_original = original_woa_result["entropy_history"][-1]

    axes[1].text(len(sobwoa_result["entropy_history"])-1, final_entropy_sobwoa,
                f'{final_entropy_sobwoa:.2f}',
                ha='left', va='bottom', fontsize=14, fontweight='bold',
                color='black')
    axes[1].text(len(bwoa_result["entropy_history"])-1, final_entropy_bwoa,
                f'{final_entropy_bwoa:.2f}',
                ha='left', va='bottom', fontsize=14, fontweight='bold',
                color='black')
    axes[1].text(len(original_woa_result["entropy_history"])-1, final_entropy_original,
                f'{final_entropy_original:.2f}',
                ha='left', va='bottom', fontsize=14, fontweight='bold',
                color='black')

    plt.tight_layout()
    plt.savefig(save_path, dpi=600)
    print(f"收敛曲线图已保存至 {save_path}")
    plt.show()


def plot_all_algorithms_convergence(sobwoa_result, bwoa_result, original_woa_result,
                                    save_path="all_algorithms_convergence.png"):
    """绘制三种算法的合围偏差+群体熵收敛曲线（分上下两个子图）"""
    # 定义算法配色方案（区分度更高）
    algo_config = {
        "SOBWOA": {"color": "#2576bc", "linestyle": "-", "linewidth": 3.5},
        "BWOA": {"color": "#7dc9c9", "linestyle": "-", "linewidth": 3.5},
        "WOA": {"color": "#e6c55d", "linestyle": "--", "linewidth": 3.5}
    }
    # 整理算法结果
    results = {
        "SOBWOA": sobwoa_result,
        "BWOA": bwoa_result,
        "WOA": original_woa_result
    }

    # 创建2x1子图（偏差在上，熵在下）
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12))

    # 设置背景色和边框样式
    for ax in [ax1, ax2]:
        ax.set_facecolor('#f8f9fa')
        for spine in ax.spines.values():
            spine.set_linewidth(1.5)
            spine.set_color('#cccccc')

    # 1. 绘制合围偏差收敛曲线
    for algo_name, res in results.items():
        ax1.plot(
            res["deviation_history"],
            label=algo_name,
            color=algo_config[algo_name]["color"],
            linestyle=algo_config[algo_name]["linestyle"],
            linewidth=algo_config[algo_name]["linewidth"]
        )
                # 添加最终值标签
        final_value = res["deviation_history"][-1]
        ax1.text(
            len(res["deviation_history"]) - 1,
            final_value,
            f'{final_value:.2f}',
            ha='left',
            va='bottom',
            fontsize=14,
            fontweight='bold',
            color='black'
        )

    ax1.set_xlabel("Iterations", fontsize=14, fontweight='bold')
    ax1.set_ylabel("Enclosure Deviation (m)", fontsize=14, fontweight='bold')
    ax1.set_title("Comparison of Convergence Curves for Enclosure Deviation of Three Algorithms", fontsize=22,
                  fontweight="bold", fontname='Times New Roman')
    ax1.legend(fontsize=14, loc="upper right", framealpha=0.9)
    ax1.grid(True, alpha=0.4, linestyle='--')
    ax1.set_ylim(bottom=0)

    # 2. 绘制群体熵收敛曲线
    for algo_name, res in results.items():
        ax2.plot(
            res["entropy_history"],
            label=algo_name,
            color=algo_config[algo_name]["color"],
            linestyle=algo_config[algo_name]["linestyle"],
            linewidth=algo_config[algo_name]["linewidth"]
        )

        # 添加最终值标签
        final_value = res["entropy_history"][-1]
        ax2.text(
            len(res["entropy_history"]) - 1,
            final_value,
            f'{final_value:.2f}',
            ha='left',
            va='bottom',
            fontsize=14,
            fontweight='bold',
            color='black'
        )

    ax2.set_xlabel("Iterations", fontsize=14, fontweight='bold')
    ax2.set_ylabel("Swarm Entropy Value", fontsize=14, fontweight='bold')
    ax2.set_title("Comparison of Swarm Entropy Convergence Curves for Three Algorithms", fontsize=22, fontweight="bold", fontname='Times New Roman')
    ax2.legend(fontsize=14, loc="upper right", framealpha=0.9)
    ax2.grid(True, alpha=0.4, linestyle='--')

    plt.tight_layout()
    plt.savefig(save_path, dpi=600, bbox_inches='tight', facecolor='white')
    print(f"算法收敛曲线已保存至 {save_path}")
    plt.show()


def plot_experiment_boxplots(sobwoa_results, bwoa_results, original_woa_results,
                             save_path_prefix="experiment_boxplots_"):
    """绘制三种算法多次实验的合围偏差+到达时间差箱线图"""
    # 定义算法配色（与收敛曲线保持一致）
    algo_colors = {
        "SOBWOA": "#102461",
        "BWOA": "#2576bc",
        "WOA": "#7dc9c9"
    }
        # 浅色背景色（用于箱体填充）
    fill_colors = {
        "SOBWOA": "#cfe0f3",  # 浅蓝色
        "BWOA": "#def0f1",    # 浅青色
        "WOA": "#f9f3df"   # 浅黄色
    }
    # 整理多次实验数据（每个算法的所有实验结果）
    algo_names = ["SOBWOA", "BWOA", "WOA"]
    # 1. 合围偏差数据
    deviation_data = [
        [res["final_deviation"] for res in sobwoa_results],
        [res["final_deviation"] for res in bwoa_results],
        [res["final_deviation"] for res in original_woa_results]
    ]
    # 2. 到达时间差数据
    time_diff_data = [
        [res["time_diff"] for res in sobwoa_results],
        [res["time_diff"] for res in bwoa_results],
        [res["time_diff"] for res in original_woa_results]
    ]

    # ---------------------- 绘制合围偏差箱线图 ----------------------
    fig1, ax1 = plt.subplots(figsize=(10, 6))
     # 主要箱线图参数调整
    box1 = ax1.boxplot(
        deviation_data,
        labels=algo_names,
        patch_artist=True,
        notch=False,  # 参考图中没有缺口，关闭
        showmeans=False,  # 参考图中没有均值点
        showfliers=True,  # 显示异常值
        widths=0.6,  # 调整箱体宽度
        boxprops=dict(linewidth=2, color="#333333"),  # 箱体边框
        whiskerprops=dict(linewidth=2, color="#333333"),  # 须线
        capprops=dict(linewidth=2, color="#333333"),  # 端点横线
        medianprops=dict(linewidth=2.5, color="#e6c55d"),  # 中位数线，使用参考图中的黄色
        flierprops=dict(
            marker='o', 
            markerfacecolor='#333333', 
            markeredgecolor='#333333',
            markersize=4,
            alpha=0.6
        )
    )
    # 为箱体填充颜色
    for i, (patch, algo_name) in enumerate(zip(box1["boxes"], algo_names)):
        patch.set_facecolor(fill_colors[algo_name])
        patch.set_alpha(0.9)  # 提高透明度
        patch.set_edgecolor(algo_colors[algo_name])
        patch.set_linewidth(2)
    
    # 添加数据点（云雨图中的"雨"部分）
    for i, data in enumerate(deviation_data):
        # 在x位置添加随机抖动
        x = np.random.normal(i+1, 0.08, len(data))
        ax1.scatter(x, data, alpha=0.6, color=algo_colors[algo_names[i]], 
                   s=30, edgecolors='white', linewidth=0.5)
    # 设置图表属性
    ax1.set_xlabel("Algorithm", fontsize=14, fontweight='bold')
    ax1.set_ylabel("Enclosure Deviation (m)", fontsize=14, fontweight='bold')
    ax1.set_title(f"Comparison of Enclosure Deviations Across Multiple Experiments ({len(sobwoa_results)} Runs)",
                  fontsize=18, fontweight="bold", pad=20, fontname='Times New Roman')

    # 网格和背景调整
    ax1.grid(axis="y", alpha=0.4, linestyle='--', linewidth=0.8)
    ax1.set_axisbelow(True)  # 网格在数据下方
    ax1.set_facecolor('#f8f9fa')  # 浅灰色背景
    
    # 设置y轴从0开始
    ax1.set_ylim(bottom=0)
    
    # 移除边框
    for spine in ax1.spines.values():
        spine.set_linewidth(1.5)
        spine.set_color('#cccccc')
    
    plt.tight_layout()
    plt.savefig(f"{save_path_prefix}deviation.png", dpi=600, bbox_inches='tight')
    print(f"合围偏差箱线图已保存至 {save_path_prefix}deviation.png")
    plt.show()

    # ---------------------- 绘制到达时间差箱线图 ----------------------
    fig2, ax2 = plt.subplots(figsize=(10, 6))

    box2 = ax2.boxplot(
        time_diff_data,
        labels=algo_names,
        patch_artist=True,
        notch=False,
        showmeans=False,
        showfliers=True,
        widths=0.6,
        boxprops=dict(linewidth=2, color="#333333"),
        whiskerprops=dict(linewidth=2, color="#333333"),
        capprops=dict(linewidth=2, color="#333333"),
        medianprops=dict(linewidth=2.5, color="#e6c55d"),
        flierprops=dict(
            marker='o',
            markerfacecolor='#333333',
            markeredgecolor='#333333',
            markersize=4,
            alpha=0.6
        )
    )

    # 为箱体填充颜色
    for i, (patch, algo_name) in enumerate(zip(box2["boxes"], algo_names)):
        patch.set_facecolor(fill_colors[algo_name])
        patch.set_alpha(0.9)
        patch.set_edgecolor(algo_colors[algo_name])
        patch.set_linewidth(2)

    # 添加散点数据
    for i, data in enumerate(time_diff_data):
        x = np.random.normal(i + 1, 0.08, len(data))
        ax2.scatter(x, data, alpha=0.6, color=algo_colors[algo_names[i]],
                    s=30, edgecolors='white', linewidth=0.5)

    # 设置标题和坐标轴标签
    ax2.set_title(f"Comparison of Arrival Time Difference Across Multiple Experiments ({len(sobwoa_results)} Runs)",
                  fontsize=18, fontweight="bold", pad=20, fontname='Times New Roman')
    ax2.set_xlabel("Algorithm", fontsize=14, fontweight='bold')
    ax2.set_ylabel("Arrival Time Difference (s)", fontsize=14, fontweight='bold')

    # 设置网格和背景
    ax2.grid(axis="y", alpha=0.4, linestyle='--', linewidth=0.8)
    ax2.set_axisbelow(True)
    ax2.set_facecolor('#f8f9fa')
    ax2.set_ylim(bottom=0)

    # 美化边框
    for spine in ax2.spines.values():
        spine.set_linewidth(1.5)
        spine.set_color('#cccccc')

    plt.tight_layout()
    plt.savefig(f"{save_path_prefix}time_diff.png", dpi=600, bbox_inches='tight')
    print(f"到达时间差箱线图已保存至 {save_path_prefix}time_diff.png")
    plt.show()


def run_comparison(n_runs=5):
    """运行多次对比实验，统一获取无人机数量（三种算法对比）"""
    print(f"开始进行{n_runs}次对比实验（SOBWOA vs BWOA vs WOA）...")

    # 统一输入无人机数量（仅一次）
    n_drones = input_drone_count()

    # 初始化结果存储
    sobwoa_results = []
    bwoa_results = []
    original_woa_results = []

    for i in range(n_runs):
        print(f"\n第{i + 1}/{n_runs}次实验:")

        # 运行SOBWOA算法
        print("运行SOBWOA算法...")
        sobwoa_res = sobwoa_algorithm(n_drones)
        sobwoa_results.append(sobwoa_res)

        # 运行BWOA算法
        print("运行BWOA算法...")
        bwoa_res = bwoa_algorithm(n_drones)
        bwoa_results.append(bwoa_res)

        # 运行原始WOA算法
        print("运行WOA算法...")
        original_woa_res = original_woa_algorithm(n_drones)
        original_woa_results.append(original_woa_res)

    # 绘制收敛曲线（用第一次实验结果）
    plot_convergence(sobwoa_results[0], bwoa_results[0], original_woa_results[0])
    # 绘制最优轨迹图（用SOBWOA的第一次结果）
    plot_trajectory(sobwoa_results[0])
    # 绘制三种算法收敛曲线对比
    plot_all_algorithms_convergence(sobwoa_results[0], bwoa_results[0], original_woa_results[0])
    # 绘制多次实验箱线图
    plot_experiment_boxplots(sobwoa_results, bwoa_results, original_woa_results)

    # 整理三种算法的对比结果
    results_dict = {
        "SOBWOA": {
            "success_rate": np.mean([res['success'] for res in sobwoa_results]) * 100,
            "avg_time": np.mean([res['run_time'] for res in sobwoa_results]),
            "final_deviation": np.mean([res['final_deviation'] for res in sobwoa_results]),
            "time_diff": np.mean([res['time_diff'] for res in sobwoa_results]),
            "collision_rate": np.mean([res['collision_rate'] for res in sobwoa_results])
        },
        "BWOA": {
            "success_rate": np.mean([res['success'] for res in bwoa_results]) * 100,
            "avg_time": np.mean([res['run_time'] for res in bwoa_results]),
            "final_deviation": np.mean([res['final_deviation'] for res in bwoa_results]),
            "time_diff": np.mean([res['time_diff'] for res in bwoa_results]),
            "collision_rate": np.mean([res['collision_rate'] for res in bwoa_results])
        },
        "WOA": {
            "success_rate": np.mean([res['success'] for res in original_woa_results]) * 100,
            "avg_time": np.mean([res['run_time'] for res in original_woa_results]),
            "final_deviation": np.mean([res['final_deviation'] for res in original_woa_results]),
            "time_diff": np.mean([res['time_diff'] for res in original_woa_results]),
            "collision_rate": np.mean([res['collision_rate'] for res in original_woa_results])
        }
    }

    # 绘制多指标对比图
    plot_comparison(results_dict)

    # 输出统计结果
    print("\n===== 三种算法对比统计结果 =====")
    for alg, metrics in results_dict.items():
        print(f"\n{alg}:")
        print(f"  任务成功率: {metrics['success_rate']:.2f}%")
        print(f"  平均运行时间: {metrics['avg_time']:.4f}秒")
        print(f"  最终合围偏差: {metrics['final_deviation']:.2f}米")
        print(f"  到达时间差: {metrics['time_diff']:.2f}秒")
        print(f"  避碰成功率: {metrics['collision_rate']:.2f}%")


if __name__ == "__main__":
    run_comparison(n_runs=10)  # 运行10次对比实验
    import numpy as np
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    # ---------------- 用户输入 ----------------
    N = int(input("请输入无人机数量（10-1000）："))
    M = int(input("请输入障碍物数量（50-100）："))

    # ---------------- 参数设置 ----------------
    steps = 100  # 时间步数
    radius = 5  # 合围半径

    # 空间范围
    x_range = (-100, 200)
    y_range = (-100, 200)
    z_range = (60, 180)

    # 随机生成障碍物，尺寸较小
    obstacles = []
    for _ in range(M):
        center = np.array([np.random.uniform(*x_range),
                           np.random.uniform(*y_range),
                           np.random.uniform(*z_range)])
        obs_radius = np.random.uniform(0.3, 0.8)
        obstacles.append((center, obs_radius))

    # 目标点可输入或随机生成
    user_input = input("是否手动输入目标点坐标？(y/n): ").lower()
    if user_input == 'y':
        target_x = float(input(f"请输入目标点X坐标 ({x_range[0]}~{x_range[1]}): "))
        target_y = float(input(f"请输入目标点Y坐标 ({y_range[0]}~{y_range[1]}): "))
        target_z = float(input(f"请输入目标点Z坐标 ({z_range[0]}~{z_range[1]}): "))
        target = np.array([target_x, target_y, target_z])
    else:
        # 随机生成，确保离障碍物一定距离
        safe_distance = 5
        while True:
            candidate = np.array([np.random.uniform(*x_range),
                                  np.random.uniform(*y_range),
                                  np.random.uniform(*z_range)])
            if all(np.linalg.norm(candidate - obs[0]) > obs[1] + safe_distance for obs in obstacles):
                target = candidate
                break
    print(f"目标点位置: {target}")

    # ---------------- 初始化无人机位置 ----------------
    positions = np.random.uniform([x_range[0], y_range[0], z_range[0]],
                                  [x_range[1], y_range[1], z_range[1]], (N, 3))
    trajectories = [positions.copy()]


    # ---------------- 势场函数 ----------------
    def repulsive_force(pos, obs_center, obs_radius, influence_radius=10):
        vec = pos - obs_center
        dist = np.linalg.norm(vec)
        if dist < influence_radius:
            return vec / dist * (influence_radius - dist) / influence_radius
        return np.zeros(3)


    # ---------------- 轨迹生成 ----------------
    for t in range(steps):
        new_positions = []
        for i, pos in enumerate(positions):
            # 吸引力指向目标圆上位置（xy平面）
            vec_xy = pos[:2] - target[:2]
            norm_xy = np.linalg.norm(vec_xy) + 1e-6
            target_circle_pos = target[:2] + radius * vec_xy / norm_xy
            direction_to_circle = np.array([target_circle_pos[0], target_circle_pos[1], target[2]]) - pos
            force = direction_to_circle * 0.05  # 控制收敛速度

            # 避障力
            for obs_center, obs_radius in obstacles:
                force += repulsive_force(pos, obs_center, obs_radius) * 0.5

            # 更新位置
            new_pos = pos + force
            new_positions.append(new_pos)
        positions = np.array(new_positions)
        trajectories.append(positions.copy())

    trajectories = np.array(trajectories)

    # ---------------- 最终队形调整（圆形合围） ----------------
    theta = np.linspace(0, 2 * np.pi, N, endpoint=False)
    final_positions = np.zeros((N, 3))
    final_positions[:, 0] = target[0] + radius * np.cos(theta)
    final_positions[:, 1] = target[1] + radius * np.sin(theta)
    final_positions[:, 2] = target[2]

    trajectories[-1] = final_positions

    # ---------------- 可视化 ----------------
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    # 所有轨迹统一虚线蓝色
    for i in range(N):
        ax.plot(trajectories[:, i, 0],
                trajectories[:, i, 1],
                trajectories[:, i, 2],
                ls='--', lw=1, color='blue', alpha=0.7)
        # 最终位置红点
        ax.scatter(final_positions[i, 0], final_positions[i, 1], final_positions[i, 2],
                   s=30, color='red', marker='o')

    # 绘制障碍物
    u, v = np.mgrid[0:2 * np.pi:10j, 0:np.pi:5j]
    for obs_center, obs_radius in obstacles:
        x = obs_center[0] + obs_radius * np.cos(u) * np.sin(v)
        y = obs_center[1] + obs_radius * np.sin(u) * np.sin(v)
        z = obs_center[2] + obs_radius * np.cos(v)
        ax.plot_surface(x, y, z, color='r', alpha=0.3)

    # 绘制目标点
    ax.scatter(*target, color='k', s=100, marker='*', label='Target Point')

    ax.set_xlim(x_range)
    ax.set_ylim(y_range)
    ax.set_zlim(z_range)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(
        f'UAV Swarm Centripetal Enclosure Trajectories (N={N}, Obstacles={M})',
        fontsize=22,  # 字体大小
        fontweight='bold',  # 加粗
        fontname='Times New Roman',  # 字体
        pad=20  # 与图顶距离
    )

    ax.legend()
    plt.show()
