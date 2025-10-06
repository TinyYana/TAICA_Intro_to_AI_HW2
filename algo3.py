import numpy as np


def ultimate_algorithm(loss_fn, start_points, field_size, steps=30):
    """
    Ultimate Algorithm (radial exploration + local hill-climb)
    - loss_fn(x,y): 返回該點的 GPA（值越大越好）
    - start_points: 主程式給的起點清單（len = number of ships）
    - field_size: (min, max) 的範圍，例如 (-8, 8)
    - steps: 每艘飛船最多的移動步數（作業會給 30）
    回傳: list_of_paths，長度 = #ships，每個 path 是一個 step-list，
           每個 step 為 (x, y, current_gpa, best_x, best_y, best_gpa)
    """
    n_ships = len(start_points)
    paths = []
    # 中心座標（地圖中心）
    center_x = (field_size[0] + field_size[1]) / 2.0
    center_y = center_x
    # 最大半徑（保證不出界）
    max_radius = min(abs(field_size[0] - center_x), abs(field_size[1] - center_y))
    # 分配探索與爬升步數
    explore_steps = max(6, steps // 3)  # 例如 30 -> 探索 10 步
    refine_steps = steps - explore_steps  # 例如 30 -> 爬山 20 步

    rng = np.random.default_rng(42)

    for i in range(n_ships):
        # 每艘船被分配一個角度（均勻分布）
        angle = 2 * np.pi * i / max(1, n_ships)
        path = []

        sx, sy = start_points[i]
        s_val = loss_fn(sx, sy)
        best_x, best_y, best_gpa = sx, sy, s_val
        path.append((sx, sy, s_val, best_x, best_y, best_gpa))

        # --- 探索階段 (radial / concentric) ---
        # 每一步向外（或從中心向外）在該角度上取一個點
        for t in range(explore_steps):
            if explore_steps == 1:
                frac = 0.0
            else:
                frac = t / (explore_steps - 1)  # 0.0 .. 1.0
            r = frac * max_radius
            x = center_x + r * np.cos(angle)
            y = center_y + r * np.sin(angle)
            # 保證不超出邊界
            x = float(np.clip(x, field_size[0], field_size[1]))
            y = float(np.clip(y, field_size[0], field_size[1]))
            val = float(loss_fn(x, y))
            if val > best_gpa:
                best_x, best_y, best_gpa = x, y, val
            path.append((x, y, val, best_x, best_y, best_gpa))

        # --- 局部精修 (local hill climbing) ---
        # 從探索到的 best 點開始，用較大的 step 嘗試鄰居
        x, y = best_x, best_y
        current_gpa = best_gpa
        step_size = 0.4  # 可以調，數值越大會跳得比較多
        for _ in range(refine_steps):
            # 產生 8 個鄰居 (上下左右 + 四個對角)
            neighbors = []
            for dx in (step_size, 0.0, -step_size):
                for dy in (step_size, 0.0, -step_size):
                    if dx == 0 and dy == 0:
                        continue
                    nx = x + dx
                    ny = y + dy
                    # 邊界檢查
                    if not (
                        field_size[0] <= nx <= field_size[1]
                        and field_size[0] <= ny <= field_size[1]
                    ):
                        continue
                    neighbors.append((float(nx), float(ny)))

            best_nx, best_ny = x, y
            best_ngpa = current_gpa
            for nx, ny in neighbors:
                ngpa = float(loss_fn(nx, ny))
                if ngpa > best_ngpa:
                    best_nx, best_ny, best_ngpa = nx, ny, ngpa

            if best_ngpa > current_gpa:
                # 找到更好的鄰居就移動
                x, y, current_gpa = best_nx, best_ny, best_ngpa
            else:
                # 沒有更好就有小機率嘗試「跳躍」（避免陷入局部）
                if rng.random() < 0.15:
                    jump = step_size * 2.0
                    theta = rng.uniform(0, 2 * np.pi)
                    nx = float(
                        np.clip(x + jump * np.cos(theta), field_size[0], field_size[1])
                    )
                    ny = float(
                        np.clip(y + jump * np.sin(theta), field_size[0], field_size[1])
                    )
                    current_gpa = float(loss_fn(nx, ny))
                    x, y = nx, ny
                else:
                    # 否則就待在原地（消耗一步燃料）
                    pass

            if current_gpa > best_gpa:
                best_x, best_y, best_gpa = x, y, current_gpa
            path.append((x, y, current_gpa, best_x, best_y, best_gpa))

        paths.append(path)

    return paths


# import numpy as np

# def ultimate_algorithm(loss_fn, start_points, field_size, steps=300):
#     starts = np.array(start_points, dtype=float).copy()
#     paths = []

#     for start in starts:
#         x, y = start
#         current_gpa = loss_fn(x, y)
#         best_x, best_y, best_gpa = x, y, current_gpa
#         path = [(x, y, current_gpa, best_x, best_y, best_gpa)]
#         step_size = 0.2
#         for _ in range(steps):
#             # --- Homework: Implement the Ultimate Algorithm ---
#             # You may either:
#             #
#             # ? Choose an existing method (e.g., Local Beam Search, Genetic Algorithm, etc.)
#             #     Example: Local Beam Search
#             #     Steps if you choose Local Beam Search:
#             #       1. For each current state, generate all 8 neighbors
#             #          (using step_size in x and/or y)
#             #       2. Collect all neighbors from all k beams
#             #       3. Evaluate each neighbor with loss_fn (GPA)
#             #       4. Select the top-k neighbors (highest GPA values)
#             #       5. These become the new states for the next iteration
#             #       6. Append the chosen states to their corresponding paths
#             #
#             # ? Modify Hill Climbing or Simulated Annealing into an improved version
#             #
#             # ? Design your own original algorithm
#             #
#             # ------------------------------------------------------

#             # --- IMPLEMENTATION START ---
#             pass

#             # --- IMPLEMENTATION END ---

#             # ------------------------------------------------
#             # "path" means the whole trajectory of one ship from start to end.
#             # Each entry in path should include:
#             #   - current ship (x, y, gpa)
#             #   - best (x, y, gpa) found so far by this ship
#             #
#             # "paths" means the collection of all ships' trajectories.
#             # For this homework, it should contain 10 paths (one per ship).
#             # You can check example_10.txt (generated by main.py) to verify if your code
#             #   produces a similar format and result.
#             # You can also refer to algo_example.py for a better understanding.
#             # Note: best_gpa should only increase over time.
#             # ------------------------------------------------
#             if current_gpa > best_gpa:
#                 best_x, best_y, best_gpa = x, y, current_gpa
#             path.append((x, y, current_gpa, best_x, best_y, best_gpa))
#         paths.append(path)
#     return paths
# algo3.py
