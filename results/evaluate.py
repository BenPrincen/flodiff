import numpy as np
import tqdm
import os

def judge_success(data, distance_th_times, collision_th, suc_dis, shortest_traj):
    # cal distance in shortest_traj
    shortest_dis = 0
    for i in range(len(shortest_traj)-1):
        shortest_dis += np.linalg.norm(shortest_traj[i][:2] - shortest_traj[i+1][:2])
 
    goal = shortest_traj[-1][:2]
    d_0 = np.linalg.norm(data[0][:2] - goal)
    collision_num = 0
    arrive = False
    cul_dis = 0
    d_t = d_0  # final distance to goal (if loop never runs)
    for i, d in enumerate(data):
        d_t = np.linalg.norm(d[:2] - goal)
        if i > 0:
            cul_dis += np.linalg.norm(d[:2] - data[i-1][:2])
        if collision_num >= collision_th:
            break
        if d[4] == 1 :
            collision_num += 1
        if np.linalg.norm(d[:2] - goal) < suc_dis:
            arrive = True
            break
    return arrive, collision_num, shortest_dis, cul_dis, d_0, d_t

# Paths: results = simulator trajectories, dataset = ground truth trajectories
RESULTS_DIR = '/home/vgmachinist/projects/flodiff/results'
EXP_NAME = 'exp_1'
TRAJ_DIR = '/home/vgmachinist/projects/flodiff/dataset/scenes_117/test/'

# Simulator trajectories are in results/<exp>/trajectory/*.txt
data_dir = os.path.join(RESULTS_DIR, EXP_NAME, 'trajectory')
distance_th_times = 3
collision_th = [1, 10, 30, 50, 5000]
suc_dis = [0.25, 0.3, 0.35, 0.4]

traj_txt_files = sorted([f for f in os.listdir(data_dir) if f.endswith('.txt')])
if not traj_txt_files:
    raise SystemExit(f"No .txt trajectory files in {data_dir}")

for c_th in collision_th:
    for s_dis in suc_dis:
        print('-------------------------------------')
        print('collision count th:', c_th, 'arrive th:', s_dis)

        arrives = []
        collision_nums = []
        shortest_diss = []
        cul_diss = []
        SPL = []
        SoftSPL = []
        skipped = 0

        for f in tqdm.tqdm(traj_txt_files, desc="Evaluating"):
            f_splits = f.split('_')
            f_splits[-1] = f_splits[-1].split('.')[0]
            scene_floor = f_splits[0] + '_' + f_splits[1]
            traj_name = f_splits[2] + '_' + f_splits[3]
            shortest_traj_path = os.path.join(TRAJ_DIR, scene_floor, traj_name, traj_name + '.npy')
            if not os.path.isfile(shortest_traj_path):
                skipped += 1
                continue
            shortest_traj = np.load(shortest_traj_path)
            data = np.loadtxt(os.path.join(data_dir, f))
            arrive, collision_num, shortest_dis, cul_dis_val, d_0, d_t = judge_success(
                data, distance_th_times, c_th, s_dis, shortest_traj
            )
            arrives.append(arrive)
            collision_nums.append(collision_num)
            shortest_diss.append(shortest_dis)
            cul_diss.append(cul_dis_val)
            SPL.append(arrive * shortest_dis / max(cul_dis_val, shortest_dis))
            SoftSPL.append((1 - d_t / d_0) * shortest_dis / max(cul_dis_val, shortest_dis))

        if skipped:
            print(f"Skipped {skipped} trajectories (no ground truth .npy).")
        n = len(arrives)
        if n == 0:
            print("No trajectories evaluated.")
            continue
        print(f"{EXP_NAME}: SR: {np.mean(arrives):.4f}, SPL: {np.mean(SPL):.4f}, SoftSPL: {np.mean(SoftSPL):.4f} (n={n})")
        print('collision mean nums:', np.mean(collision_nums))
            
            
