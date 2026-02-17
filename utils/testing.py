import os
import sys
import numpy as np
from PIL import Image
import torch
import cv2
import time

# # Resolve igibson when run from flodiff root (package lives in iGibson/)
# _flodiff_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# _igibson_root = os.path.join(_flodiff_root, "iGibson")
# if _igibson_root not in sys.path:
#     sys.path.insert(0, _igibson_root)

from igibson.render.viewer import Viewer
from igibson.simulator import Simulator
from igibson.scenes.gibson_indoor_scene import StaticIndoorScene
from igibson.render.mesh_renderer.mesh_renderer_settings import MeshRendererSettings
from igibson.render.profiler import Profiler
from utils.utils import *
from model.data_utils import img_path_to_data, resize_and_aspect_crop
from model.flona import flona, DenseNetwork
from model.flona_vint import flona_ViNT, replace_bn_with_gn
from diffusion_policy.diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

IMG_RESIZE_SIZE = (96, 96)
IMAGE_ASPECT_RATIO = 1 / 1  # 4 / 3

# ================================ Simulator Helper functions ================================
def check_collision(pos, travers_map):
    """
    check if the position is in collision
    """
    # print(pos)
    pos_in_map = (pos * 100 + np.array([travers_map.shape[0], travers_map.shape[1]]) // 2).astype(np.int16)
    return travers_map[pos_in_map[1], pos_in_map[0]] == 0

def world_to_map_pixel(pos_xy, map_shape):
    """Convert world (x, y) in meters to map pixel (col, row) for drawing. Same convention as check_collision."""
    pos_in_map = (np.array(pos_xy) * 100 + np.array([map_shape[0], map_shape[1]]) // 2).astype(np.int32)
    # travers_map is indexed [row, col] -> pos_in_map[1] is row, pos_in_map[0] is col; cv2 uses (x,y) = (col, row)
    return (int(pos_in_map[0]), int(pos_in_map[1]))


def draw_floorplan_with_trajectory(floorplan_ary, trajectory_history, current_position, map_shape, goal_pos=None,
                                   window_name="Floor plan - robot trajectory", save_path=None, show=True):
    """
    Draw floor plan with trajectory history (path so far) and current robot position.
    - trajectory: orange line
    - current position: red circle
    - goal (if provided): green circle
    """
    # RGBA -> BGR for cv2
    vis = np.array(floorplan_ary[:, :, :3].copy(), dtype=np.uint8)
    if vis.shape[2] == 3:
        vis = cv2.cvtColor(vis, cv2.COLOR_RGB2BGR)
    else:
        vis = cv2.cvtColor(vis, cv2.COLOR_RGBA2BGR)
    h, w = vis.shape[:2]
    
    # Clip points to image bounds
    def clip(p):
        return (max(0, min(p[0], w - 1)), max(0, min(p[1], h - 1)))
    # Draw trajectory (where the robot has been)
    if len(trajectory_history) >= 2:
        pts = [clip(world_to_map_pixel(t[:2], map_shape)) for t in trajectory_history]
        for i in range(len(pts) - 1):
            cv2.line(vis, pts[i], pts[i + 1], (255, 165, 0), 2)  # Orange path

    # Draw current robot position
    cur_px = clip(world_to_map_pixel(current_position[:2], map_shape))
    cv2.circle(vis, cur_px, 8, (0, 0, 255), -1)  # Red = current
    cv2.circle(vis, cur_px, 10, (255, 255, 255), 1)

    # Draw goal if provided
    if goal_pos is not None and len(goal_pos) > 0:
        g_xy = goal_pos[0][:2] if hasattr(goal_pos[0], "__len__") else goal_pos[:2]
        goal_px = clip(world_to_map_pixel(np.array(g_xy), map_shape))
        cv2.circle(vis, goal_px, 10, (0, 255, 0), 2)  # Green = goal
    if save_path:
        cv2.imwrite(save_path, vis)
    if show:
        cv2.imshow(window_name, vis)
        cv2.waitKey(1)

def l2_distance(v1, v2):
    """Returns the L2 distance between vector v1 and v2."""
    return np.linalg.norm(np.array(v1) - np.array(v2))


def camera_set_and_record(env, current_position, current_heading):
    if env.viewer is not None:
        env.viewer.initial_pos = current_position
        env.viewer.initial_view_direction = current_heading - current_position # [0,0,0] will make the image black
        env.viewer.reset_viewer()
    env.renderer.set_camera(current_position, current_heading, [0, 0, 1])
    with Profiler("Render"):
        frame = env.renderer.render(modes=("rgb"))
    img = Image.fromarray((255 * np.concatenate(frame, axis=1)[:, :, :3]).astype(np.uint8))
    # plt.imshow(img)
    # plt.show()
    resized_img = resize_and_aspect_crop(img, IMG_RESIZE_SIZE, IMAGE_ASPECT_RATIO)
    env.step()
    return resized_img
    
def camera_execute_actions(env, current_position, trajectory, cur_obs_list, context_size, travers_map, save_state, trajectory_history=None):
    '''
    let camera execute actions according to the given trajectory
    '''
    turn = False
    collision, resized_img, current_heading_point = None, None, None
    for subgoal in trajectory:
        sub_waypoints = sub_waypoints_generator(subgoal, current_position)  # generate four sub_waypoints for each next subgoal
        last_sub_waypoint = current_position
        for sub_waypoint in sub_waypoints:
            collision = bool(np.asarray(check_collision(sub_waypoint[:2], travers_map)).any())
            current_position = last_sub_waypoint
            current_heading_point = sub_waypoint
            resized_img = camera_set_and_record(env, current_position, current_heading_point)
            save_state.append(np.array([float(current_position[0]), float(current_position[1]), float(current_heading_point[0]), float(current_heading_point[1]), bool(np.asarray(collision).any())]))
            if trajectory_history is not None:
                trajectory_history.append(np.array([float(current_position[0]), float(current_position[1])]))
          
            last_sub_waypoint = sub_waypoint
            time.sleep(0.01)          
            if collision:
                print("collide")               
                break
        
        # Make sure in loop variables are not None
        # if not collision or not resized_img or not current_heading_point:
        #     return None
        
        if len(cur_obs_list) < context_size + 1:
            cur_obs_list.append(resized_img.unsqueeze(0))
        else:
            cur_obs_list.pop(0)
            cur_obs_list.append(resized_img.unsqueeze(0))
            
        if collision:
            current_yaw = np.arctan2(current_heading_point[1] - current_position[1], current_heading_point[0] - current_position[0])
            collision_num = 0
            coin = np.random.rand() 
            if coin >= 0.5:
                while collision and collision_num < 8:
                    turn = True
                    # print("turn right")
                    collision_num += 1
                    current_yaw -= 45 / 180 * np.pi
                    current_heading_point[:2] = current_position[:2] + np.array([np.cos(current_yaw), np.sin(current_yaw)]) * 0.02
                    
                    resized_img = camera_set_and_record(env, current_position, current_heading_point)
                    save_state.append(np.array([float(current_position[0]), float(current_position[1]), float(current_heading_point[0]), float(current_heading_point[1]), bool(np.asarray(collision).any())]))
                    if trajectory_history is not None:
                        trajectory_history.append(np.array([float(current_position[0]), float(current_position[1])]))
                    for i in range(context_size):
                        cur_obs_list.pop(0)
                        cur_obs_list.append(resized_img.unsqueeze(0))

                    collision = bool(np.asarray(check_collision(current_heading_point[:2], travers_map)).any())
            else:
                while collision and collision_num < 8:
                    turn = True
                    # print("turn left")
                    collision_num += 1
                    current_yaw += 45 / 180 * np.pi
                    current_heading_point[:2] = current_position[:2] + np.array([np.cos(current_yaw), np.sin(current_yaw)]) * 0.02
                    
                    resized_img = camera_set_and_record(env, current_position, current_heading_point)
                    save_state.append(np.array([float(current_position[0]), float(current_position[1]), float(current_heading_point[0]), float(current_heading_point[1]), bool(np.asarray(collision).any())]))
                    if trajectory_history is not None:
                        trajectory_history.append(np.array([float(current_position[0]), float(current_position[1])]))
                    for i in range(context_size):
                        cur_obs_list.pop(0)
                        cur_obs_list.append(resized_img.unsqueeze(0))

                    collision = bool(np.asarray(check_collision(current_heading_point[:2], travers_map)).any())
            break
    return current_position, current_heading_point, collision, turn

def camera_follow_traj(env, current_position, trajectory, orientation, cur_obs_list, context_size):
    '''
    directly excute the trajectory
    '''
    for i in range(len(trajectory)):
        if env.viewer is not None:
            env.viewer.initial_pos = trajectory[i]
            env.viewer.initial_view_direction = orientation[i] - trajectory[i]
            env.viewer.reset_viewer()
        
        frame = env.renderer.render(modes=("rgb"))
        img = Image.fromarray((255 * np.concatenate(frame, axis=1)[:, :, :3]).astype(np.uint8))
        resized_img = resize_and_aspect_crop(img, (96,96), 4 / 3)
        if len(cur_obs_list) < context_size + 1:
            cur_obs_list.append(resized_img.unsqueeze(0))
        else:
            cur_obs_list.pop(0)
            cur_obs_list.append(resized_img.unsqueeze(0))
        
        env.step()
        time.sleep(0.01)
    return trajectory[-1], orientation[-1]

def sub_waypoints_generator(subgoal, current_position):
    '''
    generate four sub_waypoints for each next subgoal
    '''
    sub_waypoints = []
    for i in range(5):
        sub_waypoints.append(current_position + (subgoal - current_position) * (i + 1) / 5)
    # sub_waypoints.append(subgoal)
    return sub_waypoints


# ================================ Input Error Checking Helper functions ================================
def check_num_trajs(config, scene_name_floor: str, floor: int):
    trajs_dir = config["testdataset"] + scene_name_floor + '_' + str(floor)
    num_trajs = 0
    for f in os.listdir(trajs_dir):
        if f.startswith('traj'):
            num_trajs += 1
    return num_trajs

def check_floor_num(config, scene_name_floor: str, floor: int):
    trajs_dir = config["testdataset"] + scene_name_floor + '_' + str(floor)
    return os.path.exists(trajs_dir)

# ================================ Main Loop Helper functions ================================
def test_single_scene(config, settings, short_exec: bool, scene_name_floor: str):
    metric_waypoint_spacing = config["metric_waypoint_spacing"]
    waypoint_spacing = config["waypoint_spacing"]
    max_steps = -1 if not short_exec else 90
    step = 0
    context_size = config["context_size"] 
    arrive_th = config["arrive_th"]
    headless = config["headless"]
    state_save_dir, img_save_dir = create_save_dirs(config)
    scene_id = scene_name_floor.split('_')[0]
    floor = int(scene_name_floor.split('_')[1])
    sim = Simulator(
        mode="gui_interactive" if not headless else "headless",
        image_width=512,
        image_height=512,
        rendering_settings=settings,
        device_idx=0
    )
    scene = StaticIndoorScene(
        scene_id,
        build_graph=True,
    )
    sim.import_scene(scene)
    if floor == 0:
        focused_map_path =  os.path.join(config["trav_maps_path"], scene_id, 'foucused_map.png')
    else:
        focused_map_path =  os.path.join(config["trav_maps_path"], scene.scene_id, 'foucused_map.png')
    
    if not headless:
        if sim.viewer is None:
            raise ValueError("Viewer is not initialized")
        elif isinstance(sim.viewer, Viewer):
            sim.viewer.initial_pos = config["initial_pos"]
            sim.viewer.initial_view_direction = config["initial_view_direction"]
            sim.viewer.reset_viewer()
        else:
            raise ValueError("Viewer is not a Viewer object")
    # floorplan and travers map
    if floor == 0:
        floorplan_path = config["trav_maps_path"] + scene_id + '/map.png'
        travers_path_8bit = config["trav_maps_path"] + scene_id + '/floor_trav_test_0_modified_8bit.png'
        travers_path_fallback = config["trav_maps_path"] + scene_id + '/floor_trav_test_0_modified.png'
    else:
        floorplan_path = config["trav_maps_path"] + scene_name_floor + '/map.png'
        travers_path_8bit = config["trav_maps_path"] + scene_name_floor + '/floor_trav_test_' + str(floor) + '_modified_8bit.png'
        travers_path_fallback = config["trav_maps_path"] + scene_name_floor + '/floor_trav_test_' + str(floor) + '_modified.png'
    travers_path = travers_path_8bit if os.path.exists(travers_path_8bit) else travers_path_fallback
    floorplan = img_path_to_data(floorplan_path, (96,96))
    if floorplan.shape[0] > 3:
        floorplan = floorplan[:3]
    floorplan = floorplan.unsqueeze(0) # 1,h,w,3
    floorplan_ary = np.array(Image.open(floorplan_path).convert('RGBA') ) # original size
    travers_map = np.array(Image.open(travers_path)) # original size
    # load traj
    trajs_dir = config["testdataset"] + scene_id + '_' + str(floor)
    trajs_id = []
    for f in os.listdir(trajs_dir):
        if f.startswith('traj'):
            trajs_id.append(int(f.split('_')[1]))
    trajs_id.sort()
    
    # iterate trajs
    for i in range(10):
        traj_name = 'traj_' + str(trajs_id[i])
        traj_file = os.path.join(config["testdataset"], f'{scene_id}_{floor}', traj_name, traj_name + '.npy')   
        scene_dir = config["scene_path"] + scene_id
        with open(os.path.join(scene_dir, "floors.txt"), "r") as ff:  # floor z coordinate
            heights = sorted(list(map(float, ff.readlines())))
        base_height = heights[floor]
        height = 0.85 + base_height       
        all_traj = np.load(traj_file)
        start_time = 4
        goal_time = -1
        # start to excute a first few positions--collect context
        trajectory_to_execute = []
        cur_obs_list = []
        trajectory_to_execute = [[all_traj[i][0], all_traj[i][1], height] for i in range(start_time - context_size, start_time + 1)]
        heading_to_execute = [[all_traj[i][2], all_traj[i][3], height] for i in range(start_time - context_size, start_time + 1)]
        for i in range(len(trajectory_to_execute)):
            trajectory_to_execute[i] = np.array(trajectory_to_execute[i])
            heading_to_execute[i] = np.array(heading_to_execute[i])
        goal_pos = np.array([[all_traj[goal_time][0], all_traj[goal_time][1]]])
        current_position, current_heading_point = camera_follow_traj(sim, trajectory_to_execute[0], trajectory_to_execute, heading_to_execute, cur_obs_list, context_size)
        # Trajectory history for floor plan visualization (where the robot has been)
        trajectory_history = [np.array(p[:2]) for p in trajectory_to_execute]
        save_num = 0
        save_state = []
        arrive = False
        turn = False
        collision_num = 0
        step = 1
        floorplan_ary = np.array(Image.open(floorplan_path).convert('RGBA') )
        traj_img_save_dir = os.path.join(img_save_dir, scene_name_floor + '_' + traj_name)
        if not os.path.exists(traj_img_save_dir):
            os.makedirs(traj_img_save_dir)
        if not headless:
            cv2.namedWindow("Floor plan - robot trajectory", cv2.WINDOW_NORMAL)
        while step <= max_steps:
            with Profiler("Simulator step"):
                if len(trajectory_to_execute) > 0:
                    current_position, current_heading_point, collision, turn = camera_execute_actions(sim, current_position, trajectory_to_execute, cur_obs_list, context_size, travers_map, save_state, trajectory_history=trajectory_history)
                    trajectory_to_execute = []
                if collision:
                    print('into a stuck situation')
                    break

                if len(cur_obs_list) == context_size + 1:
                    cur_obs = torch.cat(cur_obs_list, dim=0)
                    cur_pos = np.array([current_position[:2]])
                    cur_heading = np.array([current_heading_point[:2]])
                    cur_heading = cur_pos + (cur_heading - cur_pos) / np.linalg.norm(cur_heading - cur_pos)
                    if np.linalg.norm(cur_pos - goal_pos) < arrive_th:
                        print("arrive at the target!")
                        arrive = True
                        break
                
                    actions_2d = execute_model(model, 
                                            cur_pos, 
                                            cur_heading, 
                                            goal_pos, 
                                            cur_obs, 
                                            floorplan, 
                                            metric_waypoint_spacing, 
                                            waypoint_spacing, 
                                            transform, 
                                            device, 
                                            noise_scheduler, 
                                            floorplan_ary, 
                                            os.path.join(traj_img_save_dir ,f'{save_num}.png')) 
                    save_num += 1
                    for i in range(20):
                        action = actions_2d[i]
                        trajectory_to_execute.append(np.array([action[0], action[1], height]))
                else:
                    frame = sim.renderer.render(modes=("rgb"))
                    img = Image.fromarray((255 * np.concatenate(frame, axis=1)[:, :, :3]).astype(np.uint8))
                    resized_img = resize_and_aspect_crop(img, IMG_RESIZE_SIZE, IMAGE_ASPECT_RATIO)
                    cur_obs_list.append(resized_img.unsqueeze(0))
                # Update floor plan visualization: where the robot is and where it has been
                if not headless:
                    draw_floorplan_with_trajectory(floorplan_ary, trajectory_history, current_position, travers_map.shape, goal_pos=goal_pos, window_name="Floor plan - robot trajectory")
                step += 1
        save_state = np.array(save_state)
        state_save_file = os.path.join(state_save_dir, scene_name_floor + '_' + traj_name + '.txt')
        np.savetxt(state_save_file, save_state, fmt='%f')
        # Save final floor plan with full trajectory (where robot is and has been)
        floorplan_final_path = os.path.join(traj_img_save_dir, "floorplan_trajectory.png")
        draw_floorplan_with_trajectory(floorplan_ary, trajectory_history, current_position, travers_map.shape, goal_pos=goal_pos, window_name="Floor plan - robot trajectory", save_path=floorplan_final_path, show=not headless)