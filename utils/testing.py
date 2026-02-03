import os
import numpy as np
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

def test_single_scene(config, scene_name_floor: str):
    print("testing scene: ", scene_name_floor)  
    scene_id = scene_name_floor.split('_')[0]
    floor = int(scene_name_floor.split('_')[1])
    if floor == 0:
        foucused_map_path =  os.path.join(config["trav_maps_path"], scene_id, 'foucused_map.png')
    else:
        foucused_map_path =  os.path.join(config["trav_maps_path"], scene, 'foucused_map.png')

    
    # load scene
    s = Simulator(
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
    s.import_scene(scene)
    if not headless:
        s.viewer.initial_pos = config["initial_pos"]
        s.viewer.initial_view_direction = config["initial_view_direction"]
        s.viewer.reset_viewer()
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
        current_position, current_heading_point = camera_follow_traj(s, trajectory_to_execute[0], trajectory_to_execute, heading_to_execute, cur_obs_list, context_size)         
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
        while step <= max_steps:
            with Profiler("Simulator step"):
                if len(trajectory_to_execute) > 0:
                    current_position, current_heading_point, collision, turn = camera_execute_actions(s, current_position, trajectory_to_execute, cur_obs_list, context_size, travers_map, save_state)
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
                                                metric_waipoint_spacing, 
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
                    frame = s.renderer.render(modes=("rgb"))
                    img = Image.fromarray((255 * np.concatenate(frame, axis=1)[:, :, :3]).astype(np.uint8))
                    resized_img = resize_and_aspect_crop(img, IMG_RESIZE_SIZE, IMAGE_ASPECT_RATIO)
                    cur_obs_list.append(resized_img.unsqueeze(0))
                step += 1
        save_state = np.array(save_state)
        state_save_file = os.path.join(state_save_dir, scene_name_floor + '_' + traj_name + '.txt')
        np.savetxt(state_save_file, save_state, fmt='%f')
    s.disconnect()