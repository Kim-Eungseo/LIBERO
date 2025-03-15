import os
import numpy as np
import robosuite as suite
import matplotlib.cm as cm

from robosuite.utils.errors import RandomizationError

import libero.libero.envs.bddl_utils as BDDLUtils

from libero.libero.envs import TASK_MAPPING
from libero.libero.envs import *


class ControlEnv:
    def __init__(
        self,
        bddl_file_name,
        robots=["Panda"],
        controller="OSC_POSE",
        gripper_types="default",
        initialization_noise=None,
        use_camera_obs=True,
        has_renderer=False,
        has_offscreen_renderer=True,
        render_camera="frontview",
        render_collision_mesh=False,
        render_visual_mesh=True,
        render_gpu_device_id=-1,
        control_freq=20,
        horizon=1000,
        ignore_done=False,
        hard_reset=True,
        camera_names=[
            "agentview",
            "robot0_eye_in_hand",
        ],
        camera_heights=128,
        camera_widths=128,
        camera_depths=False,
        camera_segmentations=None,
        renderer="mujoco",
        renderer_config=None,
        **kwargs,
    ):
        assert os.path.exists(
            bddl_file_name
        ), f"[error] {bddl_file_name} does not exist!"

        controller_configs = suite.load_controller_config(default_controller=controller)

        problem_info = BDDLUtils.get_problem_info(bddl_file_name)
        # Check if we're using a multi-armed environment and use env_configuration argument if so

        # Create environment
        self.problem_name = problem_info["problem_name"]
        self.domain_name = problem_info["domain_name"]
        self.language_instruction = problem_info["language_instruction"]
        self.env = TASK_MAPPING[self.problem_name](
            bddl_file_name,
            robots=robots,
            controller_configs=controller_configs,
            gripper_types=gripper_types,
            initialization_noise=initialization_noise,
            use_camera_obs=use_camera_obs,
            has_renderer=has_renderer,
            has_offscreen_renderer=has_offscreen_renderer,
            render_camera=render_camera,
            render_collision_mesh=render_collision_mesh,
            render_visual_mesh=render_visual_mesh,
            render_gpu_device_id=render_gpu_device_id,
            control_freq=control_freq,
            horizon=horizon,
            ignore_done=ignore_done,
            hard_reset=hard_reset,
            camera_names=camera_names,
            camera_heights=camera_heights,
            camera_widths=camera_widths,
            camera_depths=camera_depths,
            camera_segmentations=camera_segmentations,
            renderer=renderer,
            renderer_config=renderer_config,
            **kwargs,
        )

    @property
    def obj_of_interest(self):
        return self.env.obj_of_interest

    def step(self, action):
        obs, _, done, info = self.env.step(action)
        reward = max(self.staged_rewards())
        return obs, reward, done, info

    def reset(self):
        success = False
        while not success:
            try:
                ret = self.env.reset()
                success = True
            except RandomizationError:
                pass
            finally:
                continue

        return ret

    def check_success(self):
        return self.env._check_success()

    @property
    def _visualizations(self):
        return self.env._visualizations

    @property
    def robots(self):
        return self.env.robots

    @property
    def sim(self):
        return self.env.sim

    def get_sim_state(self):
        return self.env.sim.get_state().flatten()

    def _post_process(self):
        return self.env._post_process()

    def _update_observables(self, force=False):
        self.env._update_observables(force=force)

    def set_state(self, mujoco_state):
        self.env.sim.set_state_from_flattened(mujoco_state)

    def reset_from_xml_string(self, xml_string):
        self.env.reset_from_xml_string(xml_string)

    def seed(self, seed):
        self.env.seed(seed)

    def set_init_state(self, init_state):
        return self.regenerate_obs_from_state(init_state)

    def regenerate_obs_from_state(self, mujoco_state):
        self.set_state(mujoco_state)
        self.env.sim.forward()
        self.check_success()
        self._post_process()
        self._update_observables(force=True)
        return self.env._get_observations()

    def close(self):
        self.env.close()
        del self.env

    def get_object(self, object_name):
        """
        Get the object from the environment.
        """

        return self.env.objects_dict[object_name]

    def _calculate_reach_reward(self, obj_names: list[str]):
        """
        로봇 그리퍼와 가장 가까운 객체 사이의 거리에 기반한 도달 보상을 계산합니다.

        Args:
            active_objs (list): 활성 객체 목록

        Returns:
            float: 도달 보상
        """
        reach_mult = 0.1
        r_reach = 0.0

        if obj_names:
            # 대상 객체까지의 최소 거리를 통해 도달 보상 계산
            dists = [
                self.env._gripper_to_target(
                    gripper=self.robots[0].gripper,
                    target=self.get_object(obj_name).root_body,
                    target_type="body",
                    return_distance=True,
                )
                for obj_name in obj_names
            ]
            r_reach = (1 - np.tanh(10.0 * min(dists))) * reach_mult

        return r_reach

    def _calculate_grasp_reward(self, obj_names: list[str]):
        """
        관심 객체를 잡고 있는지 여부에 기반한 잡기 보상을 계산합니다.

        Args:
            active_objs (list): 활성 객체 목록

        Returns:
            float: 잡기 보상
        """
        grasp_mult = 0.35

        r_grasp = (
            int(
                self.env._check_grasp(
                    gripper=self.robots[0].gripper,
                    object_geoms=[
                        g
                        for obj_name in obj_names
                        for g in self.get_object(obj_name).contact_geoms
                    ],
                )
            )
            * grasp_mult
        )

        return r_grasp

    def _calculate_lift_reward(
        self, obj_names: list[str], r_grasp: float
    ):  # TODO: this code is dependent to table manipulation
        """
        객체를 들어올리는 것에 대한 보상을 계산합니다.

        Args:
            obj_names (list): 활성 객체 목록
            r_grasp (float): 잡기 보상

        Returns:
            float: 들기 보상
        """
        grasp_mult = 0.35
        lift_mult = 0.5
        r_lift = 0.0

        if obj_names and r_grasp > 0.0:
            # 테이블 높이 + 목표 높이 (테이블 위 25cm)
            # 테이블 높이를 직접 가져오거나 적절한 기준 높이 설정
            table_height = 0.0  # 기본값

            # 테이블 객체 찾기 시도
            try:
                table_id = self._get_object_id("table")
                table_height = self.sim.data.body_xpos[table_id][2]
            except ValueError:
                # 테이블이 없으면 바닥 높이 + 약간의 오프셋 사용
                table_height = 0.0

            # 목표 높이: 테이블 위 25cm
            z_target = table_height + 0.25

            # 객체들의 현재 높이 가져오기
            object_z_locs = self.sim.data.body_xpos[
                [self._get_object_id(obj_name) for obj_name in obj_names]
            ][:, 2]

            # 목표 높이와 현재 높이의 차이 계산
            z_dists = np.maximum(z_target - object_z_locs, 0.0)

            # 들기 보상 계산
            r_lift = grasp_mult + (1 - np.tanh(15.0 * min(z_dists))) * (
                lift_mult - grasp_mult
            )

        return r_lift

    def _get_object_id(self, object_name):
        """
        Get the id of the object in the environment.
        """
        total_body_names = self.env.sim.model.body_names
        for body_name in total_body_names:
            if object_name in body_name:
                return self.env.sim.model.body_name2id(body_name)

        raise ValueError(f"Object {object_name} not found")

    def _calculate_hover_reward(self, obj_names: list[str], r_lift: float):
        """
        한 오브젝트가 다른 오브젝트 위에 있는지 확인하고 보상을 계산합니다.

        Args:
            obj_names (list[str]): 길이 2의 리스트. [위에 올려야 할 오브젝트, 아래에 있어야 할 오브젝트]
            r_lift (float): 들기 보상

        Returns:
            float: 호버링 보상
        """
        lift_mult = 0.5
        hover_mult = 0.7
        r_hover = 0.0

        # 오브젝트가 2개 있는지 확인
        if len(obj_names) != 2:
            print(f"경고: obj_names의 길이가 2가 아닙니다. 현재 길이: {len(obj_names)}")
            return r_hover

        top_obj_name = obj_names[0]  # 위에 올려야 할 오브젝트
        bottom_obj_name = obj_names[1]  # 아래에 있어야 할 오브젝트

        # 오브젝트 위치 가져오기
        top_obj_pos = self.sim.data.body_xpos[self._get_object_id(top_obj_name)]
        bottom_obj_pos = self.sim.data.body_xpos[self._get_object_id(bottom_obj_name)]

        # 수직 위치 확인 (top_obj가 bottom_obj보다 위에 있어야 함)
        vertical_check = top_obj_pos[2] > bottom_obj_pos[2]

        # 수평 위치 확인 (top_obj가 bottom_obj 위에 있어야 함)
        horizontal_dist = np.linalg.norm(top_obj_pos[:2] - bottom_obj_pos[:2])

        # 오브젝트 크기에 따라 적절한 임계값 설정 (이 값은 환경에 맞게 조정 필요)
        horizontal_threshold = 0.1  # 수평 거리 임계값
        vertical_threshold = 0.05  # 수직 거리 최소값

        # top_obj가 bottom_obj 바로 위에 있는지 확인
        is_above = vertical_check and horizontal_dist < horizontal_threshold

        # 수직 거리 계산 (적절한 높이에 있는지)
        vertical_dist = top_obj_pos[2] - bottom_obj_pos[2]
        good_height = (
            vertical_dist > vertical_threshold
            and vertical_dist < 3 * vertical_threshold
        )

        if not is_above:
            return 0.0

        if good_height:
            # 적절한 높이에 있을 때 최대 보상
            r_hover = hover_mult
        else:
            # 높이가 적절하지 않을 때 부분 보상
            height_factor = 1 - np.tanh(
                5.0 * abs(vertical_dist - 2 * vertical_threshold)
            )
            r_hover = lift_mult + height_factor * (hover_mult - lift_mult)

        return r_hover

    def staged_rewards(self):
        """
        현재 물리적 상태에 기반한 단계별 보상을 반환합니다.
        단계는 도달, 잡기, 들기, 호버링으로 구성됩니다.

        Returns:
            4-tuple:

                - (float) 도달 보상
                - (float) 잡기 보상
                - (float) 들기 보상
                - (float) 호버링 보상
        """
        # 활성 객체 목록 가져오기
        obj_of_interests = self.env.obj_of_interest

        # 각 단계별 보상 계산
        r_reach = self._calculate_reach_reward(obj_of_interests)
        r_grasp = self._calculate_grasp_reward(obj_of_interests)
        r_lift = self._calculate_lift_reward(obj_of_interests, r_grasp)
        r_hover = self._calculate_hover_reward(obj_of_interests, r_lift)

        return r_reach, r_grasp, r_lift, r_hover


class OffScreenRenderEnv(ControlEnv):
    """
    For visualization and evaluation.
    """

    def __init__(self, **kwargs):
        # This shouldn't be customized
        kwargs["has_renderer"] = False
        kwargs["has_offscreen_renderer"] = True
        super().__init__(**kwargs)


class SegmentationRenderEnv(OffScreenRenderEnv):
    """
    This wrapper will additionally generate the segmentation mask of objects,
    which is useful for comparing attention.
    """

    def __init__(
        self,
        camera_segmentations="instance",
        camera_heights=128,
        camera_widths=128,
        **kwargs,
    ):
        assert camera_segmentations is not None
        kwargs["camera_segmentations"] = camera_segmentations
        kwargs["camera_heights"] = camera_heights
        kwargs["camera_widths"] = camera_widths
        self.segmentation_id_mapping = {}
        self.instance_to_id = {}
        self.segmentation_robot_id = None
        super().__init__(**kwargs)

    def step(self, action):
        return self.env.step(action)

    def reset(self):
        obs = self.env.reset()
        self.segmentation_id_mapping = {}

        for i, instance_name in enumerate(list(self.env.model.instances_to_ids.keys())):
            if instance_name == "Panda0":
                self.segmentation_robot_id = i

        for i, instance_name in enumerate(list(self.env.model.instances_to_ids.keys())):
            if instance_name not in ["Panda0", "RethinkMount0", "PandaGripper0"]:
                self.segmentation_id_mapping[i] = instance_name

        self.instance_to_id = {
            v: k + 1 for k, v in self.segmentation_id_mapping.items()
        }
        return obs

    def get_segmentation_instances(self, segmentation_image):
        # get all instances' segmentation separately
        seg_img_dict = {}
        segmentation_image[segmentation_image > self.segmentation_robot_id] = (
            self.segmentation_robot_id + 1
        )
        seg_img_dict["robot"] = segmentation_image * (
            segmentation_image == self.segmentation_robot_id + 1
        )

        for seg_id, instance_name in self.segmentation_id_mapping.items():
            seg_img_dict[instance_name] = segmentation_image * (
                segmentation_image == seg_id + 1
            )
        return seg_img_dict

    def get_segmentation_of_interest(self, segmentation_image):
        # get the combined segmentation of obj of interest
        # 1 for obj_of_interest
        # -1.0 for robot
        # 0 for other things
        ret_seg = np.zeros_like(segmentation_image)
        for obj in self.obj_of_interest:
            ret_seg[segmentation_image == self.instance_to_id[obj]] = 1.0
        # ret_seg[segmentation_image == self.segmentation_robot_id+1] = -1.0
        ret_seg[segmentation_image == 0] = -1.0
        return ret_seg

    def segmentation_to_rgb(self, seg_im, random_colors=False):
        """
        Helper function to visualize segmentations as RGB frames.
        NOTE: assumes that geom IDs go up to 255 at most - if not,
        multiple geoms might be assigned to the same color.
        """
        # ensure all values lie within [0, 255]
        seg_im = np.mod(seg_im, 256)

        if random_colors:
            colors = randomize_colors(N=256, bright=True)
            return (255.0 * colors[seg_im]).astype(np.uint8)
        else:
            # deterministic shuffling of values to map each geom ID to a random int in [0, 255]
            rstate = np.random.RandomState(seed=2)
            inds = np.arange(256)
            rstate.shuffle(inds)
            seg_img = (
                np.array(255.0 * cm.rainbow(inds[seg_im], 10))
                .astype(np.uint8)[..., :3]
                .astype(np.uint8)
                .squeeze(-2)
            )
            print(seg_img.shape)
            cv2.imshow("Seg Image", seg_img[::-1])
            cv2.waitKey(1)
            # use @inds to map each geom ID to a color
            return seg_img


class DemoRenderEnv(ControlEnv):
    """
    For visualization and evaluation.
    """

    def __init__(self, **kwargs):
        # This shouldn't be customized
        kwargs["has_renderer"] = False
        kwargs["has_offscreen_renderer"] = True
        kwargs["render_camera"] = "frontview"

        super().__init__(**kwargs)

    def _get_observations(self):
        return self.env._get_observations()
