from abc import abstractmethod
from typing import List

import nlopt
import numpy as np
import sapien.core as sapien
import torch


class Optimizer:
    retargeting_type = "BASE"

    def __init__(
        self, 
        robot: sapien.Articulation,
        wrist_link_name: str,
        target_joint_names: List[str],
        target_link_human_indices: np.ndarray
    ):
        self.robot = robot
        self.robot_dof = robot.dof
        self.model = robot.create_pinocchio_model()
        self.wrist_link_name = wrist_link_name
        self.finger_roots_index = [5, 9, 13, 17]
        joint_names = [joint.get_name() for joint in robot.get_active_joints()]
        #print("joint_names:\n", joint_names)
        target_joint_index = []
        for target_joint_name in target_joint_names:
            if target_joint_name not in joint_names:
                raise ValueError(f"Joint {target_joint_name} given does not appear to be in robot XML.")
            target_joint_index.append(joint_names.index(target_joint_name))
        self.target_joint_names = target_joint_names
        self.target_joint_indices = np.array(target_joint_index)
        self.fixed_joint_indices = np.array([i for i in range(robot.dof) if i not in target_joint_index], dtype=int)
        self.opt = nlopt.opt(nlopt.LD_SLSQP, len(target_joint_index)) # ??? change an optimizer?
        self.dof = len(target_joint_index)

        self.root_index = self.get_link_indices(["palm"])[0]
        # self.debug_link_names = ["palm", "thtip", "fftip", "mftip", "rftip", "lftip"]
        self.debug_link_names = ["palm", "thmiddle", "thtip", "ffmiddle", "fftip", "mfmiddle", "mftip", "rfmiddle", "rftip", "lfmiddle", "lftip"]
        self.debug_link_indices = self.get_link_indices(self.debug_link_names)
        
        self.base_link_names = ["thbase", "ffknuckle", "mfknuckle", "rfknuckle", "lfmetacarpal"]
        self.base_link_indices = self.get_link_indices(self.base_link_names)
        
        # Target
        self.target_link_human_indices = target_link_human_indices

    def set_joint_limit(self, joint_limits: np.ndarray):
        #print("joint_limits:\n",joint_limits)
        if joint_limits.shape != (self.dof, 2):
            raise ValueError(f"Expect joint limits have shape: {(self.dof, 2)}, but get {joint_limits.shape}")
        self.opt.set_lower_bounds(joint_limits[:, 0].tolist())
        self.opt.set_upper_bounds(joint_limits[:, 1].tolist())

    def get_last_result(self):
        return self.opt.last_optimize_result()

    def get_link_names(self):
        return [link.get_name() for link in self.robot.get_links()]

    def get_link_indices(self, target_link_names):
        target_link_index = []
        for target_link_name in target_link_names:
            if target_link_name not in self.get_link_names():
                raise ValueError(f"Body {target_link_name} given does not appear to be in robot XML.")
            target_link_index.append(self.get_link_names().index(target_link_name))
        return target_link_index

    @abstractmethod
    def retarget(self, ref_value, fixed_qpos, last_qpos=None):
        pass

    def optimize(self, objective_fn, last_qpos):
        self.opt.set_min_objective(objective_fn)
        qpos = None
        try:
            qpos = self.opt.optimize(last_qpos)
        except RuntimeError as e:
            print(e)
            qpos = np.array(last_qpos)
        return qpos
    def get_joint_pos(self, qpos, target_link_indices, root_link_index):
        self.model.compute_forward_kinematics(qpos)

        root_link_pose = self.model.get_link_pose(root_link_index)
        target_link_poses = [self.model.get_link_pose(index) for index in target_link_indices]
        #print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        #print("root_link_index: ", root_link_index)
        #print("root_link_pose: ", root_link_pose.p)
        #print("")
        #print("target_link_poses (wrt root):\n")
        result_pos = []
        for pos in target_link_poses:
            #print(pos.p - root_link_pose.p)
            result_pos.append( [ pos.p[i] - root_link_pose.p[i] for i in range(3)] )
        #print("")
        return result_pos
    def get_position(self, qpos, ref_value, debug = False):
        joint_pos = self.get_joint_pos(qpos, self.debug_link_indices, self.root_index)
        if(debug):
            print("diff: \n")
            for i in range(len(ref_value)):
                print( abs(ref_value[i] - joint_pos[i+1]) )
        #base_link_pos = self.get_joint_pos(qpos, self.base_link_indices, self.root_index)
        #print("base_link_pos:\n", base_link_pos)
        return  joint_pos
        #body_pos = np.array([pose.p for pose in target_link_poses])
    
    def get_base_position(self, qpos):
        base_link_pos = self.get_joint_pos(qpos, self.base_link_indices, self.root_index)
        print("base_link_pos:\n", base_link_pos)
        return  base_link_pos
        #body_pos = np.array([pose.p for pose in target_link_poses])

class PositionOptimizer(Optimizer):
    retargeting_type = "POSITION"

    def __init__(
        self,
        robot: sapien.Articulation,
        wrist_link_name: str,
        target_joint_names: List[str],
        target_link_names: List[str],
        target_link_human_indices: np.ndarray,
        huber_delta=0.02,
        norm_delta=4e-3,
    ):
        super().__init__(robot, wrist_link_name, target_joint_names, target_link_human_indices)
        self.body_names = target_link_names
        self.huber_loss = torch.nn.SmoothL1Loss(beta=huber_delta)
        self.norm_delta = norm_delta

        # Sanity check and cache link indices
        self.target_link_indices = self.get_link_indices(target_link_names)

        # Use local jacobian if target link name <= 2, otherwise first cache all jacobian and then get all
        # This is only for the speed but will not affect the performance
        if len(target_link_names) <= 40:
            self.use_sparse_jacobian = True
        else:
            self.use_sparse_jacobian = False
        self.opt.set_ftol_abs(1e-5)

    def _get_objective_function(self, target_pos: np.ndarray, fixed_qpos: np.ndarray, last_qpos: np.ndarray):
        qpos = np.zeros(self.robot_dof)
        qpos[self.fixed_joint_indices] = fixed_qpos
        torch_target_pos = torch.as_tensor(target_pos)
        torch_target_pos.requires_grad_(False)

        def objective(x: np.ndarray, grad: np.ndarray) -> float:
            qpos[self.target_joint_indices] = x
            self.model.compute_forward_kinematics(qpos)
            target_link_poses = [self.model.get_link_pose(index) for index in self.target_link_indices]
            body_pos = np.array([pose.p for pose in target_link_poses])

            # Torch computation for accurate loss and grad
            torch_body_pos = torch.as_tensor(body_pos)
            torch_body_pos.requires_grad_()

            # Loss term for kinematics retargeting based on 3D position error
            huber_distance = self.huber_loss(torch_body_pos, torch_target_pos)
            # huber_distance = torch.norm(torch_body_pos - torch_target_pos, dim=1).mean()
            result = huber_distance.cpu().detach().item()

            if grad.size > 0:
                if self.use_sparse_jacobian:
                    jacobians = []
                    for i, index in enumerate(self.target_link_indices):
                        link_spatial_jacobian = self.model.compute_single_link_local_jacobian(qpos, index)[
                            :3, self.target_joint_indices
                        ]
                        link_rot = self.model.get_link_pose(index).to_transformation_matrix()[:3, :3]
                        link_kinematics_jacobian = link_rot @ link_spatial_jacobian
                        jacobians.append(link_kinematics_jacobian)
                    jacobians = np.stack(jacobians, axis=0)
                else:
                    self.model.compute_full_jacobian(qpos)
                    jacobians = [
                        self.model.get_link_jacobian(index, local=True)[:3, self.target_joint_indices]
                        for index in self.target_link_indices
                    ]

                huber_distance.backward()
                grad_pos = torch_body_pos.grad.cpu().numpy()[:, None, :]
                grad_qpos = np.matmul(grad_pos, np.array(jacobians))
                grad_qpos = grad_qpos.mean(1).sum(0)

                grad_qpos += 2 * self.norm_delta * (x - last_qpos)

                grad[:] = grad_qpos[:]

            return result

        return objective

    def retarget(self, ref_value, fixed_qpos, last_qpos=None):
        if len(fixed_qpos) != len(self.fixed_joint_indices):
            raise ValueError(
                f"Optimizer has {len(self.fixed_joint_indices)} joints but non_target_qpos {fixed_qpos} is given"
            )
        if last_qpos is None:
            last_qpos = np.zeros(self.dof)
        if isinstance(last_qpos, np.ndarray):
            last_qpos = last_qpos.astype(np.float32)
        last_qpos = list(last_qpos)
        objective_fn = self._get_objective_function(ref_value, fixed_qpos, np.array(last_qpos).astype(np.float32))
        return self.optimize(objective_fn, last_qpos)


class VectorOptimizer(Optimizer):
    retargeting_type = "VECTOR"

    def __init__(
        self,
        robot: sapien.Articulation,
        wrist_link_name: str,
        target_joint_names: List[str],
        target_origin_link_names: List[str],
        target_task_link_names: List[str],
        target_link_human_indices: np.ndarray,
        huber_delta=0.02,
        norm_delta=4e-3,
        scaling=1.0,
    ):
        super().__init__(robot, wrist_link_name, target_joint_names, target_link_human_indices)
        self.origin_link_names = target_origin_link_names
        self.task_link_names = target_task_link_names
        self.huber_loss = torch.nn.SmoothL1Loss(beta=huber_delta, reduction="mean")
        self.norm_delta = norm_delta
        self.scaling = scaling

        # Computation cache for better performance
        # For one link used in multiple vectors, e.g. hand palm, we do not want to compute it multiple times
        self.computed_link_names = list(set(target_origin_link_names).union(set(target_task_link_names)))
        self.origin_link_indices = torch.tensor(
            [self.computed_link_names.index(name) for name in target_origin_link_names]
        )
        self.task_link_indices = torch.tensor([self.computed_link_names.index(name) for name in target_task_link_names])

        # Sanity check and cache link indices
        self.robot_link_indices = self.get_link_indices(self.computed_link_names)

        # Use local jacobian if target link name <= 2, otherwise first cache all jacobian and then get all
        # This is only for the speed but will not affect the performance
        if len(self.computed_link_names) <= 40:
            self.use_sparse_jacobian = True
        else:
            self.use_sparse_jacobian = False
        self.opt.set_ftol_abs(1e-6)

    def _get_objective_function(self, target_vector: np.ndarray, fixed_qpos: np.ndarray, last_qpos: np.ndarray):
        qpos = np.zeros(self.robot_dof)
        qpos[self.fixed_joint_indices] = fixed_qpos
        torch_target_vec = torch.as_tensor(target_vector) * self.scaling
        torch_target_vec.requires_grad_(False)

        def objective(x: np.ndarray, grad: np.ndarray) -> float:
            qpos[self.target_joint_indices] = x
            self.model.compute_forward_kinematics(qpos)
            target_link_poses = [self.model.get_link_pose(index) for index in self.robot_link_indices]
            body_pos = np.array([pose.p for pose in target_link_poses])

            # Torch computation for accurate loss and grad
            torch_body_pos = torch.as_tensor(body_pos)
            torch_body_pos.requires_grad_()

            # Index link for computation
            origin_link_pos = torch_body_pos[self.origin_link_indices, :]
            task_link_pos = torch_body_pos[self.task_link_indices, :]
            robot_vec = task_link_pos - origin_link_pos

            # Loss term for kinematics retargeting based on 3D position error
            vec_dist = torch.norm(robot_vec - torch_target_vec, dim=1, keepdim=False)
            huber_distance = self.huber_loss(vec_dist, torch.zeros_like(vec_dist))
            result = huber_distance.cpu().detach().item()

            if grad.size > 0:
                if self.use_sparse_jacobian:
                    jacobians = []
                    for i, index in enumerate(self.robot_link_indices):
                        link_spatial_jacobian = self.model.compute_single_link_local_jacobian(qpos, index)[
                            :3, self.target_joint_indices
                        ]
                        link_rot = self.model.get_link_pose(index).to_transformation_matrix()[:3, :3]
                        link_kinematics_jacobian = link_rot @ link_spatial_jacobian
                        jacobians.append(link_kinematics_jacobian)
                    jacobians = np.stack(jacobians, axis=0)
                else:
                    self.model.compute_full_jacobian(qpos)
                    jacobians = [
                        self.model.get_link_jacobian(index, local=True)[:3, self.target_joint_indices]
                        for index in self.robot_link_indices
                    ]

                huber_distance.backward()
                grad_pos = torch_body_pos.grad.cpu().numpy()[:, None, :]
                grad_qpos = np.matmul(grad_pos, np.array(jacobians))
                grad_qpos = grad_qpos.mean(1).sum(0)

                grad_qpos += 2 * self.norm_delta * (x - last_qpos)

                grad[:] = grad_qpos[:]

            return result

        return objective

    def retarget(self, ref_value, fixed_qpos, last_qpos=None):
        if len(fixed_qpos) != len(self.fixed_joint_indices):
            raise ValueError(
                f"Optimizer has {len(self.fixed_joint_indices)} joints but non_target_qpos {fixed_qpos} is given"
            )
        if last_qpos is None:
            last_qpos = np.zeros(self.dof)
        last_qpos = list(last_qpos)
        objective_fn = self._get_objective_function(ref_value, fixed_qpos, np.array(last_qpos).astype(np.float32))
        return self.optimize(objective_fn, last_qpos)


class DexPilotOptimizer(Optimizer):
    """Retargeting optimizer using the method proposed in DexPilot

    This is a broader adaptation of the original optimizer delineated in the DexPilot paper.
    While the initial DexPilot study focused solely on the four-fingered Allegro Hand, this version of the optimizer
    embraces the same principles for both four-fingered and five-fingered hands. It projects the distance between the
    thumb and the other fingers to facilitate more stable grasping.
    Reference: https://arxiv.org/abs/1910.03135

    Args:
        robot:
        target_joint_names:
        finger_tip_link_names:
        wrist_link_name:
        gamma:
        project_dist:
        escape_dist:
        eta1:
        eta2:
        scaling:
    """

    retargeting_type = "DEXPILOT"

    def __init__(
        self,
        robot: sapien.Articulation,
        wrist_link_name: str,
        target_joint_names: List[str],
        finger_tip_link_names: List[str],
        huber_delta=0.03,
        norm_delta=4e-3,
        # DexPilot parameters
        # gamma=2.5e-3,
        project_dist=0.03,
        escape_dist=0.05,
        eta1=1e-4,
        eta2=3e-2,
        scaling=1.0,
    ):
        if len(finger_tip_link_names) < 4 or len(finger_tip_link_names) > 5:
            raise ValueError(f"DexPilot optimizer can only be applied to hands with four or five fingers")
        is_four_finger = len(finger_tip_link_names) == 4

        if is_four_finger:
            origin_link_index = [2, 3, 4, 3, 4, 4, 0, 0, 0, 0]
            task_link_index = [1, 1, 1, 2, 2, 3, 1, 2, 3, 4]
            self.num_fingers = 4
        else:
            origin_link_index = [2, 3, 4, 5, 3, 4, 5, 4, 5, 5, 0, 0, 0, 0, 0]
            task_link_index = [1, 1, 1, 1, 2, 2, 2, 3, 3, 4, 1, 2, 3, 4, 5]
            self.num_fingers = 5

        target_link_human_indices = (np.stack([origin_link_index, task_link_index], axis=0) * 4).astype(int)
        link_names = [wrist_link_name] + finger_tip_link_names
        target_origin_link_names = [link_names[index] for index in origin_link_index]
        target_task_link_names = [link_names[index] for index in task_link_index]

        super().__init__(robot, wrist_link_name, target_joint_names, target_link_human_indices)
        # print("target_joint_names:",target_joint_names)
        self.origin_link_names = target_origin_link_names
        self.task_link_names = target_task_link_names
        self.scaling = scaling
        self.huber_loss = torch.nn.SmoothL1Loss(beta=huber_delta, reduction="none")
        self.norm_delta = norm_delta

        # DexPilot parameters
        self.project_dist = project_dist
        self.escape_dist = escape_dist
        self.eta1 = eta1
        self.eta2 = eta2

        # Computation cache for better performance
        # For one link used in multiple vectors, e.g. hand palm, we do not want to compute it multiple times
        self.computed_link_names = list(set(target_origin_link_names).union(set(target_task_link_names)))
        self.origin_link_indices = torch.tensor(
            [self.computed_link_names.index(name) for name in target_origin_link_names]
        )
        self.task_link_indices = torch.tensor([self.computed_link_names.index(name) for name in target_task_link_names])

        # Sanity check and cache link indices
        self.robot_link_indices = self.get_link_indices(self.computed_link_names)

        # Use local jacobian if target link name <= 2, otherwise first cache all jacobian and then get all
        # This is only for the speed but will not affect the performance
        if len(self.computed_link_names) <= 40:
            self.use_sparse_jacobian = True
        else:
            self.use_sparse_jacobian = False
        self.opt.set_ftol_abs(1e-6)

        # DexPilot cache
        if is_four_finger:
            self.projected = np.zeros(6, dtype=bool)
            self.s2_project_index_origin = np.array([1, 2, 2], dtype=int)
            self.s2_project_index_task = np.array([0, 0, 1], dtype=int)
            self.projected_dist = np.array([eta1] * 3 + [eta2] * 3)
        else:
            self.projected = np.zeros(10, dtype=bool)
            self.s2_project_index_origin = np.array([1, 2, 3, 2, 3, 3], dtype=int)
            self.s2_project_index_task = np.array([0, 0, 0, 1, 1, 2], dtype=int)
            self.projected_dist = np.array([eta1] * 4 + [eta2] * 6)

    def _get_objective_function_four_finger(
        self, target_vector: np.ndarray, fixed_qpos: np.ndarray, last_qpos: np.ndarray
    ):
        target_vector = target_vector.astype(np.float32)
        qpos = np.zeros(self.robot_dof)
        qpos[self.fixed_joint_indices] = fixed_qpos

        len_proj = len(self.projected)
        len_s2 = len(self.s2_project_index_task) # 6
        len_s1 = len_proj - len_s2 # 4

        # Update projection indicator
        target_vec_dist = np.linalg.norm(target_vector[:len_proj], axis=1)
        # connection to thumbs 0 or 1
        self.projected[:len_s1][target_vec_dist[0:len_s1] < self.project_dist] = True
        self.projected[:len_s1][target_vec_dist[0:len_s1] > self.escape_dist] = False

        #connection between others
        self.projected[len_s1:len_proj] = np.logical_and(
            self.projected[:len_s1][self.s2_project_index_origin], self.projected[:len_s1][self.s2_project_index_task]
        )
        self.projected[len_s1:len_proj] = np.logical_and(
            self.projected[len_s1:len_proj], target_vec_dist[len_s1:len_proj] <= 0.03
        )

        # Update weight vector
        normal_weight = np.ones(len_proj, dtype=np.float32) * 1
        high_weight = np.array([200] * len_s1 + [400] * len_s2, dtype=np.float32)
        weight = np.where(self.projected, high_weight, normal_weight)

        # We change the weight to 10 instead of 1 here, for vector originate from wrist to fingertips
        # This ensures better intuitive mapping due wrong pose detection
        weight = torch.from_numpy(
            np.concatenate([weight, np.ones(self.num_fingers, dtype=np.float32) * len_proj + self.num_fingers])
        )

        # Compute reference distance vector
        normal_vec = target_vector * self.scaling  # (15, 3)
        dir_vec = target_vector[:len_proj] / (target_vec_dist[:, None] + 1e-6)  # (10, 3)
        projected_vec = dir_vec * self.projected_dist[:, None]  # (10, 3)

        # Compute final reference vector
        reference_vec = np.where(self.projected[:, None], projected_vec, normal_vec[:len_proj])  # (10, 3)
        reference_vec = np.concatenate([reference_vec, normal_vec[len_proj:]], axis=0)  # (15, 3)
        torch_target_vec = torch.as_tensor(reference_vec, dtype=torch.float32)
        torch_target_vec.requires_grad_(False)

        def objective(x: np.ndarray, grad: np.ndarray) -> float:
            qpos[self.target_joint_indices] = x
            self.model.compute_forward_kinematics(qpos)
            target_link_poses = [self.model.get_link_pose(index) for index in self.robot_link_indices]
            body_pos = np.array([pose.p for pose in target_link_poses])

            # Torch computation for accurate loss and grad
            torch_body_pos = torch.as_tensor(body_pos)
            torch_body_pos.requires_grad_()

            # Index link for computation
            origin_link_pos = torch_body_pos[self.origin_link_indices, :]
            task_link_pos = torch_body_pos[self.task_link_indices, :]
            robot_vec = task_link_pos - origin_link_pos

            # Loss term for kinematics retargeting based on 3D position error
            # Different from the original DexPilot, we use huber loss here instead of the squared dist
            vec_dist = torch.norm(robot_vec - torch_target_vec, dim=1, keepdim=False)
            huber_distance = (
                self.huber_loss(vec_dist, torch.zeros_like(vec_dist)) * weight / (robot_vec.shape[0])
            ).sum()
            huber_distance = huber_distance.sum()
            result = huber_distance.cpu().detach().item()

            if grad.size > 0:
                if self.use_sparse_jacobian:
                    jacobians = []
                    for i, index in enumerate(self.robot_link_indices):
                        link_spatial_jacobian = self.model.compute_single_link_local_jacobian(qpos, index)[
                            :3, self.target_joint_indices
                        ]
                        link_rot = self.model.get_link_pose(index).to_transformation_matrix()[:3, :3]
                        link_kinematics_jacobian = link_rot @ link_spatial_jacobian
                        jacobians.append(link_kinematics_jacobian)
                    jacobians = np.stack(jacobians, axis=0)
                else:
                    self.model.compute_full_jacobian(qpos)
                    jacobians = [
                        self.model.get_link_jacobian(index, local=True)[:3, self.target_joint_indices]
                        for index in self.robot_link_indices
                    ]

                huber_distance.backward()
                grad_pos = torch_body_pos.grad.cpu().numpy()[:, None, :]
                grad_qpos = np.matmul(grad_pos, np.array(jacobians))
                grad_qpos = grad_qpos.mean(1).sum(0)

                # In the original DexPilot, γ = 2.5 × 10−3 is a weight on regularizing the Allegro angles to zero
                # which is equivalent to fully opened the hand
                # In our implementation, we regularize the joint angles to the previous joint angles
                grad_qpos += 2 * self.norm_delta * (x - last_qpos)

                grad[:] = grad_qpos[:]

            return result

        return objective

    def retarget(self, ref_value, fixed_qpos, last_qpos=None):
        if len(fixed_qpos) != len(self.fixed_joint_indices):
            raise ValueError(
                f"Optimizer has {len(self.fixed_joint_indices)} joints but non_target_qpos {fixed_qpos} is given"
            )
        if last_qpos is None:
            last_qpos = np.zeros(self.dof)
        objective_fn = self._get_objective_function_four_finger(
            ref_value, fixed_qpos, np.array(last_qpos).astype(np.float32)
        )
        return self.optimize(objective_fn, last_qpos)

class CustomOptimizer(Optimizer):
    retargeting_type = "CUSTOM"

    def __init__(
        self,
        robot: sapien.Articulation,
        wrist_link_name: str,
        target_joint_names: List[str],
        target_origin_link_names: List[str],
        target_task_link_names: List[str],
        target_link_human_indices: np.ndarray,

        finger_tip_link_names: List[str],

        huber_delta=0.02,
        norm_delta=4e-3,

        # params from dexpilot
        project_dist=0.03,
        escape_dist=0.05,
        eta1=1e-4,
        eta2=3e-2,
        scaling=1.0,
    ):
        super().__init__(robot, wrist_link_name, target_joint_names, target_link_human_indices)

        # from dexpilot
        self.dexpilot_origin_link_index = [ 11, 12, 13, 14] # only between fingers
        self.dexpilot_task_link_index =   [ 10, 10, 10, 10]
        self.num_fingers = 5
        
        self.origin_link_names = target_origin_link_names
        self.task_link_names = target_task_link_names
        self.huber_loss = torch.nn.SmoothL1Loss(beta=huber_delta, reduction="mean")
        self.norm_delta = norm_delta
        #self.scaling = scaling
        self.scaling = np.array([
            [1.2, 1.2, 1.2],
            [1.2, 1.2, 1.2],

            [1.2, 1.2, 1.2],
            [1.2, 1.2, 1.2],
            
            [1.2, 1.2, 1.2],
            [1.2, 1.2, 1.2],

            [1.2, 1.2, 1.2],
            [1.2, 1.2, 1.2],
            
            [1.2, 1.2, 1.2],
            [1.2, 1.2, 1.2],

            [1.2, 1.2, 1.2],
            [1.2, 1.2, 1.2],
            [1.2, 1.2, 1.2],
            [1.2, 1.2, 1.2],
            [1.2, 1.2, 1.2],
            [1.2, 1.2, 1.2]
            ])

        # Computation cache for better performance
        # For one link used in multiple vectors, e.g. hand palm, we do not want to compute it multiple times
        #self.computed_link_names = list(set(target_origin_link_names).union(set(target_task_link_names)))
        self.computed_link_names = ["palm", 
                                    "thbase", "thmiddle", "thtip", 
                                    "ffknuckle", "ffmiddle", "fftip", 
                                    "mfknuckle","mfmiddle", "mftip", 
                                    "rfknuckle", "rfmiddle", "rftip", 
                                    "lfknuckle", "lfmiddle", "lftip"]

        # self.computed_link_names = ["palm", "thtip", "fftip", "mftip", "rftip", "lftip"]
        # self.computed_link_names = ["palm", "thdistal", "ffdistal", "mfdistal", "rfdistal", "lfdistal" ]
        

        self.origin_link_indices = torch.tensor(
            [self.computed_link_names.index(name) for name in target_origin_link_names]
        )

        # self.root_index = self.get_link_indices(["palm"])[0]

        self.task_link_indices = torch.tensor([self.computed_link_names.index(name) for name in target_task_link_names])

        # Sanity check and cache link indices
        self.robot_link_indices = self.get_link_indices(self.computed_link_names)

        # DexPilot parameters
        self.project_dist = project_dist
        self.escape_dist = escape_dist
        self.eta1 = eta1
        self.eta2 = eta2
        # DexPilot cache
        self.projected = np.zeros(10, dtype=bool)
        self.s2_project_index_origin = np.array([1, 2, 3, 2, 3, 3], dtype=int)
        self.s2_project_index_task = np.array([0, 0, 0, 1, 1, 2], dtype=int)
        self.projected_dist = np.array([eta1] * 4 + [eta2] * 6)
        
        # Use local jacobian if target link name <= 2, otherwise first cache all jacobian and then get all
        # This is only for the speed but will not affect the performance
        if len(self.computed_link_names) <= 40:
            self.use_sparse_jacobian = True
        else:
            self.use_sparse_jacobian = False
        self.opt.set_ftol_abs(1e-6)

    def _get_objective_function(self, target_vector: np.ndarray, fixed_qpos: np.ndarray, last_qpos: np.ndarray):
        qpos = np.zeros(self.robot_dof)
        qpos[self.fixed_joint_indices] = fixed_qpos

        len_proj = len(self.projected) # 10 
        len_s2 = len(self.s2_project_index_task) # 6
        len_s1 = len_proj - len_s2 # 4
        
        dexpilot_target_vector = target_vector[self.dexpilot_origin_link_index,:] - target_vector[self.dexpilot_task_link_index,:]
        # Update projection indicator
        target_vec_dist = np.linalg.norm(dexpilot_target_vector[:4], axis=1) #dist between fingers
        #print("target_vec_dist:\n", target_vec_dist)
        #print("self.project_dist:\n", self.project_dist)
        #print("target_vec_dist[0:len_s1] < self.project_dist:\n", target_vec_dist[0:len_s1] < self.project_dist)
        self.projected[:len_s1][target_vec_dist[0:len_s1] < self.project_dist] = True # dist between thumb & others
        self.projected[:len_s1][target_vec_dist[0:len_s1] > self.project_dist] = False # dist between thumb & others

        #print("self.projected[:len_s1]: \n", self.projected[:len_s1])

        reference_vec = target_vector * self.scaling
        
        # if self.projected[0]:
        #     mid_point = (reference_vec[3,:] + reference_vec[1,:] ) / 2.0
        #     print("target_vector[1,:]:\n", reference_vec[1,:])
        #     print("target_vector[3,:]:\n", reference_vec[3,:])   
        #     print("mid_point:\n", mid_point)
        #     #reference_vec[3,:] = reference_vec[1,:].copy()
        #     #reference_vec[1,:] = reference_vec[3,:].copy()
        #     #reference_vec[1,:] = mid_point
        #     #reference_vec[3,:] = mid_point            
        #     print("after reference_vec[1,:]:\n", reference_vec[1,:])
        #     print("after reference_vec[3,:]:\n", reference_vec[3,:])   
     

        # if self.projected[1]:
        #     mid_point = (reference_vec[5,:] + reference_vec[1,:] ) / 2.0
        #     print("target_vector[1,:]:\n", reference_vec[1,:])
        #     print("target_vector[5,:]:\n", reference_vec[5,:])   
        #     print("mid_point:\n", mid_point)
        #     #reference_vec[1,:] = mid_point
        #     #reference_vec[5,:] = mid_point
        #     tmp = reference_vec[1,:].copy()
        #     reference_vec[1,:] = reference_vec[5,:].copy()
        #     reference_vec[5,:] = tmp
        #     #reference_vec[5,:] = reference_vec[1,:].copy()
        #     print("after reference_vec[1,:]:\n", reference_vec[1,:])
        #     print("after reference_vec[5,:]:\n", reference_vec[5,:])   

        # if self.projected[2]:
        #     mid_point = (reference_vec[7,:] + reference_vec[1,:] ) / 2.0
        #     print("target_vector[1,:]:\n", reference_vec[1,:])
        #     print("target_vector[7,:]:\n", reference_vec[7,:])   
        #     print("mid_point:\n", mid_point)
        #     #reference_vec[7,:] = reference_vec[1,:].copy()
        #     #tmp = reference_vec[1,:].copy()
        #     #reference_vec[1,:] = reference_vec[7,:].copy()
        #     #reference_vec[7,:] = tmp
        #     reference_vec[1,:] = mid_point
        #     reference_vec[7,:] = mid_point
        #     print("after reference_vec[1,:]:\n", reference_vec[1,:])
        #     print("after reference_vec[7,:]:\n", reference_vec[7,:])  

        # if self.projected[3]:
        #     mid_point = (reference_vec[9,:] + reference_vec[1,:] ) / 2.0
        #     print("target_vector[1,:]:\n", reference_vec[1,:])
        #     print("target_vector[9,:]:\n", reference_vec[9,:])   
        #     print("mid_point:\n", mid_point)
        #     #reference_vec[9,:] = reference_vec[1,:].copy()
        #     reference_vec[1,:] = mid_point
        #     reference_vec[9,:] = mid_point
        #     print("after reference_vec[1,:]:\n", reference_vec[1,:])
        #     print("after reference_vec[9,:]:\n", reference_vec[9,:])  
        
        
        torch_target_vec = torch.as_tensor(reference_vec, dtype=torch.float32)
        torch_target_vec.requires_grad_(False)

        def objective(x: np.ndarray, grad: np.ndarray) -> float:
            qpos[self.target_joint_indices] = x
            self.model.compute_forward_kinematics(qpos)
            target_link_poses = [self.model.get_link_pose(index) for index in self.robot_link_indices]

            body_pos = np.array([pose.p for pose in target_link_poses])

            # Torch computation for accurate loss and grad
            torch_body_pos = torch.as_tensor(body_pos)
            torch_body_pos.requires_grad_()

            # Index link for computation
            origin_link_pos = torch_body_pos[self.origin_link_indices, :]
            task_link_pos = torch_body_pos[self.task_link_indices, :]
            robot_vec = task_link_pos - origin_link_pos

            # Loss term for kinematics retargeting based on 3D position error
            vec_dist = torch.norm(robot_vec - torch_target_vec, dim=1, keepdim=False)
            
            # middle_eta = 1.0
            # tip_eta = 10.0
            weight = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
            
            # if self.projected[0]:
            #     weight[1] = tip_eta
            #     weight[3] = tip_eta
            # if self.projected[1]:
            #     weight[1] = tip_eta
            #     weight[5] = tip_eta
            # if self.projected[2]:
            #     weight[1] = tip_eta
            #     weight[7] = tip_eta
            # if self.projected[3]:
            #     weight[1] = tip_eta
            #     weight[9] = tip_eta

            weight_tensor = torch.as_tensor(weight)
            vec_dist = vec_dist * torch.as_tensor(weight_tensor)
            
            huber_distance = self.huber_loss(vec_dist, torch.zeros_like(vec_dist))
            result = huber_distance.cpu().detach().item()

            if grad.size > 0:
                if self.use_sparse_jacobian:
                    jacobians = []
                    for i, index in enumerate(self.robot_link_indices):
                        link_spatial_jacobian = self.model.compute_single_link_local_jacobian(qpos, index)[
                            :3, self.target_joint_indices
                        ]
                        link_rot = self.model.get_link_pose(index).to_transformation_matrix()[:3, :3]
                        link_kinematics_jacobian = link_rot @ link_spatial_jacobian
                        jacobians.append(link_kinematics_jacobian)
                    jacobians = np.stack(jacobians, axis=0)
                else:
                    self.model.compute_full_jacobian(qpos)
                    jacobians = [
                        self.model.get_link_jacobian(index, local=True)[:3, self.target_joint_indices]
                        for index in self.robot_link_indices
                    ]

                huber_distance.backward()
                grad_pos = torch_body_pos.grad.cpu().numpy()[:, None, :]
                grad_qpos = np.matmul(grad_pos, np.array(jacobians))
                grad_qpos = grad_qpos.mean(1).sum(0)

                grad_qpos += 2 * self.norm_delta * (x - last_qpos)

                grad[:] = grad_qpos[:]

            return result

        return objective

    def retarget(self, ref_value, fixed_qpos, last_qpos=None, debug = False):
        if len(fixed_qpos) != len(self.fixed_joint_indices):
            raise ValueError(
                f"Optimizer has {len(self.fixed_joint_indices)} joints but non_target_qpos {fixed_qpos} is given"
            )
        if last_qpos is None:
            last_qpos = np.zeros(self.dof)
        last_qpos = list(last_qpos)
        objective_fn = self._get_objective_function(ref_value, fixed_qpos, np.array(last_qpos).astype(np.float32))
        result = self.optimize(objective_fn, last_qpos)
        if debug:
            target_vec = ref_value * self.scaling
            print("target_vec:\n", target_vec)    
        return result
    def get_root_base_pos(self, target_vector, hand_type = "Right"):
        reference_vec = target_vector * self.scaling
        # palm_to_thumb_base= target_vector[10,:] - target_vector[1,:]
        # palm_to_ff_base= target_vector[11,:] - target_vector[3,:]
        # palm_to_mf_base= target_vector[12,:] - target_vector[5,:]
        # palm_to_rf_base= target_vector[13,:] - target_vector[7,:]
        palm_to_lf_base = reference_vec[10,:]

        if(hand_type == "Right"):
            palm_to_thumb_base = np.array( [0.008579908, 0.033999946, 0.029000089] )
            palm_to_ff_base = np.array( [0.0, 0.033, 0.09499999] )
            palm_to_mf_base = np.array( [0.0, 0.011, 0.099] )
            palm_to_rf_base = np.array( [0.0, -0.011, 0.09499999] )
            #palm_to_lf_base = np.array( [0.0, -0.033000004, 0.020710003] )
        else:
            #print("WTF")
            palm_to_thumb_base = np.array( [0.008579908, -0.033999946, 0.029000089] )
            palm_to_ff_base = np.array( [0.0, -0.033, 0.09499999] )
            palm_to_mf_base = np.array( [0.0, -0.011, 0.099] )
            palm_to_rf_base = np.array( [0.0, 0.011, 0.09499999] )
            #palm_to_lf_base = np.array( [0.0, 0.033000004, 0.020710003] )            
        result = []
        
        thumb_mid = palm_to_thumb_base + reference_vec[0,:]
        thumb_tip = palm_to_thumb_base + reference_vec[1,:]
        result.append(thumb_mid)
        result.append(thumb_tip)

        ff_mid = palm_to_ff_base + reference_vec[2,:]
        ff_tip = palm_to_ff_base + reference_vec[3,:]
        result.append(ff_mid)
        result.append(ff_tip)

        mf_mid = palm_to_mf_base + reference_vec[4,:]
        mf_tip = palm_to_mf_base + reference_vec[5,:]
        result.append(mf_mid)
        result.append(mf_tip)

        rf_mid = palm_to_rf_base + reference_vec[6,:]
        rf_tip = palm_to_rf_base + reference_vec[7,:]
        result.append(rf_mid)
        result.append(rf_tip)

        lf_mid = palm_to_lf_base + reference_vec[8,:]
        lf_tip = palm_to_lf_base + reference_vec[9,:]
        result.append(lf_mid)
        result.append(lf_tip)

        result = np.array(result)

        return result