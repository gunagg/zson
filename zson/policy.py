import os
from typing import Dict, Optional, Tuple

import clip
import torch
import torch.nn as nn
import torch.nn.functional as F
from gym import spaces
from habitat import logger
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.rl.models.rnn_state_encoder import build_rnn_state_encoder
from habitat_baselines.rl.ppo.policy import Net, Policy

from zson.sensors import (
    CachedGoalSensor,
    ObjectGoalKShotImagePromptSensor,
    ObjectGoalPromptSensor,
    PanoramicImageGoalSensor,
)
from zson.transforms import get_transform
from zson.visual_encoder import VisualEncoder


class ZSONPolicyNet(Net):
    def __init__(
        self,
        observation_space: spaces.Dict,
        action_space: spaces.Discrete,
        backbone: str,
        baseplanes: int,
        hidden_size: int,
        rnn_type: str,
        rnn_num_layers: int,
        use_data_aug: bool,
        use_clip_obs_encoder: bool,
        run_type: str,
        clip_model: str,
        obs_clip_model: str,
        pretrained_encoder: Optional[str] = None,
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.use_clip_obs_encoder = use_clip_obs_encoder
        rnn_input_size = 0

        # visual encoder
        if self.use_clip_obs_encoder:
            name = "clip"
            if use_data_aug and run_type == "train":
                name = "clip+weak"
            self.visual_transform = get_transform(name=name, size=224)
            self.visual_encoder, _ = clip.load(obs_clip_model)
            for p in self.visual_encoder.parameters():
                p.requires_grad = False
            self.visual_encoder.eval()
            visual_size = self.visual_encoder.visual.output_dim
            assert pretrained_encoder is None
        else:
            name = "resize"
            if use_data_aug and run_type == "train":
                name = "resize+weak"
            self.visual_transform = get_transform(name=name, size=128)
            self.visual_encoder = VisualEncoder(
                backbone=backbone, baseplanes=baseplanes, spatial_size=128
            )
            visual_size = self.visual_encoder.output_size

            if pretrained_encoder is not None:
                assert os.path.exists(pretrained_encoder)
                checkpoint = torch.load(pretrained_encoder, map_location="cpu")
                state_dict = checkpoint["teacher"]
                state_dict = {
                    k.replace("module.", ""): v for k, v in state_dict.items()
                }
                msg = self.visual_encoder.load_state_dict(
                    state_dict=state_dict, strict=False
                )
                logger.info("Using weights from {}: {}".format(pretrained_encoder, msg))

        # visual encoder mlp
        self.policy_mlp = nn.Sequential(
            nn.Linear(visual_size, hidden_size // 2),
            nn.ReLU(True),
            nn.Linear(hidden_size // 2, hidden_size),
        )

        # update rnn input size
        self.visual_keys = [k for k in observation_space.spaces if k.startswith("rgb")]
        rnn_input_size += len(self.visual_keys) * hidden_size

        # goal embedding
        goal_uuids = [
            PanoramicImageGoalSensor.cls_uuid,
            ObjectGoalPromptSensor.cls_uuid,
            CachedGoalSensor.cls_uuid,
            ObjectGoalKShotImagePromptSensor.cls_uuid,
        ]
        goal_uuid = [uuid for uuid in observation_space.spaces if uuid in goal_uuids]
        assert len(goal_uuid) == 1
        goal_uuid = goal_uuid[0]

        # CLIP goal encoder
        if goal_uuid != CachedGoalSensor.cls_uuid:
            assert run_type == "eval"
            self.clip_transform = get_transform(name="clip", size=224)
            self.clip, _ = clip.load(clip_model)
            for p in self.clip.parameters():
                p.requires_grad = False
            self.clip.eval()
            goal_size = self.clip.visual.output_dim
        else:
            goal_size = 1024 if clip_model == "RN50" else 768

        # goal embedding size
        rnn_input_size += goal_size

        # previous action embedding
        self.prev_action_embedding = nn.Embedding(action_space.n + 1, 32)
        rnn_input_size += 32

        # state encoder
        self.state_encoder = build_rnn_state_encoder(
            input_size=rnn_input_size,
            hidden_size=hidden_size,
            rnn_type=rnn_type,
            num_layers=rnn_num_layers,
        )

    @property
    def output_size(self):
        return self.hidden_size

    @property
    def is_blind(self):
        return False

    @property
    def num_recurrent_layers(self):
        return self.state_encoder.num_recurrent_layers

    def forward(
        self,
        observations: Dict[str, torch.Tensor],
        rnn_hidden_states,
        prev_actions,
        masks,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """In this function `N` will denote the number of environments, `T` the
        number of timesteps, `V` the number of views and `K` is number of shots.
        """

        x = []

        # stack rgb observations
        obs = torch.stack([observations[k] for k in self.visual_keys], dim=1)

        # get shapes
        TN = obs.size(0)
        N = rnn_hidden_states.size(0)
        T = TN // N
        OV = obs.size(1)

        # visual encoder
        obs = obs.flatten(0, 1)  # TN * OV x H x W x 3
        rgb = self.visual_transform(obs, T, N, OV)  # TN * OV x h x w x 3
        if self.use_clip_obs_encoder:
            rgb = self.visual_encoder.encode_image(rgb).float()
        else:
            rgb = self.visual_encoder(rgb)  # TN * OV x D
        rgb = self.policy_mlp(rgb)  # TN * OV x d
        rgb = rgb.reshape(TN, OV, -1)  # TN x OV x d
        rgb = rgb.flatten(1)  # TN x OV * d
        x.append(rgb)

        # goal embedding
        if CachedGoalSensor.cls_uuid in observations:
            goal = observations[CachedGoalSensor.cls_uuid].float()
            GV = goal.size(1)
            goal = goal.flatten(0, 1)  # TN * GV x D
            goal /= goal.norm(dim=-1, keepdim=True)  # TN * GV x D

        if PanoramicImageGoalSensor.cls_uuid in observations:
            goal = observations[PanoramicImageGoalSensor.cls_uuid]
            GV = goal.size(1)
            goal = goal.flatten(0, 1)  # TN * GV x H x W x 3
            with torch.no_grad():
                goal = self.clip_transform(goal, T, N, GV)
                goal = self.clip.encode_image(goal).float()
                goal /= goal.norm(dim=-1, keepdim=True)

        if ObjectGoalPromptSensor.cls_uuid in observations:
            goal = observations[ObjectGoalPromptSensor.cls_uuid]  # TN x 1 x F
            GV = goal.size(1)
            goal = goal.flatten(0, 1)  # TN * GV x F
            with torch.no_grad():
                goal = self.clip.encode_text(goal).float()
                goal /= goal.norm(dim=-1, keepdim=True)

        if ObjectGoalKShotImagePromptSensor.cls_uuid in observations:
            goal = observations[ObjectGoalKShotImagePromptSensor.cls_uuid]  # TN x K x D
            goal /= goal.norm(dim=-1, keepdim=True)  # TN x K x D
            with torch.no_grad():
                rgb = self.clip_transform(obs, T, N, OV)  # TN * OV x 3 x h x w
                rgb = self.clip.encode_image(rgb).float()  # TN * OV x D
                rgb /= rgb.norm(dim=-1, keepdim=True)  # TN * OV x D
            # assume V=1 for our case
            cosine_similarity = torch.einsum("nkd,nd->nk", goal, rgb).unsqueeze(
                dim=-1
            )  # TN x K x 1
            goal_weights = F.softmax(cosine_similarity, dim=1)  # TN x K x 1
            goal = torch.sum(goal * goal_weights, dim=1, keepdim=True)  # TN x 1 x D
            goal /= goal.norm(dim=-1, keepdim=True)  # TN x 1 x D
            GV = 1

        # average pool goal embedding (Note: this is a no-op for object goal
        # representations and single view goals because GV == 1)
        goal = goal.reshape(TN, GV, -1)  # TN x GV x D
        goal = goal.mean(1)  # TN x D

        x.append(goal)

        # previous action embedding
        prev_actions = prev_actions.squeeze(-1)  # TN
        start_token = torch.zeros_like(prev_actions)  # TN
        prev_actions = self.prev_action_embedding(
            torch.where(masks.view(-1), prev_actions + 1, start_token)
        )  # TN x 32
        x.append(prev_actions)  # TN x 32

        # state encoding
        out = torch.cat(x, dim=1)
        out, rnn_hidden_states = self.state_encoder(out, rnn_hidden_states, masks)

        return out, rnn_hidden_states


@baseline_registry.register_policy
class ZSONPolicy(Policy):
    def __init__(
        self,
        observation_space: spaces.Dict,
        action_space: spaces.Discrete,
        backbone: str = "resnet18",
        baseplanes: int = 32,
        hidden_size: int = 512,
        rnn_type: str = "GRU",
        rnn_num_layers: int = 1,
        use_data_aug: bool = False,
        run_type: str = "train",
        clip_model: str = "RN50",
        obs_clip_model: str = "RN50",
        pretrained_encoder: Optional[str] = None,
        use_clip_obs_encoder: bool = False,
    ):
        super().__init__(
            net=ZSONPolicyNet(
                observation_space=observation_space,
                action_space=action_space,
                backbone=backbone,
                baseplanes=baseplanes,
                hidden_size=hidden_size,
                rnn_type=rnn_type,
                rnn_num_layers=rnn_num_layers,
                use_data_aug=use_data_aug,
                run_type=run_type,
                clip_model=clip_model,
                obs_clip_model=obs_clip_model,
                pretrained_encoder=pretrained_encoder,
                use_clip_obs_encoder=use_clip_obs_encoder,
            ),
            dim_actions=action_space.n,
        )

    @classmethod
    def from_config(cls, config, observation_space, action_space):
        return cls(
            observation_space=observation_space,
            action_space=action_space,
            backbone=config.RL.POLICY.backbone,
            baseplanes=config.RL.POLICY.baseplanes,
            hidden_size=config.RL.POLICY.hidden_size,
            rnn_type=config.RL.POLICY.rnn_type,
            rnn_num_layers=config.RL.POLICY.rnn_num_layers,
            use_data_aug=config.RL.POLICY.use_data_aug,
            run_type=config.RUN_TYPE,
            clip_model=config.RL.POLICY.CLIP_MODEL,
            obs_clip_model=config.RL.POLICY.OBS_CLIP_MODEL,
            pretrained_encoder=config.RL.POLICY.pretrained_encoder,
            use_clip_obs_encoder=config.RL.POLICY.use_clip_obs_encoder,
        )
