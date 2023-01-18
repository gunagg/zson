from typing import Any, Optional

import clip
import numpy as np
from gym import Space, spaces
from habitat.config import Config
from habitat.core.dataset import Dataset
from habitat.core.registry import registry
from habitat.core.simulator import RGBSensor, Sensor, SensorTypes, Simulator
from habitat.tasks.nav.nav import NavigationEpisode
from habitat.utils.geometry_utils import quaternion_from_coeff


@registry.register_sensor
class PanoramicImageGoalSensor(Sensor):
    cls_uuid: str = "panoramic-imagegoal"

    def __init__(self, *args: Any, sim: Simulator, config: Config, **kwargs: Any):
        self._sim = sim
        sensors = self._sim.sensor_suite.sensors
        self._uuids = [
            uuid for uuid, sensor in sensors.items() if isinstance(sensor, RGBSensor)
        ]
        self._current_episode_id: Optional[str] = None
        self._current_image_goal = None
        super().__init__(config=config)

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.cls_uuid

    def _get_sensor_type(self, *args: Any, **kwargs: Any) -> SensorTypes:
        return SensorTypes.PATH

    def _get_observation_space(self, *args: Any, **kwargs: Any) -> Space:
        h, w, d = self._sim.sensor_suite.observation_spaces.spaces[self._uuids[0]].shape
        return spaces.Box(
            low=0, high=255, shape=(len(self._uuids), h, w, d), dtype=np.uint8
        )

    def _get_panoramic_image_goal(self, episode: NavigationEpisode):
        goal_position = list(episode.goals[0].position)

        if self.config.SAMPLE_GOAL_ANGLE:
            angle = np.random.uniform(0, 2 * np.pi)
        else:
            seed = abs(hash(episode.episode_id)) % (2**32)  # deterministic angle
            rng = np.random.RandomState(seed)
            angle = rng.uniform(0, 2 * np.pi)

        # set the goal rotation
        goal_rotation = [0, np.sin(angle / 2), 0, np.cos(angle / 2)]
        episode.goals[0].rotation = quaternion_from_coeff(goal_rotation)

        # get the goal observation
        goal_observation = self._sim.get_observations_at(
            position=goal_position, rotation=goal_rotation
        )
        return np.stack([goal_observation[k] for k in self._uuids])

    def get_observation(
        self,
        *args: Any,
        episode: NavigationEpisode,
        **kwargs: Any,
    ):
        episode_uniq_id = f"{episode.scene_id} {episode.episode_id}"
        if episode_uniq_id == self._current_episode_id:
            return self._current_image_goal

        self._current_image_goal = self._get_panoramic_image_goal(episode)
        self._current_episode_id = episode_uniq_id

        return self._current_image_goal


@registry.register_sensor
class ObjectGoalPromptSensor(Sensor):
    r"""A sensor for Object Goal specification as observations which is used in
    ObjectGoal Navigation. The goal is expected to be specified by object_id or
    semantic category id, and we will generate the prompt corresponding to it
    so that it's usable by CLIP's text encoder.
    Args:
        sim: a reference to the simulator for calculating task observations.
        config: a config for the ObjectGoalPromptSensor sensor. Can contain field
            GOAL_SPEC that specifies which id use for goal specification,
            GOAL_SPEC_MAX_VAL the maximum object_id possible used for
            observation space definition.
        dataset: a Object Goal navigation dataset that contains dictionaries
        of categories id to text mapping.
    """
    cls_uuid: str = "objectgoalprompt"

    def __init__(
        self,
        *args: Any,
        config: Config,
        **kwargs: Any,
    ):
        super().__init__(config=config)

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.cls_uuid

    def _get_sensor_type(self, *args: Any, **kwargs: Any):
        return SensorTypes.SEMANTIC

    def _get_observation_space(self, *args: Any, **kwargs: Any):
        return spaces.Box(low=0, high=np.inf, shape=(77,), dtype=np.int64)

    def get_observation(
        self,
        *args: Any,
        episode: Any,
        **kwargs: Any,
    ) -> Optional[int]:
        category = episode.object_category if hasattr(episode, "object_category") else ""
        if self.config.HAS_ATTRIBUTE:
            tokens = category.split("_")
            attr, cat = tokens[0], " ".join(tokens[1:])  # assume one word attributes
        else:
            attr, cat = None, category.replace("_", " ")
        # use `attr` and `cat` in prompt templates
        prompt = self.config.PROMPT.format(attr=attr, cat=cat)
        return clip.tokenize(prompt, context_length=77).numpy()


@registry.register_sensor
class CachedGoalSensor(Sensor):
    cls_uuid: str = "cached-goal"

    def __init__(self, *args: Any, config: Config, dataset: Dataset, **kwargs: Any):
        self._current_episode_id: Optional[str] = None
        self._current_goal = None

        self._data = {}
        for scene_id in dataset.scene_ids:
            scene = dataset.scene_from_scene_path(scene_id)
            path = config.DATA_PATH.format(split=config.DATA_SPLIT, scene=scene)
            self._data[scene_id] = np.load(path, mmap_mode="r")

        super().__init__(config=config)

    def _get_goal(self, episode):
        """order: left, front, right, back."""
        scene_id, episode_id = episode.scene_id, episode.episode_id
        obs = self._data[scene_id][episode_id]

        # default episode angle
        seed = abs(hash(episode.episode_id)) % (2**32)  # deterministic angle
        rng = np.random.RandomState(seed)
        angle = rng.uniform(0, 2 * np.pi)

        obs_idx = 1  # default observation index
        if self.config.SINGLE_VIEW and self.config.SAMPLE_GOAL_ANGLE:
            # sample an observation index and offset the default angle
            obs_idx = np.random.choice(4)
            angle += [1 / 2 * np.pi, 0.0, 3 / 2 * np.pi, np.pi][obs_idx]

        # set the goal rotation
        goal_rotation = [0, np.sin(angle / 2), 0, np.cos(angle / 2)]
        episode.goals[0].rotation = quaternion_from_coeff(goal_rotation)

        # return the observation embedding
        if self.config.SINGLE_VIEW:
            return obs[obs_idx][None]
        else:
            return obs

    def _get_example_observation(self):
        scene_id = list(self._data.keys())[0]
        obs = self._data[scene_id][0]
        if self.config.SINGLE_VIEW:
            return obs[0][None]
        else:
            return obs

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.cls_uuid

    def _get_sensor_type(self, *args: Any, **kwargs: Any) -> SensorTypes:
        return SensorTypes.PATH

    def _get_observation_space(self, *args: Any, **kwargs: Any) -> Space:
        obs = self._get_example_observation()
        return spaces.Box(
            low=float("-inf"), high=float("inf"), shape=obs.shape, dtype=obs.dtype
        )

    def get_observation(
        self,
        *args: Any,
        episode: NavigationEpisode,
        **kwargs: Any,
    ):
        episode_uniq_id = f"{episode.scene_id} {episode.episode_id}"
        if episode_uniq_id == self._current_episode_id:
            return self._current_goal

        self._current_episode_id = episode_uniq_id
        self._current_goal = self._get_goal(episode)

        return self._current_goal


@registry.register_sensor
class ObjectGoalKShotImagePromptSensor(Sensor):
    r"""A sensor for Object Goal specification as observations which is used in
    ObjectGoal Navigation. The goal is expected to be specified by object_id or
    semantic category id, and we will use sample object image prompt corresponding to it
    so that it's usable by CLIP's image encoder.
    Args:
        sim: a reference to the simulator for calculating task observations.
        config: a config for the ObjectGoalKShotImagePromptSensor sensor. Can contain field
            GOAL_SPEC that specifies which id use for goal specification,
            GOAL_SPEC_MAX_VAL the maximum object_id possible used for
            observation space definition.
        dataset: a Object Goal navigation dataset that contains dictionaries
        of categories id to text mapping.
    """
    cls_uuid: str = "objectgoal_kshot_image_prompt"

    def __init__(
        self,
        *args: Any,
        config: Config,
        **kwargs: Any,
    ):

        self._data = {}
        self._data = np.load(config.DATA_PATH, allow_pickle=True).item()
        super().__init__(config=config)

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.cls_uuid

    def _get_sensor_type(self, *args: Any, **kwargs: Any):
        return SensorTypes.SEMANTIC

    def _get_observation_space(self, *args: Any, **kwargs: Any):
        return spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.config.SHOTS, self.config.CLIP_SIZE),
            dtype=np.float32,
        )

    def get_observation(
        self,
        *args: Any,
        episode: Any,
        **kwargs: Any,
    ) -> Optional[int]:
        if not hasattr(episode, "object_category"):
            return np.zeros((self.config.SHOTS, self.config.CLIP_SIZE))
        image_goal_samples = self._data[episode.object_category]
        if self.config.SHOTS >= len(image_goal_samples):
            return image_goal_samples
        shots_ind = np.random.permutation(len(image_goal_samples))[: self.config.SHOTS]
        return image_goal_samples[shots_ind]
