import argparse
import os
import string
import sys

import clip
import habitat
import habitat_sim
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from zson.transforms import get_transform

# VERSION = "v1"
# IMG_WIDTH = 512
# IMG_HEIGHT = 512
# SENSOR_HFOV = 90
# SENSOR_HEIGHT = 1.25
# DATASET_VERSION = "v1"

VERSION = "v2"
IMG_WIDTH = 640
IMG_HEIGHT = 480
SENSOR_HFOV = 90
SENSOR_HEIGHT = 0.88
DATASET_VERSION = "v2"

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--chunks",
        default=8,
        type=int,
        help="total number of gpus to parallelize",
    )
    parser.add_argument(
        "--chunk_index",
        required=True,
        type=int,
        help="the chunk index",
    )
    parser.add_argument(
        "--clip_model",
        type=str,
        choices=["RN50", "ViT-B/32", "ViT-B/16", "ViT-L/14"],
        default="RN50",
        help="the CLIP model to use for extracting features",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="hm3d",
        choices=["hm3d", "gibson"],
        help="Dataset to cache goal embeddings for",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        choices=["train", "val", "val_combined", "val_easy", "val_medium", "val_hard"],
        help="Dataset split split",
    )
    args = parser.parse_args()
    return args


def get_sensor_spec(uuid, angle):
    spec = habitat_sim.CameraSensorSpec()
    spec.uuid = uuid
    spec.sensor_type = habitat_sim.SensorType.COLOR
    spec.resolution = [IMG_HEIGHT, IMG_WIDTH]
    spec.position = [0.0, SENSOR_HEIGHT, 0.0]
    spec.orientation = [0.0, angle, 0.0]
    spec.hfov = SENSOR_HFOV
    return spec


def make_simulator(scene_id):
    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.scene_id = scene_id

    agent_cfg = habitat_sim.agent.AgentConfiguration()
    agent_cfg.sensor_specifications = []

    names = ["left", "front", "right", "back"]
    angles = [1 / 2 * np.pi, 0.0, 3 / 2 * np.pi, np.pi]
    for n, a in zip(names, angles):
        agent_cfg.sensor_specifications.append(get_sensor_spec(n, a))
    print(agent_cfg.sensor_specifications)

    cfg = habitat_sim.Configuration(sim_cfg, [agent_cfg])

    return habitat_sim.Simulator(cfg)


def get_goal_observation(sim, episode):
    assert len(episode.goals) == 1
    goal = episode.goals[0]
    position = goal.position

    seed = abs(hash(episode.episode_id)) % (2**32)
    rng = np.random.RandomState(seed)
    angle = rng.uniform(0, 2 * np.pi)
    rotation = [0, np.sin(angle / 2), 0, np.cos(angle / 2)]

    agent = sim.get_agent(0)
    state = agent.get_state()
    state.position = position
    state.rotation = rotation
    agent.set_state(state)

    return sim.get_sensor_observations()


def to_tensor(obs, keys=["left", "front", "right", "back"]):
    imgs = [np.array(Image.fromarray(obs[k]).convert("RGB")) for k in keys]
    imgs = torch.from_numpy(np.stack(imgs))
    if torch.cuda.is_available():
        imgs = imgs.cuda()
    return imgs


def main(args):
    # remove punctuation and lowercase
    clip_model_str = args.clip_model.translate(
        str.maketrans("", "", string.punctuation)
    ).lower()

    clip_transform = get_transform("clip", 224)
    clip_model, _ = clip.load(args.clip_model)
    clip_size = clip_model.visual.output_dim
    clip_model.eval()

    # dataset config
    config = habitat.config.Config()
    config.TYPE = "PointNav-v1"
    config.SPLIT = args.split
    config.SCENES_DIR = "data/scene_datasets/"
    config.CONTENT_SCENES = ["*"]
    config.DATA_PATH = os.path.join(
        "data",
        "datasets",
        "imagenav",
        args.dataset,
        DATASET_VERSION,
        "{split}",
        "{split}.json.gz",
    )
    config.freeze()

    # construct output path
    output_path = os.path.join(
        "data",
        "goal_datasets",
        "imagenav",
        args.dataset,
        f"{VERSION}-{clip_model_str}",
        args.split,
        "content",
        "{scene}.npy",
    )

    # dataset
    dataset = habitat.make_dataset(config.TYPE, config=config)

    # chunk scenes
    N = len(dataset.scene_ids)
    indices = torch.arange(N).chunk(args.chunks)[args.chunk_index]
    scene_ids = [dataset.scene_ids[i] for i in indices]

    # loop over scenes
    for scene_id in scene_ids:
        scene = dataset.scene_from_scene_path(scene_id)
        path = output_path.format(split=config.SPLIT, scene=scene)
        if os.path.exists(path):
            continue
        os.makedirs(os.path.dirname(path), exist_ok=True)

        # create simulator
        try:
            sim.close()
        except NameError:
            pass
        sim = make_simulator(scene_id)

        # loop over episodes for scene
        episodes = dataset.get_scene_episodes(scene_id)
        episode_ids = [int(e.episode_id) for e in episodes]
        assert len(episode_ids) == len(np.unique(episode_ids))
        N = max(episode_ids) + 1
        if len(episode_ids) < N:
            print("*" * 80)
            print("Warning: {} has missing episodes".format(scene))
            print("*" * 80)
        data = np.empty((N, 4, clip_size), dtype="float16")
        for ep in tqdm(episodes):
            eid = int(ep.episode_id)
            obs = get_goal_observation(sim, ep)
            imgs = to_tensor(obs)
            with torch.no_grad():
                imgs = clip_transform(imgs)
                data[eid] = clip_model.encode_image(imgs).cpu().numpy()

        np.save(path, data)


if __name__ == "__main__":
    args = parse_args()
    assert (
        args.chunk_index < args.chunks
    ), "The chunk index should be less than total chunks"
    sys.exit(main(args))