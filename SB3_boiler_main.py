import gymnasium as gym
from gymnasium import spaces
import random 
import time
import torch as th
from torch import nn
from torch.optim import lr_scheduler
import torch.optim as optim

from SB3_boiler_policies import CnnPolicy, BaseFeaturesExtractor
from stable_baselines3.common.preprocessing import get_flattened_obs_dim, is_image_space
# from stable_baselines3.common.torch_layers import NatureCNN

env = gym.make("ALE/SpaceInvaders-v5", render_mode = "human")

####### Checking environment shape ################
if isinstance(env.observation_space, gym.spaces.Box):
    print("Observation space is a Box space.")
    print("Environment: ", env.observation_space)

    # Check the shape of the sample observation
    print("Sample observation shape:", env.observation_space.shape)
else:
    print("Observation space is not a Box space.")

####### Defining the CNN ##########################
    
class NatureCNN(BaseFeaturesExtractor):
    """
    CNN from DQN Nature paper:
        Mnih, Volodymyr, et al.
        "Human-level control through deep reinforcement learning."
        Nature 518.7540 (2015): 529-533.

    :param observation_space:
    :param features_dim: Number of features extracted.
        This corresponds to the number of unit for the last layer.
    :param normalized_image: Whether to assume that the image is already normalized
        or not (this disables dtype and bounds checks): when True, it only checks that
        the space is a Box and has 3 dimensions.
        Otherwise, it checks that it has expected dtype (uint8) and bounds (values in [0, 255]).
    """

    def __init__(
        self,
        observation_space: gym.Space,
        features_dim: int = 512,
        normalized_image: bool = False,
    ) -> None:
        assert isinstance(observation_space, spaces.Box), (
            "NatureCNN must be used with a gym.spaces.Box ",
            f"observation space, not {observation_space}",
        )
        super().__init__(observation_space, features_dim)
        # We assume CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper
        assert is_image_space(observation_space, check_channels=False, normalized_image=normalized_image), (
            "You should use NatureCNN "
            f"only with images not with {observation_space}\n"
            "(you are probably using `CnnPolicy` instead of `MlpPolicy` or `MultiInputPolicy`)\n"
            "If you are using a custom environment,\n"
            "please check it using our env checker:\n"
            "https://stable-baselines3.readthedocs.io/en/master/common/env_checker.html.\n"
            "If you are using `VecNormalize` or already normalized channel-first images "
            "you should pass `normalize_images=False`: \n"
            "https://stable-baselines3.readthedocs.io/en/master/guide/custom_env.html"
        )
        n_input_channels = observation_space.shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with th.no_grad():
            n_flatten = self.cnn(th.as_tensor(observation_space.sample()[None]).float()).shape[1] # used to be shape[1]
            print("Worked")
        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())
        print("Worked again")



    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.linear(self.cnn(observations))

####### APPLYING CUSTOM CNN TO DQN ################
observation_space = env.observation_space

# Ensure that the observation space is in CHW format (channels, height, width)
# You need to pass a new Box space with CHW shape to NatureCNN
# Have this match the printed environment
chw_observation_space = gym.spaces.Box(
    low=0, high=255, shape=(observation_space.shape[2], observation_space.shape[0], observation_space.shape[1]), dtype=observation_space.dtype)

print("CHW OBS: ", chw_observation_space)

    
nature_model= NatureCNN(observation_space=chw_observation_space)
optimizer = optim.Adam(nature_model.parameters(), lr=0.001)
scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9)
# nature_model= NatureCNN(observation_space=env.observation_space)

def my_lr_schedule(epoch: int) -> float:
    if epoch < 10:
        return 0.1  # Initial learning rate for the first 10 epochs
    else:
        return 0.01  # Lower learning rate after the first 10 epochs

DQN_custom = CnnPolicy(
                        observation_space=chw_observation_space, # make sure to change this to chw
                        action_space= env.action_space, 
                        lr_schedule= my_lr_schedule,
                        activation_fn= th.nn.modules.activation.ReLU, 
                        features_extractor_class= NatureCNN, # Can customize CNN or use a pretrained
                        normalize_images=True, 
                        optimizer_class= optim.Adam, # Can change optimizer class
                        optimizer_kwargs=None
                         )



env.reset()

for steps in range(5): 
    obs, info = env.reset()
    done = False
    
    while not done: 
        action, _states = DQN_custom.predict(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated: 
            obs, info = env.reset()
            print("Just died on episode ", steps)

env.close()