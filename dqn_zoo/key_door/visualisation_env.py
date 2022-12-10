from typing import Dict, List, Optional, Tuple, Union

import numpy as np

from dqn_zoo.key_door import constants, wrapper

try:
    import cv2
    import matplotlib
    from matplotlib import cm
    from matplotlib import pyplot as plt
    from mpl_toolkits.axes_grid1 import make_axes_locatable
except:
    raise AssertionError(
        "To use visualisation wrapper, further package requirements need to be satisfied. Please consult README."
    )


class VisualisationEnv(wrapper.Wrapper):

    COLORMAP = cm.get_cmap("plasma")

    def __init__(self, env):
        super().__init__(env=env)

    def render(
        self,
        save_path: Optional[str] = None,
        dpi: Optional[int] = 60,
        format: str = "state",
    ) -> None:
        if format == constants.STATE:
            assert (
                self._env.active
            ), "To render map with state, environment must be active."
            "call reset_environment() to reset environment and make it active."
            "Else render stationary environment skeleton using format='stationary'"
        if save_path:
            fig = plt.figure()
            plt.imshow(
                self._env._env_skeleton(
                    rewards=format,
                    keys={k: format for k in self._env._key_ids},
                    doors=format,
                    agent=format,
                    cue=format,
                ),
                origin="lower",
            )
            fig.savefig(save_path, dpi=dpi)
        else:
            plt.imshow(
                self._env._env_skeleton(
                    rewards=format,
                    keys={k: format for k in self._env._key_ids},
                    doors=format,
                    agent=format,
                    cue=format,
                ),
                origin="lower",
            )

    def visualise_episode_history(
        self, save_path: str, history: Union[str, List[np.ndarray]] = "train"
    ) -> None:
        """Produce video of episode history.

        Args:
            save_path: name of file to be saved.
            history: "train", "test" to plot train or test history, else provide an independent history.
        """
        if isinstance(history, str):
            if history == constants.TRAIN:
                history = self._env.train_episode_history
            elif history == constants.TEST:
                history = self._env.test_episode_history
            elif history == constants.TRAIN_PARTIAL:
                history = self._env.train_episode_partial_history
            elif history == constants.TEST_PARTIAL:
                history = self._env.test_episode_partial_history

        SCALING = 20
        FPS = 30

        map_shape = history[0].shape
        frameSize = (SCALING * map_shape[1], SCALING * map_shape[0])

        out = cv2.VideoWriter(
            filename=save_path,
            fourcc=cv2.VideoWriter_fourcc("a", "v", "c", "1"),
            fps=FPS,
            frameSize=frameSize,
        )

        for frame in history:
            bgr_frame = frame[..., ::-1].copy()
            flipped_frame = np.flip(bgr_frame, 0)
            scaled_up_frame = np.kron(flipped_frame, np.ones((SCALING, SCALING, 1)))
            out.write((scaled_up_frame * 255).astype(np.uint8))

        out.release()

    def plot_heatmap_over_env(
        self,
        heatmap: Dict[Tuple[int, int], float],
        fig: Optional[matplotlib.figure.Figure] = None,
        ax: Optional[matplotlib.axes.Axes] = None,
        save_name: Optional[str] = None,
    ) -> None:
        assert (
            ax is not None and fig is not None
        ) or save_name is not None, "Either must provide axis to plot heatmap over,"
        "r file name to save separate figure."
        environment_map = self._env._env_skeleton(
            rewards=None,
            keys={k: None for k in self._env._key_ids},
            doors=None,
            agent=None,
            cue=None,
        )

        all_values = list(heatmap.values())
        current_max_value = np.max(all_values)
        current_min_value = np.min(all_values)

        for position, value in heatmap.items():
            # remove alpha from rgba in colormap return
            # normalise value for color mapping
            environment_map[position[::-1]] = self.COLORMAP(
                (value - current_min_value) / (current_max_value - current_min_value)
            )[:-1]

        fig = plt.figure()
        if save_name is not None:
            plt.imshow(environment_map, origin="lower", cmap=self.COLORMAP)
            plt.colorbar()
            fig.savefig(save_name, dpi=60)
        else:
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            im = ax.imshow(environment_map, origin="lower", cmap=self.COLORMAP)
            fig.colorbar(im, ax=ax, cax=cax, orientation="vertical")
        plt.close()
