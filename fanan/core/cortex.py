import logging

import jax
import numpy as np
import tensorflow as tf
from jax.experimental import mesh_utils
from tqdm import tqdm

from fanan.config import Config
from fanan.modeling.architectures import get_architecture


class Cortex:
    """The Cortex class represents the core component of the neural network
    model. It is responsible for initializing the model, training the model,
    and storing the model state.

    Args:
        config (Config): The configuration object containing the model settings.

    Attributes:
        config (Config): The configuration object containing the model settings.
        devices (list): The list of devices used for computation.
        mesh (Mesh): The mesh object representing the distributed computation mesh.
        architecture (Architecture): The architecture object representing the neural network architecture.
        state (TrainState): The train state object representing the current state of the model.

    Methods:
        __init__(self, config: Config) -> None: Initializes the Cortex object.
        initialize_train_state(self) -> None: Initializes the train state of the model.
        train(self, dataset) -> None: Trains the model using the given dataset.
    """

    def __init__(self, config: Config) -> None:
        self.config = config
        self.devices = mesh_utils.create_device_mesh(
            devices=jax.devices(),
            mesh_shape=(
                self.config.mesh.n_data_parallel,
                self.config.mesh.n_fsdp_parallel,
                self.config.mesh.n_sequence_parallel,
                self.config.mesh.n_tensors_parallel,
            ),
            contiguous_submeshes=True,
        )
        logging.info(f"{self.devices=}")

        self.mesh = jax.sharding.Mesh(
            devices=self.devices,
            axis_names=self.config.mesh.mesh_axis_names,
        )
        logging.info(f"{self.mesh=}")

        self.architecture = get_architecture(self.config)
        self._writer = tf.summary.create_file_writer("./logs")

    def train(self, train_dataloader_iter, val_dataloader_iter) -> None:
        """Trains the model using the given dataset.

        This method trains the model using the given dataset by iterating over the dataset
        and performing training steps for each batch.

        Args:
            dataset: The dataset used for training.

        Returns:
            None
        """

        # main loop
        losses = []
        pbar = tqdm(range(self.config.training.total_steps))
        for step in pbar:
            batch = next(train_dataloader_iter)
            loss = self.architecture.train_step(batch=batch)
            losses.append(loss)

            if step % self.config.training.eval_every_steps == 0:
                batch = next(val_dataloader_iter)
                generated_images = self.architecture.eval_step(batch=batch)
                with self._writer.as_default():
                    tf.summary.image("generated", generated_images, step=step, max_outputs=8)

            avg_loss = np.mean(losses)
            pbar.set_postfix(
                {"step_loss": f"{loss:.5f}", "avg_loss": f"{avg_loss:.5f}",}
            )

            with self._writer.as_default():
                tf.summary.scalar("loss", avg_loss, step=step)
