"""Channel generation functions using Sionna and TensorFlow.

This module provides functionality for generating wireless communication channels.
"""

# Standard library imports
import os
from typing import Optional

# GPU setup must happen before any TensorFlow imports
GPU_IDX = 0
os.environ["CUDA_VISIBLE_DEVICES"] = str(GPU_IDX)  # Select only one GPU
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "3"  # Silence TensorFlow.

# Third-party imports
import sionna  # type: ignore
import tensorflow as tf  # type: ignore

from dataclasses import dataclass
import numpy as np

@dataclass
class TopologyConfig:
    ut_loc: np.ndarray  # [batch size, num_ut, 3], tf.float
    bs_loc: np.ndarray  # [batch size, num_bs, 3], tf.float
    ut_orientations: np.ndarray  # [batch size, num_ut, 3], tf.float
    bs_orientations: np.ndarray  # [batch size, num_bs, 3], tf.float
    ut_velocities: np.ndarray  # [batch size, num_ut, 3], tf.float
    in_state: np.ndarray  # [batch size, num_ut], tf.bool
    los: Optional[bool]  # tf.bool or None


class SionnaChannelGenerator(tf.keras.Model):
    """Generator class for Sionna channels."""
    def __init__(self, num_prbs: int, channel_name: str = 'UMa', batch_size: int = 1, 
                 n_rx: int = 1, n_tx: int = 1, normalize: bool = True, n_ue: int = 1,
                 seed: Optional[int] = None, topology: Optional[TopologyConfig] = None,
                 ue_speed: float = 1):
        """
        Initializor for a Sionna Channel Generator.

        It defines an anntena array and a resource grid in order to generate channels conveniently.

        For simplicity, we currently hardcode for single user, single antenna, single layer,
        and several other parameters like frequency, delay spread, link direction, etc.
        """
        super().__init__()

        if seed is not None:
            sionna.config.seed = seed

        self.num_prbs = num_prbs
        self.batch_size = batch_size
        self.normalize = normalize
        self.topology_config = topology
        
        # parameters for channel modeling
        self.channel_model = channel_name
        self.fc = 3.5e9                  # Frequency [Hz]
        self.link_direction = 'downlink' # Link direction (direction of the signal)
        self.delay_spread = 100e-9       # Nominal delay spread [s]
        self.ue_speed = ue_speed        # User speed [m/s]
        self.n_ue = n_ue

        self.ue_array = sionna.channel.tr38901.PanelArray(
            num_rows_per_panel=1,
            num_cols_per_panel=n_rx,
            polarization='single',
            polarization_type='V',
            antenna_pattern='omni',
            carrier_frequency=self.fc)

        self.gnb_array = sionna.channel.tr38901.PanelArray(
            num_rows_per_panel=1,
            num_cols_per_panel=n_tx,
            polarization='single',
            polarization_type='V',
            antenna_pattern='omni',
            carrier_frequency=self.fc)

        self.channel = self.make_channel()

        self.rg = sionna.ofdm.ResourceGrid(
            num_ofdm_symbols=1,
            fft_size=self.num_prbs * 12,  # Num. subcarriers
            subcarrier_spacing=30e3,      # [kHz]
            num_tx=1,
            num_streams_per_tx=1)

        self.channel_generator = sionna.channel.GenerateOFDMChannel(
            self.channel,
            self.rg,
            normalize_channel=self.normalize)

        self.awgn = sionna.channel.AWGN()

        sionna.config.xla_compat = True  # Necessary for proper model compilation

    def make_channel(self):
        """Create the channel object, using one of the 3GPP defined channel models.

        Available channel models in Sionna: https://nvlabs.github.io/sionna/api/channel.html
        This function further requires the batch size for the number of channels to sample,
        and transmit and receive arrays for the dimensions of the channel, and some
        information to determine the delay/doppler spread to include in the channel,
        like user speeds and carrier frequency.
        """
        rx_array = self.gnb_array if self.link_direction == 'uplink' else self.ue_array
        tx_array = self.ue_array if self.link_direction == 'uplink' else self.gnb_array

        num_rx_ant, num_tx_ant = rx_array.num_ant, tx_array.num_ant

        squeeze_options = (1, 2, 3, 4, 5) if num_rx_ant + num_tx_ant <= 2 else (1, 3, 5)
        self.shape_squeeze = squeeze_options if self.n_ue == 1 else (3, 5)

        # Setup network topology (required in UMi, UMa, and RMa)
        if self.channel_model in ['UMi', 'UMa']:
            default_topology = sionna.channel.gen_single_sector_topology(
                batch_size=self.batch_size,
                num_ut=1,
                scenario=self.channel_model.lower(),
                min_ut_velocity=self.ue_speed,
                max_ut_velocity=self.ue_speed,
            )

        # Configure a channel impulse response (CIR) generator for the channel models
        if self.channel_model == "Rayleigh":
            ch_model = sionna.channel.RayleighBlockFading(
                num_rx=1,
                num_rx_ant=num_rx_ant,
                num_tx=1,
                num_tx_ant=num_tx_ant
            )
        elif "CDL" in self.channel_model:
            ch_model = sionna.channel.tr38901.CDL(
                model=self.channel_model[-1],
                delay_spread=self.delay_spread,
                carrier_frequency=self.fc,
                ut_array=self.ue_array,
                bs_array=self.gnb_array,
                direction=self.link_direction,
                min_speed=self.ue_speed
            )
        elif 'UMi' in self.channel_model or 'UMa' in self.channel_model:
            if 'UMa' in self.channel_model:
                model = sionna.channel.tr38901.UMa
            else:
                model = sionna.channel.tr38901.UMi

            ch_model = model(carrier_frequency=self.fc,
                             o2i_model='low',
                             bs_array=self.gnb_array,
                             ut_array=self.ue_array,
                             direction=self.link_direction,
                             enable_pathloss=False)
            if self.topology_config is not None:
                ch_model.set_topology(**self.topology_config.__dict__)
            else:
                ch_model.set_topology(*default_topology, los=None)

        elif 'TDL' in self.channel_model:
            ch_model = sionna.channel.tr38901.TDL(
                model=self.channel_model[-1],
                delay_spread={'A': 30e-9, 'B': 100e-9, 'C': 300e-9}[self.channel_model[-1]],
                carrier_frequency=self.fc,
                num_rx_ant=num_rx_ant,
                num_tx_ant=num_tx_ant,
                min_speed=self.ue_speed
            )
        else:
            raise ValueError(f"Invalid channel model {self.channel_model}!")

        return ch_model

    @tf.function(jit_compile=True)
    def gen_channel_jit(self, snr_db):
        """Sample channel and add noise based on an SNR value in dB.

        Args:
            snr_db: Signal-to-noise ratio in dB

        Returns:
            Tuple of channel and noisy channel
        """
        h = self.channel_generator(self.batch_size)
        
        # Add noise
        No = tf.math.pow(10., tf.cast(-snr_db, tf.float32) / 10.)
        h_n = self.awgn([h, No])

        # Squeeze: tx/rx id, tx/rx ant id, ofdm symb dimensions
        return tf.squeeze(h, self.shape_squeeze), tf.squeeze(h_n, self.shape_squeeze)
