import jax 
import jax.numpy as jnp
import numpy as np

from jaxtypes import Array

jax.config.update("jax_enable_x64", True)

#########################
### UTILITY FUNCTIONS ###
#########################

def semi_major_minor_from_e(e: Array) -> tuple[Array, Array]:
    a = 1.0 / jnp.sqrt(2.0 - (e * e))
    b = a * jnp.sqrt(1.0 - (e * e))
    return a, b

class SineGaussianJax():
    """
    Callable class for generating sine-Gaussian waveforms.

    Args:
        sample_rate: Sample rate of waveform
        duration: Duration of waveform
    """

    def __init__(self, sample_rate: float, duration: float):
        # super().__init__()
        
        # TODO determine this
        # determine times based on requested duration and sample rate
        # and shift so that the waveform is centered at t=0

        num = int(duration * sample_rate)
        times = jnp.arange(num) / sample_rate
        times -= duration / 2.0

        # self.register_buffer("times", times)

    def __call__(
        self,
        quality: float,
        frequency: float,
        hrss: float,
        phase: float,
        eccentricity: float,
    ):
        """
        Generate lalinference implementation of a sine-Gaussian waveform.
        See
        git.ligo.org/lscsoft/lalsuite/-/blob/master/lalinference/lib/LALInferenceBurstRoutines.c#L381
        for details on parameter definitions.

        Args:
            frequency:
                Central frequency of the sine-Gaussian waveform
            quality:
                Quality factor of the sine-Gaussian waveform
            hrss:
                Hrss of the sine-Gaussian waveform
            phase:
                Phase of the sine-Gaussian waveform
            eccentricity:
                Eccentricity of the sine-Gaussian waveform.
                Controls the relative amplitudes of the
                hplus and hcross polarizations.
        Returns:
            Tensors of cross and plus polarizations
        """

        # add dimension for calculating waveforms in batch
        frequency = frequency.view(-1, 1)
        quality = quality.view(-1, 1)
        hrss = hrss.view(-1, 1)
        phase = phase.view(-1, 1)
        eccentricity = eccentricity.view(-1, 1)

        # calculate relative hplus / hcross amplitudes based on eccentricity
        # as well as normalization factors
        a, b = semi_major_minor_from_e(eccentricity)
        norm_prefactor = quality / (4.0 * frequency * torch.sqrt(pi))
        cosine_norm = norm_prefactor * (1.0 + torch.exp(-quality * quality))
        sine_norm = norm_prefactor * (1.0 - torch.exp(-quality * quality))

        cos_phase, sin_phase = torch.cos(phase), torch.sin(phase)

        h0_plus = (
            hrss
            * a
            / torch.sqrt(
                cosine_norm * (cos_phase**2) + sine_norm * (sin_phase**2)
            )
        )
        h0_cross = (
            hrss
            * b
            / torch.sqrt(
                cosine_norm * (sin_phase**2) + sine_norm * (cos_phase**2)
            )
        )

        # cast the phase to a complex number
        phi = 2 * pi * frequency * self.times
        complex_phase = torch.complex(torch.zeros_like(phi), (phi - phase))

        # calculate the waveform and apply a tukey window to taper the waveform
        fac = torch.exp(phi**2 / (-2.0 * quality**2) + complex_phase)

        cross = fac.imag * h0_cross
        plus = fac.real * h0_plus

        # TODO dtype as argument?
        cross = cross.double()
        plus = plus.double()

        return cross, plus
