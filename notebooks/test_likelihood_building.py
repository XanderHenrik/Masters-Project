import os
import sacc
import numpy as np

from firecrown.likelihood.likelihood import NamedParameters
import firecrown.likelihood.weak_lensing as wl
import firecrown.likelihood.number_counts as nc
from firecrown.likelihood.two_point import TwoPoint
from firecrown.likelihood.gaussian import ConstGaussian
from firecrown.modeling_tools import ModelingTools
from firecrown.ccl_factory import CCLFactory


# import cosmosis
# print(cosmosis.__file__)

def build_likelihood(params: NamedParameters):
    """
    Build a likelihood using Firecrown. We will build a theoretical likelihood, starting with the 3x2pt analysis, based on the DES year 1 data.
    This might be extended to a 6x2pt analysis a we go.

    We will first initiate the sources, build the angular power spectra and finally the TwoPoint objects from which we will get the likelihood
    """

    # The tracers we define and use to build the 3x2pt likelihood will be stored in a dictionary:
    sources: dict[str, wl.WeakLensing | nc.NumberCounts] = {}

    """
    Now we will define the sources, being the source galaxies being lensed in the WL tracer. Next to that we will define the lenses
    which are the galaxies that lens the source galaxies aswell as form the galaxy sample in the galaxy clustering tracer.
    Along side the tracers, we define the systematics we will apply to these tracers

    Starting with the weak lensing sources and the systematics that apply to it:
    """
    lai_systematic = wl.LinearAlignmentSystematic(sacc_tracer="") # This should be the intrinsic alignment, characteristic to the weak lensing tracer

    # Next to the intrinsic alignment, that applies to all bins;
    for i in range(4):
        # Define the multiplicative bias for WL:
        mult_bias = wl.MultiplicativeBiasSystematic(sacc_tracer=f"src{i}")
        
        # Define the photo-z shift for WL:
        wl_photo_z = wl.PhotoZShift(sacc_tracer=f"src{i}")

        # Defining the tracer in each bin:
        sources[f'src{i}'] = wl.WeakLensing(
            sacc_tracer=f'src{i}', systematics=[lai_systematic, mult_bias, wl_photo_z],
        )
    
    # For the galaxy clustering:
    for i in range(5):
        # Define the photo-z shift for the galaxy clustering:
        gc_photo_z = nc.PhotoZShift(sacc_tracer=f'lens{i}')

        # Define the tracers:
        sources[f'lens{i}'] = nc.NumberCounts(
            sacc_tracer=f'lens{i}', systematics=[gc_photo_z], derived_scale=True,
        )
    
    # With the tracers defined, we can now correlate them to build auto and croww correlated angular power spectra:

    # Store the TwoPoint objects in the directory:
    statistics = {}
    for stat, sacc_stat in [
        ("xip", "galaxy_shear_xi_plus"),
        ("xim", "galaxy_shear_xi_minus"),
    ]:
        # Determine the auto & cross-correlations of weak lensing:
        for i in range(4):
            for j in range(i, 4):
                # Define the TwoPoint object:
                statistics[f'{stat}_src{i}_src{j}'] = TwoPoint(
                    source0 = sources[f'src{i}'],
                    source1 = sources[f'src{j}'],
                    sacc_data_type = sacc_stat,
                )
    # For the auto correlations
    for j in range(5):
        for i in range(4):
            statistics[f"gammat_lens{j}_src{i}"] = TwoPoint(
                source0=sources[f"lens{j}"],
                source1=sources[f"src{i}"],
                sacc_data_type="galaxy_shearDensity_xi_t",
            )
    # Finally the auto correlated galaxy clustering angular power spectra:
    for i in range(5):
        statistics[f"wtheta_lens{i}_lens{i}"] = TwoPoint(
            source0=sources[f"lens{i}"],
            source1=sources[f"lens{i}"],
            sacc_data_type="galaxy_density_xi",
        )

    # Given the definitions of the TwoPoint objects; we now turn to initializing the likelihoods:
    likelihood = ConstGaussian(statistics=list(statistics.values()))
    
    # To properly initiate the likelihood, we read in the SACC file:
    sacc_file = params.get_string("sacc_file")
    # Translate envriornment variables, if needed.
    sacc_file = os.path.expandvars(sacc_file)
    sacc_data = sacc.Sacc.load_fits(sacc_file)

    # Configure the likelihood with the SACC data and set the modeling tools:
    likelihood.read(sacc_data)
    modeling_tools = ModelingTools(ccl_factory=CCLFactory(require_nonlinear_pk=True))

    return likelihood, modeling_tools