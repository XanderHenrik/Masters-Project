import fitsio
import numpy as np
import yaml
import sacc
import glass.observations as glass_obs
import matplotlib.pyplot as plt
import pyccl

# From Firecrown, we need the following imports:
from firecrown.metadata_types import (
    Galaxies,
    InferredGalaxyZDist,
    TwoPointXY, 
    TwoPointHarmonic,
    Measurement,
    TwoPointCorrelationSpace,
)

from firecrown.generators.two_point import (
    LogLinearElls,
)

from firecrown.modeling_tools import (
    ModelingTools,
)

from firecrown.likelihood.two_point import (
    TwoPoint, 
    TwoPointFactory, 
    NumberCountsFactory,
    WeakLensingFactory,
)

from firecrown.likelihood import (
    number_counts as nc,
    weak_lensing as wl,
    two_point as tp,
)

import load_config#.py as load_config

class sacc_generator:
        
    """
    Here we create a SACC file to accomodate the 3x2pt/6x2pt or even en nx2pt Fisher forecast.
    The application is to be determined but the clear goal is the generation of a full firecrown likeihood.

    To do so, we need to include a couple of elements into the SACC file:
    - metadata; the nuisance parameter specification
    - galaxy redshift distributions for the tracers in the analysis
    - (Mock) data; to facilitate the forecasting
    - The covariance

    The SACC file is then read and used to generate the theory vector and then the likelihood
    """

    def __init__(self, config):
        self.config = load_config.load_config(config)

        # For the redshift domain of the analysis;
        self.z_start = self.config['analysis specs']['z_start']
        self.z_stop = self.config['analysis specs']['z_stop']



    def calc_redshift_distributions(self, tracer: str): 
        
        # Define the the specified number of bins for the given tracer:
        tracer_bins: dict[str, InferredGalaxyZDist] = {}

        n_bins, z_start, z_stop, z_err, z0 = (
            self.config['tracers'][tracer]['n_bins'],
            self.config['tracers'][tracer]['z_start'],
            self.config['tracers'][tracer]['z_stop'],
            self.config['tracers'][tracer]['z_err'],
            self.config['tracers'][tracer]['z0']
        )

        # Redshift distribution from doi:10.1111/j.1365-2966.2007.12271.x:
        z = np.linspace(z_start, z_stop, n_bins + 1)
        # print(f"Redshift bins for {tracer}: {z}")
        alpha = 2.0
        beta = 1.5
        dndz = (0.02 / np.sqrt(2 * np.pi)) * (z / z0)**alpha * np.exp(-0.5 * (z / z0)**beta) # Following Smail et al. 1994, eq. 2 (from Glass)
        n_bar = 30 * (60**2) * (np.pi/180)**2  # number density in steradians

        """
        TODO: look at reading the redshift distribution from a '.txt' file; we will be using a redshift
        distribution given to us in a '.txt' file or using the default redshift distributions defined in Firecrown.

        We can incorporate gaussian error in redshift binning in this '.txt' file?
        """

        if tracer == 'lens_spec':
            # z_spec = np.linspace(z_start, z_stop, n_bins + 1)
            for i in range(n_bins):
                # Define the spec z distribution for the i-th bin:
                spec_z = (np.heaviside((z[i] - z_start), 1) - np.heaviside((z[i+1] - z_stop), 1)) * dndz
                # print(f"Spec z distribution for {tracer}{i}: {spec_z}")
                tracer_bins[f'{tracer}{i}'] = InferredGalaxyZDist(
                    bin_name = f'{tracer}{i}',
                    z = z,
                    dndz= spec_z * n_bar,
                    measurements={Galaxies.COUNTS},
                )
            # plt.plot(z,
            #          (np.heaviside((z - z_start), 1) - np.heaviside((z - z_stop), 1)) * dndz, 'o-', label='Step it up')
            # plt.show()
                
        elif tracer == 'lens':
            for i in range(n_bins):
                tracer_bins[f'{tracer}{i}'] = InferredGalaxyZDist(
                    bin_name = f'{tracer}{i}',
                    z = z,
                    dndz= dndz * n_bar,
                    measurements={Galaxies.COUNTS},
                )
        
        elif tracer == 'src':
            for i in range(n_bins):
                tracer_bins[f'{tracer}{i}'] = InferredGalaxyZDist(
                    bin_name = f'{tracer}{i}',
                    z = z,
                    dndz= dndz * n_bar,
                    measurements={Galaxies.SHEAR_E},
                )

        return tracer_bins
    

    def get_statistics(self, tracer: str):
        """
        Initiate the systematics we apply to the tracers when building the harmonic
        TwoPoint object.

        In the config file, the the systematics are given a bolean value; our treatment
        of the systematics therefore contains quite some assumptions:
        - If True: linear alignment is always applied to the whole redshift space
        - The other statistics are applied per bin. This includes:
            - multiplicative bias
            - photo-z shift
        Other systematics might follow later

        Parameters:
        tracer: str, the tracer for which we want to get the systematics.

        returns:
        systematics: list of systematics that will be applied per bin
        global_systematics: list of systematics that will be applied to the entire redshift space
        (These will be used in the TwoPointHarmonic object, to obtain the angular power spectra)
        """
        # Initialize the systematics and global_systematics lists:
        systematics = []
        global_systematics = []

        if tracer == 'lens_spec':
            for i in range(self.config['tracers'][f'{tracer}']['n_bins']):
                if self.config['tracers'][f'{tracer}']['systematics']['mult_bias']:
                    systematics.append(
                        wl.MultiplicativeShearBias(sacc_tracer=f'{tracer}{i}')
                    )
                if self.config['tracers'][f'{tracer}']['systematics']['photo_z_shift']:
                    systematics.append(
                        wl.PhotoZShift(sacc_tracer=f'{tracer}{i}')
                    )
                if self.config['tracers'][f'{tracer}']['systematics']['linear_alignment']:
                        global_systematics.append(
                            wl.LinearAlignmentSystematic(sacc_tracer=f'{tracer}{i}')
                        )

        if tracer == 'lens':
            for i in range(self.config['tracers'][f'{tracer}']['n_bins']):
                if self.config['tracers'][f'{tracer}']['systematics']['photo_z_shift']:
                    systematics.append(
                        nc.PhotoZShift(sacc_tracer=f'{tracer}{i}')
                    )
        if tracer == 'src':
            for i in range(self.config['tracers'][tracer]['n_bins']):
                if self.config['tracers'][f'{tracer}']['systematics']['photo_z_shift']:
                    systematics.append(
                        wl.PhotoZShift(sacc_tracer=f'{tracer}{i}')
                    )

        return systematics, global_systematics
    

    def correlate_two_bins(self, tracer_a: InferredGalaxyZDist, tracer_b: InferredGalaxyZDist):
        """
        Here we correlate bins of tracers. We combine to both auto- and cross-correlated
        bins of the tracers in the analysis.
        Since we obtain a dictionary containing all bins of one tracer, we input the tracers:

        Parameters:
        tracer_a: InferredGalaxyZDist, tracer bins defined by 'get_lens_statistics'
        tracer_b: InferredGalaxyZDist, tracer bins defined by 'get_lens_statistics'

        Return(s):
        correlated_bins: dict, containing the correlated bins of the tracers
        """
        correlated_bins: dict[str, TwoPointXY] = {}

        # Correlate the given bins in all possible ways:
        tracer_type_a = [tracer_a[key].bin_name[:-1] for key in tracer_a.keys()]
        tracer_type_b = [tracer_b[key].bin_name[:-1] for key in tracer_b.keys()]

        print(tracer_type_a[0], tracer_type_b[0])


        # For thev auto-correlated tracers:
        if tracer_a == tracer_b:
            for i in range(len(tracer_a.keys())):
                for j in range(len(tracer_b.keys())):
                    if i <= j:
                        correlated_bins[f"{tracer_type_a[i]}_{tracer_type_a[j]}"] = [TwoPointXY(
                            x = tracer_a[f'{tracer_type_a[i]}'],
                            y = tracer_a[f'{tracer_type_a[j]}'],
                            
                            # Extract the correct measurements from the bin definitions:
                            x_measurement = next(iter(tracer_a[f'{tracer_type_a[i]}'].measurements)),
                            y_measurement = next(iter(tracer_a[f'{tracer_type_a[j]}'].measurements)),
                        )]

        # For cross-correlated tracers, we don't have to cut half of the correlations
        else:
            for key_a in tracer_a.keys():
                for key_b in tracer_b.keys():
                    # print(f"Correlating {key_a} with {key_b}")
                    correlated_bins[f"{tracer_a[key_a]}_{tracer_b[key_b]}"] = [TwoPointXY(
                        x = tracer_a[key_a],
                        y = tracer_b[key_b],
                        
                        # Extract the correct measurements from the bin definitions:
                        x_measurement = next(iter(tracer_a[key_a].measurements)),
                        y_measurement = next(iter(tracer_b[key_b].measurements)),
                    )]

        return correlated_bins


    def collect_6x2pt_correlated_bins(self,
                                      tracer_a: InferredGalaxyZDist,
                                      tracer_b: InferredGalaxyZDist,
                                      tracer_c: InferredGalaxyZDist, 
                                      ):
        """
        In this method, we build the 6x2pt correlated bins from which we can then initiate the TwoPointHarmonic
        objects.

        Parameters:
        The 3 tracers we build the 6x2pt from:
        tracer_a: dict, containing the redshift distributions of the lens probe (photometric galaxy clustering)
        tracer_b: dict, containing the redshift distributions of the source probe (sources of the weak lensing probe)
        tracer_c: dict, containing the redshift distributions of the lens_spec probe (spectroscopic galaxy clustering)

        Returns:
        correlated_bins: dict, containing the correlated bins of the tracers
        """

        all_correlated_bins = {'lens_lens': [], 'src_src': [], 'lens_spec_lens_spec': [], 
                               'lens_src': [], 'lens_lens_spec': [], 'src_lens_spec': []}

        # Now definen the correlated bins in auto and cross-correlations:
        # Auto-correlations:
        all_correlated_bins['lens_lens'].append(
            self.correlate_two_bins(tracer_a, tracer_a)
        )

        all_correlated_bins['src_src'].append(
            self.correlate_two_bins(tracer_b, tracer_b)
        )

        all_correlated_bins['lens_spec_lens_spec'].append(
            self.correlate_two_bins(tracer_c, tracer_c)
        )

        # Cross-correlations
        all_correlated_bins['lens_src'].append(
            self.correlate_two_bins(tracer_a, tracer_b)
        )

        all_correlated_bins['src_lens_spec'].append(
            self.correlate_two_bins(tracer_b, tracer_c)
        )

        all_correlated_bins['lens_lens_spec'].append(
            self.correlate_two_bins(tracer_a, tracer_c)
        )

        return all_correlated_bins


    def init_two_point_harmonic(self, correlated_bins):

        return

    def get_multipole_bins(self):

        l_start = self.config['analysis specs']['ell_min']
        l_stop = self.config['analysis specs']['ell_max']
        n_ells = self.config['analysis specs']['n_ell']

        ell_bins = LogLinearElls(
            minimum = l_start,
            midpoint = (l_start + l_stop) / 10,
            maximum = l_stop,
            n_log = n_ells,
        )

        return ell_bins


    def get_modelling_tools(self):
        return


    def calc_gaussian_cov(self):
        return

if __name__ == "__main__":

    # Initialize the sacc_generator with the given configuration:
    sacc_gen = sacc_generator('6x2pt_config.yaml')

    # Example usage of the galaxy_redshift_distributions method
    lens_bins = sacc_gen.calc_redshift_distributions('lens')
    src_bins = sacc_gen.calc_redshift_distributions('src')
    lens_spec_bins = sacc_gen.calc_redshift_distributions('lens_spec')
    
    # print(f"Lens spec bins: {lens_spec_bins['src1']}")
    # print(f"Lens spec bins: {lens_spec_bins['src1'].dndz}")

    # for key in lens_spec_bins.keys():
    #     print(key[:-1])
        # print(lens_spec_bins[key].bin_name[:-1])

    # print(lens_spec_bins['lens_spec0'].measurements == {Galaxies.COUNTS})

    syst_spec, global_syst_spec = sacc_gen.get_statistics('src')
    # print(f"Systematics for lens_spec: {syst_spec}")
    # print(f"Global systematics for lens_spec: {global_syst_spec}")

    # auto_lens_spec = sacc_gen.correlate_two_bins(lens_spec_bins, lens_spec_bins)
    # print(f"Auto-correlated bins for lens_spec: {auto_lens_spec['lens_lens']}")


# angles = """\=
# ## angle_range_xip_1_1 = 7.195005 250.0
# ## angle_range_xip_1_2 = 7.195005 250.0
# ## angle_range_xip_1_3 = 5.715196 250.0
# ## angle_range_xip_1_4 = 5.715196 250.0
# ## angle_range_xip_2_1 = 7.195005 250.0
# ## angle_range_xip_2_2 = 4.539741 250.0
# ## angle_range_xip_2_3 = 4.539741 250.0
# ## angle_range_xip_2_4 = 4.539741 250.0
# ## angle_range_xip_3_1 = 5.715196 250.0
# ## angle_range_xip_3_2 = 4.539741 250.0
# ## angle_range_xip_3_3 = 3.606045 250.0
# ## angle_range_xip_3_4 = 3.606045 250.0
# ## angle_range_xip_4_1 = 5.715196 250.0
# ## angle_range_xip_4_2 = 4.539741 250.0
# ## angle_range_xip_4_3 = 3.606045 250.0
# ## angle_range_xip_4_4 = 3.606045 250.0
# ## angle_range_xim_1_1 = 90.579750 250.0
# ## angle_range_xim_1_2 = 71.950053 250.0
# ## angle_range_xim_1_3 = 71.950053 250.0
# ## angle_range_xim_1_4 = 71.950053 250.0
# ## angle_range_xim_2_1 = 71.950053 250.0
# ## angle_range_xim_2_2 = 57.151958 250.0
# ## angle_range_xim_2_3 = 57.151958 250.0
# ## angle_range_xim_2_4 = 45.397414 250.0
# ## angle_range_xim_3_1 = 71.950053 250.0
# ## angle_range_xim_3_2 = 57.151958 250.0
# ## angle_range_xim_3_3 = 45.397414 250.0
# ## angle_range_xim_3_4 = 45.397414 250.0
# ## angle_range_xim_4_1 = 71.950053 250.0
# ## angle_range_xim_4_2 = 45.397414 250.0
# ## angle_range_xim_4_3 = 45.397414 250.0
# ## angle_range_xim_4_4 = 36.060448 250.0
# ## angle_range_gammat_1_1 = 64.0 250.0
# ## angle_range_gammat_1_2 = 64.0 250.0
# ## angle_range_gammat_1_3 = 64.0 250.0
# ## angle_range_gammat_1_4 = 64.0 250.0
# ## angle_range_gammat_2_1 = 40.0 250.0
# ## angle_range_gammat_2_2 = 40.0 250.0
# ## angle_range_gammat_2_3 = 40.0 250.0
# ## angle_range_gammat_2_4 = 40.0 250.0
# ## angle_range_gammat_3_1 = 30.0 250.0
# ## angle_range_gammat_3_2 = 30.0 250.0
# ## angle_range_gammat_3_3 = 30.0 250.0
# ## angle_range_gammat_3_4 = 30.0 250.0
# ## angle_range_gammat_4_1 = 24.0 250.0
# ## angle_range_gammat_4_2 = 24.0 250.0
# ## angle_range_gammat_4_3 = 24.0 250.0
# ## angle_range_gammat_4_4 = 24.0 250.0
# ## angle_range_gammat_5_1 = 21.0 250.0
# ## angle_range_gammat_5_2 = 21.0 250.0
# ## angle_range_gammat_5_3 = 21.0 250.0
# ## angle_range_gammat_5_4 = 21.0 250.0
# ## angle_range_wtheta_1_1 = 43.0 250.0
# ## angle_range_wtheta_2_2 = 27.0 250.0
# ## angle_range_wtheta_3_3 = 20.0 250.0
# ## angle_range_wtheta_4_4 = 16.0 250.0
# ## angle_range_wtheta_5_5 = 14.0 250.0"""

# # here we munge them to a dict of dicts with structure:
# #
# # {'xip': {(1, 1): [7.195005, 250.0],
# #   (1, 2): [7.195005, 250.0],
# #   (1, 3): [5.715196, 250.0],
# #   ...
# #  'xim': {(1, 1): [90.57975, 250.0],
# #   (1, 2): [71.950053, 250.0],
# #   (1, 3): [71.950053, 250.0],
# #   ...
# #  'gammat': {(1, 1): [64.0, 250.0],
# #   (1, 2): [64.0, 250.0],
# #   (1, 3): [64.0, 250.0],
# #   ...
# #  'wtheta': {(1, 1): [43.0, 250.0],
# #   (2, 2): [27.0, 250.0],
# #   (3, 3): [20.0, 250.0],
# #   ...
# #   'spec': {}

# # Type specifications for the bin information.
# Bin = tuple[float, float]
# BinIndex = tuple[int, int]

# bin_limits: dict[str, dict[BinIndex, Bin]] = {}
# for line in angles.split("\n"):
#     items = line.split()
#     keys = items[1].replace("angle_range_", "").split("_")
#     topkey = keys[0]
#     binkeys = (int(keys[1]), int(keys[2]))
#     if topkey not in bin_limits:
#         bin_limits[topkey] = {}
#     bin_limits[topkey][binkeys] = (float(items[-2]), float(items[-1]))


# # finally we read the data, cut each part, and write to disk
# # the order of the covmat is xip, xim, gammat, wtheta
# # these elements range from
# #   xip: [0, 200)
# #   xip: [200, 400)
# #   gammat: [400, 800)
# #   wtheta: [800, 900)
# # there are 20 angular bins per data vector
# # there are 4 source bins
# # there are 5 lens bins
# # there are 5 spectroscopic bins
# # only the autocorrelation wtheta bins are kept
# n_srcs = 4
# n_lens = 5
# n_spec = 10 # If we define spectrocopic bins on the same resdhift interval as the photometric bins, we must have more bins due to the higher resolution of the spectroscopic data.

# # this holds a global mask of which elements of the data vector to keep
# tot_msk = []

# sacc_data = sacc.Sacc()

# with fitsio.FITS("/home/xander/Masters-Project/notebooks/Firecrown_cosmoSIS_example/2pt_NG_mcal_1110.fits") as data:
#     # nz_lens
#     dndz = data["nz_lens"].read()
#     for i in range(1, n_lens + 1):
#         sacc_data.add_tracer("NZ", f"lens{i - 1}", dndz["Z_MID"], dndz[f"BIN{i}"])

#     # nz_src
#     dndz = data["nz_source"].read()
#     for i in range(1, n_srcs + 1):
#         sacc_data.add_tracer("NZ", f"src{i - 1}", dndz["Z_MID"], dndz[f"BIN{i}"])

#     # nz_spec
#     dndz = data["nz_spec"].read()
#     for i in range(1, n_spec + 1):
#         sacc_data.add_tracer("NZ", f"spec{i - 1}", dndz["Z_MID"], dndz[f"BIN{i}"])

#     # xip
#     xip = data["xip"].read()
#     for i in range(1, n_srcs + 1):
#         for j in range(i, n_srcs + 1):
#             theta_min, theta_max = bin_limits["xip"][(i, j)]

#             ij_msk = (xip["BIN1"] == i) & (xip["BIN2"] == j)
#             xip_ij = xip[ij_msk]
#             msk = (xip_ij["ANG"] > theta_min) & (xip_ij["ANG"] < theta_max)

#             tot_msk.extend(msk.tolist())

#             sacc_data.add_theta_xi(
#                 "galaxy_shear_xi_plus",
#                 f"src{i - 1}",
#                 f"src{j - 1}",
#                 xip_ij["ANG"][msk],
#                 xip_ij["VALUE"][msk],
#             )

#     # xim
#     xim = data["xim"].read()
#     for i in range(1, n_srcs + 1):
#         for j in range(i, n_srcs + 1):
#             theta_min, theta_max = bin_limits["xim"][(i, j)]

#             ij_msk = (xim["BIN1"] == i) & (xim["BIN2"] == j)
#             xim_ij = xim[ij_msk]
#             msk = (xim_ij["ANG"] > theta_min) & (xim_ij["ANG"] < theta_max)

#             tot_msk.extend(msk.tolist())

#             sacc_data.add_theta_xi(
#                 "galaxy_shear_xi_minus",
#                 f"src{i - 1}",
#                 f"src{j - 1}",
#                 xim_ij["ANG"][msk],
#                 xim_ij["VALUE"][msk],
#             )

#     # gammat
#     gammat = data["gammat"].read()
#     for i in range(1, n_lens + 1):
#         for j in range(1, n_srcs + 1):
#             theta_min, theta_max = bin_limits["gammat"][(i, j)]

#             ij_msk = (gammat["BIN1"] == i) & (gammat["BIN2"] == j)
#             gammat_ij = gammat[ij_msk]
#             msk = (gammat_ij["ANG"] > theta_min) & (gammat_ij["ANG"] < theta_max)

#             tot_msk.extend(msk.tolist())

#             sacc_data.add_theta_xi(
#                 "galaxy_shearDensity_xi_t",
#                 f"lens{i - 1}",
#                 f"src{j - 1}",
#                 gammat_ij["ANG"][msk],
#                 gammat_ij["VALUE"][msk],
#             )

#     # wtheta
#     wtheta = data["wtheta"].read()
#     for i in range(1, n_lens + 1):
#         theta_min, theta_max = bin_limits["wtheta"][(i, i)]

#         ii_msk = (wtheta["BIN1"] == i) & (wtheta["BIN2"] == i)
#         wtheta_ii = wtheta[ii_msk]
#         msk = (wtheta_ii["ANG"] > theta_min) & (wtheta_ii["ANG"] < theta_max)

#         tot_msk.extend(msk.tolist())

#         sacc_data.add_theta_xi(
#             "galaxy_density_xi",
#             f"lens{i - 1}",
#             f"lens{i - 1}",
#             wtheta_ii["ANG"][msk],
#             wtheta_ii["VALUE"][msk],
#         )
    
#     # spec
#     spec = data["spec"].read()
#     for i in range(1, n_spec + 1):
#         theta_min, theta_max = bin_limits["spec"][(i, i)]

#         ii_msk = (spec["BIN1"] == i) & (spec["BIN2"] == i)
#         spec_ii = spec[ii_msk]
#         msk = (spec_ii["ANG"] > theta_min) & (spec_ii["ANG"] < theta_max)

#         tot_msk.extend(msk.tolist())

#         sacc_data.add_theta_xi(
#             "spec",
#             f"lens_spec{i - 1}",
#             f"lens_spec{i - 1}",
#             spec_ii["ANG"][msk],
#             spec_ii["VALUE"][msk],
#         )

#     # covmat
#     msk_inds = np.where(tot_msk)[0]
#     n_cov = np.sum(tot_msk)
#     old_cov = data["COVMAT"].read()
#     new_cov = np.zeros((np.sum(tot_msk), np.sum(tot_msk)))

#     for new_cov_i, old_cov_i in enumerate(msk_inds):
#         for new_cov_j, old_cov_j in enumerate(msk_inds):
#             new_cov[new_cov_i, new_cov_j] = old_cov[old_cov_i, old_cov_j]

#     sacc_data.add_covariance(new_cov)

# sacc_data.save_fits("sacc_data.fits", overwrite=True)