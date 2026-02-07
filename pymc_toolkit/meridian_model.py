import logging
import numpy as np
import pandas as pd
import arviz as az

from meridian.model import prior_distribution
from meridian.analysis import visualizer
from meridian.data import load
from meridian.model import model
from meridian.model import spec

from typing import Optional, List
from meridian_toolkit.utils import create_media_mapping, compute_holdout_id

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class MeridianModel:
    """
    Class to generate Media Mix Models (MMM) within Meridian with support for
    prior configuration, saturation functions, and adstock functions.
    
    Parameters
    ----------
    client_data_path : str
        Path to CSV file with data
    channel_names : List[str]
        List of media spend column names
    impression_names : List[str]
        List of impression column names
    date_var : str, default "date"
        Name of date column
    control_names : Optional[List[str]], default None
        List of control variable names
    target_name : str, default "y"
        Name of target/KPI column
    depvar_type : str, default 'revenue'
        Type of dependent variable ('revenue' or 'conversion')
    n_test : int, default 12
        Number of holdout observations for testing
    lag_max : int, default 8
        Maximum lag for adstock effects
    adstock : str, default "binomial"
        Adstock decay specification
    time_varying_intercept : bool, default False
        Whether to use time-varying intercept
    knots_percents : float, default 0.7
        Percentage of observations to use as knots (if time_varying_intercept=True)
    prior_type : str, default 'roi'
        Prior type for media effects
    """
    def __init__(
        self,
        client_data_path: str,
        channel_names: List[str],
        impression_names: List[str],
        date_var: str = "date",
        control_names: Optional[List[str]] = None, 
        target_name: str = "y",
        depvar_type: str = 'revenue',
        revenue_per_kpi: str = None,
        n_test: int = 12,
        lag_max: int = 8,
        adstock: str = "binomial",
        time_varying_intercept: bool = False,
        knots_percents: float = 0.7,
        prior_type: str = 'roi'):
        
        logger.info("Initializing MeridianModel...")
        
        # Store parameters
        self.client_data_path = client_data_path
        self.channel_names = channel_names
        self.impression_names = impression_names
        self.date_var = date_var
        self.control_names = control_names
        self.target_name = target_name
        self.revenue_per_kpi = revenue_per_kpi
        self.n_test = n_test
        self.lag_max = lag_max
        self.adstock = adstock
        self.time_varying_intercept = time_varying_intercept
        self.knots_percents = knots_percents
        self.depvar_type = depvar_type if depvar_type == "revenue" else "non_revenue"
        self.prior_type = prior_type if prior_type == "roi" else "contribution"
        self.sampled_model = False

        # Load data
        logger.info("Loading client data...")
        self.client_data = pd.read_csv(client_data_path)
        self.n_times = len(self.client_data)
        
        # Compute holdout
        logger.info(f"Computing holdout mask (n_test={n_test})...")
        self.holdout_id = compute_holdout_id(
            data_real=self.client_data, 
            n_test=self.n_test
        )
        
        # Compute knots
        if time_varying_intercept:
            self.n_knots = round(knots_percents * self.n_times)
            logger.info(f"Time-varying intercept enabled with {self.n_knots} knots")
        else: 
            self.n_knots = 1
            logger.info("Fixed intercept (n_knots=1)")
        
        # Build Meridian data loader
        logger.info("Building Meridian data loader...")
        coord_to_columns = load.CoordToColumns(
            time=self.date_var,
            controls=self.control_names,
            kpi=self.target_name,
            revenue_per_kpi=self.revenue_per_kpi,
            media=self.impression_names,
            media_spend=self.channel_names
        )
        
        loader = load.CsvDataLoader(
            csv_path=client_data_path,
            kpi_type=self.depvar_type,
            coord_to_columns=coord_to_columns,
            media_to_channel=create_media_mapping(
                media_list=self.impression_names, 
                suffix='_impressions'
            ),
            media_spend_to_channel=create_media_mapping(
                media_list=self.channel_names, 
                suffix='_spend'
            )
        )
        
        self.meridian_data = loader.load()
        logger.info("Data loaded successfully")
        
        # Build model specification
        logger.info("Building Meridian model specification...")
        self.model_spec = spec.ModelSpec(
            prior=prior_distribution.PriorDistribution(),
            media_effects_dist='log_normal',
            hill_before_adstock=False,
            max_lag=self.lag_max,
            unique_sigma_for_each_geo=False,
            media_prior_type=self.prior_type,
            roi_calibration_period=None,
            rf_prior_type=self.prior_type,
            rf_roi_calibration_period=None,
            organic_media_prior_type='contribution',
            organic_rf_prior_type='contribution',
            non_media_treatments_prior_type='contribution',
            knots=self.n_knots,
            baseline_geo=None,
            holdout_id=self.holdout_id,
            control_population_scaling_id=None,
            adstock_decay_spec=self.adstock,
            enable_aks=False
        )
        logger.info("Model specification created")
        
        # Build Meridian model
        logger.info("Building Meridian model...")
        self.mmm = model.Meridian(
            input_data=self.meridian_data, 
            model_spec=self.model_spec
        )
        logger.info("MeridianModel initialized successfully")
    
    def __repr__(self):
        return (f"MeridianModel(channels={len(self.channel_names)}, "
                f"n_times={self.n_times}, n_test={self.n_test}, "
                f"adstock='{self.adstock}')")
    
    def fit(self, n_chains:int = 4, n_draws:int = 1000, n_tune:int = 1000, seed:int = None, cores:int = None):
        
        cores = n_chains if cores is None else cores

        logger.info(f"Sample {n_tune} draws from the prior")
        self.mm.sample_prior(n_draws=n_tune, seed=seed)

        logger.info(f"Sample {n_chains} chains of {n_draws} draws each and burn the first {n_tune} draws.")
        self.mmm.sample_posterior(n_chains=n_chains, 
                            n_keep=n_draws,
                            n_adapt=n_tune,
                            n_burnin=n_tune,
                            seed=seed,
                            parallel_iterations=cores)
        logger.info("Model sampling completed.")
        self.sampled_model = True
        self.model_variables = list(self.mmm.inference_data.posterior.data_vars)
    
    def mcmc_summary(self, var_names:list = None, probs = 0.90):
        
        var_names = var_names if var_names is not None else self.model_variables

        if self.sampled_model:
            return az.summary(
                self.mmm.inference_data,
                hdi_prob=probs, 
                var_names=var_names
            )
        else:
            logger.warning("The model has not been sampled")

    def mcmc_plot(self, var_names:list = None):

        var_names = var_names if var_names is not None else self.model_variables
        
        if self.sampled_model:
            for param in var_names:
               return az.plot_trace(
                    self.mmm.inference_data,
                    var_names=param,
                    compact=False,
                    backend_kwargs={"constrained_layout": True},
                )
        else: 
            logger.warning("The model has not been sampled")

