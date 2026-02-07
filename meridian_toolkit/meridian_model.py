import logging
import pandas as pd
import arviz as az
import matplotlib.pyplot as plt

from meridian.model import prior_distribution
from meridian.data import load
from meridian.model import model
from meridian.model import spec

from typing import Optional, List
from meridian_toolkit.utils import create_media_mapping, compute_holdout_id

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class MeridianModel:
    """
    Wrapper class for Meridian Media Mix Models (MMM) with simplified interface.
    
    This class provides a high-level interface to build, fit, and analyze
    Meridian MMM models with support for various adstock functions, saturation
    curves, and time-varying effects.
    
    Parameters
    ----------
    client_data_path : str
        Path to CSV file containing the marketing data.
    channel_names : List[str]
        List of media spend column names (e.g., ['meta_spend', 'google_spend']).
    impression_names : List[str]
        List of impression/frequency column names (e.g., ['meta_impressions']).
    date_var : str, default "date"
        Name of the date column in the dataset.
    control_names : Optional[List[str]], default None
        List of control variable column names (e.g., trend, seasonality).
    target_name : str, default "y"
        Name of the target/KPI column to model.
    depvar_type : str, default 'revenue'
        Type of dependent variable. Options: 'revenue' or 'conversion'.
    revenue_per_kpi : str, optional
        Column name for revenue per KPI conversion (for non-revenue KPIs).
    n_test : int, default 12
        Number of observations to hold out for testing (from the end).
    lag_max : int, default 8
        Maximum lag for adstock carryover effects.
    adstock : str, default "binomial"
        Adstock decay specification. Options: 'binomial', 'geometric', 'weibull'.
    time_varying_intercept : bool, default False
        Whether to use time-varying intercept with splines.
    knots_percents : float, default 0.7
        Percentage of observations to use as knots when time_varying_intercept=True.
    prior_type : str, default 'roi'
        Prior type for media effects. Options: 'roi' or 'contribution'.
    
    Attributes
    ----------
    mmm : meridian.model.Meridian
        The underlying Meridian model object.
    meridian_data : meridian.data.InputData
        Loaded and processed input data.
    model_spec : meridian.model.ModelSpec
        Model specification with priors and hyperparameters.
    sampled_model : bool
        Whether the model has been fitted (sampled).
    model_variables : List[str]
        List of parameter names in the posterior (available after fitting).
    
    Examples
    --------
    >>> model = MeridianModel(
    ...     client_data_path='data.csv',
    ...     channel_names=['meta_spend', 'google_spend'],
    ...     impression_names=['meta_impressions', 'google_impressions'],
    ...     target_name='revenue',
    ...     n_test=12
    ... )
    >>> model.fit(n_chains=4, n_draws=1000, n_tune=1000)
    >>> summary = model.mcmc_summary()
    >>> model.mcmc_plot(var_names=['roi'])
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
        revenue_per_kpi: Optional[str] = None,
        n_test: int = 12,
        lag_max: int = 8,
        adstock: str = "binomial",
        time_varying_intercept: bool = False,
        knots_percents: float = 0.7,
        prior_type: str = 'roi'
    ):
        
        logger.info("Initializing MeridianModel...")
        
        # Store parameters
        self.client_data_path = client_data_path
        self.channel_names = channel_names
        self.impression_names = impression_names
        self.date_var = date_var
        self.control_names = control_names or []
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
        self.model_variables = []

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
            controls=self.control_names if self.control_names else None,
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
        status = "fitted" if self.sampled_model else "not fitted"
        return (f"MeridianModel(channels={len(self.channel_names)}, "
                f"n_times={self.n_times}, n_test={self.n_test}, "
                f"adstock='{self.adstock}', status='{status}')")
    
    def fit(
        self, 
        n_chains: int = 4, 
        n_draws: int = 1000, 
        n_tune: int = 1000, 
        seed: Optional[int] = None, 
        cores: Optional[int] = None
    ):
        """
        Fit the Meridian model using MCMC sampling.
        
        This method first samples from the prior distribution, then performs
        posterior sampling using TensorFlow Probability's MCMC implementation.
        
        Parameters
        ----------
        n_chains : int, default 4
            Number of parallel MCMC chains to run.
        n_draws : int, default 1000
            Number of posterior samples to keep per chain.
        n_tune : int, default 1000
            Number of tuning/adaptation steps (also used as burn-in).
        seed : Optional[int], default None
            Random seed for reproducibility.
        cores : Optional[int], default None
            Number of CPU cores to use. If None, uses n_chains.
        
        Returns
        -------
        None
            Updates the model in-place and sets sampled_model=True.
        
        Examples
        --------
        >>> model.fit(n_chains=4, n_draws=2000, n_tune=1000, seed=42)
        """
        cores = n_chains if cores is None else cores

        logger.info(f"Sampling {n_tune} draws from the prior...")
        self.mmm.sample_prior(n_draws=n_tune, seed=seed)

        logger.info(f"Sampling {n_chains} chains of {n_draws} draws each...")
        logger.info(f"Burn-in: {n_tune} draws, Adaptation: {n_tune} draws")
        
        self.mmm.sample_posterior(
            n_chains=n_chains, 
            n_keep=n_draws,
            n_adapt=n_tune,
            n_burnin=n_tune,
            seed=seed,
            parallel_iterations=cores
        )
        
        logger.info("Model sampling completed successfully.")
        self.sampled_model = True
        self.model_variables = list(self.mmm.inference_data.posterior.data_vars)
        logger.info(f"Sampled {len(self.model_variables)} parameters")
    
    def mcmc_summary(
        self, 
        var_names: Optional[List[str]] = None, 
        hdi_prob: float = 0.90
    ) -> pd.DataFrame:
        """
        Generate a summary table of MCMC posterior samples.
        
        Uses ArviZ to compute summary statistics including mean, standard deviation,
        HDI (Highest Density Interval), effective sample size, and R-hat diagnostics.
        
        Parameters
        ----------
        var_names : Optional[List[str]], default None
            List of parameter names to summarize. If None, summarizes all parameters.
        hdi_prob : float, default 0.90
            Probability mass for the HDI interval (e.g., 0.90 for 90% HDI).
        
        Returns
        -------
        pd.DataFrame
            Summary statistics table with columns: mean, sd, hdi_lower, hdi_upper,
            ess_bulk, ess_tail, r_hat.
        
        Raises
        ------
        Warning
            If the model has not been fitted yet.
        
        Examples
        --------
        >>> summary = model.mcmc_summary(var_names=['roi', 'beta_m'], hdi_prob=0.95)
        >>> print(summary)
        """
        if not self.sampled_model:
            logger.warning("The model has not been sampled yet. Call fit() first.")
            return None
        
        var_names = var_names if var_names is not None else self.model_variables
        
        logger.info(f"Generating summary for {len(var_names)} parameters...")
        summary = az.summary(
            self.mmm.inference_data,
            hdi_prob=hdi_prob, 
            var_names=var_names
        )
        
        return summary
    
    def mcmc_plot(
        self, 
        var_names: Optional[List[str]] = None,
        compact: bool = False,
        figsize: Optional[tuple] = None
    ):
        """
        Create trace plots for MCMC diagnostics.
        
        Generates trace plots showing the sampling history and posterior distributions
        for specified parameters. Useful for assessing convergence and mixing.
        
        Parameters
        ----------
        var_names : Optional[List[str]], default None
            List of parameter names to plot. If None, plots all parameters.
        compact : bool, default False
            If True, uses compact plot format combining chains.
        figsize : Optional[tuple], default None
            Figure size as (width, height). If None, uses default.
        
        Returns
        -------
        matplotlib.figure.Figure or List[matplotlib.figure.Figure]
            Figure object(s) containing the trace plots.
        
        Raises
        ------
        Warning
            If the model has not been fitted yet.
        
        Examples
        --------
        >>> # Plot all parameters
        >>> model.mcmc_plot()
        
        >>> # Plot specific parameters
        >>> model.mcmc_plot(var_names=['roi', 'beta_m'], compact=True)
        """
        if not self.sampled_model:
            logger.warning("The model has not been sampled yet. Call fit() first.")
            return None
        
        var_names = var_names if var_names is not None else self.model_variables
        
        logger.info(f"Generating trace plots for {len(var_names)} parameters...")
        
        backend_kwargs = {"constrained_layout": True}
        if figsize is not None:
            backend_kwargs["figsize"] = figsize
        
        axes = az.plot_trace(
            self.mmm.inference_data,
            var_names=var_names,
            compact=compact,
            backend_kwargs=backend_kwargs,
        )
        
        plt.show()
        return axes