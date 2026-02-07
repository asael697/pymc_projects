import os
import tempfile
import logging
import numpy as np
import arviz as az
import pandas as pd
import matplotlib.pyplot as plt

from meridian.model import prior_distribution
from meridian.analysis import visualizer
from meridian.analysis import analyzer
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
    n_validate : int, default 12
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
        n_validate: int = 12,
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
        self.n_validate = n_validate
        self.lag_max = lag_max
        self.adstock = adstock
        self.time_varying_intercept = time_varying_intercept
        self.knots_percents = knots_percents
        self.depvar_type = depvar_type if depvar_type == "revenue" else "non_revenue"
        self.prior_type = prior_type if prior_type == "roi" else "contribution"
        self.sampled_model = False
        self.model_variables = []
        self.mcmc_diagnostics = {}

        # Load data
        logger.info("Loading client data...")
        self.client_data = pd.read_csv(client_data_path)
        self.n_times = len(self.client_data)
        
        # Compute holdout
        logger.info(f"Computing holdout mask (n_validate={n_validate})...")
        self.holdout_id = compute_holdout_id(
            data_real=self.client_data, 
            n_test=self.n_validate
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
                f"n_times={self.n_times}, n_validate={self.n_validate}, "
                f"adstock='{self.adstock}', status='{status}')")

    def _get_mcmc_diagnostics(self):
        """
        Get MCMC sampling diagnostics including divergences and tree depth.
        Returns diagnostic information about the MCMC sampling process,
        including divergent transitions and maximum tree depth warnings.
        
        Returns
        -------
        dict
          Dictionary containing:
           - 'n_divergences': Total number of divergent transitions
           - 'divergences_by_chain': List of divergences per chain
           - 'pct_divergences': Percentage of divergent samples
           - 'max_treedepth': Maximum tree depth reached
           - 'n_draws': Total number of draws (excluding warmup)
           - 'n_chains': Number of chains
        
        Raises
        ------
        Warning
          If the model has not been fitted yet.
        """
        if not self.sampled_model:
            logger.warning("The model has not been sampled yet. Call fit() first.")
            return None
        
        logger.info("Extracting MCMC diagnostics...")
        sample_stats = self.mmm.inference_data.sample_stats
        
        if 'diverging' in sample_stats:
            divergences = sample_stats['diverging'].values
            n_divergences = int(divergences.sum())
            divergences_by_chain = divergences.sum(axis=1).tolist()
        else:
            n_divergences = 0
            divergences_by_chain = [0] * len(sample_stats.chain)
        
        if 'tree_depth' in sample_stats:
            max_treedepth = int(sample_stats['tree_depth'].max())
        else:
            max_treedepth = None
        
        n_chains = len(sample_stats.chain)
        n_draws = len(sample_stats.draw)
        total_samples = n_chains * n_draws
        
        pct_divergences = (n_divergences / total_samples) * 100 if total_samples > 0 else 0

        self.mcmc_diagnostics =  {
            'n_divergences': n_divergences,
            'divergences_by_chain': divergences_by_chain,
            'pct_divergences': pct_divergences,
            'max_treedepth': max_treedepth,
            'n_draws': n_draws,
            'n_chains': n_chains,
            'total_samples': total_samples
        }
        
        if n_divergences > 0:
            logger.warning(f"Found {n_divergences} divergent transitions ({pct_divergences:.2f}%)")
        else:
            logger.info("✓ No divergent transitions detected")

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
        self._get_mcmc_diagnostics()

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
    
    def get_model_fit(self, confidence_level: float = 0.90) -> pd.DataFrame:
        """
        Get model fit with mean and confidence intervals. 
        Extracts the expected KPI values from the model fit, including the mean
        and confidence intervals, and reshapes into a wide format with time as
        rows and metrics (ci_lo, mean, ci_hi) as columns. Also includes a column
        indicating train/test split. 
        
        Parameters
        ----------
        confidence_level : float, default 0.90
          Confidence level for the credible intervals (e.g., 0.90 for 90% CI).
        
        Returns
        -------
        pd.DataFrame
          DataFrame with shape (n_times, 4) where:
           - Index: Time periods from the dataset
           - Columns: ['ci_lo', 'mean', 'ci_hi', 'holdout']
           - holdout: Boolean indicating test set (True) vs train set (False) 
        Raises
        ------
          Warning
            If the model has not been fitted yet.
        """
        if not self.sampled_model:
            logger.warning("The model has not been sampled yet. Call fit() first.")
            return None
        
        logger.info(f"Extracting model fit with {confidence_level*100}% CI...")
        model_fit = visualizer.ModelFit(self.mmm, confidence_level=confidence_level)
        expected_df = model_fit.model_fit_data.expected.to_dataframe()
        expected_df = expected_df.reset_index()
        
        fit_wide = expected_df.pivot(
            index='time',
            columns='metric',
            values='expected'
        )
        
        fit_wide = fit_wide[['ci_lo', 'mean', 'ci_hi']]
        
        fit_wide['holdout'] = self.holdout_id
        return fit_wide
    
    def plot_predict(
            self,
            train_window: Optional[int] = 10,
            confidence_level: float = 0.90,
            figsize: tuple = (14, 6),
            cumulative: bool = False,
            ax=None):
        """
        Plot model fit for train and test data with confidence intervals.
        
        Displays time series predictions with credible intervals, optionally
        including the last N training points. Can also show cumulative predictions.
        
        Parameters
        ----------
        train_window : Optional[int], default 10
          Number of most recent training points to display. 
          If None, training is not shown. Ignored if cumulative=True.
        confidence_level : float, default 0.90
          Credible interval level (e.g., 0.90 for 90% CI).
        figsize : tuple, default (14, 6)
          Figure size (width, height). Used only if ax is None.
        cumulative : bool, default False
          If True, plot cumulative sums and ignore train_window.
        ax : matplotlib.axes.Axes, optional
          Axes object to plot on. If None, a new figure and axes are created.
        
        Returns
        -------
        matplotlib.figure.Figure or None
          Figure object if a new one was created; otherwise None.
        
        Raises
        ------
        Warning
          If the model has not been fitted yet.
        """
        if not self.sampled_model:
            logger.warning("The model has not been sampled yet. Call fit() first.")
            return None
        
        fit_df = self.get_model_fit(confidence_level=confidence_level)
        train_df = fit_df[~fit_df['holdout']].copy()
        test_df = fit_df[fit_df['holdout']].copy()
        
        y_train = self.client_data.loc[~self.holdout_id, self.target_name].values
        y_test = self.client_data.loc[self.holdout_id, self.target_name].values
        
        if cumulative:
            train_window = None
            train_df['mean'] = train_df['mean'].cumsum()
            train_df['ci_lo'] = train_df['ci_lo'].cumsum()
            train_df['ci_hi'] = train_df['ci_hi'].cumsum()
            y_train_plot = np.cumsum(y_train)
            
            test_df['mean'] = test_df['mean'].cumsum()
            test_df['ci_lo'] = test_df['ci_lo'].cumsum()
            test_df['ci_hi'] = test_df['ci_hi'].cumsum()
            y_test_plot = np.cumsum(y_test)
        else:
            y_train_plot = y_train
            y_test_plot = y_test
        fig = None
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        
        if train_window is not None and not cumulative:
            train_window_df = train_df.iloc[-train_window:]
            y_train_window = y_train_plot[-train_window:]
            start_idx = max(0, len(train_df) - train_window)
            
            train_idx = np.arange(start_idx, start_idx + len(train_window_df))
            ax.plot(
                train_idx, 
                y_train_window, 
                label="Train (Observed)", 
                color="black", 
                linewidth=2)
            
            ax.axvline(
                start_idx + len(train_window_df) - 0.5, 
                color="gray", 
                linestyle="--", 
                alpha=0.7
            )
            test_idx = np.arange(
                start_idx + len(train_window_df),
                start_idx + len(train_window_df) + len(test_df)
            )
        else:
            test_idx = np.arange(len(test_df))
        
        ax.plot(test_idx, 
           y_test_plot, 
           label="Test (Observed)", 
           color="green", 
           marker="o", 
           linestyle="-"
        )
        ax.plot(
            test_idx, 
            test_df['mean'].values, 
            label="Prediction (Mean)", 
            color="royalblue", 
            linewidth=2
        )
        
        ax.fill_between(
            test_idx, 
            test_df['ci_lo'].values, 
            test_df['ci_hi'].values, 
            color="royalblue", 
            alpha=0.25, 
            label=f"{int(confidence_level * 100)}% CI"
        )
        # Labels and formatting
        
        ax.set_xlabel("Time Steps")
        ax.set_ylabel("Target (Cumulative)" if cumulative else "Target")
        ax.set_title("Model Fit" + (" (Cumulative)" if cumulative else ""), fontsize=16)
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        
        return fig
    
    def get_channel_contribution(self) -> pd.DataFrame:
        """
        Get channel contributions to incremental outcome.
        
        Returns
        -------
        pd.DataFrame
          DataFrame with columns:
          - channel: Channel name
          - incremental_outcome: Incremental KPI contribution
        
        Raises
        ------
        Warning
          If the model has not been fitted yet.
        """
        if not self.sampled_model:
            logger.warning("The model has not been sampled yet. Call fit() first.")
            return None
        
        logger.info("Computing channel contributions...")
        media_summary = visualizer.MediaSummary(self.mmm)
        contrib_df = media_summary.contribution_metrics(include_non_paid=True, aggregate_times=True)        
        contrib_df = contrib_df.drop(columns=['incremental_outcome'])
        
        # Rename and convert to percentage
        contrib_df = contrib_df.rename(columns={'pct_of_contribution': 'contributions'})
        contrib_df['contributions'] = contrib_df['contributions'] * 100

        return contrib_df.reset_index(drop=True)
    
    def get_channel_roi(self) -> pd.DataFrame:
        """
        Get channel ROI (Return on Investment).
        
        Returns
        -------
        pd.DataFrame
          DataFrame with ROI metrics by channel
        Raises
        ------
        Warning
          If the model has not been fitted yet.
        """
        if not self.sampled_model:
            logger.warning("The model has not been sampled yet. Call fit() first.")
            return None
        
        logger.info("Computing channel ROI...")
        media_summary = visualizer.MediaSummary(self.mmm)
        media_summary.contribution_metrics(include_non_paid=True, aggregate_times=True)
        
        roi_df = media_summary._summary_metric_to_df(metric="roi")
        
        return roi_df
    
    def get_test_predictions(self, confidence_level: float = 0.90) -> pd.DataFrame:
        """
        Get predictions for the test (holdout) set.
        Extracts model predictions and observed values for holdout observations only.
        
        Parameters
        ----------
        confidence_level : float, default 0.90
          Confidence level for prediction intervals
        
        Returns
        -------
        pd.DataFrame
          DataFrame with columns:
          - time: Date/time index
          - ci_lo: Lower bound of prediction interval
          - mean: Mean prediction
          - ci_hi: Upper bound of prediction interval
          - holdout: Boolean (always True)
          - observed: Actual observed values
        
        Raises
        ------
        Warning
          If the model has not been fitted yet or no holdout data exists.
        """
        if not self.sampled_model:
            logger.warning("The model has not been sampled yet. Call fit() first.")
            return None
        
        logger.info("Extracting test set predictions...")
        df = self.get_model_fit(confidence_level=confidence_level)
        df = df.reset_index()
        df['observed'] = self.client_data[self.target_name].values
        
        df_test = df[df['holdout']].copy()
        if len(df_test) == 0:
            logger.warning("No holdout data found.")
            return None
        
        return df_test
        
    def compute_validation_errors(self, confidence_level: float = 0.90) -> dict:
        """
        Compute validation metrics on the test (holdout) set.
        
        Calculates prediction error metrics including MAPE, RMSE, MAE, and R²
        for the holdout observations.
        
        Parameters
        ----------
        confidence_level : float, default 0.90
          Confidence level for model fit extraction
        
        Returns
        -------
        dict
          Dictionary containing:
            - 'mape': Mean Absolute Percentage Error (%)
            - 'rmse': Root Mean Squared Error
            - 'mae': Mean Absolute Error
            - 'r_squared': R-squared
            - 'n_test': Number of test observations 
        
        Raises
        ------
          Warning
             If the model has not been fitted yet or no holdout data exists.
        """
        if not self.sampled_model:
            logger.warning("The model has not been sampled yet. Call fit() first.")
            return None
        
        logger.info("Computing validation errors on holdout set...")
        
        df = self.get_model_fit(confidence_level=confidence_level)
        df = df.reset_index()
        
        df['observed'] = self.client_data[self.target_name].values
        
        df_test = df[df['holdout']].copy()
        
        if len(df_test) == 0:
            logger.warning("No holdout data found. Cannot compute validation errors.")
            return None
        
        # Extract predictions and actuals
        y_pred = df_test['mean'].values
        y_true = df_test['observed'].values
        
        # Compute metrics
        errors = y_pred - y_true
        abs_errors = np.abs(errors)
        pct_errors = abs_errors / np.abs(y_true)
        
        # MAPE (%)
        mape = np.mean(pct_errors) * 100
        
        # RMSE
        mse = np.mean(errors ** 2)
        rmse = np.sqrt(mse)
        
        # MAE
        mae = np.mean(abs_errors)
        
        return{
            'mape': float(mape),
            'rmse': float(rmse),
            'mae': float(mae),
            'n_test': len(df_test)
        }
        
    def predict(self,new_data: pd.DataFrame,confidence_level: float = 0.90) -> pd.DataFrame:
        """
        Generate predictions for new data using the fitted Meridian model.
        
        Takes new data with media spend, impressions, and control variables,
        applies the fitted model to generate predictions with confidence intervals.
        
        Parameters
        ----------
        new_data : pd.DataFrame
          New input data containing:
          - Media spend columns (matching self.channel_names)
          - Impression columns (matching self.impression_names)
          - Control columns (matching self.control_names)
          - Date column (matching self.date_var)
        confidence_level : float, default 0.90
          Confidence level for prediction intervals
        
        Returns
        -------
        pd.DataFrame
          DataFrame with columns:
          - time: Date/time index
          - ci_lo: Lower bound of prediction interval
          - mean: Mean prediction
          - ci_hi: Upper bound of prediction interval
        
        Raises
        ------
        Warning
          If the model has not been fitted yet.
        ValueError
          If new_data is missing required columns
        """
        if not self.sampled_model:
            logger.warning("The model has not been sampled yet. Call fit() first.")
            return None
        
        required_cols = ([self.date_var] + self.channel_names + self.impression_names + self.control_names)
        missing_cols = set(required_cols) - set(new_data.columns)
        if missing_cols:
            raise ValueError(f"Missing required columns in new_data: {missing_cols}")
        
        logger.info(f"Generating predictions for {len(new_data)} new observations...")
        full_data = pd.concat([self.client_data, new_data], ignore_index=True)
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as tmp:
            full_data.to_csv(tmp.name, index=False)
            tmp_path = tmp.name
        try:
            # Create new data loader with full data
            coord_to_columns = load.CoordToColumns(
                time=self.date_var,
                controls=self.control_names if self.control_names else None,
                kpi=self.target_name,
                revenue_per_kpi=self.revenue_per_kpi,
                media=self.impression_names,
                media_spend=self.channel_names)
                
            loader = load.CsvDataLoader(
                csv_path=tmp_path,
                kpi_type=self.depvar_type,
                coord_to_columns=coord_to_columns,
                media_to_channel=create_media_mapping(media_list=self.impression_names, suffix='_impressions'),
                media_spend_to_channel=create_media_mapping(media_list=self.channel_names, suffix='_spend')
            )
            new_input_data = loader.load()
            
            # Get predictions using analyzer        
            mmm_analyzer = analyzer.Analyzer(self.mmm)
            
            # Get expected outcome with new data
            predictions = mmm_analyzer.expected_outcome(
                new_data=new_input_data,
                aggregate_times=False,
                aggregate_geos=True,
                use_kpi=True,
                inverse_transform_outcome=True)
            
            # Extract only the new predictions (last n rows)
            n_new = len(new_data)
            pred_new = predictions[:, :, -n_new:]  # shape: (n_chains, n_draws, n_new)
            # Reshape to (n_draws_total, n_new)
            pred_flat = pred_new.numpy().reshape(-1, n_new)
            
            # Compute statistics
            pred_mean = np.mean(pred_flat, axis=0)
            pred_lower = np.percentile(pred_flat, (1 - confidence_level) / 2 * 100, axis=0)
            pred_upper = np.percentile(pred_flat, (1 + confidence_level) / 2 * 100, axis=0)
            # Create output DataFrame
            predictions_df = pd.DataFrame({
                'time': new_data[self.date_var].values,
                'ci_lo': pred_lower,
                'mean': pred_mean,
                'ci_hi': pred_upper
            })
            
            logger.info(f"Predictions generated successfully for {len(predictions_df)} observations")
            return predictions_df

        finally:
            # Clean up temporary file
            if os.path.exists(tmp_path):
                os.remove(tmp_path)