from meridian import constants
from meridian.analysis import analyzer
from meridian.analysis import optimizer
from meridian.analysis import summarizer
from meridian.analysis import visualizer
from meridian.analysis.review import reviewer

from meridian.data import load
from meridian.model import model
from meridian.model import prior_distribution
from meridian.model import spec

from typing import Optional, Dict, Union, List
from meridian_toolkit.utils import create_media_mapping,compute_holdout_id

import numpy as np
import pandas as pd
import arviz as az

# check if GPU is available
import tensorflow as tf
import tensorflow_probability as tfp


date_var = "date"
target = "subscriptions"
target_type = "non_revenue"
revenue_per_kpi = None

media = ['meta_spend', 'google_spend', 'snapchat_spend', 
            'tiktok_spend', 'moloco_spend','liveintent_spend',
            'beehiiv_spend','amazon_spend']

controls = ["trend","seasonal"]

impressions = ['meta_impressions', 'google_impressions', 'snapchat_impressions', 
            'tiktok_impressions', 'moloco_impressions','liveintent_impressions',
            'beehiiv_impressions','amazon_impressions']

adstock='binomial'
time_varying_intercept = False
n_tests = 12
knots_percents = 0.7
max_lags = 8

logger = logging.getLogger(__name__)

class MeridianModel:
    """
    Class to generate Media Mix Models (MMM) within Meridian with support for
    prior configuration, saturation functions, and adstock functions.
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
      lag_max: int = 8,
      adstock: str = "binomial",
      time_varying_intercept: bool = False,
      knots_percents = 0.7,
      prior_type = 'roi'):
    
    logger.info("Updating priors with user-defined values.")
    self.client_data = pd.read_csv(client_data_path)
    self.n_times = len(self.client_data[date_var])  
    self.channel_names = channel_names
    self.impression_names = impression_names
    self.control_names = control_names or []
    self.target_name = target_name
    self.depvar_type = depvar_type
    self.lag_max = lag_max
    self.adstock = adstock
    self.time_varying_intercept = time_varying_intercept
    self.knots_percents = knots_percents

    self.holdout_id = compute_holdout_id(data_real=self.client_data,n_test=self.n_tests)
    if time_varying_intercept:
        self.n_knots = round(knots_percents * self.n_times)
    else: 
         n_knots = 1

    # build Meridian's data loader
    coord_to_columns = load.CoordToColumns(
        time=date_var,
        controls=controls,
        kpi=target,
        revenue_per_kpi=None,
        media=impressions,
        media_spend= media)
     
    loader = load.CsvDataLoader(
        csv_path=client_data_path,
        kpi_type=target_type,
        coord_to_columns=coord_to_columns,
        media_to_channel=create_media_mapping(media_list=impressions,suffix='_impressions'),
        media_spend_to_channel=create_media_mapping(media_list=media,suffix='_spend')
        )
    
    self.meridian_data_loader = loader.load()

    ## build model specifications 
    self.model_spec = spec.ModelSpec(
        prior=prior_distribution.PriorDistribution(),
        media_effects_dist='log_normal',
        hill_before_adstock=False,
        max_lag=self.lag_max,
        unique_sigma_for_each_geo=False,
        media_prior_type = prior_type,
        roi_calibration_period=None,
        rf_prior_type = prior_type,
        rf_roi_calibration_period=None,
        organic_media_prior_type='contribution',
        organic_rf_prior_type='contribution',
        non_media_treatments_prior_type='contribution',
        knots=n_knots,
        baseline_geo=None,
        holdout_id=self.holdout_id,
        control_population_scaling_id=None,
        adstock_decay_spec=self.adstock,
        enable_aks=False
    )

#########################
# fit the model 
#########################
mmm = model.Meridian(input_data=data, model_spec=model_spec)
mmm.sample_prior(500)
mmm.sample_posterior(n_chains=4, n_adapt=500, n_burnin=500, n_keep=1000,seed=1697,parallel_iterations=4)


model_diagnostics = visualizer.ModelDiagnostics(mmm)
model_diagnostics.plot_rhat_boxplot()

parameters_to_plot=["roi_m"]

for params in parameters_to_plot:
  az.plot_trace(
      mmm.inference_data,
      var_names=params,
      compact=False,
      backend_kwargs={"constrained_layout": True},
  )

media_summary = visualizer.MediaSummary(mmm)
media_summary.contribution_metrics(include_non_paid=True)
media_summary.plot_roi_bar_chart()

media_effects = visualizer.MediaEffects(mmm)
media_effects.plot_response_curves(confidence_level=0.40, plot_separately=False)

media_effects.plot_adstock_decay(confidence_level=0.4)

########################################
#  model predict    
########################################

model_diagnostics = visualizer.ModelDiagnostics(mmm)
model_diagnostics.predictive_accuracy_table()

model_fit = visualizer.ModelFit(mmm)
model_fit.plot_model_fit()
