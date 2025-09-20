from lstm_eviction_policy.config.config_io.config_loader import load_config
from lstm_eviction_policy.config.utils.Config import Config
from lstm_eviction_policy.config.validation.data_params.data_params_validator import validate_data_distribution_params, \
    validate_data_access_pattern_zipf_params, validate_data_access_behavior_pattern_params, \
    validate_data_access_temporal_pattern_params, validate_data_sequence_params, validate_data_dataset_params
from lstm_eviction_policy.config.validation.evaluation_params.evaluation_params_validator import \
    validate_evaluation_general_params
from lstm_eviction_policy.config.validation.inference_params.inference_params_validator import \
    validate_inference_confidence_intervals_params
from lstm_eviction_policy.config.validation.model_params.model_params_validator import validate_model_general_params, \
    validate_model_params
from lstm_eviction_policy.config.validation.simulation_params.simulation_params_validator import \
    validate_simulation_general_params, validate_simulation_lstm_cache_params
from lstm_eviction_policy.config.validation.testing_params.testing_params_validator import \
    validate_testing_general_params
from lstm_eviction_policy.config.validation.training_params.training_params_validator import \
    validate_training_optimizer_params, validate_training_general_params, validate_training_early_stopping_params
from lstm_eviction_policy.config.validation.validation_params.validation_params_validator import validate_cv_params, \
    validate_validation_early_stopping_params, validate_search_space_params
from lstm_eviction_policy.utils.logs.log_utils import phase_var, info


def prepare_config() -> Config:
    """
    Prepare configuration required to run
    the entire pipeline.

    This function prepares the pipeline's configuration
    by orchestrating loading and validation of YAML configuration file.
    If the whole YAML configuration file is valid, it builds
    and returns the data class representing the configuration settings.

    Returns:
        Config: Data class representing the configuration settings of
            the entire pipeline.
    """
    # Set the new pipeline state
    phase_var.set("config")

    # Load the YAML configuration file
    config_file = load_config()

    # Validate data distribution configuration
    (
        seed,
        distribution_type,
        num_requests,
        num_keys,
        first_key,
        last_key,
    ) = validate_data_distribution_params(config_file)

    # Validate Zipfian data access pattern configuration
    (
        zipf_alpha,
        zipf_alpha_start,
        zipf_alpha_end,
        zipf_time_steps,
    ) = validate_data_access_pattern_zipf_params(config_file)

    # Validate behavioral data access pattern configuration
    (
        repetition_interval,
        repetition_offset,
        toggle_interval,
        cycle_base,
        cycle_mod,
        cycle_divisor,
        distortion_interval,
        noise_range,
        memory_interval,
        memory_offset,
    ) = validate_data_access_behavior_pattern_params(config_file)

    # Validate temporal data access pattern configuration
    (
        burst_high,
        burst_low,
        burst_hour_start,
        burst_hour_end,
        periodic_base_scale,
        periodic_amplitude,
    ) = validate_data_access_temporal_pattern_params(config_file)

    # Validate data sequence configuration
    (seq_len, embedding_dim, num_requests) = validate_data_sequence_params(
        config_file, num_requests
    )

    # Validate dataset configuration
    (
        training_perc,
        validation_perc,
        static_save_path,
        dynamic_save_path,
    ) = validate_data_dataset_params(config_file)

    # Validate general model configuration
    (num_features, model_save_path) = validate_model_general_params(config_file)

    # Validate model parameters configuration
    (
        model_params,
        hidden_size,
        num_layers,
        bias,
        batch_first,
        dropout,
        bidirectional,
        proj_size,
    ) = validate_model_params(config_file)

    # Validate training general configuration
    (training_num_epochs, training_batch_size) = validate_training_general_params(
        config_file
    )

    # Validate training optimizer configuration
    (
        optimizer_type,
        learning_rate,
        weight_decay,
        momentum,
    ) = validate_training_optimizer_params(config_file)

    # Validate training early stopping configuration
    (
        training_early_stopping_patience,
        training_early_stopping_delta,
    ) = validate_training_early_stopping_params(config_file)

    # Validate cross validation configuration
    (cv_num_folds, validation_num_epochs) = validate_cv_params(config_file)

    # Validate validation early stopping configuration
    (
        validation_early_stopping_patience,
        validation_early_stopping_delta,
    ) = validate_validation_early_stopping_params(config_file)

    # Validate search space configuration
    (
        search_space,
        hidden_size_range,
        num_layers_range,
        dropout_range,
        learning_rate_range,
    ) = validate_search_space_params(config_file)

    # Validate general evaluation configuration
    top_k = validate_evaluation_general_params(config_file)

    # Validate general testing configuration
    testing_batch_size = validate_testing_general_params(config_file)

    # Validate inference confidence intervals configuration
    (
        confidence_level,
        mc_dropout_num_samples,
    ) = validate_inference_confidence_intervals_params(config_file)

    # Validate general simulation configuration
    (cache_size, ttl) = validate_simulation_general_params(config_file)

    # Validate LSTM cache simulation configuration
    (prediction_interval, threshold_score) = validate_simulation_lstm_cache_params(
        config_file
    )

    info("Configuration completed")

    return Config(
        config_file=config_file,
        seed=seed,
        distribution_type=distribution_type,
        num_requests=num_requests,
        num_keys=num_keys,
        first_key=first_key,
        last_key=last_key,
        zipf_alpha=zipf_alpha,
        zipf_alpha_start=zipf_alpha_start,
        zipf_alpha_end=zipf_alpha_end,
        zipf_time_steps=zipf_time_steps,
        burst_high=burst_high,
        burst_low=burst_low,
        burst_hour_start=burst_hour_start,
        burst_hour_end=burst_hour_end,
        periodic_base_scale=periodic_base_scale,
        periodic_amplitude=periodic_amplitude,
        repetition_interval=repetition_interval,
        repetition_offset=repetition_offset,
        toggle_interval=toggle_interval,
        cycle_base=cycle_base,
        cycle_mod=cycle_mod,
        cycle_divisor=cycle_divisor,
        distortion_interval=distortion_interval,
        noise_range=noise_range,
        memory_interval=memory_interval,
        memory_offset=memory_offset,
        seq_len=seq_len,
        embedding_dim=embedding_dim,
        training_perc=training_perc,
        validation_perc=validation_perc,
        static_save_path=static_save_path,
        dynamic_save_path=dynamic_save_path,
        num_features=num_features,
        model_save_path=model_save_path,
        model_params=model_params,
        hidden_size=hidden_size,
        num_layers=num_layers,
        bias=bias,
        batch_first=batch_first,
        dropout=dropout,
        bidirectional=bidirectional,
        proj_size=proj_size,
        training_num_epochs=training_num_epochs,
        training_batch_size=training_batch_size,
        optimizer_type=optimizer_type,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        momentum=momentum,
        training_early_stopping_patience=training_early_stopping_patience,
        training_early_stopping_delta=training_early_stopping_delta,
        cv_num_folds=cv_num_folds,
        validation_num_epochs=validation_num_epochs,
        validation_early_stopping_patience=validation_early_stopping_patience,
        validation_early_stopping_delta=validation_early_stopping_delta,
        search_space=search_space,
        hidden_size_range=hidden_size_range,
        num_layers_range=num_layers_range,
        dropout_range=dropout_range,
        learning_rate_range=learning_rate_range,
        top_k=top_k,
        testing_batch_size=testing_batch_size,
        confidence_level=confidence_level,
        mc_dropout_num_samples=mc_dropout_num_samples,
        cache_size=cache_size,
        ttl=ttl,
        prediction_interval=prediction_interval,
        threshold_score=threshold_score,
    )
