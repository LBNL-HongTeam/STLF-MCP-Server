"""
Pydantic models for request/response schemas.

These schemas document the expected structure of tool inputs and outputs.
They can be used for validation if needed.
"""

from typing import Optional
from pydantic import BaseModel, Field


class ColumnMapping(BaseModel):
    """Column mapping for CSV data."""

    datetime: str = Field(description="Column containing timestamps")
    target: str = Field(description="Column containing load values to forecast")
    past_covariates: Optional[list[str]] = Field(
        default=None, description="Optional weather/feature columns"
    )


class ModelMetrics(BaseModel):
    """Metrics from model evaluation."""

    rmse: Optional[float] = Field(description="Root Mean Square Error")
    mae: Optional[float] = Field(description="Mean Absolute Error")
    mape: Optional[float] = Field(description="Mean Absolute Percentage Error (%)")
    cv_rmse: Optional[float] = Field(
        description="Coefficient of Variation of RMSE (%)"
    )
    r_squared: Optional[float] = Field(description="R² score (1.0 = perfect)")


class DataSummary(BaseModel):
    """Summary of input data."""

    total_samples: int
    training_samples: Optional[int] = None
    validation_samples: Optional[int] = None
    start_date: str
    end_date: str
    target_column: str
    covariate_columns: list[str] = Field(default_factory=list)
    frequency_detected: str
    missing_value_count: int = 0


class TrainingInfo(BaseModel):
    """Information about training process."""

    training_time_seconds: float
    lookback_hours: int
    horizon_hours: int
    darts_model_class: str


class TrainModelRequest(BaseModel):
    """Request schema for train_forecast_model."""

    csv_path: str = Field(description="Path to CSV file with datetime and load data")
    model_type: str = Field(
        default="LinearRegression",
        description="Model type to train",
    )
    lookback_hours: int = Field(
        default=24,
        ge=1,
        le=168,
        description="Hours of history for model input",
    )
    horizon_hours: int = Field(
        default=6,
        ge=1,
        le=48,
        description="Hours ahead to forecast",
    )
    frequency: str = Field(
        default="h",
        description="Data frequency (15min, 30min, h)",
    )
    validation_split: float = Field(
        default=0.2,
        ge=0.1,
        le=0.3,
        description="Fraction for validation",
    )
    building_name: Optional[str] = Field(
        default=None,
        description="Building identifier",
    )
    model_name: Optional[str] = Field(
        default=None,
        description="Custom model name",
    )
    column_mapping: Optional[ColumnMapping] = Field(
        default=None,
        description="Map CSV columns to roles",
    )


class TrainModelResponse(BaseModel):
    """Response schema for train_forecast_model."""

    success: bool
    error: Optional[str] = None
    model_id: Optional[str] = None
    model_path: Optional[str] = None
    model_type: Optional[str] = None
    training_metrics: Optional[ModelMetrics] = None
    validation_metrics: Optional[ModelMetrics] = None
    data_summary: Optional[DataSummary] = None
    training_info: Optional[TrainingInfo] = None


class EvaluateModelRequest(BaseModel):
    """Request schema for evaluate_forecast_model."""

    model_id: str = Field(description="ID of trained model")
    csv_path: str = Field(description="Path to test data CSV")
    column_mapping: Optional[ColumnMapping] = Field(
        default=None,
        description="Column mapping (uses training mapping if not provided)",
    )
    return_predictions: bool = Field(
        default=True,
        description="Include predictions in output",
    )
    output_csv_path: Optional[str] = Field(
        default=None,
        description="Optional path to save predictions CSV",
    )
    include_residual_analysis: bool = Field(
        default=False,
        description="Include residual statistics",
    )


class Prediction(BaseModel):
    """Single prediction record."""

    timestamp: str
    actual: float
    predicted: float
    residual: float


class ComparisonToValidation(BaseModel):
    """Comparison of test metrics to validation metrics."""

    cv_rmse_diff: Optional[float] = Field(
        description="test_cv_rmse - validation_cv_rmse"
    )
    performance_status: str = Field(
        description="similar, degraded, or improved"
    )


class ResidualAnalysis(BaseModel):
    """Residual statistics for model diagnostics."""

    mean_residual: Optional[float] = Field(
        description="Should be ~0 for unbiased model"
    )
    std_residual: Optional[float] = None
    autocorrelation_lag1: Optional[float] = Field(
        description="Should be ~0 for good model"
    )


class TestSummary(BaseModel):
    """Summary of test data."""

    test_samples: int
    start_date: str
    end_date: str
    column_mapping_source: str


class EvaluateModelResponse(BaseModel):
    """Response schema for evaluate_forecast_model."""

    success: bool
    error: Optional[str] = None
    model_id: Optional[str] = None
    model_type: Optional[str] = None
    test_metrics: Optional[ModelMetrics] = None
    comparison_to_validation: Optional[ComparisonToValidation] = None
    test_summary: Optional[TestSummary] = None
    predictions: Optional[list[Prediction]] = None
    residual_analysis: Optional[ResidualAnalysis] = None
    output_csv_path: Optional[str] = None


class ListModelsRequest(BaseModel):
    """Request schema for list_models."""

    building_name: Optional[str] = Field(
        default=None,
        description="Filter by building",
    )
    model_type: Optional[str] = Field(
        default=None,
        description="Filter by model type",
    )
    sort_by: str = Field(
        default="created_at",
        description="Sort field",
    )
    limit: int = Field(
        default=20,
        description="Maximum results",
    )


class ModelSummary(BaseModel):
    """Summary of a trained model."""

    model_id: str
    model_type: str
    building_name: Optional[str] = None
    created_at: str
    validation_cv_rmse: Optional[float] = None
    lookback_hours: Optional[int] = None
    horizon_hours: Optional[int] = None


class ListModelsResponse(BaseModel):
    """Response schema for list_models."""

    success: bool
    error: Optional[str] = None
    models: list[ModelSummary] = Field(default_factory=list)
    total_count: int = 0
    filters_applied: Optional[dict] = None
