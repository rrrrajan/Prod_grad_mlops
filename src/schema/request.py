from pydantic import BaseModel, Field


class CustomerRequest(BaseModel):
    """
    Request schema for customer churn prediction.
    """

    gender: str = Field(..., example="Female")
    SeniorCitizen: int = Field(..., ge=0, le=1, example=0)
    Partner: str = Field(..., example="Yes")
    Dependents: str = Field(..., example="No")
    tenure: int = Field(..., ge=0, example=24)
    PhoneService: str = Field(..., example="Yes")
    MultipleLines: str = Field(..., example="No")
    InternetService: str = Field(..., example="Fiber optic")
    OnlineSecurity: str = Field(..., example="No")
    OnlineBackup: str = Field(..., example="Yes")
    DeviceProtection: str = Field(..., example="No")
    TechSupport: str = Field(..., example="No")
    StreamingTV: str = Field(..., example="Yes")
    StreamingMovies: str = Field(..., example="Yes")
    Contract: str = Field(..., example="Month-to-month")
    PaperlessBilling: str = Field(..., example="Yes")
    PaymentMethod: str = Field(..., example="Electronic check")
    MonthlyCharges: float = Field(..., ge=0, example=89.15)
    TotalCharges: float = Field(..., ge=0, example=2135.75)