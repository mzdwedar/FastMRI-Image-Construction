"""Gateway settings: `GATEWAY_*` environment variables, validated at startup."""

from typing import Literal

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="GATEWAY_", extra="ignore")

    env: Literal["dev", "prod"] = "prod"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    workers: int = Field(default=2, ge=1)

    triton_url: str = "triton:8001"
    triton_model_name: str = "unet_model"
    triton_timeout_s: float = Field(default=10.0, gt=0)

    max_upload_mb: int = Field(default=256, ge=1)
    max_slices: int = Field(default=64, ge=1)
    max_inflight: int = Field(default=32, ge=1)

    api_keys_file: str = "/run/secrets/api_keys"
    auth_disabled: bool = False
    otlp_endpoint: str | None = None

    @model_validator(mode="after")
    def _auth_bypass_only_in_dev(self) -> "Settings":
        if self.auth_disabled and self.env != "dev":
            raise ValueError("GATEWAY_AUTH_DISABLED=true is only allowed when GATEWAY_ENV=dev")
        return self
