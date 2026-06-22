from __future__ import annotations

import asyncio
import hashlib
import os
import threading
from typing import Any, Dict, List, Optional

import boto3
import structlog
from botocore.config import Config
from botocore.exceptions import ClientError
from pydantic import BaseModel, Field, ValidationError

# Configure structured logging
log = structlog.get_logger(__name__)

class S3Config(BaseModel):
    """Configuration for S3 storage with validation."""
    bucket_name: str = Field(..., min_length=1)
    region: str = Field("us-east-1")
    access_key_id: Optional[str] = None
    secret_access_key: Optional[str] = None
    endpoint_url: Optional[str] = None
    max_pool_connections: int = Field(50, ge=1)
    retry_max_attempts: int = Field(3, ge=0)

class S3Manager:
    """
    Production-ready asynchronous S3 file manager.
    Handles file validation, streaming uploads, and presigned URL generation.
    """
    
    ALLOWED_EXTENSIONS: set[str] = {".pdf", ".md", ".markdown", ".txt", ".rst", ".docx"}
    MAX_FILE_SIZE_MB: int = 100
    
    def __init__(self) -> None:
        self.config = self._load_config()
        self._client: Any = None
        self._lock = threading.Lock()

    def _load_config(self) -> S3Config:
        """Loads and validates S3 configuration from environment variables."""
        try:
            return S3Config(
                bucket_name=os.getenv("S3_BUCKET_NAME", ""),
                region=os.getenv("AWS_REGION", "us-east-1"),
                access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
                secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
                endpoint_url=os.getenv("S3_ENDPOINT_URL"),
                max_pool_connections=int(os.getenv("S3_MAX_POOL_CONNECTIONS", "50")),
                retry_max_attempts=int(os.getenv("S3_RETRY_MAX_ATTEMPTS", "3")),
            )
        except (ValidationError, ValueError) as e:
            log.critical("s3_config_invalid", error=str(e))
            # Fallback to a minimal shell if needed, but usually better to fail fast
            raise RuntimeError(f"Failed to initialize S3 configuration: {e}") from e

    @property
    def client(self) -> Any:
        """Lazy-initialized, thread-safe S3 client."""
        if self._client is None:
            with self._lock:
                if self._client is None:
                    # Optimized botocore config for high concurrency
                    boto_config = Config(
                        region_name=self.config.region,
                        signature_version="s3v4",
                        retries={
                            "max_attempts": self.config.retry_max_attempts,
                            "mode": "standard"
                        },
                        max_pool_connections=self.config.max_pool_connections
                    )
                    
                    self._client = boto3.client(
                        "s3",
                        aws_access_key_id=self.config.access_key_id,
                        aws_secret_access_key=self.config.secret_access_key,
                        endpoint_url=self.config.endpoint_url,
                        config=boto_config
                    )
        return self._client

    def _validate_file(self, filename: str, content_length: int) -> None:
        """Validates file extension and size before upload."""
        ext = os.path.splitext(filename)[1].lower()
        if ext not in self.ALLOWED_EXTENSIONS:
            log.warn("s3_invalid_file_type", filename=filename, extension=ext)
            raise ValueError(
                f"Unsupported file type '{ext}'. Allowed extensions: {', '.join(sorted(self.ALLOWED_EXTENSIONS))}"
            )
        
        max_bytes = self.MAX_FILE_SIZE_MB * 1024 * 1024
        if content_length > max_bytes:
            log.warn("s3_file_too_large", filename=filename, size=content_length)
            raise ValueError(
                f"File size ({content_length / (1024 * 1024):.1f}MB) exceeds "
                f"the maximum allowed limit of {self.MAX_FILE_SIZE_MB}MB."
            )

    async def upload_file(
        self, 
        file_path: str, 
        user_id: str, 
        project_id: str = "default"
    ) -> str:
        """
        Uploads a local file to S3 using streaming to prevent memory exhaustion.
        Returns the unique object key.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found at: {file_path}")

        filename = os.path.basename(file_path)
        stat = await asyncio.to_thread(os.stat, file_path)
        self._validate_file(filename, stat.st_size)

        # Deterministic but unique key generation
        # Structure: users/<user_id>/<project_id>/<timestamp>_<hash>_<filename>
        name_hash = hashlib.sha256(filename.encode()).hexdigest()[:8]
        timestamp = int(stat.st_mtime)
        object_key = f"users/{user_id}/{project_id}/{timestamp}_{name_hash}_{filename}"

        try:
            # Use boto3's upload_file which handles multipart and streaming naturally
            await asyncio.to_thread(
                self.client.upload_file,
                Filename=file_path,
                Bucket=self.config.bucket_name,
                Key=object_key,
                ExtraArgs={"ContentType": "application/octet-stream"}
            )
            log.info("s3_upload_success", user_id=user_id, project_id=project_id, key=object_key)
            return object_key
        except ClientError as e:
            log.error("s3_upload_failed", user_id=user_id, key=object_key, error=str(e))
            raise
        except Exception as e:
            log.error("s3_unhandled_upload_error", user_id=user_id, error=str(e))
            raise

    async def generate_presigned_url(
        self, 
        object_key: str, 
        expiration_seconds: int = 3600
    ) -> str:
        """Generates a temporary pre-signed URL for downloading a file."""
        try:
            url = await asyncio.to_thread(
                self.client.generate_presigned_url,
                "get_object",
                Params={
                    "Bucket": self.config.bucket_name,
                    "Key": object_key
                },
                ExpiresIn=expiration_seconds,
            )
            return url
        except ClientError as e:
            log.error("s3_presigned_url_failed", key=object_key, error=str(e))
            raise

    async def delete_file(self, object_key: str) -> bool:
        """
        Deletes an object from S3.
        Returns True on success, False otherwise.
        """
        try:
            await asyncio.to_thread(
                self.client.delete_object,
                Bucket=self.config.bucket_name,
                Key=object_key
            )
            log.info("s3_delete_success", key=object_key)
            return True
        except ClientError as e:
            log.error("s3_delete_failed", key=object_key, error=str(e))
            return False

    async def health_check(self) -> bool:
        """Verifies S3 connectivity and bucket access."""
        try:
            await asyncio.to_thread(
                self.client.head_bucket,
                Bucket=self.config.bucket_name
            )
            return True
        except Exception as e:
            log.error("s3_health_check_failed", error=str(e))
            return False