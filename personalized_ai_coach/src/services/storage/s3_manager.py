from __future__ import annotations

import asyncio
import hashlib
import os
from contextlib import asynccontextmanager
from typing import AsyncGenerator, BinaryIO, List, Optional

import boto3
import structlog
from botocore.exceptions import ClientError
from pydantic import BaseModel, Field, ValidationError

log = structlog.get_logger(__name__)

class S3Config(BaseModel):
    bucket_name: str = Field(..., min_length=1)
    region: str = Field("us-east-1")
    access_key_id: str | None = None
    secret_access_key: str | None = None
    endpoint_url: str | None = None

class S3Manager:
    ALLOWED_EXTENSIONS = {".pdf", ".md", ".markdown", ".txt", ".rst", ".docx"}
    MAX_FILE_SIZE_MB = 100

    def __init__(self):
        self.config = self._load_config()
        self._client = None

    def _load_config(self) -> S3Config:
        return S3Config(
            bucket_name=os.getenv("S3_BUCKET_NAME", ""),
            region=os.getenv("AWS_REGION", "us-east-1"),
            access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
            secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
            endpoint_url=os.getenv("S3_ENDPOINT_URL"),
        )

    @property
    def client(self):
        if self._client is None:
            self._client = boto3.client(
                "s3",
                region_name=self.config.region,
                aws_access_key_id=self.config.access_key_id,
                aws_secret_access_key=self.config.secret_access_key,
                endpoint_url=self.config.endpoint_url,
            )
        return self._client

    def _validate_file(self, filename: str, content_length: int) -> None:
        ext = os.path.splitext(filename)[1].lower()
        if ext not in self.ALLOWED_EXTENSIONS:
            raise ValueError(f"Unsupported file type: {ext}. Allowed: {self.ALLOWED_EXTENSIONS}")
        if content_length > self.MAX_FILE_SIZE_MB * 1024 * 1024:
            raise ValueError(f"File too large: {content_length/1024/1024:.1f}MB > {self.MAX_FILE_SIZE_MB}MB")

    async def upload_file(self, file_path: str, user_id: str, project_id: str = "default") -> str:
        """Upload a local file to S3, return object key."""
        import aiofiles
        filename = os.path.basename(file_path)
        stat = await asyncio.to_thread(os.stat, file_path)
        self._validate_file(filename, stat.st_size)

        key = f"users/{user_id}/{project_id}/{int(stat.st_mtime)}_{hashlib.sha256(filename.encode()).hexdigest()[:8]}_{filename}"
        async with aiofiles.open(file_path, "rb") as f:
            data = await f.read()
        try:
            await asyncio.to_thread(
                self.client.put_object,
                Bucket=self.config.bucket_name,
                Key=key,
                Body=data,
                ContentType="application/octet-stream",
            )
            log.info("s3_upload_success", user=user_id, key=key)
            return key
        except ClientError as e:
            log.error("s3_upload_failed", error=str(e))
            raise

    async def generate_presigned_url(self, object_key: str, expiration_seconds: int = 3600) -> str:
        """Generate a temporary download URL."""
        try:
            url = await asyncio.to_thread(
                self.client.generate_presigned_url,
                "get_object",
                Params={"Bucket": self.config.bucket_name, "Key": object_key},
                ExpiresIn=expiration_seconds,
            )
            return url
        except ClientError as e:
            log.error("presigned_url_failed", key=object_key, error=str(e))
            raise

    async def delete_file(self, object_key: str) -> bool:
        """Delete an object from S3."""
        try:
            await asyncio.to_thread(
                self.client.delete_object, Bucket=self.config.bucket_name, Key=object_key
            )
            log.info("s3_delete_success", key=object_key)
            return True
        except ClientError:
            return False

    async def health_check(self) -> bool:
        try:
            await asyncio.to_thread(self.client.head_bucket, Bucket=self.config.bucket_name)
            return True
        except Exception:
            return False