import os

import boto3
from botocore.exceptions import ClientError
from dotenv import load_dotenv


load_dotenv()


class S3Client:

    def __init__(self):
        self.bucket_name = os.getenv("AWS_S3_BUCKET")
        self.region = os.getenv("AWS_REGION")

        if not self.bucket_name:
            raise ValueError("AWS_S3_BUCKET is missing from .env")

        if not self.region:
            raise ValueError("AWS_REGION is missing from .env")

        self.client = boto3.client(
            "s3",
            region_name=self.region,
        )

    def upload_file(
        self,
        local_path: str,
        s3_key: str,
        metadata: dict | None = None,
        content_type: str | None = None,
    ) -> None:

        extra_args = {}

        if metadata:
            extra_args["Metadata"] = metadata

        if content_type:
            extra_args["ContentType"] = content_type

        if extra_args:
            self.client.upload_file(
                Filename=local_path,
                Bucket=self.bucket_name,
                Key=s3_key,
                ExtraArgs=extra_args,
            )

        else:
            self.client.upload_file(
                Filename=local_path,
                Bucket=self.bucket_name,
                Key=s3_key,
            )

    def file_exists(self, s3_key: str) -> bool:

        try:
            self.client.head_object(
                Bucket=self.bucket_name,
                Key=s3_key,
            )

            return True

        except ClientError as exc:

            error_code = exc.response["Error"]["Code"]

            if error_code in (
                "404",
                "NoSuchKey",
                "NotFound",
            ):
                return False

            raise

    def download_file(
        self,
        s3_key: str,
        local_path: str,
    ) -> None:

        self.client.download_file(
        Bucket=self.bucket_name,
        Key=s3_key,
        Filename=local_path,
        )
    
    def list_objects(
    self,
    prefix: str,
) -> list[dict]:

        paginator = self.client.get_paginator(
            "list_objects_v2"
        )

        objects = []

        for page in paginator.paginate(
            Bucket=self.bucket_name,
            Prefix=prefix,
        ):

            objects.extend(
                page.get("Contents", [])
            )

        return objects
    
    def delete_file(
        self,
        s3_key: str,
    ) -> None:

        self.client.delete_object(

            Bucket=self.bucket_name,

            Key=s3_key,
        )