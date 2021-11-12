import os
import boto3
from dotenv import load_dotenv

load_dotenv()


def get_s3_bucket():
    s3 = boto3.resource(
        "s3",
        aws_access_key_id=os.getenv("ACCOUNT_ID"),
        aws_secret_access_key=os.getenv("ACCESS_KEY"),
    )
    source_bucket = s3.Bucket(os.getenv("BUCKET_NAME"))
    return source_bucket


def upload_file(bucket, file_path, key_file):
    bucket.upload_file(file_path, key_file)
    os.remove(file_path)


def download_file(bucket, file_name, file_path):
    bucket.download_file(file_name, file_path)
