import os 
import boto3

bucket_name = "mlops-mhofmann"
s3 = boto3.client('s3')

def download_dir_from_s3(local_path, s3_prefix):
    """Download a directory from an S3 bucket"""
    if not os.path.exists(local_path):
        os.makedirs(local_path, exist_ok=True)

    paginator = s3.get_paginator("list_objects_v2")
    for result in paginator.paginate(Bucket=bucket_name, Prefix=s3_prefix):
        if "Contents" in result:
            for obj in result["Contents"]:
                s3_key = obj["Key"]
                print(s3_key)
                local_file_path = os.path.join(local_path, s3_key.replace(s3_prefix, ""))
                os.makedirs(os.path.dirname(local_file_path), exist_ok=True)

                s3.download_file(bucket_name, s3_key, local_file_path)
                print(f"Downloaded {obj['Key']} to {local_file_path}.")

def upload_image_to_s3(file_name, s3_prefix="ml-images", object_name=None):
    """Upload an image to an S3 bucket and return a presigned URL."""
    if object_name is None:
        object_name = os.path.basename(file_name)

    object_name = os.path.join(s3_prefix, object_name)
    try:
        s3.upload_file(file_name, bucket_name, object_name)
        response = s3.generate_presigned_url(
            'get_object',
            Params={'Bucket': bucket_name, 'Key': object_name},
            ExpiresIn=3600  # URL expires in 1 hour
        )
        print(f"Uploaded {file_name} to s3://{bucket_name}/{object_name}")
        print(f"Generated presigned URL: {response}")
        return response
    except Exception as e:
        print(f"Failed to upload {file_name} to s3://{bucket_name}/{object_name}: {e}")