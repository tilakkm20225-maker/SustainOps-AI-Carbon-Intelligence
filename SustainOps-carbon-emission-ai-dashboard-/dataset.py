import kagglehub

# Download latest version
path = kagglehub.dataset_download("stefancomanita/hourly-electricity-consumption-and-production")

print("Path to dataset files:", path)