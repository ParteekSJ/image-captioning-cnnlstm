# Define a constant path prefix
data_dir="./data_dir/"

# Linux
apt-get install wget

# Mac
brew install wget

# Create a data directory
mkdir "$data_dir"

# Download images and annotations to the data directory
wget http://images.cocodataset.org/annotations/annotations_trainval2014.zip -P "$data_dir"
wget http://images.cocodataset.org/zips/train2014.zip -P "$data_dir"
wget http://images.cocodataset.org/zips/val2014.zip -P "$data_dir"

# Extract zipped images and annotations and remove the zip files
unzip "$data_dir/annotations_trainval2014.zip" -d "$data_dir"
rm "$data_dir/annotations_trainval2014.zip"
unzip "$data_dir/train2014.zip" -d "$data_dir"
rm "$data_dir/train2014.zip"
unzip "$data_dir/val2014.zip" -d "$data_dir"
rm "$data_dir/val2014.zip"