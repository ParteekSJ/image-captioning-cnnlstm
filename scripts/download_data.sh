# linux
apt-get install wget
# mac
brew install wget
 
# create a data directory
mkdir data_dir
 
# download images and annotations to the data directory
wget http://msvocds.blob.core.windows.net/annotations-1-0-3/captions_train-val2014.zip -P ./data_dir/
wget http://images.cocodataset.org/zips/train2014.zip -P ./data_dir/
wget http://images.cocodataset.org/zips/val2014.zip -P ./data_dir/
    
# extract zipped images and annotations and remove the zip files
unzip ./data_dir/captions_train-val2014.zip -d ./data_dir/
rm ./data_dir/captions_train-val2014.zip
unzip ./data_dir/train2014.zip -d ./data_dir/
rm ./data_dir/train2014.zip 
unzip ./data_dir/val2014.zip -d ./data_dir/ 
rm ./data_dir/val2014.zip