mkdir -p datasets
cd datasets
wget -c --tries=100 --timeout=30 https://cvg-data.inf.ethz.ch/nice-slam/data/Replica.zip
unzip Replica.zip
mv Replica replica