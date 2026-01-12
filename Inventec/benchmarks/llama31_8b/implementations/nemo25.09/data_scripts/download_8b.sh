mkdir -p "$DATADIR"
pushd "$DATADIR"
bash <(curl -s https://raw.githubusercontent.com/mlcommons/r2-downloader/refs/heads/main/mlc-r2-downloader.sh) -d 8b https://training.mlcommons-storage.org/metadata/llama-3-1-8b-preprocessed-c4-dataset.uri
bash <(curl -s https://raw.githubusercontent.com/mlcommons/r2-downloader/refs/heads/main/mlc-r2-downloader.sh) -d 8b/tokenizer https://training.mlcommons-storage.org/metadata/llama-3-1-8b-tokenizer.uri
popd
#bash data_scripts/cleanup_8b.sh
