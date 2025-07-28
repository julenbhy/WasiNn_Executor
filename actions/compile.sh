#!/bin/bash

set -e

WASMTIME=${WASMTIME_PATH:-"/opt/wasmtime-v34.0.1-x86_64-linux/wasmtime"}
export WASMTIME


# Check if the necessary arguments are passed
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <filename>"
    exit 1
fi

# Input variables
INPUT_FILE="$1"        # Original filename
METHOD="$2"            # Method selected by the user


# Check if the version matches the required version (34.0.1)
if [ "$($WASMTIME --version)" != "wasmtime 34.0.1 (ebdadc4e0 2025-06-24)" ]; then
    echo "The version of wasmtime is not 34.0.1. Please install the correct version."
    exit 1
fi

FILENAME=$(basename "$INPUT_FILE" .rs) # Filename without the path and extension
BUILDER="action-builder"

# Check if the file containing the bytes exists
if [ ! -f "$INPUT_FILE" ]; then
    echo "The file '$INPUT_FILE' does not exist."
    exit 1
fi

# Prepare the builder
cp "$BUILDER/Cargo_template.toml" "$BUILDER/Cargo.toml"

# Add the necessary dependencies to the builder
crate_names=$(grep -Eo 'use [a-zA-Z0-9_]+(::)?' "$INPUT_FILE" | awk '{print $2}' | sed 's/::$//' | sort | uniq)
pwd
echo "Detected dependencies: $crate_names"
for crate in $crate_names; do
  if ! grep -q "^$crate =" $BUILDER/Cargo.toml; then
    echo "Adding dependency $crate to Cargo.toml"
    if ! cargo add --manifest-path "$BUILDER/Cargo.toml" "$crate"; then
      echo "Failed to add crate '$crate'. It may not be compatible or required."
    fi
  else
    echo "Dependency $crate already added to Cargo.toml" 
  fi
done

# Copy the file to the builder
mkdir -p "$BUILDER/examples/"
cp "$INPUT_FILE" "$BUILDER/examples/"

# Add the METHOD feature to the builder
# if the method is the tensor method, to expose the preprocess and postprocess functions
sed -i "1i action_builder::memory_nn_tensors_method!(preprocess, postprocess);" "$BUILDER/examples/$FILENAME.rs"

# Compile the file with the selected METHOD feature
echo "Compiling with $METHOD method."
cargo build --manifest-path ./"$BUILDER"/Cargo.toml --release --example "$FILENAME" --target wasm32-wasip1

# Check if the compilation was successful
if [ $? -ne 0 ]; then
    echo "Compilation failed."
    exit 1
fi

mkdir -p "actions/compiled"

# Compile the WASM to a .cwasm file
$WASMTIME compile "target/wasm32-wasip1/release/examples/$FILENAME.wasm" -o "./actions/compiled/$FILENAME.cwasm"

# Package the .cwasm file into a zip
zip "./actions/compiled/$FILENAME.zip" "./actions/compiled/$FILENAME.cwasm"
