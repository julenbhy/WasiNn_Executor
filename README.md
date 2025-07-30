<div align="center">
  <h1>WebAssembly-flavored OpenWhisk</h1>

<strong>A WebAssembly-based container runtime for the Apache OpenWhisk serverless platform.
</strong>
</div>

This project implements a WebAssembly-based executor based on the Apache OpenWhisk API, allowing you to run machine learning models in a serverless environment. It supports running AI models split into multiple WebAssembly modules, executed in a concurrent pipeline. Each step is a separate instance.

The executor automatically splits the input batch and distributes it across parallel instances of the pipeline, increasing throughput.

wasi-nn does not natively support direct access to tensors from the host. Data must be serialized and deserialized multiple times between pipeline steps, introducing significant overhead. This version introduces direct tensor access from the host, removing unnecessary serialization and enabling efficient memory sharing across pipeline stages.

#### Features:
- Shared tensors with zero-copy access.
- Automatic parallelization of inference tasks.
- Configurable batch size for dynamic workload splitting.


## Quickstart: Wasmtime Runtime

As a small tutorial, let's build the wasmtime invoker and run one of the examples.

1. Install wsk-cli from https://github.com/apache/openwhisk-cli/releases/tag/1.2.0


2. Clone the openwhisk repo, checkout the appropriate branch and run OpenWhisk in a separate terminal:

```sh
git clone https://github.com/julenbhy/openwhisk
cd openwhisk
git checkout burst-openwasm
./gradlew core:standalone:bootRun
```

This will print something like the following:

```
[ WARN  ] Configure wsk via below command to connect to this server as [guest]

wsk property set --apihost 'http://172.17.0.1:3233' --auth '23bc46b1-71f6-4ed5-8c54-816aa4f8c502:123zO3xZCLrMN6v2BKK1dXYFpXlPkccOFqm12CdAsMgRU4VrNZ9lyGVCGuMDGIwP'
```

Execute this command.

3. In a new terminal, run the desired wasmtime invoker with the following command from the root of this repository:

```sh
cargo run --release
```

**_NOTE:_**  Take care you will need to install torch 2.4.0 to use the wasi-nn invoker:

For using torch:
```sh
sudo apt install libarchive-tools
mkdir -p "$HOME/libtorch-2.4.0" && wget -qO- "https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-2.4.0%2Bcpu.zip" | bsdtar -xvf- -C "$HOME/libtorch-2.4.0"
export LIBTORCH=$HOME/libtorch-2.4.0/libtorch
export LD_LIBRARY_PATH=$LIBTORCH/lib:$LD_LIBRARY_PATH
```

4. Next, build the `pytorch_image_classification` example with:

```sh
./actions/compile.sh actions/pytorch_image_classification.rs
```

This will add all the required dependencies for the selected execution model and compile it using the action builder crate. The script will also add the function to OpenWhisk.

**_NOTE:_**  The precompilation step performed by the script requires wasmtime-cli 34.0.1 to be installed.

**_NOTE:_**  The `compile.sh` script detects and adds the required crates to `action-builder/Cargo.toml`. If a specific version or feature is required for a certain crate, manually add it to `action-builder/Cargo_template.toml` before running the script.

5. Run the test_client to call an action:

```sh
python examples/image-classification/test.py
```

6. For benchmarking a function, use the following benchmarking tool:
[`openwhisk-bench`](https://github.com/julenbhy/openwhisk-bench/tree/main)
