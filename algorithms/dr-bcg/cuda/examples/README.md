# Examples

## Building

To build the examples, provide the `DR_BCG_BUILD_EXAMPLES` build flag when building using CMake. This can be done from the project root directory like so:

```bash
cmake -B build -S . -DDR_BCG_BUILD_EXAMPLES=ON
```

After building, this will add an additional `examples/` subdirectory under `build/` containing executables for each example.

## Running

See the `README.md` files in each example's directory for directions on how to run them.