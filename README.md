# liballoc

C binding of `talc` crate for x86 or x86_64 OS to create a scalable, high performance heap.

## Usage

Download the header file and lib from [releases](https://github.com/plos-clan/liballoc/releases/tag/release).

Link the library to your project.

## Build

Build directly to get the two target files:

```bash
cargo build --release
```

The production build will be in `target/release/<target>/` directory.

And use `cbindgen` to generate the header file:

```bash
cargo install cbindgen
cbindgen --output alloc.h
```

## Example

```c
#include <stdint.h>
#include "alloc.h"

void test_alloc() {
    uint8_t alloc_test[500];
    bool result = heap_init((uint8_t *)alloc_test, 500);
    if (result) {
        void *ptr1 = malloc(10);
        free(ptr1);
    }
}
```

For general use, you can map the address space of the memory you want to use as a heap in the page table, and then pass the start address and size to the `heap_init`.

```c
#include <stdint.h>
#include "alloc.h"

const uint8_t *heap_start = (uint8_t *)0x100000;
const size_t heap_size = 8 * 1024 * 1024;

void test_alloc() {
    map_page(heap_start, heap_size);
    if (heap_init(heap_start, heap_size)) {
        void *ptr1 = malloc(10);
        free(ptr1);
    }
}
```
