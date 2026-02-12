#![no_std]
#![no_main]

use core::alloc::Layout;
use core::cmp;
use core::ffi::c_void;
use core::mem::{align_of, size_of};
use core::panic::PanicInfo;
use core::ptr::{self, NonNull};
use core::sync::atomic::{AtomicPtr, AtomicUsize, Ordering};
use talc::{OomHandler, Span, Talc, Talck};

// Yes if panic just loops forever and you know died.
#[panic_handler]
unsafe fn panic(_info: &PanicInfo) -> ! {
    loop {}
}

// Represents a raw memory span returned by the external OOM callback.
#[repr(C)]
pub struct MemorySpan {
    /// Pointer to the start of the new memory block.
    /// Must be non-null and properly aligned if the system requires it.
    ptr: *mut u8,
    /// The size of the memory block in bytes.
    /// If `size` is 0, the allocator considers the OOM handling failed.
    size: usize,
}

// Function pointer type for the OOM callback.
type OomCallback = unsafe extern "C" fn(usize) -> MemorySpan;

// Define a type for our error handling function pointer.
// It will receive an enum indicating the error type and the invalid pointer.
#[repr(C)]
pub enum HeapError {
    InvalidFree, // Covers double free, invalid pointer, corrupted metadata
    LayoutError, // Internal layout calculation failed
}

/// Type alias for the error handler function pointer.
pub type ErrorHandler = unsafe extern "C" fn(error: HeapError, ptr: *mut c_void);

// Holds the user-provided OOM callback.
static OOM_CALLBACK: AtomicPtr<c_void> = AtomicPtr::new(ptr::null_mut());

// A static variable to hold the user-provided error handler.
// Use AtomicPtr for safe, lock-free access (even if we only set it once).
static ERROR_HANDLER: AtomicPtr<c_void> = AtomicPtr::new(ptr::null_mut());

// A constant checksum value to verify metadata integrity.
static CHECKSUM_SALT: AtomicUsize = AtomicUsize::new(0x5A17_C0DE);

/// Metadata struct stored *before* the user pointer's aligned position.
/// We store the requested size and the alignment used for the user pointer.
#[repr(C)] // Ensure predictable layout for size/alignment calculations
#[derive(Debug, Clone, Copy)]
struct Metadata {
    /// A checksum derived from size, alignment, and a salt.
    checksum: usize,
    /// The size requested by the user for the allocation.
    size: usize,
    /// The alignment requested by the user (or default for malloc).
    alignment: usize,
}

// Calculates a simple checksum for metadata integrity verification.
///
/// This function mixes the allocation size, alignment, and a global salt to produce
/// a verification value.
///
/// # Algorithm
/// It uses a lightweight mixing step (similar to MurmurHash3's finalizer) to ensure
/// that small bit changes in size/alignment result in large changes in the checksum.
///
/// # Stale Pointer Protection
/// The inclusion of `CHECKSUM_SALT` (which rotates on `heap_init`) ensures that
/// pointers from a previous heap generation cannot be successfully freed or reallocated
/// after the allocator has been reset, preventing use-after-reset UB.
#[inline(always)]
fn calculate_checksum(size: usize, alignment: usize) -> usize {
    let mut x = size;
    // Mix in the alignment and the global salt.
    // The salt changes every time the heap is re-initialized.
    x ^= alignment;
    x ^= CHECKSUM_SALT.load(Ordering::Relaxed);

    // Apply a fast avalanche function to spread the bits.
    // This helps detect partial corruption or linear overflows.
    x = (x ^ (x >> 15)).wrapping_mul(0xd168aaad);
    x ^= x >> 15;
    x
}

// Use a spinlock mutex for thread safety in concurrent environments (if applicable)
static ALLOCATOR: Talck<spin::Mutex<()>, OomHandlerImpl> = Talc::new(OomHandlerImpl).lock();

// Size and alignment of the metadata header itself.
const METADATA_SIZE: usize = size_of::<Metadata>();
const METADATA_ALIGN: usize = align_of::<Metadata>();

struct OomHandlerImpl;

impl OomHandler for OomHandlerImpl {
    fn handle_oom(talc: &mut Talc<Self>, layout: Layout) -> Result<(), ()> {
        let callback_ptr = OOM_CALLBACK.load(Ordering::SeqCst);

        if callback_ptr.is_null() {
            return Err(());
        }

        let callback: OomCallback = unsafe { core::mem::transmute(callback_ptr) };
        let ffi_span = unsafe { callback(layout.size()) };
        let new_span = Span::from_base_size(ffi_span.ptr, ffi_span.size);

        if new_span.is_empty() {
            return Err(());
        }

        unsafe { talc.claim(new_span).map(|_| ()) }
    }
}

/// Calculates the layout required for the underlying allocation from `talc`.
/// This layout must be large enough to hold the Metadata struct plus the requested
/// user data size, potentially with padding in between to ensure the user data
/// pointer meets its required alignment.
///
/// Returns:
/// - `Ok((total_layout, user_data_offset))` where:
///   - `total_layout`: The layout object to request from `talc`.
///   - `user_data_offset`: The offset from the beginning of the `talc` allocation
///     to where the correctly aligned user pointer should be.
/// - `Err(())` if layout calculation overflows or results in an invalid layout.
#[inline]
fn layout_for_allocation(
    size: usize,
    alignment: usize, // Alignment requested for the user pointer
) -> Result<(Layout, usize), ()> {
    // The alignment for the total block must satisfy both metadata and user alignment.
    let total_align = cmp::max(alignment, METADATA_ALIGN);

    // Calculate the offset for the user data pointer.
    // The user pointer must start *after* the metadata AND satisfy `alignment`.
    let user_data_offset = {
        // Align this offset up to the required user alignment.
        // Formula: (value + align - 1) & !(align - 1) assuming align is power of 2.
        METADATA_SIZE.checked_add(alignment - 1).ok_or(())? & !(alignment - 1)
    };

    // Calculate the total size needed for the allocation block.
    let total_size = user_data_offset.checked_add(size).ok_or(())?;

    // Create the final layout for the allocator.
    let total_layout = Layout::from_size_align(total_size, total_align).map_err(|_| ())?;

    Ok((total_layout, user_data_offset))
}

/// Reports an error via the user-provided error handler, if set.
/// If no handler is set, this function does nothing.
///
/// # Safety
/// The caller must ensure that the error handler, if set, is a valid function
/// pointer and can be safely called with the provided arguments.
/// This function is unsafe because it involves calling a raw function pointer.
#[inline]
unsafe fn report_error(error: HeapError, ptr: *mut c_void) {
    let handler_ptr = ERROR_HANDLER.load(Ordering::SeqCst);
    if !handler_ptr.is_null() {
        let handler: ErrorHandler = core::mem::transmute(handler_ptr);
        handler(error, ptr);
    }
}

/// Gets the pointer to the Metadata struct from the user pointer.
/// Assumes the user pointer is valid and was allocated by this allocator.
/// The Metadata struct resides `METADATA_SIZE` bytes immediately *before*
/// the address pointed to by `user_ptr`.
///
/// # Safety
/// `user_ptr` must point to the start of the user data area of a valid allocation
/// managed by this allocator. The memory layout must be as expected (Metadata immediately preceding).
#[inline]
unsafe fn get_metadata_ptr(user_ptr: NonNull<u8>) -> Result<NonNull<Metadata>, ()> {
    // Calculate the address directly before the user pointer.
    let metadata_ptr = user_ptr.as_ptr().sub(METADATA_SIZE);
    let metadata = &*(metadata_ptr.cast::<Metadata>());

    // Verify the checksum to ensure metadata integrity.
    let expected = calculate_checksum(metadata.size, metadata.alignment);

    // Check the checksum to verify integrity.
    if metadata.checksum != expected {
        Err(())
    } else {
        Ok(NonNull::new_unchecked(metadata_ptr.cast::<Metadata>()))
    }
}

/// Recovers the original allocation pointer (as returned by `talc`) and the
/// `Layout` object used for that original allocation, based on the user pointer.
///
/// # Safety
/// `user_ptr` must be a non-null pointer returned by `malloc`, `aligned_alloc`, or `realloc`
/// from this allocator. The metadata preceding it must be intact.
#[inline]
unsafe fn recover_alloc_info(user_ptr: NonNull<u8>) -> Result<(NonNull<u8>, Layout, Metadata), ()> {
    // 1. Get pointer to metadata and read it to find original size and alignment.
    let metadata_ptr = get_metadata_ptr(user_ptr)?;
    // Safety: Pointer is assumed valid by caller.
    let metadata = metadata_ptr.as_ptr().read();

    // 2. Recalculate the layout and user_data_offset used for the original allocation.
    //    Use the size and alignment stored *in the metadata*.
    let (original_layout, user_data_offset) =
        layout_for_allocation(metadata.size, metadata.alignment)?;

    // 3. Calculate the original allocation start pointer address.
    //    user_ptr = alloc_ptr + user_data_offset
    //    alloc_ptr = user_ptr - user_data_offset
    let start_ptr_addr = user_ptr.as_ptr().wrapping_sub(user_data_offset);
    let start_ptr = NonNull::new_unchecked(start_ptr_addr);

    Ok((start_ptr, original_layout, metadata))
}

// == C API Implementation ==

/// Initializes (or RESETS) the heap memory arena.
///
/// # Warning
/// Calling this function a second time will **wipe** the allocator state.
/// Any pointers allocated before the reset will become "leaked" (safe to use,
/// but calling free() on them later is Undefined Behavior/Double Free because
/// the new allocator doesn't know about them).
///
/// # Safety
/// - `address` must be a valid pointer to the start of a contiguous, writable memory block.
/// - `size` must be the correct size of that memory block in bytes.
/// - The memory range `[address, address + size)` must be exclusively available to the allocator
///   (it must not be accessed or modified by other code while managed by the allocator).
/// - Thread safety is handled internally by the allocator lock.
#[no_mangle]
pub unsafe extern "C" fn heap_init(address: *mut u8, size: usize) -> bool {
    // Basic sanity checks
    if address.is_null() || size < METADATA_SIZE * 2 {
        return false;
    }

    // Update checksum salt to help catch stale metadata across resets.
    CHECKSUM_SALT.fetch_add(1, Ordering::SeqCst);

    // Lock the allocator
    let mut allocator_guard = ALLOCATOR.lock();

    // Overwrite the existing Talc instance with a fresh one.
    // We construct a new Talc exactly how the static one was initialized.
    // This effectively "forgets" all previous spans and allocations.
    *allocator_guard = Talc::new(OomHandlerImpl);

    // Give the memory span to the allocator.
    let arena = Span::from_base_size(address, size);
    allocator_guard.claim(arena).is_ok()
}

/// Extends the heap with a new memory block.
///
/// # Safety
/// - `address` must be a valid pointer to the start of a new, available memory block.
/// - `size` must be the correct size of that memory block.
/// - The memory block must not overlap with currently managed memory (unless extending the end).
/// - Thread safety is handled internally by the allocator lock.
#[no_mangle]
pub unsafe extern "C" fn heap_extend(address: *mut u8, size: usize) -> bool {
    // Basic sanity checks
    if address.is_null() || size == 0 {
        return false;
    }

    // Minimal size check similar to heap_init to avoid useless fragmentation
    // imply that the block is at least big enough for some headers.
    if size < METADATA_SIZE {
        return false;
    }

    // Create the span for the new memory
    let arena = Span::from_base_size(address, size);

    // Lock the allocator and claim the new memory.
    // Talc handles multiple spans automatically. If 'address' is immediately
    // following an existing heap chunk, Talc will merge them.
    ALLOCATOR.lock().claim(arena).is_ok()
}

/// Sets a custom error handler function to be called on heap errors.
/// Passing `None` clears any previously set handler.
///
/// # Safety
/// - The `handler` function pointer must be valid and callable.
#[no_mangle]
pub unsafe extern "C" fn heap_onerror(handler: ErrorHandler) {
    ERROR_HANDLER.store(handler as *mut c_void, Ordering::SeqCst);
}

/// Registers a custom Out-Of-Memory (OOM) handler.
///
/// # Purpose
/// This handler acts as a hook that is triggered when the allocator **runs out of memory**.
/// It provides a mechanism to **automatically extend the heap** on demand.
///
/// Instead of immediately returning `NULL` when the heap is full, the allocator will:
/// 1. Call this function.
/// 2. If this function returns a valid new memory span, the allocator adds it to the heap.
/// 3. The allocator retries the original allocation request.
///
/// # Safety
/// - The `callback` function pointer must be valid and callable.
/// - The callback implementation must ensure the returned memory is valid and not already in use.
/// - The callback itself must be thread-safe if the allocator is accessed concurrently.
#[no_mangle]
pub unsafe extern "C" fn heap_set_oom_handler(callback: OomCallback) {
    OOM_CALLBACK.store(callback as *mut c_void, Ordering::SeqCst);
}

/// Returns the usable size of the memory block pointed to by `ptr`.
/// This corresponds to the size originally requested during allocation (`malloc`, `aligned_alloc`, `realloc`).
/// Returns 0 if `ptr` is null.
///
/// # Safety
/// - `ptr` must be null or a pointer previously returned by `malloc`, `realloc`,
///   or `aligned_alloc` from this specific allocator instance and implementation.
///   Passing any other pointer (including pointers offset from the original user pointer)
///   leads to Undefined Behavior.
/// - The behavior is undefined if the metadata preceding `ptr` has been corrupted.
#[no_mangle]
pub unsafe extern "C" fn usable_size(ptr: *mut c_void) -> usize {
    // Standard behavior: usable_size(NULL) returns 0.
    let Some(user_ptr) = NonNull::new(ptr.cast::<u8>()) else {
        return 0;
    };

    // Check and retrieve the metadata.
    match get_metadata_ptr(user_ptr) {
        Ok(metadata_ptr) => metadata_ptr.as_ref().size,
        Err(_) => {
            report_error(HeapError::InvalidFree, ptr);
            0
        }
    }
}

/// Internal allocation function implementing the core logic for `malloc` and `aligned_alloc`.
/// Handles layout calculation, allocation via `talc`, metadata writing, and returns the aligned user pointer.
///
/// # Safety
/// `alignment` must be a power of two.
/// `size` should be non-zero for `aligned_alloc`, `malloc` handles size 0 separately.
#[inline]
unsafe fn allocate_internal(size: usize, alignment: usize) -> *mut c_void {
    // Calculate the layout required for the talc allocation and the offset to the user pointer.
    let (total_layout, user_data_offset) = match layout_for_allocation(size, alignment) {
        Ok(l) => l,
        Err(_) => return ptr::null_mut(), // Layout calculation failed (e.g., overflow, invalid size/align)
    };

    // Allocate the memory block using the calculated total layout.
    match ALLOCATOR.lock().malloc(total_layout) {
        Ok(alloc_ptr) => {
            // `alloc_ptr` points to the start of the talc allocation block.
            // Calculate the user pointer address based on the start and the offset.
            // Safety: alloc_ptr is valid and non-null, user_data_offset calculated correctly.
            let user_ptr = alloc_ptr.as_ptr().add(user_data_offset);

            // Calculate the metadata pointer address (immediately before user_ptr).
            // Safety: user_ptr_addr is valid, and layout_for_allocation ensures
            // user_data_offset >= METADATA_SIZE.
            let metadata_ptr = user_ptr.sub(METADATA_SIZE);
            let metadata_ptr = metadata_ptr.cast::<Metadata>();

            // Write the metadata (original requested size and alignment).
            // Safety: metadata_ptr points to valid, allocated space of METADATA_SIZE within the block.
            metadata_ptr.write(Metadata {
                checksum: calculate_checksum(size, alignment),
                size,
                alignment,
            });

            // Return the user pointer (correctly aligned).
            user_ptr as *mut c_void
        }
        Err(_) => ptr::null_mut(), // Allocation failed (OOM)
    }
}

/// Allocates memory with default alignment (`align_of::<usize>`).
/// Stores metadata (size, alignment) before the returned pointer.
///
/// # Safety
/// Caller is responsible for handling the returned pointer (e.g., checking for null)
/// and eventually freeing it with `free`.
#[no_mangle]
pub unsafe extern "C" fn malloc(size: usize) -> *mut c_void {
    if size == 0 {
        // Standard C malloc behavior for size 0 is implementation-defined.
        // Returning null is common and simple. Some implementations return a unique pointer
        // that can be passed to free. Returning null avoids allocating for zero size.
        return ptr::null_mut();
    }
    // Use the natural alignment of usize as the default alignment for malloc.
    allocate_internal(size, align_of::<usize>())
}

/// Allocates memory for an array of `nmemb` elements of `size` bytes each
/// and initializes all bytes in the allocated storage to zero.
///
/// # Safety
/// Caller is responsible for handling the returned pointer and freeing it.
/// Returns NULL on integer overflow or allocation failure.
#[no_mangle]
pub unsafe extern "C" fn calloc(nmemb: usize, size: usize) -> *mut c_void {
    // This is a critical security check. If nmemb * size overflows usize,
    // we must return null to prevent allocating a small buffer for a large request.
    let total_size = match nmemb.checked_mul(size) {
        Some(s) => s,
        None => return ptr::null_mut(),
    };

    // Check for zero-sized allocation.
    if total_size == 0 {
        return ptr::null_mut();
    }

    // Allocate using the internal allocator with default alignment.
    // Calloc typically uses the same alignment as malloc (align_of::<usize>).
    let ptr = allocate_internal(total_size, align_of::<usize>());

    // Zero-initialize the memory if allocation succeeded.
    if !ptr.is_null() {
        // ptr::write_bytes is Rust's equivalent of C's memset(ptr, 0, len).
        // It is highly optimized (usually compiles to efficient SIMD instructions).
        ptr::write_bytes(ptr.cast::<u8>(), 0, total_size);
    }

    ptr
}

/// Allocates memory with specified alignment.
/// Stores metadata (size, alignment) before the returned pointer.
///
/// # Safety
/// Caller is responsible for handling the returned pointer and eventually freeing it.
/// `alignment` must be a power of two.
/// `size` must be non-zero (as per C standard `aligned_alloc`).
/// Behavior is undefined if `size` is not a multiple of `alignment` (C standard requirement).
#[no_mangle]
pub unsafe extern "C" fn aligned_alloc(alignment: usize, size: usize) -> *mut c_void {
    // Check alignment validity (must be power of two).
    // `layout_for_allocation` relies on this for correct alignment calculation.
    if !alignment.is_power_of_two() {
        // Consider setting errno to EINVAL if integrating deeply with C stdlib.
        return ptr::null_mut();
    }

    // Check size validity (non-zero, multiple of alignment - C standard requirement).
    if size == 0 || !size.is_multiple_of(alignment) {
        // Consider setting errno to EINVAL.
        return ptr::null_mut();
    }

    // Delegate to the internal allocation function.
    allocate_internal(size, alignment)
}

/// Frees memory previously allocated by `malloc`, `realloc`, or `aligned_alloc`.
///
/// # Safety
/// - `ptr` must be null or a pointer previously returned by `malloc`, `realloc`, or `aligned_alloc`
///   from *this* allocator instance.
/// - Calling `free` multiple times on the same non-null pointer leads to double-free (UB).
/// - Using the pointer after `free` leads to use-after-free (UB).
#[no_mangle]
pub unsafe extern "C" fn free(ptr: *mut c_void) {
    // `free(null)` is defined as a no-op.
    let Some(user_ptr) = NonNull::new(ptr.cast::<u8>()) else {
        return;
    };

    // Recover the original allocation pointer and layout using the metadata stored before user_ptr.
    match recover_alloc_info(user_ptr) {
        Ok((start_ptr, original_layout, _)) => {
            // Before freeing, corrupt the checksum to help catch use-after-free
            // if the user tries to free it again.
            if let Ok(metadata_ptr) = get_metadata_ptr(user_ptr) {
                (*metadata_ptr.as_ptr()).checksum = 0; // Set to a non-checksum value
            }

            // Safety: `start_ptr` and `original_layout` must correspond to a previous
            // allocation made by this allocator via `allocate_internal` or `realloc`.
            // `recover_alloc_info` guarantees this if `user_ptr` was valid.
            ALLOCATOR.lock().free(start_ptr, original_layout);
        }
        Err(_) => {
            // Could not recover allocation info - indicates corruption or invalid pointer.
            // Report error but do not attempt to free.
            report_error(HeapError::InvalidFree, ptr);
        }
    }
}

/// Reallocates memory previously allocated by `malloc`, `realloc`, or `aligned_alloc`.
/// Attempts to use `talc`'s underlying realloc for efficiency, preserving the original alignment.
///
/// # Safety
/// - `ptr` must be null or a pointer previously returned by `malloc`, `realloc`, or `aligned_alloc`.
/// - `size` is the desired size for the new allocation.
/// - If reallocation fails, the original pointer `ptr` remains valid and must still be freed.
/// - If reallocation succeeds, the original `ptr` is invalidated and the new pointer should be used.
#[no_mangle]
pub unsafe extern "C" fn realloc(ptr: *mut c_void, size: usize) -> *mut c_void {
    // Handle null pointer: standard requires this is equivalent to malloc(size).
    let Some(user_ptr) = NonNull::new(ptr.cast::<u8>()) else {
        // Use default alignment consistent with malloc.
        return malloc(size);
    };

    // Handle size == 0: standard requires this is equivalent to free(ptr) and returns null.
    if size == 0 {
        free(ptr);
        return ptr::null_mut();
    }

    // --- Attempt Optimized Realloc ---

    // 1. Recover original allocation info (pointer returned by talc and its layout).
    let (old_ptr, old_layout, old_metadata) = match recover_alloc_info(user_ptr) {
        Ok(res) => res,
        Err(_) => {
            report_error(HeapError::InvalidFree, ptr);
            return ptr::null_mut();
        }
    };

    // 2. Extract original alignment from metadata.
    let alignment = old_metadata.alignment;

    // 3. Compare sizes and choose strategy.
    match size.cmp(&old_metadata.size) {
        core::cmp::Ordering::Equal => {
            // Sizes are the same, no operation needed.
            ptr
        }

        core::cmp::Ordering::Less => {
            // Calculate the new total layout for the smaller size.
            let (new_total_layout, _) = match layout_for_allocation(size, alignment) {
                Ok(l) => l,
                Err(_) => return ptr::null_mut(),
            };

            // Attempt to shrink in place.
            ALLOCATOR
                .lock()
                .shrink(old_ptr, old_layout, new_total_layout.size());

            // Successfully shrunk in place (or no-op if already small enough).
            let metadata_ptr = user_ptr.as_ptr().sub(METADATA_SIZE).cast::<Metadata>();

            // Safety: metadata_ptr is valid and points to the correct location.
            metadata_ptr.write(Metadata {
                checksum: calculate_checksum(size, alignment),
                size,
                alignment,
            });

            // Return original pointer
            ptr
        }

        core::cmp::Ordering::Greater => {
            // Calculate the required new total layout based on new user size and original alignment.
            let (new_total_layout, _) = match layout_for_allocation(size, alignment) {
                Ok(l) => l,
                Err(_) => return ptr::null_mut(), // New layout calculation failed.
            };

            // Attempt to grow in-place first.
            // We need to lock the allocator.
            let mut allocator_lock = ALLOCATOR.lock();

            // Safety: `old_ptr` and `old_layout` are valid.
            match allocator_lock.grow_in_place(old_ptr, old_layout, new_total_layout.size()) {
                Ok(_returned_ptr) => {
                    // Successfully grew in place! _returned_ptr should == old_ptr
                    // The memory block is larger, but user pointer hasn't moved.
                    // We MUST update the metadata *in place*.
                    drop(allocator_lock); // Release lock before writing metadata

                    // Update metadata with new size.
                    let metadata_ptr = user_ptr.as_ptr().sub(METADATA_SIZE).cast::<Metadata>();

                    // Safety: metadata_ptr is valid as pointer hasn't moved.
                    metadata_ptr.write(Metadata {
                        checksum: calculate_checksum(size, alignment),
                        size,
                        alignment,
                    });

                    // Return the original user pointer.
                    ptr
                }
                Err(_) => {
                    // grow_in_place failed, fall back to malloc-copy-free.
                    // Release the lock as allocate_internal will acquire it.
                    drop(allocator_lock);

                    // Allocate a completely new block.
                    let new_ptr_void = allocate_internal(size, alignment);
                    if new_ptr_void.is_null() {
                        // Allocation failed, original pointer `ptr` is still valid.
                        return ptr::null_mut();
                    }
                    let new_user_ptr = new_ptr_void.cast::<u8>();

                    // Copy data from the old user pointer area to the new user pointer area.
                    let copy_size = old_metadata.size;

                    // When growing, copy the original size.
                    // Safety: `ptr` and `new_user_ptr` are valid, non-overlapping.
                    ptr::copy_nonoverlapping(user_ptr.as_ptr(), new_user_ptr, copy_size);

                    // Free the *original* allocation block (using recovered ptr and layout).
                    // `free` will handle locking internally.
                    // NOTE: We use the original `ptr` here, which free can handle.
                    free(ptr);

                    // Return the pointer to the newly allocated and populated block.
                    new_ptr_void
                }
            }
        }
    }
}
