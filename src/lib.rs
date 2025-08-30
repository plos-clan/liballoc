#![no_std]
#![no_main]

use core::alloc::Layout;
use core::cmp;
use core::ffi::c_void;
use core::mem::{align_of, size_of};
use core::panic::PanicInfo;
use core::ptr::{self, NonNull};
use core::sync::atomic::{AtomicPtr, Ordering};
use talc::{ClaimOnOom, Span, Talc, Talck};

// Yes if panic just loops forever and you know died.
#[panic_handler]
unsafe fn panic(_info: &PanicInfo) -> ! {
    loop {}
}

// Define a type for our error handling function pointer.
// It will receive an enum indicating the error type and the invalid pointer.
#[repr(C)]
pub enum HeapError {
    InvalidFree, // Covers double free, invalid pointer, corrupted metadata
    LayoutError, // Internal layout calculation failed
}

/// Type alias for the error handler function pointer.
pub type ErrorHandler = unsafe extern "C" fn(error: HeapError, ptr: *mut c_void);

// A static variable to hold the user-provided error handler.
// Use AtomicPtr for safe, lock-free access (even if we only set it once).
static ERROR_HANDLER: AtomicPtr<c_void> = AtomicPtr::new(ptr::null_mut());

// A constant canary value to verify metadata integrity.
const CANARY_VALUE: usize = 0xDEC0DED;

/// Metadata struct stored *before* the user pointer's aligned position.
/// We store the requested size and the alignment used for the user pointer.
#[repr(C)] // Ensure predictable layout for size/alignment calculations
#[derive(Debug, Clone, Copy)]
struct Metadata {
    /// A canary value to help detect memory corruption.
    canary: usize,
    /// The size requested by the user for the allocation.
    size: usize,
    /// The alignment requested by the user (or default for malloc).
    alignment: usize,
}

// Use a spinlock mutex for thread safety in concurrent environments (if applicable)
static ALLOCATOR: Talck<spin::Mutex<()>, ClaimOnOom> =
    Talc::new(unsafe { ClaimOnOom::new(Span::empty()) }).lock();

// Size and alignment of the metadata header itself.
const METADATA_SIZE: usize = size_of::<Metadata>();
const METADATA_ALIGN: usize = align_of::<Metadata>();

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
    let handler_ptr = ERROR_HANDLER.load(Ordering::Relaxed);
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

    // Check the canary to verify integrity.
    if metadata.canary != CANARY_VALUE {
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

/// Initializes the heap memory arena.
///
/// # Safety
/// - `address` must be a valid pointer to the start of a memory block.
/// - `size` must be the correct size of that memory block.
/// - This function should only be called once.
/// - The memory block should ideally be aligned to at least `align_of::<Metadata>()`.
#[no_mangle]
pub unsafe extern "C" fn heap_init(address: *mut u8, size: usize) -> bool {
    // Ensure the address is not null.
    if address.is_null() {
        return false;
    }
    // Basic check: Ensure size is somewhat reasonable. Talc itself has minimal overhead.
    // We need enough space for at least one metadata block and minimal user data.
    if size < METADATA_SIZE * 2 {
        // Arbitrary minimal check
        return false; // Arena too small.
    }

    // Give the memory span to the allocator.
    let arena = Span::from_base_size(address, size);
    ALLOCATOR.lock().claim(arena).is_ok()
}

/// Sets a custom error handler function to be called on heap errors.
/// Passing `None` clears any previously set handler.
///
/// # Safety
/// - The `handler` function pointer must be valid and callable.
#[no_mangle]
pub unsafe extern "C" fn heap_onerror(handler: Option<ErrorHandler>) {
    let ptr = match handler {
        Some(h) => h as *mut c_void,
        None => ptr::null_mut(),
    };
    ERROR_HANDLER.store(ptr, Ordering::SeqCst);
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
                canary: CANARY_VALUE,
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
    if size == 0 || size % alignment != 0 {
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
            // Before freeing, corrupt the canary to help catch use-after-free
            // if the user tries to free it again.
            if let Ok(metadata_ptr) = get_metadata_ptr(user_ptr) {
                (*metadata_ptr.as_ptr()).canary = 0; // Set to a non-canary value
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
        return allocate_internal(size, align_of::<usize>());
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
            // --- Shrink ---
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
                canary: CANARY_VALUE,
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
                        canary: CANARY_VALUE, // <-- THE FIX
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
