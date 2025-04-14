#![no_std]
#![no_main]

use core::alloc::Layout;
use core::cmp;
use core::ffi::c_void;
use core::mem::{align_of, size_of};
use core::ptr::{self, NonNull};
use talc::{ClaimOnOom, Span, Talc, Talck};
use core::panic::PanicInfo;

// Yes if panic just loops forever and you know died.
#[panic_handler]
unsafe fn panic(_info: &PanicInfo) -> ! {
    loop {}
}

// Use a spinlock mutex for thread safety in concurrent environments (if applicable)
// or a simpler mutex/no mutex if targeting single-threaded `no_std`.
static ALLOCATOR: Talck<spin::Mutex<()>, ClaimOnOom> =
    Talc::new(unsafe { ClaimOnOom::new(Span::empty()) }).lock();

// We store the allocation size *just before* the pointer returned to the user.
const METADATA_SIZE: usize = size_of::<usize>();
// Use the alignment of the metadata itself for the layout calculations involving metadata.
const METADATA_ALIGN: usize = align_of::<usize>();

/// Unsafely gets the pointer to the metadata (the stored size) from the user pointer.
#[inline]
unsafe fn get_metadata_ptr(ptr: NonNull<u8>) -> NonNull<usize> {
    // User pointer points *after* the metadata.
    let metadata_ptr = ptr.as_ptr().cast::<usize>().sub(1);
    // Safety: Assumes the pointer was allocated by this allocator and has metadata prepended.
    NonNull::new_unchecked(metadata_ptr)
}

/// Unsafely gets the pointer to the start of the allocation (including metadata) from the user pointer.
#[inline]
unsafe fn get_allocation_start_ptr(ptr: NonNull<u8>) -> NonNull<u8> {
    // User pointer points *after* the metadata.
    let alloc_start_ptr = ptr.as_ptr().sub(METADATA_SIZE);
    // Safety: Assumes the pointer was allocated by this allocator and has metadata prepended.
    NonNull::new_unchecked(alloc_start_ptr)
}

/// Creates the layout for an allocation, including space for metadata.
/// The alignment must accommodate both the requested alignment and the metadata alignment.
#[inline]
fn create_layout_with_metadata(size: usize, alignment: usize) -> Result<Layout, ()> {
    let layout_align = cmp::max(alignment, METADATA_ALIGN);
    // Check for overflow before calling Layout::from_size_align
    let layout_size = size.checked_add(METADATA_SIZE).ok_or(())?;
    Layout::from_size_align(layout_size, layout_align).map_err(|_| ())
}

// == C API Implementation ==

/// Initializes the heap memory arena.
///
/// # Safety
/// - `address` must be a valid pointer to the start of a memory block.
/// - `size` must be the correct size of that memory block.
/// - This function should only be called once.
#[no_mangle]
pub unsafe extern "C" fn heap_init(address: *mut u8, size: usize) -> bool {
    // Ensure the address is not null before creating the span.
    if address.is_null() {
        return false;
    }
    let arena = Span::from_base_size(address, size);
    ALLOCATOR.lock().claim(arena).is_ok()
}

/// Allocates memory with default alignment.
/// Stores the requested size before the returned pointer.
///
/// # Safety
/// Caller is responsible for handling the returned pointer (e.g., checking for null)
/// and eventually freeing it with `free`.
#[no_mangle]
pub unsafe extern "C" fn malloc(size: usize) -> *mut c_void {
    if size == 0 {
        // Standard C malloc behavior for size 0 is implementation-defined.
        // Returning null is a common choice.
        return ptr::null_mut();
    }

    // Request layout with space for metadata, using default alignment (usize).
    let layout = match create_layout_with_metadata(size, METADATA_ALIGN) {
        Ok(l) => l,
        Err(_) => return ptr::null_mut(), // Layout calculation failed
    };

    match ALLOCATOR.lock().malloc(layout) {
        Ok(alloc_ptr) => {
            // Pointer returned by allocator is the start of the whole block (metadata + data).
            // Store the *requested* size in the metadata slot.
            let metadata_ptr = alloc_ptr.as_ptr().cast::<usize>();
            // Safety: alloc_ptr is valid and points to sufficient space.
            metadata_ptr.write(size);

            // Return the pointer *after* the metadata.
            let user_ptr = alloc_ptr.as_ptr().add(METADATA_SIZE);
            user_ptr as *mut c_void
        }
        Err(_) => ptr::null_mut(), // Allocation failed (OOM)
    }
}

/// Frees memory previously allocated by `malloc`, `realloc`, or `aligned_alloc`.
///
/// # Safety
/// - `ptr` must be null or a pointer previously returned by `malloc`, `realloc`, or `aligned_alloc`.
/// - Calling `free` multiple times on the same non-null pointer leads to double-free (UB).
/// - Using the pointer after `free` leads to use-after-free (UB).
#[no_mangle]
pub unsafe extern "C" fn free(ptr: *mut c_void) {
    // `free(null)` is a no-op.
    let Some(user_ptr) = NonNull::new(ptr.cast::<u8>()) else {
        return;
    };

    // Retrieve the stored size and calculate the original allocation pointer and layout.
    // Safety: Assumes `ptr` came from this allocator's functions.
    let metadata_ptr = get_metadata_ptr(user_ptr);
    let size = metadata_ptr.as_ptr().read();

    let alloc_start_ptr = get_allocation_start_ptr(user_ptr);

    // Reconstruct the layout used for the original allocation.
    let layout = match create_layout_with_metadata(size, METADATA_ALIGN) {
        Ok(l) => l,
        Err(_) => {
            // This should not happen if allocation succeeded, but handle defensively.
            // Perhaps log an error in a real scenario.
            return;
        }
    };

    // Safety: alloc_start_ptr and layout must correspond to a previous allocation.
    ALLOCATOR.lock().free(alloc_start_ptr, layout);
}

/// Reallocates memory previously allocated by `malloc`, `realloc`, or `aligned_alloc`.
/// Attempts to use `talc`'s underlying realloc for efficiency.
///
/// # Safety
/// - `ptr` must be null or a pointer previously returned by `malloc`, `realloc`, or `aligned_alloc`.
/// - `new_size` is the desired size for the new allocation.
/// - If reallocation fails, the original pointer `ptr` remains valid and must still be freed.
/// - If reallocation succeeds, the original `ptr` is invalidated.
#[no_mangle]
pub unsafe extern "C" fn realloc(ptr: *mut c_void, new_size: usize) -> *mut c_void {
    // Handle null pointer: equivalent to malloc(new_size).
    let Some(user_ptr_non_null) = NonNull::new(ptr.cast::<u8>()) else {
        return malloc(new_size);
    };
    let user_ptr = user_ptr_non_null.as_ptr(); // Back to *mut u8 for pointer math/copying

    // Handle new_size == 0: equivalent to free(ptr).
    if new_size == 0 {
        free(ptr);
        return ptr::null_mut();
    }

    // Retrieve old size from metadata.
    // Safety: Assumes `ptr` came from this allocator's functions.
    let old_size = get_metadata_ptr(user_ptr_non_null).as_ptr().read();

    // If size hasn't changed, do nothing (as per C standard).
    if new_size == old_size {
        return ptr;
    }

    // Allocate new block
    let new_ptr = malloc(new_size);
    if new_ptr.is_null() {
        // Allocation failed, original pointer is still valid.
        return ptr::null_mut();
    }

    // Copy data from old block to new block.
    let copy_size = cmp::min(old_size, new_size);
    // Safety: `ptr` and `new_ptr` are valid for reads/writes of `copy_size` bytes respectively,
    // and `malloc` guarantees they don't overlap if `new_ptr` is non-null.
    ptr::copy_nonoverlapping(user_ptr, new_ptr.cast::<u8>(), copy_size);

    // Free the old block.
    // Safety: `ptr` is valid and was allocated by this allocator.
    free(ptr);

    // Return the new block.
    new_ptr
}

/// Returns the usable size of the memory block pointed to by `ptr`.
///
/// In this specific implementation, this corresponds to the size originally requested
/// during the allocation (`malloc`, `aligned_alloc`, `realloc`). It does *not*
/// necessarily reflect the true size of the underlying block allocated by `talc`,
/// which might be larger due to internal padding or alignment.
///
/// Returns 0 if `ptr` is null.
///
/// # Safety
/// - `ptr` must be null or a pointer previously returned by `malloc`, `realloc`,
///   or `aligned_alloc` from this specific allocator instance and implementation.
///   Passing any other pointer (including pointers offset from the original)
///   leads to Undefined Behavior.
/// - The behavior is undefined if the metadata preceding `ptr` has been corrupted.
#[no_mangle]
pub unsafe extern "C" fn usable_size(ptr: *mut c_void) -> usize {
    // Standard behavior: usable_size(NULL) returns 0.
    let Some(user_ptr) = NonNull::new(ptr.cast::<u8>()) else {
        return 0;
    };

    // Retrieve the stored size from the metadata located just before the user pointer.
    // Safety: Assumes `ptr` is valid and points *after* our metadata header.
    // Relies on the metadata structure defined in `malloc`, `realloc`, `aligned_alloc`.
    get_metadata_ptr(user_ptr).as_ptr().read()
}


/// Allocates memory with specified alignment.
/// Stores metadata similarly to `malloc`.
///
/// NOTE: While this allocates with the requested alignment *for the block given to talc*,
/// the returned user pointer (offset by METADATA_SIZE) might not meet the requested
/// alignment if `alignment > align_of::<usize>()`.
///
/// # Safety
/// Caller is responsible for handling the returned pointer and eventually freeing it.
/// `alignment` must be a power of two.
/// `size` must be a multiple of `alignment` (as per C standard).
#[no_mangle]
pub unsafe extern "C" fn aligned_alloc(alignment: usize, size: usize) -> *mut c_void {
    // Check alignment validity (power of two)
    if alignment == 0 || alignment & (alignment - 1) != 0 {
        return ptr::null_mut();
    }

    // Check size validity (multiple of alignment, non-zero)
    if size == 0 || size % alignment != 0 {
        return ptr::null_mut();
    }

    // Request layout with space for metadata, using the specified alignment.
    let layout = match create_layout_with_metadata(size, alignment) {
        Ok(l) => l,
        Err(_) => return ptr::null_mut(), // Layout calculation failed
    };

    match ALLOCATOR.lock().malloc(layout) {
        Ok(alloc_ptr) => {
            // Store the *requested* size in the metadata slot.
            let metadata_ptr = alloc_ptr.as_ptr().cast::<usize>();
            // Safety: alloc_ptr is valid and points to sufficient space.
            metadata_ptr.write(size);

            // Return the pointer *after* the metadata.
            // WARNING: This might not be `alignment`-aligned if `alignment > METADATA_ALIGN`.
            let user_ptr = alloc_ptr.as_ptr().add(METADATA_SIZE);
            user_ptr as *mut c_void
        }
        Err(_) => ptr::null_mut(), // Allocation failed (OOM)
    }
}
